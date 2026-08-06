"""Conditional latent-diffusion predictor for Cross-Modal Diffusion-JEPA.

Motivation
----------
A deterministic JEPA predictor maps a context latent to a single target latent
and therefore collapses an inherently *one-to-many* mapping (one tissue
phenotype is compatible with a distribution of genetic programs) to its mean.
We instead model the full conditional distribution p(t | c) of the target latent
``t`` given the context latent ``c`` with a small conditional DDPM operating
**in latent space** (no raw-gene decoding). The denoising score-matching loss is
the JEPA latent-prediction objective generalized to a distribution; the standard
deterministic JEPA cosine/MSE predictor is recovered as the single-step limit.

The predictor exposes:
  * ``loss(x0, cond)``           -- denoising objective (the JEPA loss term)
  * ``sample_stats(cond, n, k)`` -- posterior mean + per-patient uncertainty
                                    (std across ``n`` DDIM samples)
This gives a principled cross-modal *uncertainty* used to gate fusion.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alpha_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine noise schedule -> alphas_cumprod (T,)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    return ac[1:].clamp(1e-5, 0.9999)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: (B,) float
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / max(1, half - 1))
        ang = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class FiLMResBlock(nn.Module):
    """Residual MLP block modulated (FiLM) by a time+condition embedding."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)
        self.act = nn.SiLU()

    def forward(self, h, ctx):
        scale, shift = self.film(ctx).chunk(2, dim=-1)
        x = self.norm(h) * (1 + scale) + shift
        x = self.fc2(self.act(self.fc1(self.act(x))))
        return h + x


class ConditionalDenoiser(nn.Module):
    """Predicts the noise eps added to a target latent, conditioned on (t, cond)."""

    def __init__(self, dim: int, cond_dim: int, hidden: int = 256,
                 time_dim: int = 128, depth: int = 3):
        super().__init__()
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cond = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU())
        self.in_proj = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList(
            [FiLMResBlock(hidden, hidden) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x_t, t, cond):
        ctx = self.time(t) + self.cond(cond)
        h = self.in_proj(x_t)
        for blk in self.blocks:
            h = blk(h, ctx)
        return self.out(h)


class ConditionalLatentDiffusion(nn.Module):
    def __init__(self, dim: int, cond_dim: int, timesteps: int = 1000,
                 sample_steps: int = 8, hidden: int = 256, depth: int = 3,
                 x0_clamp: float = 3.0):
        super().__init__()
        self.dim = dim
        self.timesteps = timesteps
        self.sample_steps = sample_steps
        # Latent targets are (Layer)normalised / whitened to ~unit scale, so the
        # x0 estimate is clamped to a few standard deviations. This is essential:
        # at high noise levels x0 = (x - sqrt(1-ab) eps)/sqrt(ab) divides by a
        # tiny sqrt(ab) and otherwise explodes with few-step DDIM sampling.
        self.x0_clamp = x0_clamp
        self.denoiser = ConditionalDenoiser(dim, cond_dim, hidden, depth=depth)
        ab = cosine_alpha_bar(timesteps)
        self.register_buffer("alpha_bar", ab)

    # ---- training objective (= the JEPA latent-prediction loss) ----
    def loss(self, x0, cond):
        B = x0.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        ab = self.alpha_bar[t].unsqueeze(-1)
        noise = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        eps = self.denoiser(x_t, t.float(), cond)
        return F.mse_loss(eps, noise)

    # ---- DDIM sampling (deterministic given the start noise) ----
    def _ddim(self, cond, steps, generator=None):
        B = cond.size(0)
        dev = cond.device
        x = torch.randn(B, self.dim, device=dev, generator=generator)
        ts = torch.linspace(self.timesteps - 1, 0, steps, device=dev).long()
        x0 = x
        for i in range(steps):
            t = ts[i]
            ab_t = self.alpha_bar[t]
            eps = self.denoiser(x, t.float().expand(B), cond)
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if self.x0_clamp is not None:
                x0 = x0.clamp(-self.x0_clamp, self.x0_clamp)
            t_prev = ts[i + 1] if i + 1 < steps else torch.tensor(0, device=dev)
            ab_prev = self.alpha_bar[t_prev] if i + 1 < steps else \
                torch.tensor(1.0, device=dev)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x0

    @torch.no_grad()
    def sample_stats(self, cond, n_samples: int = 8, steps: int = None):
        """Return (posterior_mean (B,dim), uncertainty (B,)).
        Uncertainty is the mean over dims of the per-dimension std across samples.
        """
        steps = steps or self.sample_steps
        samples = torch.stack([self._ddim(cond, steps) for _ in range(n_samples)],
                              dim=0)                       # (n, B, dim)
        mean = samples.mean(dim=0)                          # (B, dim)
        unc = samples.std(dim=0).mean(dim=-1)               # (B,)
        return mean, unc

    @torch.no_grad()
    def sample(self, cond, steps: int = None):
        return self._ddim(cond, steps or self.sample_steps)
