"""Survival metrics (optional). Classification is the primary task; these are
provided for cohorts that carry survival_time / survival_event columns."""
import numpy as np


def concordance_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C-index. `risk` is higher-is-riskier (shorter survival)."""
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    n = len(time)
    num = 0.0
    den = 0.0
    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if time[j] > time[i]:
                den += 1
                if risk[i] > risk[j]:
                    num += 1.0
                elif risk[i] == risk[j]:
                    num += 0.5
    return float(num / den) if den > 0 else float("nan")
