#!/bin/bash
# One-command LaTeX build for the JPathGene MICCAI/MULTITAB submission.
# Requires a TeX distribution (texlive) with pdflatex + bibtex.
#   bash build.sh           # build main.pdf
#   bash build.sh clean     # remove build artifacts
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "clean" ]; then
  rm -f main.aux main.bbl main.blg main.log main.out main.toc main.pdf \
        sections/*.aux 2>/dev/null || true
  echo "cleaned."; exit 0
fi

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "ERROR: pdflatex not found. Install TeX Live, e.g.:" >&2
  echo "  Ubuntu: sudo apt-get install texlive-full" >&2
  echo "  conda:  conda install -c conda-forge texlive-core" >&2
  exit 1
fi

echo "[1/4] pdflatex (pass 1)"; pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null
echo "[2/4] bibtex";          bibtex main >/dev/null || true
echo "[3/4] pdflatex (pass 2)"; pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null
echo "[4/4] pdflatex (pass 3)"; pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null

echo "------------------------------------------------------------"
echo "built main.pdf ($(grep -c '^' main.log 2>/dev/null) log lines)"
# surface any unresolved references / citations
grep -i "undefined" main.log | head -20 || true
grep -i "warning: Citation" main.log | head -10 || true
echo "done. open main.pdf"
