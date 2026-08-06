"""Lightweight stdout + CSV logging (no external deps)."""
import csv
import os
import sys
import time
from typing import Dict, List, Optional


class Logger:
    def __init__(self, name: str = "pathgene", logfile: Optional[str] = None):
        self.name = name
        self.logfile = logfile
        if logfile:
            os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)
        self._t0 = time.time()

    def info(self, msg: str):
        line = f"[{self.name} +{time.time() - self._t0:7.1f}s] {msg}"
        print(line, flush=True)
        if self.logfile:
            with open(self.logfile, "a") as f:
                f.write(line + "\n")


class MetricCSVLogger:
    """Appends one row per epoch to a CSV; header is written on first row."""

    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._wrote_header = os.path.exists(path) and os.path.getsize(path) > 0

    def log(self, row: Dict):
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._wrote_header:
                w.writeheader()
                self._wrote_header = True
            w.writerow({k: row.get(k, "") for k in self.fieldnames})


def banner(msg: str):
    print("=" * 70, flush=True)
    print(msg, flush=True)
    print("=" * 70, flush=True)
