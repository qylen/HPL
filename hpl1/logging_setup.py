"""Structured logging with rotating file handlers."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def setup_logging(log_dir: str | Path = "logs", level: int = logging.INFO,
                  console: bool = True, name: str = "hpl1") -> logging.Logger:
    root = logging.getLogger(name)
    if getattr(root, "_hpl1_configured", False):
        return root
    root.setLevel(level)
    fmt = logging.Formatter(_FORMAT)
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
        root.addHandler(ch)
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(p / f"{name}.log", maxBytes=5 * 1024 * 1024,
                                              backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    root._hpl1_configured = True  # type: ignore[attr-defined]
    return root


def check_disk_space(path: str | Path = ".", min_free_gb: float = 1.0) -> dict:
    """Fail-closed disk-space check used before downloads and paper runs."""
    import shutil

    total, used, free = shutil.disk_usage(str(Path(path).resolve() or "."))
    ok = free / 1e9 >= min_free_gb
    return {"free_gb": round(free / 1e9, 2), "total_gb": round(total / 1e9, 2),
            "min_free_gb": min_free_gb, "ok": ok}
