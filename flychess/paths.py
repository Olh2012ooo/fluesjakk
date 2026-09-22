"""Filesystem locations. Everything lives under FLYCHESS_HOME (default: the repo root)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOME = Path(os.environ.get("FLYCHESS_HOME", REPO_ROOT))

DATA_DIR = HOME / "data"
CONNECTOME_DIR = DATA_DIR / "connectome"
PGN_DIR = DATA_DIR / "pgn"
SHARDS_DIR = DATA_DIR / "shards"
BRAIN_DIR = DATA_DIR / "brain"
RUNS_DIR = HOME / "runs"
WEB_DIR = REPO_ROOT / "web"
WEB_MODEL_DIR = WEB_DIR / "model"
VECTORS_DIR = REPO_ROOT / "tests" / "vectors"


def run_dir(run: str) -> Path:
    return RUNS_DIR / run


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
