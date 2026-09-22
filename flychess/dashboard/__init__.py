"""Live training dashboard: FastAPI + websocket tailing ``runs/<run>/metrics.jsonl`` (docs/SPEC.md §10)."""
from flychess.dashboard.server import create_app, run_dashboard, serve

__all__ = ["create_app", "run_dashboard", "serve"]
