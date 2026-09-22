"""Play against the fly: engine (§9 difficulties), terminal UI and the local website server.

The module object is callable: ``flychess.play(run, difficulty=..., color=..., gui=False)`` is the
public entry point (SPEC §0) and importing this sub-package rebinds ``flychess.play`` to the module,
so the module forwards calls to the implementation in ``flychess/__init__.py``.
"""
from __future__ import annotations

import sys
import types
from typing import Any

from .engine import DIFFICULTIES, FlyEngine, load_checkpoint


class _CallableModule(types.ModuleType):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from flychess import _play

        return _play(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule


def play_terminal(*args, **kwargs):  # lazy: keeps ``rich`` out of the import path of the trainer
    from .terminal import play_terminal as _play

    return _play(*args, **kwargs)


def serve_local(*args, **kwargs):
    from .local_web import serve_local as _serve

    return _serve(*args, **kwargs)


__all__ = ["DIFFICULTIES", "FlyEngine", "load_checkpoint", "play_terminal", "serve_local"]
