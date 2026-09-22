"""Training: metrics logging (§7), imitation learning, MCTS, self-play and the orchestrating trainer (§6).

The module object is callable: ``flychess.train(run, stage=..., config=..., **overrides)`` is the
public training entry point (SPEC §6) and importing this sub-package rebinds ``flychess.train`` to
the module, so the module forwards calls to :func:`flychess.train.trainer.train` (see ``flychess/__init__.py``).
"""
from __future__ import annotations

import sys
import types
from typing import Any


class _CallableModule(types.ModuleType):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from flychess import _train

        return _train(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
