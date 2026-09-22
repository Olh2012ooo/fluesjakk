"""fly-chess: a chess engine whose neural network *is* the FlyWire fruit-fly connectome.

Public API (docs/SPEC.md §0 / §6)::

    import flychess
    flychess.train(run="fly1", stage="all", config="configs/default.yaml", max_steps=1000)
    model, graph = flychess.load_brain("fly1")          # run name, run dir or checkpoint path
    flychess.play("fly1", difficulty="fly", color="white")   # terminal UI; gui=True serves web/

Heavy modules (torch, rich, fastapi) are imported lazily so that ``import flychess`` stays cheap
and ``fly --help`` works without loading a GPU stack.

``flychess.train`` / ``flychess.play`` are also the names of the ``flychess.train`` and
``flychess.play`` sub-packages: importing either sub-package rebinds the attribute on this package
to the module (Python's import machinery does that). Both sub-packages therefore make their module
object *callable* with exactly the same signature (see their ``__init__``), so ``flychess.train(...)``
and ``flychess.play(...)`` work whether or not the sub-packages have been imported.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"

if TYPE_CHECKING:  # pragma: no cover - typing only
    from flychess.connectome.graph import BrainGraph
    from flychess.model.flybrain import FlyBrain


def _train(run: str | None = None, stage: str = "all", config: Any = None, resume: bool = False,
           **overrides: Any) -> Path:
    """Train a run (imitation and/or self-play); returns the path of ``runs/<run>/latest.pt``.

    ``config`` is a yaml path, a dict, a :class:`~flychess.train.config.TrainConfig` or ``None``
    (defaults); keyword ``overrides`` win (``max_steps=100``, ``**{'brain.steps': 4}``). See
    :func:`flychess.train.trainer.train`.
    """
    from flychess.train.trainer import train as _train

    return _train(run, stage=stage, config=config, resume=resume, **overrides)


def load_brain(path_or_run: str | Path, device: str | None = None) -> tuple[FlyBrain, BrainGraph]:
    """``(FlyBrain in eval mode, BrainGraph)`` from a run name, a run directory or a checkpoint path."""
    import torch

    from flychess.train.trainer import load_checkpoint

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, graph, _ = load_checkpoint(path_or_run, device=dev)
    return model, graph


def _play(run_or_ckpt: str | Path | None = None, difficulty: str = "fly", color: str = "white",
          gui: bool = False, device: str | None = None, port: int = 8000, **kwargs: Any) -> Any:
    """Play against the fly: in the terminal (default) or in the browser (``gui=True`` serves ``web/``).

    ``run_or_ckpt=None`` picks the most recently updated run under ``runs/`` that has a ``latest.pt``.
    """
    if gui:
        from flychess.play.local_web import serve_local

        return serve_local(run_or_ckpt if run_or_ckpt is not None else latest_run(), port=port, **kwargs)
    from flychess.play.terminal import play_terminal

    return play_terminal(run_or_ckpt, difficulty=difficulty, color=color, device=device, **kwargs)


def latest_run() -> str:
    """Name of the most recently updated run under ``runs/`` that has a ``latest.pt`` (FileNotFoundError if none)."""
    from flychess.play.terminal import _latest_run

    return _latest_run()


train = _train
play = _play

__all__ = ["__version__", "latest_run", "load_brain", "play", "train"]
