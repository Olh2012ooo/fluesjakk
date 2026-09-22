"""Orchestrates a training run: graph → FlyBrain → imitation → self-play, with checkpoints (docs/SPEC.md §6).

Checkpoint format (``runs/<run>/ckpt-<step>.pt`` and ``runs/<run>/latest.pt``)::

    {'config': TrainConfig dict, 'brain_config': BrainConfig dict, 'graph_path': str,
     'model': state_dict, 'optim': state_dict | None, 'step': int,
     'stage': 'imitation' | 'selfplay', 'elo': {opponent: elo_estimate}, 'saved_at': unix time}
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from flychess import paths
from flychess.connectome.graph import BrainGraph
from flychess.model.config import BrainConfig
from flychess.model.flybrain import FlyBrain
from flychess.train.config import STAGES, TrainConfig, load_config, resolve_graph_path
from flychess.train.metrics import MetricsLogger, read_metrics

LATEST = "latest.pt"


def cuda_mem_gb(peak: bool = True, reset: bool = False) -> float:
    """Peak (or current) CUDA memory allocated in GB; 0 without CUDA. ``reset`` clears the peak counter."""
    if not torch.cuda.is_available():
        return 0.0
    mem = torch.cuda.max_memory_allocated() if peak else torch.cuda.memory_allocated()
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return mem / 1e9


def checkpoint_path(run: str, step: int | None = None) -> Path:
    d = paths.run_dir(run)
    return d / LATEST if step is None else d / f"ckpt-{step}.pt"


def latest_elo(run_dir: str | Path, last_n: int = 100) -> dict[str, float]:
    """``{opponent: elo_estimate}`` from the most recent ``elo`` records of a run (empty if none)."""
    out: dict[str, float] = {}
    for rec in read_metrics(run_dir, last_n=last_n, kinds=["elo"]):  # chronological: later records win
        if rec.get("opponent") is not None and rec.get("elo_estimate") is not None:
            out[str(rec["opponent"])] = float(rec["elo_estimate"])
    return out


def save_checkpoint(
    run_dir: str | Path,
    model: FlyBrain,
    cfg: TrainConfig,
    graph_path: str | Path,
    step: int,
    stage: str,
    optim: torch.optim.Optimizer | dict | None = None,
    elo: dict[str, float] | None = None,
) -> Path:
    """Write ``ckpt-<step>.pt`` atomically and copy it to ``latest.pt``; returns the ``ckpt-<step>.pt`` path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    optim_state = optim.state_dict() if isinstance(optim, torch.optim.Optimizer) else optim
    ckpt = {
        "config": cfg.to_dict(),
        "brain_config": model.config.to_dict(),
        "graph_path": str(graph_path),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optim": optim_state,
        "step": int(step),
        "stage": str(stage),
        "elo": dict(elo if elo is not None else latest_elo(run_dir)),
        "saved_at": time.time(),
    }
    path = run_dir / f"ckpt-{int(step)}.pt"
    tmp = path.with_suffix(f".pt.tmp{os.getpid()}")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    latest_tmp = run_dir / f"{LATEST}.tmp{os.getpid()}"
    shutil.copyfile(path, latest_tmp)
    os.replace(latest_tmp, run_dir / LATEST)
    return path


def resolve_checkpoint(path_or_run: str | Path) -> Path:
    """A checkpoint file, a run directory (→ its ``latest.pt``) or a run name (→ ``runs/<run>/latest.pt``)."""
    p = Path(path_or_run).expanduser()
    if p.is_file():
        return p
    if p.is_dir() and (p / LATEST).exists():
        return p / LATEST
    run_latest = paths.run_dir(str(path_or_run)) / LATEST
    if run_latest.exists():
        return run_latest
    raise FileNotFoundError(f"no checkpoint at {path_or_run!r} (tried the path, <dir>/{LATEST} and {run_latest})")


def load_checkpoint(path_or_run: str | Path, device: str | torch.device = "cpu",
                    graph: BrainGraph | None = None) -> tuple[FlyBrain, BrainGraph, dict[str, Any]]:
    """Load a checkpoint → ``(model on device (eval mode), graph, ckpt dict)``.

    The graph is loaded from ``ckpt['graph_path']`` (falling back to the config's ``graph`` name if that
    file moved); pass ``graph`` to reuse one already in memory.
    """
    path = resolve_checkpoint(path_or_run)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if graph is None:
        gp = Path(ckpt.get("graph_path", ""))
        if not gp.exists():
            cfg_graph = (ckpt.get("config") or {}).get("graph")
            alt = resolve_graph_path(cfg_graph) if cfg_graph else None
            if alt is None or not alt.exists():
                raise FileNotFoundError(f"graph {gp} of checkpoint {path} not found")
            gp = alt
        graph = BrainGraph.load(gp)
    bcfg = BrainConfig.from_dict({**ckpt["brain_config"], "graph_path": str(ckpt.get("graph_path", ""))})
    model = FlyBrain.from_checkpoint(ckpt["model"], bcfg, graph).to(device)
    model.eval()
    ckpt["path"] = str(path)
    return model, graph, ckpt


def build_model(cfg: TrainConfig, graph: BrainGraph | None = None,
                device: str | torch.device | None = None) -> tuple[FlyBrain, BrainGraph, Path]:
    """Graph (from ``cfg.graph``) + freshly initialised FlyBrain on ``device``."""
    gp = resolve_graph_path(cfg)
    if graph is None:
        if not gp.exists():
            raise FileNotFoundError(f"brain graph {gp} not found (run `fly build-brain` or set graph=...)")
        graph = BrainGraph.load(gp)
    model = FlyBrain(graph, cfg.brain_config(gp)).to(device or cfg.device)
    model.grad_checkpoint = bool(cfg.grad_checkpoint)
    return model, graph, gp


def calibrate_model(model: FlyBrain, cfg: TrainConfig, device: torch.device | str, batch: int = 256) -> dict[str, float]:
    """Run :meth:`FlyBrain.calibrate_gains` on a batch of real training positions (data-dependent init)."""
    from flychess.data.shards import ShardDataset, collate, load_split

    train_files, _ = load_split(cfg.shards_dir, cfg.val_fraction, cfg.seed, cfg.shard_name)
    it = iter(ShardDataset(train_files[:4], seed=cfg.seed + 7))
    x = collate([next(it) for _ in range(batch)])["planes"].to(device)
    return model.calibrate_gains(x)


# num_workers / seed: the imitation stage replays the loader past the already-trained batches of the epoch on
# resume, and the batch sequence is only identical for the same worker count and seed
RESUME_DRIFT_FIELDS = ("graph", "brain", "shards_dir", "shard_name", "batch_size", "lr", "weight_decay",
                       "warmup_steps", "epochs", "max_steps", "val_fraction", "num_workers", "seed")


def _warn_config_drift(ckpt_cfg: TrainConfig, cfg: TrainConfig) -> list[str]:
    """Print (and return) the schedule-relevant fields on which ``cfg`` differs from the checkpoint's config."""
    a, b = ckpt_cfg.to_dict(), cfg.to_dict()
    drift = [f for f in RESUME_DRIFT_FIELDS if a.get(f) != b.get(f)]
    for f in drift:
        print(f"[train] resume: config.{f} differs from the checkpoint ({a.get(f)!r} -> {b.get(f)!r})")
    return drift


def train(
    run: str | None = None,
    stage: str = "all",
    config: TrainConfig | str | Path | dict | None = None,
    resume: bool = False,
    **overrides: Any,
) -> Path:
    """Run the imitation and/or self-play stage for ``run``; returns the path of ``latest.pt``.

    ``config`` may be a yaml path, a dict, a TrainConfig or None (defaults); ``overrides`` are applied
    on top (``max_steps=100``, ``brain.steps=4`` via ``**{'brain.steps': 4}``). ``resume=True`` continues
    from ``runs/<run>/latest.pt`` (model, optimizer, step counter). Ctrl-C saves a checkpoint and returns.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    # On resume the checkpoint's own config is the base (so `train(run, resume=True)` continues with the
    # graph / batch / lr / shards the run was created with); an explicit config or overrides win over it.
    resume_ckpt: dict[str, Any] | None = None
    if resume:
        run_name = run if run is not None else load_config(config, **overrides).run
        latest_existing = paths.run_dir(run_name) / LATEST
        if latest_existing.exists():
            resume_ckpt = torch.load(latest_existing, map_location="cpu", weights_only=False)
    if resume_ckpt is not None and config is None:
        cfg = load_config(TrainConfig.from_dict(resume_ckpt["config"]), **overrides)
    else:
        cfg = load_config(config, **overrides)
    if run is not None:
        cfg = cfg.replace(run=run)
    if resume_ckpt is not None:
        _warn_config_drift(TrainConfig.from_dict(resume_ckpt["config"]), cfg)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    run_dir = paths.ensure(cfg.run_dir())
    ckpt: dict[str, Any] | None = None
    latest = run_dir / LATEST
    if resume_ckpt is not None:
        model, graph, ckpt = load_checkpoint(latest, device=device)
        model.train()
        model.grad_checkpoint = bool(cfg.grad_checkpoint)
        graph_path = Path(ckpt["graph_path"])
        step = int(ckpt["step"])
    else:
        if resume:
            print(f"[train] resume requested but {latest} does not exist: starting from scratch")
        model, graph, graph_path = build_model(cfg, device=device)
        step = 0
        if getattr(model, "homeostatic", False):
            stats = calibrate_model(model, cfg, device)
            print(f"[train] homeostatic gain calibration: {stats}")

    logger = MetricsLogger(run_dir)
    logger.write_run_json(
        cfg.to_dict(), graph.meta,
        extra={"graph_path": str(graph_path), "stage": stage, "device": str(device),
               "parameters": model.count_parameters(), "resumed_from_step": step if ckpt else None},
    )

    def save(step_: int, stage_: str, optim: torch.optim.Optimizer | dict | None = None) -> Path:
        return save_checkpoint(run_dir, model, cfg, graph_path, step_, stage_, optim=optim)

    print(f"[train] run={cfg.run} stage={stage} device={device} step={step} {graph.summary()}")
    print(f"[train] parameters: {model.count_parameters()}")
    try:
        done_imitation = ckpt is not None and ckpt.get("stage") == "selfplay"
        if stage in ("imitation", "all") and not done_imitation:
            from flychess.train.imitation import run_imitation_stage

            optim_state = ckpt.get("optim") if (ckpt and ckpt.get("stage") == "imitation") else None
            step = run_imitation_stage(model, graph, cfg, logger, step, save, device, optim_state=optim_state)
        if stage in ("selfplay", "all"):
            try:
                from flychess.train.selfplay import run_selfplay_stage
            except ImportError as e:  # module owned by the self-play agent; may not exist yet
                raise RuntimeError(
                    "the self-play stage (flychess.train.selfplay.run_selfplay_stage) is not available: "
                    f"{e}. Run with stage='imitation' or install the self-play module."
                ) from e
            step = run_selfplay_stage(model, graph, cfg, logger, step, save, device)
    except KeyboardInterrupt:
        print(f"[train] interrupted at step {step}; latest checkpoint: {latest}")
        logger.log_status(step, message="interrupted", stage=stage, total_steps=step, eta_s=None)
    finally:
        logger.close()
    if not latest.exists():  # e.g. epochs=0: still leave a loadable checkpoint behind
        save(step, "imitation" if stage != "selfplay" else "selfplay")
    return latest


__all__ = [
    "LATEST",
    "RESUME_DRIFT_FIELDS",
    "build_model",
    "checkpoint_path",
    "cuda_mem_gb",
    "latest_elo",
    "load_checkpoint",
    "resolve_checkpoint",
    "save_checkpoint",
    "train",
]
