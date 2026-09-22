"""Stage 1 — imitation learning on human (Lichess) positions (docs/SPEC.md §6).

Loss: plain cross-entropy of the human move over *all* 4168 move logits + ``value_loss_weight`` × MSE of
the value head against the game result.  The shards (§5) do not store legal-move masks, so — unlike
MCTS / self-play, which mask illegal moves — imitation trains the unmasked softmax; the brain therefore
also learns to put mass only on legal moves, and every eval/play path applies the legal mask on top.

Optimiser: AdamW with weight decay on the dense matrices (``w_in``, heads) and none on the 1-D
per-neuron parameters (``bias``, ``leak_logit``, ``b_in``, head biases) nor — under Dale's law — on
``syn_gain``: there ``|w| = softplus(syn_gain)``, so decaying the *logit* toward 0 would pull every synapse
toward ln 2 ≈ 0.69 (an order of magnitude above the initial magnitudes), i.e. toward a stronger, not sparser, connectome.  With
``dale=False`` ``w = syn_gain`` itself and decay is a genuine shrinkage, so it stays on.  Linear warm-up
then cosine decay to 10 % of ``lr`` over the planned number of steps.  bf16 autocast covers the dense parts only (the SpMM
in :class:`FlyBrain` disables autocast internally and runs in fp32).

Resuming continues the global step counter and the LR schedule, and the data stream continues where the
checkpoint stopped: the epoch ``start_step // steps_per_epoch`` is re-iterated (the shard permutation and the
in-shard shuffles are seeded by ``(seed, epoch, worker)``, so the loader reproduces the same batch sequence)
and its first ``start_step % steps_per_epoch`` batches are skipped without training.  Without the skip a
resumed one-epoch run would train twice on the prefix and never see the tail of the data.  The batch
sequence is only reproducible for the same ``num_workers`` (shards are split across workers), so resuming
with a different worker count skips the right *amount* of data but not exactly the same batches.
"""
from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from flychess.connectome.graph import BrainGraph
from flychess.data.shards import VAL_SUFFIX, ShardDataset, collate, count_positions, load_split, shard_series
from flychess.model.flybrain import FlyBrain, metrics_to_float, policy_value_loss
from flychess.train.config import TrainConfig
from flychess.train.metrics import MetricsLogger

STAGE = "imitation"
ACTIVITY_NEURONS = 2048
ELO_MAX_PLIES = 200
NO_DECAY_NAMES = ("bias", "leak_logit", "b_in", "b_ret", "central_b", "log_gain")
ADAM_BETAS = (0.9, 0.95)


# ---------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------
def param_groups(model: FlyBrain, weight_decay: float,
                 decay_syn_gain: bool | None = None) -> list[dict[str, Any]]:
    """Two AdamW groups: decayed dense matrices and un-decayed per-neuron biases & leaks.

    ``syn_gain`` joins the un-decayed group when the model enforces Dale's law (``|w| = softplus(gain)``:
    shrinking the logit would push every synapse toward softplus(0) = ln 2, not toward 0) and the decayed
    group otherwise (``w = gain``).  ``decay_syn_gain`` overrides that choice; ``True`` reproduces the
    layout of checkpoints written before this rule (see :func:`adapt_optim_state`).
    """
    if decay_syn_gain is None:
        decay_syn_gain = not bool(getattr(model, "dale", True))
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        leaf = name.split(".")[-1]
        if leaf in NO_DECAY_NAMES or (leaf == "bias" and p.dim() == 1) or (leaf == "syn_gain" and not decay_syn_gain):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _legacy_layouts(model: FlyBrain, weight_decay: float) -> list[list[dict[str, Any]]]:
    """Param-group layouts older checkpoints were saved with (``syn_gain`` decayed; self-play's single group)."""
    return [param_groups(model, weight_decay, decay_syn_gain=True),
            [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": weight_decay}]]


def adapt_optim_state(state: dict[str, Any], model: FlyBrain, weight_decay: float,
                      groups: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Re-partition a checkpoint's AdamW ``param_groups`` into the current :func:`param_groups` layout.

    ``Optimizer.load_state_dict`` pairs saved and live parameters positionally and refuses a state whose
    group sizes differ, so a checkpoint written when ``syn_gain`` was still decayed (or by the old
    single-group self-play optimizer) could not be resumed after the layout changed.  When the saved
    layout matches one of the known legacy layouts, the per-parameter moments are kept and only the
    group membership (and ``weight_decay``) is rewritten; anything else is returned unchanged.
    """
    groups = param_groups(model, weight_decay) if groups is None else groups
    saved_groups = state.get("param_groups")
    if not isinstance(saved_groups, list) or not saved_groups:
        return state
    saved_sizes = [len(g["params"]) for g in saved_groups]
    if saved_sizes == [len(g["params"]) for g in groups]:
        return state
    for legacy in _legacy_layouts(model, weight_decay):
        if [len(g["params"]) for g in legacy] != saved_sizes:
            continue
        saved_index = {id(p): i for g, sg in zip(legacy, saved_groups, strict=True)
                       for p, i in zip(g["params"], sg["params"], strict=True)}
        if any(id(p) not in saved_index for g in groups for p in g["params"]):
            continue
        hparams = {k: v for k, v in saved_groups[0].items() if k != "params"}
        new_groups = [dict(hparams, params=[saved_index[id(p)] for p in g["params"]], weight_decay=g["weight_decay"])
                      for g in groups]
        return dict(state, param_groups=new_groups)
    return state


def make_optimizer(model: FlyBrain, cfg: TrainConfig, lr: float | None = None) -> torch.optim.AdamW:
    return torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=lr if lr is not None else cfg.lr,
                             betas=ADAM_BETAS, fused=next(model.parameters()).is_cuda)


def lr_at(step: int, base_lr: float, warmup_steps: int, total_steps: int, final_ratio: float = 0.1) -> float:
    """Linear warm-up over ``warmup_steps`` then cosine decay to ``final_ratio * base_lr`` at ``total_steps``."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    span = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / span, 0.0), 1.0)
    return base_lr * (final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def set_lr(optim: torch.optim.Optimizer, lr: float) -> None:
    for g in optim.param_groups:
        g["lr"] = lr


def restore_group_hparams(optim: torch.optim.Optimizer, model: FlyBrain, cfg: TrainConfig) -> None:
    """Re-apply the *current* config's per-group hyper-parameters after ``optim.load_state_dict``.

    ``load_state_dict`` restores the checkpoint's ``weight_decay`` / ``betas`` into every param group, which
    would silently override a changed ``cfg.weight_decay`` on resume (``lr`` is re-set every step anyway).
    """
    for g, fresh in zip(optim.param_groups, param_groups(model, cfg.weight_decay), strict=True):
        g["weight_decay"] = fresh["weight_decay"]
        g["betas"] = ADAM_BETAS


def make_loader(files: list[Path], cfg: TrainConfig, device: torch.device, shuffle: bool = True,
                num_workers: int | None = None, batch_size: int | None = None) -> tuple[ShardDataset, DataLoader]:
    ds = ShardDataset(files, seed=cfg.seed, shuffle=shuffle)
    workers = cfg.num_workers if num_workers is None else num_workers
    workers = min(workers, len(files))  # a worker without a shard would only idle
    kwargs: dict[str, Any] = {}
    if workers > 0:
        kwargs.update(prefetch_factor=4, persistent_workers=False)
    loader = DataLoader(
        ds, batch_size=batch_size or cfg.batch_size, collate_fn=collate, num_workers=workers, drop_last=True,
        pin_memory=device.type == "cuda", **kwargs,
    )
    return ds, loader


def _to_device(batch: dict[str, Tensor], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    nb = device.type == "cuda"
    return (batch["planes"].to(device, non_blocking=nb), batch["move"].to(device, non_blocking=nb),
            batch["value"].to(device, non_blocking=nb))


def planned_steps(cfg: TrainConfig, n_train: int) -> tuple[int, int]:
    """``(total_steps, steps_per_epoch)``: ``epochs × steps_per_epoch`` capped by ``max_steps``.

    ``epochs == 0`` with ``max_steps`` set means "cycle the data until max_steps".
    """
    steps_per_epoch = max(n_train // cfg.batch_size, 1)
    if cfg.epochs == 0:
        total = cfg.max_steps or 0
    else:
        total = cfg.epochs * steps_per_epoch
        if cfg.max_steps is not None:
            total = min(total, cfg.max_steps)
    return int(total), int(steps_per_epoch)


def _call_save(save_checkpoint: Callable[..., Path], step: int, stage: str, optim: torch.optim.Optimizer) -> Path:
    """Pass the optimizer to ``save_checkpoint`` when its signature accepts it (shared contract: 2 args)."""
    try:
        accepts_optim = "optim" in inspect.signature(save_checkpoint).parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(save_checkpoint).parameters.values()
        )
    except (TypeError, ValueError):
        accepts_optim = False
    return save_checkpoint(step, stage, optim=optim) if accepts_optim else save_checkpoint(step, stage)


class ValCache:
    """The first ``n_batches`` validation batches, held in (pinned) host memory for repeatable evals."""

    def __init__(self, files: list[Path], cfg: TrainConfig, device: torch.device, n_batches: int) -> None:
        _, loader = make_loader(files, cfg, device, shuffle=False, num_workers=0)
        self.batches: list[dict[str, Tensor]] = []
        for batch in loader:
            if device.type == "cuda":
                batch = {k: v.pin_memory() for k, v in batch.items()}
            self.batches.append(batch)
            if len(self.batches) >= n_batches:
                break
        if not self.batches:
            raise RuntimeError(f"validation shards {files} hold fewer than {cfg.batch_size} positions")
        self.positions = sum(int(b["move"].shape[0]) for b in self.batches)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        return iter(self.batches)


@torch.no_grad()
def evaluate(model: FlyBrain, val: ValCache, cfg: TrainConfig, device: torch.device,
             activity_idx: Tensor | None = None) -> tuple[dict[str, float], np.ndarray | None]:
    """Mean loss / accuracy over the cached val batches, plus mean activity of ``activity_idx`` (first batch)."""
    was_training = model.training
    model.eval()
    sums: dict[str, Tensor] = {}
    activity = None
    try:
        for i, batch in enumerate(val):
            x, move, value_t = _to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"):
                if i == 0 and activity_idx is not None:
                    logits, value, h = model(x, return_activity=True)
                    activity = h.float().index_select(1, activity_idx).mean(0).cpu().numpy()
                else:
                    logits, value = model(x)[:2]
            _, m = policy_value_loss(logits, value, move, value_t, None, cfg.value_loss_weight)
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + v
        n = len(val.batches)
        means = metrics_to_float({k: v / n for k, v in sums.items()})
    finally:
        model.train(was_training)
    return means, activity


def quick_elo(model: FlyBrain, cfg: TrainConfig, device: torch.device, logger: MetricsLogger, step: int,
              games: int | None = None) -> dict[str, float]:
    """Fly (argmax policy) vs RandomPlayer and GreedyMaterialPlayer; logs ``elo`` records and one sample game."""
    from flychess.eval.elo import play_match
    from flychess.eval.opponents import BrainPolicyPlayer, GreedyMaterialPlayer, RandomPlayer

    games = cfg.elo_games if games is None else games
    fly = BrainPolicyPlayer(model, device=device, temperature=0.0, seed=cfg.seed + step, amp=cfg.amp)
    out: dict[str, float] = {}
    logged_game = False
    for opp in (RandomPlayer(seed=cfg.seed), GreedyMaterialPlayer(seed=cfg.seed)):
        t0 = time.perf_counter()
        m = play_match(fly, opp, games, max_plies=ELO_MAX_PLIES, seed=cfg.seed + step, keep_pgns=1)
        out[opp.name] = m.elo
        logger.log_elo(step, opponent=opp.name, games=m.games, wins=m.wins, draws=m.draws, losses=m.losses,
                       elo_estimate=m.elo, truncated=m.truncated, mean_plies=float(np.mean(m.plies)),
                       seconds=time.perf_counter() - t0, stage=STAGE)
        if not logged_game and m.pgns:
            logged_game = logger.log_game(step, pgn=m.pgns[0], result=m.results[0], moves=m.plies[0],
                                          source="eval", opponent=opp.name, fly_white=True)
        print(f"[imitation] step {step}: {m}")
    return out


# ---------------------------------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------------------------------
def run_imitation_stage(
    model: FlyBrain,
    graph: BrainGraph,
    cfg: TrainConfig,
    logger: MetricsLogger,
    start_step: int,
    save_checkpoint: Callable[..., Path],
    device: str | torch.device,
    optim_state: dict[str, Any] | None = None,
) -> int:
    """Train ``model`` on the shards under ``cfg.shards_dir``; returns the final global step.

    ``save_checkpoint(step, 'imitation'[, optim=...])`` is called every ``checkpoint_every`` steps, at the
    end and on ``KeyboardInterrupt`` (which is re-raised after saving).
    """
    device = torch.device(device)
    model.to(device).train()
    torch.manual_seed(cfg.seed + start_step)
    use_amp = bool(cfg.amp and device.type == "cuda")

    # ---- data ----
    train_files, val_files = load_split(cfg.shards_dir, cfg.val_fraction, cfg.seed, cfg.shard_name)
    if not val_files:
        print("[imitation] no validation shard (val_fraction=0 or a single shard): evaluating on a train shard")
        val_files = train_files[:1]
    elif (shard_series(val_files[0]) or "").endswith(VAL_SUFFIX):
        print(f"[imitation] validation set: {len(val_files)} game-level hold-out shard(s) "
              f"({shard_series(val_files[0])}-*.npz, disjoint from training at the game level)")
    else:
        print(f"[imitation] validation set: {len(val_files)} held-out train shard(s) — NOT game-disjoint "
              "(positions of one game sit on both sides); rebuild shards with `fly build-shards --val-every K`")
    n_train = count_positions(train_files)
    total_steps, steps_per_epoch = planned_steps(cfg, n_train)
    step = int(start_step)
    if step >= total_steps:
        logger.log_status(step, message=f"imitation already complete ({step}/{total_steps} steps)", stage=STAGE,
                          total_steps=total_steps, eta_s=0.0)
        return step
    ds, loader = make_loader(train_files, cfg, device)
    val = ValCache(val_files, cfg, device, cfg.eval_batches)
    rng = np.random.default_rng(cfg.seed)
    activity_idx_np = np.sort(rng.choice(graph.n, size=min(ACTIVITY_NEURONS, graph.n), replace=False))
    activity_idx = torch.as_tensor(activity_idx_np, dtype=torch.int64, device=device)

    # ---- optimiser ----
    optim = make_optimizer(model, cfg)
    if optim_state is not None:
        try:
            optim.load_state_dict(adapt_optim_state(optim_state, model, cfg.weight_decay))
        except (ValueError, KeyError) as e:
            print(f"[imitation] optimizer state not restored ({e}); starting AdamW afresh")
        else:
            restore_group_hparams(optim, model, cfg)  # the checkpoint's weight_decay/betas must not win over cfg

    msg = (f"imitation: {n_train:,} train positions in {len(train_files)} shards, {val.positions:,} val positions, "
           f"{steps_per_epoch:,} steps/epoch, {total_steps:,} planned steps, batch {cfg.batch_size}, "
           f"amp={'bf16' if use_amp else 'off'}, device={device}")
    print(f"[imitation] {msg}")
    logger.log_status(step, message=msg, stage=STAGE, total_steps=total_steps, eta_s=None)

    def do_eval(step_: int) -> dict[str, float]:
        means, activity = evaluate(model, val, cfg, device, activity_idx)
        logger.log_eval(step_, val_loss=means["loss"], val_top1=means["top1"], val_top3=means["top3"],
                        val_value_mse=means["value_loss"], val_policy_loss=means["policy_loss"],
                        positions=val.positions, stage=STAGE)
        if activity is not None:
            logger.log_activity(step_, neuron_idx=activity_idx_np.tolist(), values=activity.tolist(), stage=STAGE)
        print(f"[imitation] eval step {step_}: loss {means['loss']:.4f} top1 {means['top1']:.3f} "
              f"top3 {means['top3']:.3f} value_mse {means['value_loss']:.4f}")
        return means

    def save(step_: int) -> Path:
        path = _call_save(save_checkpoint, step_, STAGE, optim)
        print(f"[imitation] checkpoint -> {path}")
        return path

    do_eval(step)
    last_saved = -1
    epoch = step // steps_per_epoch  # also with epochs=0 (cycle until max_steps): continue the cycle on resume
    # batches of that epoch already trained before the checkpoint: replay the (deterministic) loader past them
    skip = step - epoch * steps_per_epoch
    if skip:
        print(f"[imitation] resuming inside epoch {epoch}: skipping its first {skip:,} batches "
              f"({skip * cfg.batch_size:,} positions already trained)")
        logger.log_status(step, message=f"resuming: skipping {skip:,} already-trained batches of epoch {epoch}",
                          stage=STAGE, total_steps=total_steps, eta_s=None)
    t_window = time.perf_counter()
    pos_window = 0
    step_time_ema: float | None = None
    try:
        while step < total_steps:
            ds.set_epoch(epoch)
            seen = 0  # batches yielded by the loader this epoch (skipped ones included)
            for batch in loader:
                seen += 1
                if seen <= skip:
                    if seen == skip:  # the skip is over: pos/s and ETA measure training steps only
                        t_window, pos_window = time.perf_counter(), 0
                    continue
                x, move, value_t = _to_device(batch, device)
                lr = lr_at(step, cfg.lr, cfg.warmup_steps, total_steps)
                set_lr(optim, lr)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits, value = model(x)[:2]
                loss, metrics = policy_value_loss(logits, value, move, value_t, None, cfg.value_loss_weight)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()
                step += 1
                pos_window += int(x.shape[0])

                if step % cfg.log_every == 0 or step == total_steps:
                    m = metrics_to_float(metrics)  # single device sync
                    now = time.perf_counter()
                    elapsed = max(now - t_window, 1e-9)
                    steps_in_window = pos_window / cfg.batch_size
                    step_time = elapsed / max(steps_in_window, 1e-9)
                    step_time_ema = step_time if step_time_ema is None else 0.8 * step_time_ema + 0.2 * step_time
                    eta_s = (total_steps - step) * step_time_ema
                    logger.log_train(step, loss=m["loss"], policy_loss=m["policy_loss"], value_loss=m["value_loss"],
                                     top1=m["top1"], top3=m["top3"], lr=lr, pos_per_sec=pos_window / elapsed,
                                     gpu_mem_gb=_gpu_mem_gb(), epoch=step / steps_per_epoch, stage=STAGE)
                    logger.log_status(step, message=f"imitation step {step}/{total_steps} loss {m['loss']:.3f}",
                                      stage=STAGE, total_steps=total_steps, eta_s=eta_s)
                    print(f"[imitation] step {step}/{total_steps} loss {m['loss']:.4f} top1 {m['top1']:.3f} "
                          f"lr {lr:.2e} {pos_window / elapsed:,.0f} pos/s eta {eta_s / 60:.1f} min")
                    t_window, pos_window = now, 0
                if step >= total_steps:
                    break  # the final eval / Elo / checkpoint happen below
                side_work = False
                if step % cfg.eval_every == 0:
                    do_eval(step)
                    side_work = True
                if cfg.elo_every and step % cfg.elo_every == 0:
                    quick_elo(model, cfg, device, logger, step)
                    side_work = True
                if step % cfg.checkpoint_every == 0:
                    save(step)
                    last_saved = step
                    side_work = True
                if side_work:  # keep pos_per_sec / ETA a measure of training steps only
                    t_window, pos_window = time.perf_counter(), 0
            if seen == 0:
                raise RuntimeError(f"the training shards yielded no full batch of {cfg.batch_size} positions")
            # `seen <= skip` (no batch trained) is legitimate: steps_per_epoch = n_train // batch_size slightly
            # overestimates what a multi-worker loader with drop_last yields, so a checkpoint taken within a few
            # steps of the epoch boundary can have consumed the whole epoch; training continues in the next one
            skip = 0
            epoch += 1
        # the final eval / Elo run inside the try: a Ctrl-C there still saves the end-of-training weights
        do_eval(step)
        if cfg.elo_every:
            quick_elo(model, cfg, device, logger, step)
    except KeyboardInterrupt:
        print(f"[imitation] interrupted at step {step}: saving checkpoint")
        save(step)
        logger.log_status(step, message="interrupted", stage=STAGE, total_steps=total_steps, eta_s=None)
        raise

    if last_saved != step:
        save(step)
    logger.log_status(step, message=f"imitation finished ({step} steps)", stage=STAGE, total_steps=total_steps,
                      eta_s=0.0)
    return step


def _gpu_mem_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


__all__ = [
    "ValCache",
    "adapt_optim_state",
    "evaluate",
    "lr_at",
    "make_loader",
    "make_optimizer",
    "param_groups",
    "planned_steps",
    "quick_elo",
    "restore_group_hparams",
    "run_imitation_stage",
]
