"""``fly`` — the command-line interface (docs/SPEC.md §9).

Every subcommand is a thin wrapper around a public function of the package; heavy imports (torch,
fastapi, rich) happen inside the command so that ``fly --help`` is instant::

    fly download [--connectome] [--games] [--months 2014-01,...]
    fly build-brain [--region full|central] [--max-neurons N] [--out data/brain/<name>.npz] [--tiny]
    fly build-shards [--months ...] [--pgn FILE ...] [--min-elo 1800] [--workers 16] [--max-games N] [--val-every 50]
    fly train --run NAME [--stage imitation|selfplay|all] [--config cfg.yaml] [--resume] [--steps N] [--tiny]
    fly dashboard [--run NAME] [--port 8765]
    fly play [--run NAME | --ckpt PATH] [--difficulty larva|fly|superfly] [--color white|black] [--gui]
    fly eval --run NAME [--games 50] [--opponent random|material|stockfish|<other-run>]
    fly export-web --run NAME [--out web/model] [--quant f16|i8]
    fly test-vectors --run NAME
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from flychess import __version__, paths

STAGES = ("imitation", "selfplay", "all")
DIFFICULTIES = ("larva", "fly", "superfly")
TINY_NEURONS = 2000
TINY_IO = 128
DEFAULT_VAL_EVERY = 50   # fly build-shards: 1 game in 50 (~2 %) -> <name>.val-NNNNN.npz
DASHBOARD_PORT = 8765
WEB_PORT = 8000


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def _months(s: str | None) -> list[str]:
    return [m.strip() for m in s.split(",") if m.strip()] if s else []


def _scalar(raw: str) -> Any:
    """``'3e-4'`` → 0.0003, ``'false'`` → False, ``'null'`` → None, ``'8'`` → 8, else yaml / the string."""
    import yaml

    s = raw.strip()
    if s == "":
        return None
    for conv in (int, float):
        try:
            return conv(s)
        except ValueError:
            pass
    return yaml.safe_load(s)


def _parse_set(items: list[str] | None) -> dict[str, Any]:
    """``--set key=value`` pairs → dict (``lr=3e-4``, ``amp=false``, ``shard_name=null``, ``brain.steps=4``)."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = _scalar(raw)
    return out


def _run_or_ckpt(args: argparse.Namespace) -> str:
    """``--ckpt`` wins, then ``--run``, else the newest run with a ``latest.pt``."""
    if getattr(args, "ckpt", None):
        return str(args.ckpt)
    if getattr(args, "run", None):
        return str(args.run)
    from flychess import latest_run

    name = latest_run()
    print(f"[fly] no --run given: using the latest run '{name}'")
    return name


def _print_kv(title: str, d: dict[str, Any]) -> None:
    print(title)
    for k, v in d.items():
        print(f"  {k:<14} {v}")


# ------------------------------------------------------------------------------------------------
# commands
# ------------------------------------------------------------------------------------------------
def cmd_download(args: argparse.Namespace) -> int:
    from flychess.connectome.download import download_connectome, download_games

    both = not args.connectome and not args.games
    if args.connectome or both:
        download_connectome(dest=paths.CONNECTOME_DIR, force=args.force)
    if args.games or both:
        months = _months(args.months) or ["2014-01"]
        download_games(months, dest=paths.PGN_DIR, force=args.force)
    return 0


def cmd_build_brain(args: argparse.Namespace) -> int:
    from flychess.connectome.graph import GraphConfig, load_or_build

    if args.tiny:
        cfg = GraphConfig(region=args.region, max_neurons=TINY_NEURONS, max_inputs=TINY_IO, max_outputs=TINY_IO,
                          name=args.name or "tiny")
    else:
        default_name = args.region if args.max_neurons is None else f"{args.region}-{args.max_neurons}"
        cfg = GraphConfig(region=args.region, max_neurons=args.max_neurons, min_syn=args.min_syn,
                          max_inputs=args.max_inputs, max_outputs=args.max_outputs, name=args.name or default_name)
    out = Path(args.out) if args.out else paths.BRAIN_DIR / f"{cfg.name}.npz"
    if out.exists() and args.force:
        out.unlink()
    if out.exists():
        print(f"[fly] brain graph already exists: {out} (use --force to rebuild)")
    g = load_or_build(cfg, out_path=out)
    print(f"[fly] {g.summary()}\n[fly] -> {out}")
    return 0


def cmd_build_shards(args: argparse.Namespace) -> int:
    from flychess.data.lichess import build_shards

    pgns: list[Path] = [Path(p) for p in (args.pgn or [])]
    for m in _months(args.months):
        pgns.append(paths.PGN_DIR / f"lichess_db_standard_rated_{m}.pgn.zst")
    if not pgns:
        pgns = sorted(paths.PGN_DIR.glob("*.pgn*"))
    if not pgns:
        raise SystemExit(f"no PGN files given and none found in {paths.PGN_DIR}; run `fly download --games` first")
    missing = [p for p in pgns if not p.exists()]
    if missing:
        raise SystemExit("PGN file(s) not found: " + ", ".join(map(str, missing)) + " (run `fly download --games`)")
    out = Path(args.out) if args.out else paths.SHARDS_DIR
    print(f"[fly] building shards '{args.name}' from {len(pgns)} file(s) -> {out}")
    stats = build_shards(pgns, out, name=args.name, min_elo=args.min_elo, workers=args.workers,
                         max_games=args.max_games, max_positions=args.max_positions,
                         skip_openings=args.skip_openings, shard_size=args.shard_size, seed=args.seed,
                         val_every=args.val_every)
    print("[fly] " + stats.summary())
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from flychess.train.config import TrainConfig
    from flychess.train.trainer import train

    overrides = _parse_set(args.set)
    for key in ("graph", "shards_dir", "shard_name", "batch_size", "lr", "epochs", "device", "num_workers", "seed"):
        v = getattr(args, key, None)
        if v is not None:
            overrides[key] = v
    if args.steps is not None:
        overrides["max_steps"] = args.steps
    if args.tiny:
        config: Any = TrainConfig.tiny(run=args.run)
        if args.config:
            print("[fly] --tiny given: ignoring --config", file=sys.stderr)
    else:
        config = args.config
    print(f"[fly] dashboard: run `fly dashboard --run {args.run}` in another terminal "
          f"-> http://127.0.0.1:{DASHBOARD_PORT}/  (metrics: {paths.run_dir(args.run) / 'metrics.jsonl'})")
    latest = train(args.run, stage=args.stage, config=config, resume=args.resume, **overrides)
    print(f"[fly] done: {latest}")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from flychess.dashboard.server import run_dashboard

    run_dashboard(run=args.run, port=args.port, open_browser=args.open, host=args.host)
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    target = _run_or_ckpt(args)
    if args.gui:
        from flychess.play.local_web import serve_local

        serve_local(target, port=args.port, open_browser=not args.no_browser)
        return 0
    from flychess.play.terminal import play_terminal

    kw: dict[str, Any] = {}
    if args.sims is not None:
        kw["sims"] = args.sims
    if args.seed is not None:
        kw["seed"] = args.seed
    play_terminal(target, difficulty=args.difficulty, color=args.color, device=args.device, **kw)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from flychess.eval.elo import evaluate_run

    opponents: list[str] = []
    for item in args.opponent or []:
        opponents += [o.strip() for o in item.split(",") if o.strip()]
    opponents = opponents or ["random", "material"]
    results = evaluate_run(args.run, games=args.games, opponents=opponents, device=args.device,
                           max_plies=args.max_plies, temperature=args.temperature, seed=args.seed,
                           log=not args.no_log)
    print(f"[fly] {args.run}: " + "  ".join(
        f"vs {opp}: +{r['wins']} ={r['draws']} -{r['losses']} (Elo {r['elo_estimate']:+.0f})"
        for opp, r in results.items()))
    return 0


def _export(run_or_ckpt: str, out: Path, quant: str) -> dict[str, Any]:
    import json

    from flychess.export.web import flyb_size_report
    from flychess.play.local_web import ensure_export

    ensure_export(run_or_ckpt, out, quant=quant, force=True)
    header = json.loads((out / "brain.json").read_text())
    print(flyb_size_report(header))
    gz = out / "brain.flyb.gz"
    print(f"[fly] exported run '{header.get('run_name', '')}' (step {header.get('train_steps', 0)}, {quant}) -> {out}"
          + (f"  [{gz.stat().st_size / 1e6:.2f} MB gzipped]" if gz.exists() else ""))
    return header


def cmd_export_web(args: argparse.Namespace) -> int:
    from flychess.export.testvectors import DEFAULT_PATH, write_model_vectors

    target = _run_or_ckpt(args)
    out = Path(args.out) if args.out else paths.WEB_MODEL_DIR
    _export(target, out, args.quant)
    if not args.no_vectors:
        vec_out = Path(args.vectors) if args.vectors else DEFAULT_PATH
        write_model_vectors(model_dir=out, out=vec_out)
        print(f"[fly] test vectors -> {vec_out}  (check with: node --test web/test/parity.test.mjs)")
    return 0


def cmd_test_vectors(args: argparse.Namespace) -> int:
    import json

    from flychess.export.testvectors import DEFAULT_PATH, check_model_vectors, write_model_vectors
    from flychess.train.trainer import load_checkpoint

    target = _run_or_ckpt(args)
    out = Path(args.out) if args.out else DEFAULT_PATH
    model_dir = Path(args.model_dir) if args.model_dir else paths.WEB_MODEL_DIR
    header_path = model_dir / "brain.json"
    use_export = False
    if header_path.exists() and (model_dir / "brain.flyb").exists():
        header = json.loads(header_path.read_text())
        run_name = Path(target).parent.name if target.endswith(".pt") else Path(target).name
        use_export = header.get("run_name") == run_name
        if not use_export:
            print(f"[fly] {model_dir} holds run '{header.get('run_name')}', not '{run_name}': "
                  "exporting to a temporary directory instead")
    if use_export:
        write_model_vectors(model_dir=model_dir, out=out)
        deltas = check_model_vectors(model_dir, out)
        print(f"[fly] vectors from {model_dir} -> {out}  (self-check max Δlogit {deltas['max_logit_delta']:.2g})")
    else:
        model, graph, ckpt = load_checkpoint(target, device="cpu")
        cfg = ckpt.get("config") or {}
        run_name = cfg.get("run", "") if isinstance(cfg, dict) else ""
        write_model_vectors(model, graph, out=out, quant=args.quant, run_name=run_name,
                            extra_meta={"train_steps": int(ckpt.get("step", 0))})
        print(f"[fly] vectors from a fresh {args.quant} export of {target} -> {out}")
    print("[fly] run `node --test web/test/parity.test.mjs` (needs web/model/ from `fly export-web`)")
    return 0


# ------------------------------------------------------------------------------------------------
# parser
# ------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fly",
        description="fly-chess: train and play a chess engine whose network is the FlyWire fruit-fly connectome.",
        epilog="Typical pipeline: fly download -> fly build-brain -> fly build-shards -> fly train --run fly1 "
               "-> fly dashboard --run fly1 -> fly play --run fly1 -> fly export-web --run fly1. "
               "Add --tiny to build-brain / train for a 2000-neuron smoke run.",
    )
    p.add_argument("--version", action="version", version=f"fly-chess {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug-level log output")
    sub = p.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # download
    s = sub.add_parser("download", help="download the FlyWire connectome tables and/or Lichess PGN dumps",
                       description="Download the FlyWire v783 tables into data/connectome/ and Lichess monthly "
                                   "PGN dumps into data/pgn/. With neither flag both are downloaded.")
    s.add_argument("--connectome", action="store_true", help="FlyWire connectome tables (~60 MB)")
    s.add_argument("--games", action="store_true", help="Lichess standard rated dumps (see --months)")
    s.add_argument("--months", default=None, help="comma-separated YYYY-MM list (default 2014-01)")
    s.add_argument("--force", action="store_true", help="re-download existing files")

    # build-brain
    s = sub.add_parser("build-brain", help="turn the connectome into a BrainGraph npz (data/brain/<name>.npz)",
                       description="Select the neurons / synapses that form the network (SPEC §2.3) and save "
                                   "the CSR graph. Default: the whole brain (~134k neurons, 2.7M connections / 34M synapses).")
    s.add_argument("--region", choices=("full", "central"), default="full")
    s.add_argument("--max-neurons", type=int, default=None, help="keep the top-N neurons by synapse count")
    s.add_argument("--min-syn", type=int, default=5, help="minimum synapses per (pre, post) pair")
    s.add_argument("--max-inputs", type=int, default=2048, help="sensory/ascending neurons receiving the board")
    s.add_argument("--max-outputs", type=int, default=2048, help="descending/motor neurons read by the heads")
    s.add_argument("--name", default=None, help="graph name (default: region[-N]); used as data/brain/<name>.npz")
    s.add_argument("--out", default=None, help="output npz path (default data/brain/<name>.npz)")
    s.add_argument("--tiny", action="store_true", help=f"{TINY_NEURONS}-neuron smoke graph named 'tiny'")
    s.add_argument("--force", action="store_true", help="rebuild even if the npz exists")

    # build-shards
    s = sub.add_parser("build-shards", help="stream PGN dumps into shuffled training shards (data/shards/)",
                       description="Filter rated standard games (both players >= --min-elo, decided or drawn), "
                                   "encode every position and write shuffled npz shards (SPEC §5).")
    s.add_argument("--months", default=None, help="comma-separated YYYY-MM list of downloaded dumps")
    s.add_argument("--pgn", nargs="*", default=None, help="explicit PGN / .pgn.zst files (instead of --months)")
    s.add_argument("--min-elo", type=int, default=1800)
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--max-games", type=int, default=None, help="stop after N games (smoke runs)")
    s.add_argument("--max-positions", type=int, default=None)
    s.add_argument("--skip-openings", type=int, default=4, help="skip the first N plies of every game")
    s.add_argument("--shard-size", type=int, default=262144)
    s.add_argument("--name", default="lichess", help="shard prefix: <out>/<name>-NNNNN.npz")
    s.add_argument("--out", default=None, help=f"output directory (default {paths.SHARDS_DIR})")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--val-every", dest="val_every", type=int, default=DEFAULT_VAL_EVERY,
                   help="hold out one game in N *whole* as the validation set <out>/<name>.val-NNNNN.npz "
                        "(game-disjoint from training; 0 = off, training then splits off whole shards, "
                        "which is not game-disjoint)")

    # train
    s = sub.add_parser("train", help="train a run: imitation (stage 1) and/or self-play (stage 2)",
                       description="Train the fly brain. Checkpoints go to runs/<run>/ckpt-<step>.pt + latest.pt, "
                                   "metrics to runs/<run>/metrics.jsonl (watch them with `fly dashboard`).")
    s.add_argument("--run", required=True, help="run name (runs/<run>/)")
    s.add_argument("--stage", choices=STAGES, default="all")
    s.add_argument("--config", default=None, help="yaml TrainConfig (default: built-in defaults / --tiny preset)")
    s.add_argument("--resume", action="store_true", help="continue from runs/<run>/latest.pt")
    s.add_argument("--steps", type=int, default=None, help="cap the imitation stage at N steps (max_steps)")
    s.add_argument("--tiny", action="store_true", help="smoke preset: tiny graph, batch 32, 30 steps, 1 self-play iter")
    s.add_argument("--graph", default=None, help="graph name under data/brain/ or an npz path")
    s.add_argument("--shards-dir", dest="shards_dir", default=None)
    s.add_argument("--shard-name", dest="shard_name", default=None, help="shard series to train on: comma-separated names with optional repeat factors, e.g. 'lichess2014,lichess2015,evals:3'")
    s.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    s.add_argument("--lr", type=float, default=None)
    s.add_argument("--epochs", type=int, default=None)
    s.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    s.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override any TrainConfig field (repeatable; e.g. --set selfplay_iters=2 --set brain.steps=4)")

    # dashboard
    s = sub.add_parser("dashboard", help=f"live training dashboard (http://127.0.0.1:{DASHBOARD_PORT}/)",
                       description="Serve the dashboard: run selector, live loss / accuracy / Elo charts, "
                                   "latest self-play game, neuron activity heatmap and log tail.")
    s.add_argument("--run", default=None, help="run opened by default")
    s.add_argument("--port", type=int, default=DASHBOARD_PORT)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--open", action="store_true", help="open the browser")

    # play
    s = sub.add_parser("play", help="play against the fly in the terminal (or --gui in the browser)",
                       description="Play a game against the fly brain. Without --run/--ckpt the most recently "
                                   "updated run with a latest.pt is used. --gui exports the brain to web/model/ "
                                   "and serves the website locally.")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--run", default=None, help="run name (runs/<run>/latest.pt)")
    g.add_argument("--ckpt", default=None, help="checkpoint path")
    s.add_argument("--difficulty", choices=DIFFICULTIES, default="fly",
                   help="larva = sampled policy (T=1.2); fly = argmax + 1-ply value check; superfly = MCTS 200 sims")
    s.add_argument("--color", choices=("white", "black", "random"), default="white", help="your colour")
    s.add_argument("--gui", action="store_true", help="serve the website locally instead of the terminal UI")
    s.add_argument("--port", type=int, default=WEB_PORT, help="port for --gui")
    s.add_argument("--no-browser", action="store_true", help="--gui: do not open the browser")
    s.add_argument("--device", default=None)
    s.add_argument("--sims", type=int, default=None, help="MCTS simulations for superfly (default 200)")
    s.add_argument("--seed", type=int, default=None)

    # eval
    s = sub.add_parser("eval", help="play matches against labelled opponents and estimate Elo",
                       description="The run's fly brain (argmax policy) plays --games games per opponent; results "
                                   "are printed and appended as `elo` records to the run's metrics.")
    s.add_argument("--run", required=True)
    s.add_argument("--games", type=int, default=50)
    s.add_argument("--opponent", action="append", default=None,
                   help="random | material | stockfish[:depth[:skill]] | <other-run> | ckpt-<step> "
                        "(repeatable or comma-separated; default random,material)")
    s.add_argument("--max-plies", dest="max_plies", type=int, default=200)
    s.add_argument("--temperature", type=float, default=0.0, help="policy sampling temperature (0 = argmax)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default=None)
    s.add_argument("--no-log", action="store_true", help="do not append elo records to the run's metrics")

    # export-web
    s = sub.add_parser("export-web", help="export a run's brain to web/model/ (brain.json + brain.flyb[.gz])",
                       description="Write the website model files (SPEC §8) and the cross-language test vectors "
                                   "tests/vectors/model.json.")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--run", default=None)
    g.add_argument("--ckpt", default=None)
    s.add_argument("--out", default=None, help=f"output directory (default {paths.WEB_MODEL_DIR})")
    s.add_argument("--quant", choices=("f16", "f32", "i8"), default="f16")
    s.add_argument("--vectors", default=None, help="test-vector path (default tests/vectors/model.json)")
    s.add_argument("--no-vectors", action="store_true", help="skip writing the test vectors")

    # test-vectors
    s = sub.add_parser("test-vectors", help="write tests/vectors/model.json for the JS parity test",
                       description="Compute reference logits / values from the exported f16 weights. Uses the "
                                   "export in --model-dir when it belongs to the run, else a temporary export.")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--run", default=None)
    g.add_argument("--ckpt", default=None)
    s.add_argument("--model-dir", dest="model_dir", default=None, help=f"exported model dir (default {paths.WEB_MODEL_DIR})")
    s.add_argument("--out", default=None, help="output json (default tests/vectors/model.json)")
    s.add_argument("--quant", choices=("f16", "f32", "i8"), default="f16")

    return p


COMMANDS = {
    "download": cmd_download,
    "build-brain": cmd_build_brain,
    "build-shards": cmd_build_shards,
    "train": cmd_train,
    "dashboard": cmd_dashboard,
    "play": cmd_play,
    "eval": cmd_eval,
    "export-web": cmd_export_web,
    "test-vectors": cmd_test_vectors,
}


class _ShortName(logging.Formatter):
    """``[selfplay] message`` — the same style the imitation stage prints with."""

    def format(self, record: logging.LogRecord) -> str:
        record.short = record.name.rsplit(".", 1)[-1]
        return super().format(record)


def configure_logging(verbose: bool = False) -> None:
    """Route the package's ``logging`` output (self-play, local web server...) to stdout at INFO (DEBUG with -v)."""
    handler = logging.StreamHandler(sys.stdout)  # same stream as the print()-based stages: ordered output
    handler.setFormatter(_ShortName("[%(short)s] %(message)s"))
    pkg = logging.getLogger("flychess")
    if not pkg.handlers:
        pkg.addHandler(handler)
    pkg.setLevel(logging.DEBUG if verbose else logging.INFO)
    pkg.propagate = False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(getattr(args, "verbose", False))
    try:
        return int(COMMANDS[args.command](args) or 0)
    except KeyboardInterrupt:
        print("\n[fly] interrupted", file=sys.stderr)
        return 130
    except FileNotFoundError as e:
        print(f"[fly] error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
