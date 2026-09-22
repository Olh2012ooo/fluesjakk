#!/usr/bin/env python
"""Build (and optionally upload) the Hugging Face model repository for the trained fly brains.

    .venv/bin/python scripts/publish-hf.py --repo <user>/fly-chess --out ~/Scrivania/fly-chess-hf
    .venv/bin/python scripts/publish-hf.py --repo <user>/fly-chess --out <folder> --upload [--private]

The folder is a complete, self-describing model repo:
  README.md                      model card (YAML metadata, description, usage, evaluation, licence)
  LICENSE                        CC BY-NC 4.0 legal text (the weights derive from the FlyWire Codex release)
  ATTRIBUTION.md                 data sources, citations, code licence
  CITATION.cff                   how to cite
  manifest.json                  every file with size and sha256; training/eval summary per checkpoint
  .gitattributes                 LFS rules for the binary files
  checkpoints/<name>.pt          weights WITHOUT optimizer state (+ config, brain_config, step, stage, elo)
  checkpoints/<name>.json        the checkpoint's TrainConfig / BrainConfig / metrics in plain JSON
  graph/full.npz                 the BrainGraph every checkpoint needs (CSR connectome, signs, retina)
  web/brain.json|flyb|flyb.gz    browser blob of the flagship brain (what the website loads)
Upload needs `huggingface_hub` and a login (`.venv/bin/hf auth login` or HF_TOKEN).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = {  # published name -> checkpoint on disk
    "fly1-imitation": ROOT / "runs/fly1/imitation-final.pt",
    "fly2-imitation": ROOT / "runs/fly2/latest.pt",
    "fly3-imitation": ROOT / "runs/fly3/imitation-final.pt",
    "fly3-selfplay": ROOT / "runs/fly3/best.pt",
}
FLAGSHIP = "fly3-selfplay"


def strip_checkpoint(src: Path, dst: Path) -> dict:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ck.pop("optim", None)
    ck["graph_path"] = "graph/full.npz"  # relative to the model repo
    torch.save(ck, dst)
    return {k: ck.get(k) for k in ("step", "stage", "elo")}


def model_card(repo: str, meta: dict[str, dict]) -> str:
    rows = "\n".join(f"| `checkpoints/{n}.pt` | {m['step']:,} | {m['stage']} | {m['mb']:.0f} MB | {m['elo_note']} |"
                     for n, m in meta.items())
    return f"""---
license: cc-by-nc-4.0
language:
  - en
tags:
  - chess
  - connectome
  - drosophila
  - flywire
  - neuroscience
  - recurrent-neural-network
  - reinforcement-learning
  - pytorch
library_name: flychess
pipeline_tag: reinforcement-learning
---

# fly-chess brains

Chess-playing neural networks whose wiring **is** the FlyWire adult *Drosophila melanogaster* connectome:
134,209 neurons, 2,700,513 neuron-to-neuron connections (34.2 M synapses), every synapse's sign fixed by the
presynaptic neuron's neurotransmitter (acetylcholine excitatory; GABA and glutamate inhibitory). Training
learns only synaptic strength magnitudes, per-neuron biases / leaks / gains, and the board-input and
move-readout projections — no connection is added, removed or re-signed.

Code, training pipeline and the website that runs these brains in the browser:
**https://github.com/cesp99/fly-chess** (MIT).

## Model description

The network is a recurrent rate model on the connectome: `h_t+1 = (1-a) h_t + a · f(W h_t + bias + input)`,
`W` sparse with the connectome's pattern and signs, unrolled 8 or 16 timesteps per position. The board
(20 planes × 64 squares, always from the side to move) enters through learned projections into sensory
neurons; the move policy (4,168 logits) and value are linear/MLP read-outs of the descending and motor
neurons. `fly3` additionally lets the fly **see** the board: 5,543 real photoreceptors (R1–6, R7, R8,
placed on their ommatidial columns; left eye files a–d, right eye e–h) each receive one square, and the
signal travels through the real lamina → medulla → lobula → central brain wiring. Details:
`docs/SPEC.md` and `docs/RETINA.md` in the code repository.

## Files

| file | step | stage | size | notes |
|---|---|---|---|---|
{rows}
| `graph/full.npz` | — | — | 37 MB | the BrainGraph (CSR connectome, signs, retina mapping) every checkpoint needs |
| `web/brain.json`, `web/brain.flyb`, `web/brain.flyb.gz` | — | — | 60 / 50 MB | browser blob of `{FLAGSHIP}` (SPEC §8 format; the website loads it from here) |

Each `checkpoints/<name>.json` holds the exact training configuration and evaluation numbers of that
checkpoint; `manifest.json` lists sizes and sha256 for every file.

## Generations

| run | recipe | held-out top-1 / top-3 (same 5,120 human positions) |
|---|---|---|
| `fly1-imitation` | 8 timesteps, ReLU, board via 2,048 sensory/ascending neurons, Lichess 2014 | 34.1% / 58.4% |
| `fly2-imitation` | 16 timesteps, saturating rates, Lichess 2014+2015, fp32 | 33.1% / 57.2% |
| `fly3-imitation` | fly2 + retina input, homeostatic gains, multi-timestep readout, neuromodulatory gating, central-brain readout, + 30 M Stockfish-evaluated positions | 36.9% / 62.3% |
| **`fly3-selfplay`** | fly3-imitation + gated self-play (5 of 9 iterations promoted) | **37.6% / 63.4%** |

Head-to-head with 100-simulation MCTS over 100 games on 50 paired openings: `fly3-selfplay` vs `fly1`
**+50 =50 −0**, vs `fly2` **+100 =0 −0**; vs its own imitation checkpoint (200 sims, 20 games) 20–0; vs a
1-ply material-greedy bot +38 =2 −0 with search and +16 =4 −0 without. It is a beatable club-level-ish
opponent without search and a real fight with it; it is not an engine.

## Use

```bash
pip install git+https://github.com/cesp99/fly-chess
hf download {repo} --local-dir fly-chess-models
```

```python
import torch, chess
from flychess.connectome.graph import BrainGraph
from flychess.model.config import BrainConfig
from flychess.model.flybrain import FlyBrain
from flychess.play.engine import FlyEngine

ck = torch.load("fly-chess-models/checkpoints/{FLAGSHIP}.pt", map_location="cpu", weights_only=False)
graph = BrainGraph.load("fly-chess-models/graph/full.npz")
model = FlyBrain.from_checkpoint(ck["model"], BrainConfig.from_dict(ck["brain_config"]), graph).eval()
engine = FlyEngine(model, graph, device="cpu")          # "cuda" if available
move, info = engine.choose_move(chess.Board(), difficulty="fly")   # larva | fly | superfly (MCTS)
print(move, info["policy_top"], info["value"])
```

Or with the CLI from the code repository: `fly play --ckpt fly-chess-models/checkpoints/{FLAGSHIP}.pt --gui`.

## Training data

* Lichess monthly databases 2014-01 … 2015-12 (CC0): rated standard games, both players ≥ 1800 Elo, not
  bullet; 229 M positions (policy target = the human move, value target = the game result).
* Lichess evaluation database: 30 M positions with Stockfish evaluations (policy target = engine best move,
  value target = win probability from the centipawn score); used for `fly3`.
* Self-play games of the network itself (gated, with human/engine positions rehearsed in every batch).

## Limitations

A fixed random-for-chess topology with sign constraints is a poor chess architecture: the brains play
plausible openings and positional chess and still miss tactics; value estimates are weak without search.
The rate model is a caricature of real neural dynamics (no spikes, gap junctions, dendrites or real
neuromodulation). Input/output neuron choices are modelling decisions, not biology.

## Licence and attribution

The FlyWire Codex data release the weights derive from is **CC BY-NC 4.0**, so the weights are published
under **CC BY-NC 4.0** (`LICENSE`): non-commercial use only, with attribution. See `ATTRIBUTION.md` for the
full list of sources and citations. Training code: MIT.

## Citation

```bibtex
@misc{{flychess2026,
  title  = {{fly-chess: a chess engine wired like the FlyWire fruit-fly connectome}},
  author = {{Esposito, Carlo}},
  year   = {{2026}},
  url    = {{https://github.com/cesp99/fly-chess}}
}}
```
Please also cite the connectome papers listed in `ATTRIBUTION.md`.
"""


ATTRIBUTION = """# Attribution

These weights are derived from, and are only meaningful together with, the following sources.

## Connectome (network wiring, synapse signs, neuron positions, retina columns)

FlyWire whole-brain connectome of an adult female *Drosophila melanogaster*, Codex snapshot 783
(`connections.csv.gz`, `neurons.csv.gz`, `classification.csv.gz`, `consolidated_cell_types.csv.gz`,
`coordinates.csv.gz`, `column_assignment.csv.gz`). Licence of the Codex release: **CC BY-NC 4.0**
(https://flywire.ai/guidelines). The Zenodo mirror of the connectivity data is CC BY 4.0
(doi:10.5281/zenodo.10676866).

* Dorkenwald, S., Matsliah, A., Sterling, A. R., Schlegel, P., Yu, S.-C., McKellar, C. E., et al.
  Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138 (2024). doi:10.1038/s41586-024-07558-y
* Schlegel, P., Yin, Y., Bates, A. S., et al. Whole-brain annotation and multi-connectome cell typing of
  *Drosophila*. *Nature* 634, 139–152 (2024). doi:10.1038/s41586-024-07686-5
* Matsliah, A., Yu, S.-C., Kruk, K., et al. Neuronal parts list and wiring diagram for a visual system.
  *Nature* 634, 166–180 (2024). doi:10.1038/s41586-024-07981-1 (optic-lobe column assignments)
* Eckstein, N., Bates, A. S., Champion, A., et al. Neurotransmitter classification from electron microscopy
  images at synaptic sites in *Drosophila melanogaster*. *Cell* 187, 2574–2594 (2024) (transmitter predictions
  behind the synapse signs)
* The FlyWire Consortium. FlyWire Whole-brain Connectome Connectivity Data (783.0) [Data set]. Zenodo (2024).

## Training data

* Lichess open database, https://database.lichess.org — monthly standard rated games 2014-01 … 2015-12 and
  the evaluation database (`lichess_db_eval.jsonl.zst`). Released under CC0 1.0.
* Stockfish (https://stockfishchess.org, GPL-3.0) produced the evaluations in the Lichess evaluation database;
  no Stockfish code is included here.

## Modelling ideas

* Lappalainen, J. K., et al. Connectome-constrained networks predict neural activity across the fly visual
  system. *Nature* 634, 1132–1140 (2024) — connectome-constrained training with sign constraints.
* Shiu, P. K., et al. A *Drosophila* computational brain model reveals sensorimotor processing.
  *Nature* 634, 210–219 (2024) — whole-FlyWire-brain leaky integrate-and-fire modelling.

## Software

* Training and inference code: https://github.com/cesp99/fly-chess (MIT, Carlo Esposito).
* PyTorch (BSD-3), python-chess (GPL-3.0, used as a library by the training code only; not part of the weights).
"""


CITATION_CFF = """cff-version: 1.2.0
message: "If you use these weights, please cite the fly-chess repository and the FlyWire connectome papers listed in ATTRIBUTION.md."
title: "fly-chess brains: chess-playing networks wired like the FlyWire Drosophila connectome"
authors:
  - family-names: Esposito
    given-names: Carlo
year: 2026
url: "https://github.com/cesp99/fly-chess"
license: CC-BY-NC-4.0
"""


GITATTRIBUTES = """*.pt filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.flyb filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. cesp99/fly-chess")
    ap.add_argument("--out", required=True, help="folder to build (created / emptied)")
    ap.add_argument("--upload", action="store_true", help="also upload the folder to the Hugging Face repo")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    if out.exists():
        shutil.rmtree(out)
    for d in ("checkpoints", "graph", "web"):
        (out / d).mkdir(parents=True)

    meta: dict[str, dict] = {}
    for name, src in CHECKPOINTS.items():
        if not src.exists():
            print(f"skip {name}: {src} missing")
            continue
        dst = out / "checkpoints" / f"{name}.pt"
        ck = torch.load(src, map_location="cpu", weights_only=False)
        ck.pop("optim", None)
        ck["graph_path"] = "graph/full.npz"           # relative to this repo
        torch.save(ck, dst)
        elo = {k: round(float(v), 1) for k, v in (ck.get("elo") or {}).items()}
        info = {"name": name, "step": int(ck["step"]), "stage": ck["stage"], "saved_at": ck.get("saved_at"),
                "elo_estimates": elo, "brain_config": ck["brain_config"], "train_config": ck["config"],
                "graph": "graph/full.npz", "source_run": str(src.relative_to(ROOT))}
        (out / "checkpoints" / f"{name}.json").write_text(json.dumps(info, indent=2, default=str))
        m = {"step": info["step"], "stage": info["stage"], "mb": dst.stat().st_size / 1e6,
             "elo_note": ", ".join(f"{k} {v:+.0f}" for k, v in elo.items()) or "—"}
        meta[name] = m
        print(f"{name}: step {m['step']} {m['stage']} {m['mb']:.0f} MB  elo {elo}")

    shutil.copy(ROOT / "data/brain/full.npz", out / "graph/full.npz")
    for f in ("brain.json", "brain.flyb", "brain.flyb.gz"):
        shutil.copy(ROOT / "web/model" / f, out / "web" / f)
    header = json.loads((out / "web/brain.json").read_text())
    print(f"web blob: run {header.get('run_name')} step {header.get('train_steps')} exported {header.get('exported_at')}")

    (out / "README.md").write_text(model_card(args.repo, meta))
    (out / "LICENSE").write_text((ROOT / "docs/LICENSE-CC-BY-NC-4.0.txt").read_text())
    (out / "ATTRIBUTION.md").write_text(ATTRIBUTION)
    (out / "CITATION.cff").write_text(CITATION_CFF)
    (out / ".gitattributes").write_text(GITATTRIBUTES)

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "repo": args.repo,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "flagship": FLAGSHIP,
        "web_blob": {k: header.get(k) for k in ("run_name", "train_steps", "exported_at", "n", "nnz", "n_ret", "steps", "activation")},
        "graph": {"file": "graph/full.npz", "n": 134209, "connections": 2700513},
        "checkpoints": {n: {"step": m["step"], "stage": m["stage"]} for n, m in meta.items()},
        "files": [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(f["bytes"] for f in manifest["files"])
    print(f"built {out}: {len(manifest['files'])} files, {total / 1e6:.0f} MB")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(out),
                          commit_message="Upload fly-chess brains (fly1, fly2, fly3 imitation + self-play) and the web blob")
        print(f"uploaded -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
