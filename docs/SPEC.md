# fly-chess — Architecture & Interface Specification

This document is the contract between all modules. Every implementer must follow it exactly;
if something here is impossible, change the SPEC *and* every dependent module, never silently diverge.

## 0. Principles

1. **The fly brain really plays.** The player is a recurrent neural network whose neurons and
   synapses are the real FlyWire adult *Drosophila melanogaster* connectome. Nothing else picks moves
   (no hidden engine, no random-move fallback presented as "the fly"). The browser runs the exact
   same network as Python (bit-for-bit up to float rounding), verified by cross-language test vectors.
2. **Connectome-faithful parameterisation.** Synapse *existence* comes from the connectome and is
   never learned. Synapse *sign* comes from the presynaptic neurotransmitter (Dale's law) and is
   never learned. Only synaptic *strength magnitudes* (initialised from synapse counts), per-neuron
   biases / time constants, and the sensory-input and motor-readout projections are learned.
3. **One shared encoding.** Board → planes and move ↔ index are defined once here and implemented
   identically in `flychess/chessenv/encoding.py` and `web/engine/encoding.js`.
4. **Everything runs from the CLI**: `fly download`, `fly build-brain`, `fly train`, `fly play`,
   `fly dashboard`, `fly export-web`, `fly eval`, and from Python: `flychess.train(...)`, `flychess.play(...)`.

## 1. Repository layout

```
fly-chess/
  pyproject.toml            # uv/hatch project, console script `fly`
  README.md                 # user documentation (quickstart, science, credits)
  docs/SPEC.md              # this file
  flychess/
    __init__.py             # exports train(), play(), load_brain(), __version__
    cli.py                  # argparse CLI, subcommands listed in §9
    paths.py                # DATA_DIR / RUNS_DIR / CACHE_DIR resolution (env FLYCHESS_HOME)
    connectome/
      download.py           # download FlyWire tables (Zenodo/Codex mirrors), verify sizes
      load.py               # parse tables -> Connectome dataclass (pandas -> numpy)
      graph.py              # subgraph selection, neuron role assignment, CSR build -> BrainGraph
    chessenv/
      encoding.py           # board->planes, move<->index (§3)
      env.py                # ChessEnv (reset/step/legal_mask/result), perspective handling
    model/
      spmm.py               # SpMM autograd.Function (CSR fwd, transposed-CSR bwd, chunked dW)
      flybrain.py           # FlyBrain nn.Module (§4)
      config.py             # BrainConfig dataclass, yaml (de)serialisation
    data/
      lichess.py            # download monthly PGN .zst, stream-parse, filter, write shards
      shards.py             # npz/parquet shard format (§5), ShardDataset, collate
    train/
      metrics.py            # MetricsLogger: JSONL appends to runs/<run>/metrics.jsonl (§7)
      imitation.py          # stage 1 supervised trainer
      mcts.py               # batched MCTS using FlyBrain (policy+value)
      selfplay.py           # stage 2 self-play generation + training loop
      trainer.py            # train(config) orchestrating both stages, checkpoints (§6)
    eval/
      elo.py                # round-robin / vs-random / vs-previous-checkpoint Elo estimation
      opponents.py          # RandomPlayer, GreedyMaterialPlayer, optional Stockfish (if installed)
    play/
      engine.py             # FlyEngine: choose_move(board, difficulty) using FlyBrain (+MCTS)
      terminal.py           # rich terminal UI to play vs the fly
      local_web.py          # `fly play --gui`: serves web/ locally with the current brain
    dashboard/
      server.py             # FastAPI + websocket, tails metrics.jsonl, serves static/
      static/index.html, dashboard.js, dashboard.css
    export/
      web.py                # export trained FlyBrain -> web/model/brain.flyb (+ brain.json) (§8)
      testvectors.py        # writes tests/vectors/*.json for JS parity tests
  web/                      # static website (GitHub Pages / any static host)
    index.html, style.css, app.js
    engine/encoding.js, engine/flybrain.js, engine/mcts.js, engine/worker.js, engine/loader.js
    vendor/chess.js         # vendored (BSD-2-Clause)
    assets/                 # piece SVGs, fly sprites, sounds
    model/                  # gitignored; produced by `fly export-web`
    test/parity.test.mjs    # node test: JS engine == Python vectors
  tests/                    # pytest
  scripts/                  # helper shell scripts (deploy pages, full pipeline)
  data/  runs/              # gitignored: downloaded data, training runs
```

## 2. Connectome (`flychess/connectome`)

### 2.1 Source files (FlyWire public release, Codex snapshot 783)
Public, no-login URLs (verified 2026-09-12): `https://storage.googleapis.com/flywire-data/codex/data/fafb/783/<file>`.
Downloaded into `data/connectome/`:
- `connections.csv.gz` (50.3 MB) — columns `pre_root_id, post_root_id, neuropil, syn_count, nt_type`.
  3,869,878 rows, one per (pre, post, neuropil); aggregate over neuropil → 2,700,513 (pre, post) pairs between
  134,181 neurons, threshold ≥ 5 synapses per pair already applied (per-row syn_count may be < 5). nt_type per edge
  ∈ {ACH, GABA, GLUT, DA, SER, OCT}. Sum of syn_count = 34.15 M.
- `neurons.csv.gz` (1.7 MB) — `root_id, group, nt_type, nt_type_score, da_avg, ser_avg, gaba_avg, glut_avg, ach_avg, oct_avg`; 139,255 rows.
- `classification.csv.gz` (0.9 MB) — `root_id, flow, super_class, class, sub_class, hemilineage, side, nerve`
  (NOTE: no cell_type column any more). super_class values & counts: optic 77,873; central 32,381; sensory 16,938;
  visual_projection 7,684; ascending 1,750; descending 1,305; sensory_ascending 612; visual_centrifugal 522; motor 110; endocrine 80.
  flow ∈ {intrinsic, afferent, efferent}.
- `consolidated_cell_types.csv.gz` (0.9 MB) — `root_id, primary_type, additional_type(s)` → use `primary_type` as `cell_type`.
- `coordinates.csv.gz` — `root_id, position, supervoxel_id` where position is a string `"[x y z]"` in nm, multiple rows
  per neuron → take the first row per root_id (or the mean).
- `cell_stats.csv.gz` — `root_id, length_nm, area_nm, size_nm` (optional). `names.csv.gz` — `root_id, name, group` (optional).
Licensing: the Codex CSV release is CC BY-NC 4.0 per FlyWire guidelines (the Zenodo mirror 10.5281/zenodo.10676866 is
CC BY 4.0). Cite Dorkenwald et al. 2024 (doi:10.1038/s41586-024-07558-y) and Schlegel et al. 2024
(doi:10.1038/s41586-024-07686-5). `download.py` downloads with streaming progress, verifies gzip integrity, and
prints manual instructions if a URL fails (the user can also drop files in `data/connectome/`). `fly download --connectome`.
The Lichess dumps are CC0 (`https://database.lichess.org/standard/lichess_db_standard_rated_YYYY-MM.pgn.zst`,
2014-01 = 111 MB / 697k games, 2013-01 = 17.8 MB / 121k games).

### 2.2 `Connectome` (load.py)
```python
@dataclass
class Connectome:
    root_ids: np.ndarray[int64]      # (N,) sorted unique neuron ids
    super_class: np.ndarray[str]     # (N,)
    cell_class: np.ndarray[str]      # (N,)
    cell_type: np.ndarray[str]       # (N,)
    side: np.ndarray[str]            # (N,) 'left'/'right'/'center'/''
    nt_type: np.ndarray[str]         # (N,) presynaptic neurotransmitter: ACH/GABA/GLUT/DA/SER/OCT/''
    position: np.ndarray[float32]    # (N,3) nm, NaN if unknown
    pre: np.ndarray[int32]           # (E,) index into root_ids
    post: np.ndarray[int32]          # (E,)
    syn_count: np.ndarray[int32]     # (E,) summed over neuropils
    neuropil: np.ndarray[str] | None # optional dominant neuropil per edge
```
`load_connectome(data_dir) -> Connectome` caches a `connectome.npz` after first parse.

### 2.3 `BrainGraph` (graph.py) — what the model consumes
```python
@dataclass
class BrainGraph:
    n: int                            # neurons in the trained subgraph
    root_ids: np.ndarray[int64]       # (n,)
    csr_indptr: np.ndarray[int32]     # (n+1,)  rows = POST-synaptic neuron
    csr_indices: np.ndarray[int32]    # (nnz,)  PRE-synaptic neuron per synapse
    syn_count: np.ndarray[float32]    # (nnz,)
    sign: np.ndarray[int8]            # (nnz,) +1 / -1  from presynaptic nt_type
    input_idx: np.ndarray[int32]      # (n_in,)  sensory neurons that receive the board
    output_idx: np.ndarray[int32]     # (n_out,) descending/motor neurons read out by the heads
    super_class: np.ndarray[str]      # (n,)
    position: np.ndarray[float32]     # (n,3)
    meta: dict                        # selection parameters, counts per super_class, source hash
```
Sign rule (Dale's law, fly): `ACH → +1`, `GABA → -1`, `GLUT → -1` (glutamate is predominantly
inhibitory in the fly CNS via GluCl), `DA/SER/OCT/unknown → +1` (modulatory, treated excitatory).
The sign is a property of the presynaptic *neuron*, never of a single synapse: a neuron carries one
sign on every outgoing connection (Dale's law; `BrainGraph.validate()` asserts it). Neurons with no
`nt_type` in `neurons.csv` (~14%) get one label from the synapse-count-weighted majority of the
per-connection `nt_type` predictions over *all* their outgoing connections in the full connectome
(`neuron_nt_type`; ties alphabetical; independent of the selected subgraph); a neuron with no
labelled outgoing connection at all keeps +1.

Selection (`build_brain_graph(conn, cfg: GraphConfig)`), `GraphConfig` fields:
- `region: 'full' | 'central'` — `central` drops `super_class in {optic, visual_projection, visual_centrifugal}`.
- `max_neurons: int | None` — if set, keep the top-k neurons by total synapse count (in+out),
  but *always* keep the input and output sets.
- `min_syn: int = 5`
- `input_super_classes: ['sensory', 'ascending', 'sensory_ascending']`, `max_inputs: 2048` — choose deterministically:
  sort candidates by (out-degree desc, root_id) and take the first `max_inputs`.
- `output_super_classes: ['descending', 'motor']`, `max_outputs: 2048` — same rule with in-degree.
- Remove self-loops; remove neurons with zero degree after filtering (except input/output sets).
- Rows must be sorted by (post, pre) — CSR canonical form. `graph.save(path)` / `BrainGraph.load(path)` as npz.
Default for the real run: `region='full', max_neurons=None` (whole brain, ~139k neurons, ~2.7M synapses).
A `tiny` config (`max_neurons=2000`) exists for tests/smoke runs. Tests also use a random `toy_graph(n, nnz)`
generator, which must NEVER be used outside `tests/`.

## 3. Chess encoding (`chessenv/encoding.py` ⟷ `web/engine/encoding.js`)

### 3.1 Perspective
All encodings are from the **side to move's perspective**: if black is to move, mirror the board
vertically (`chess.Board.mirror()` semantics: swap colours *and* flip ranks) so that the mover is
always "white" moving up the board. Move indices are likewise in mirrored coordinates when black moves.
Square numbering: `chess.Square` (a1=0 … h8=63, index = rank*8+file). Mirroring a square = `sq ^ 56`.

### 3.2 Board planes — `encode_board(board) -> float32[NUM_PLANES, 8, 8]`, `NUM_PLANES = 20`
Plane order (all values 0/1 unless noted), indexing `[plane, rank, file]`:
- 0–5: mover's P, N, B, R, Q, K
- 6–11: opponent's P, N, B, R, Q, K
- 12: mover can castle kingside (all ones) · 13: mover queenside · 14: opponent kingside · 15: opponent queenside
- 16: en-passant target square (single 1)
- 17: constant ones (bias plane)
- 18: halfmove clock / 100, clipped to 1 (all squares)
- 19: repetition: 1 if the current position already occurred before in the game (all squares), else 0
`FLAT_INPUT = NUM_PLANES * 64 = 1280`; flatten in C order (plane-major).

### 3.3 Move index — `move_to_index(move, board) -> int`, `index_to_move(idx, board) -> chess.Move`, `NUM_MOVES = 4168`
Work in perspective coordinates (mirror from/to squares when black to move):
- Non-promotion, or promotion to queen: `idx = from * 64 + to` (0…4095).
- Under-promotion (N, B, R): `idx = 4096 + (from_file * 3 + dir) * 3 + piece` where
  `dir ∈ {0: capture-left (to_file = from_file-1), 1: push (same file), 2: capture-right}`,
  `piece ∈ {0: knight, 1: bishop, 2: rook}`. from_file 0..7 → 72 indices (4096…4167).
`legal_move_mask(board) -> bool[NUM_MOVES]`. Castling is encoded as the king's from/to squares
(e1g1 etc., python-chess standard UCI, not king-takes-rook).

### 3.4 Value target
`+1` mover wins, `0` draw, `-1` mover loses. Model value head outputs `tanh` in the same convention.

### 3.5 Test vectors
`tests/vectors/encoding.json`: list of `{fen, planes_sha256, legal_indices:[...], moves:{uci: idx}}`.
Generated by Python, checked by both pytest and `web/test/parity.test.mjs`.

## 4. Model (`model/flybrain.py`)

```python
@dataclass
class BrainConfig:
    graph_path: str
    steps: int = 8            # recurrent timesteps per position
    alpha: float = 0.5        # initial leak (learned per neuron via sigmoid(logit))
    activation: str = 'relu'  # 'relu' | 'gelu' | 'tanh' | 'satrelu' (sat * tanh(relu(x) / sat): bounded firing rates)
    sat: float = 10.0         # firing-rate ceiling for 'satrelu'
    input_dim: int = 1280
    num_moves: int = 4168
    weight_init_scale: float = 1.0   # multiplies log1p(syn_count) init
    dale: bool = True         # enforce signs
    dtype: str = 'float32'
    value_hidden: int = 256   # hidden width of the value MLP (0 = plain Linear(n_out, 1))
    vision: bool = True       # retina input (graph.retina_idx photoreceptors, one board square each) when the graph has one
    sensory_input: bool = True       # dense w_in projection into input_idx (at least one of vision / sensory_input)
    readout_steps: tuple[int, ...] = ()  # 1-based timesteps whose output_idx activity feeds the heads; () = final step
    neuromod: bool = False    # DA / SER / OCT synapses gate the others multiplicatively instead of adding
    central_dim: int = 0
    homeostatic: bool = False # per-neuron intrinsic gain exp(log_gain) on the synaptic input, calibrated on data at init
```
`readout_steps` entries must lie in `1..steps` and be strictly increasing (`effective_readout_steps` resolves `()`
to `(steps,)`); `to_dict()` stores it as a list (YAML-safe). Old configs without the new keys get these defaults.
Parameters:
- `syn_gain: (nnz,)` — one parameter over ALL synapses in canonical graph order; `w = sign * softplus(syn_gain)`;
  init `syn_gain = inverse_softplus(scale * log1p(syn_count) / normaliser)`
  where `normaliser` makes the expected row sum of |w| ≈ 1 (spectral-radius-ish control; document formula in code).
  With `dale=False` `w = syn_gain` (free sign), except under `neuromod`, where the modulatory synapses are always
  `softplus(syn_gain)` (positive gains).
- `bias: (n,)` init 0 · `leak_logit: (n,)` init logit(alpha).
- `w_in: (n_in, input_dim)` + `b_in: (n_in,)` — board planes injected into `input_idx` neurons each step (`sensory_input`;
  with `sensory_input=False` they are not created at all — absent from the state_dict and the optimiser).
- `w_ret: (n_ret, NUM_PLANES=20)` + `b_ret: (n_ret,)` — retina (`vision`, only when `graph.has_retina`): photoreceptor
  `k` (graph neuron `retina_idx[k]`) receives `w_ret[k] · planes[:, retina_square[k]] + b_ret[k]` each step, where
  `planes = x.view(B, 20, 64)` and `retina_square` is the square index `rank*8+file` in the mover's perspective (§3).
  Init `w_ret ~ N(0, 1/sqrt(20))`, `b_ret = 0`. Implemented as one batched matmul per square (photoreceptors grouped
  by square, padded). The model would simply add both injections to a neuron that is both a photoreceptor and an
  `input_idx` neuron, but `BrainGraph.validate()` keeps the two sets disjoint. `config.vision=True` on a graph without a
  retina runs with the sensory path only and **warns** (the builder produces such a graph silently when the column
  table is missing); with `sensory_input=False` as well the constructor **raises** — a network the board cannot reach
  must never train.
- `central_proj: Linear(n_central, central_dim)` (`central_dim > 0`) on the final activity of the neurons with
  `super_class == 'central'` (`central_idx`, buffer), in fp32 under the dense dtype; raises if the graph has none.
- `policy_head: Linear(head_in, num_moves)`, `value_head: Linear(head_in, 1)` (or small MLP `head_in→256→1`), with
  `head_in = n_out * len(readout_steps) + central_dim`: the concatenation of the `output_idx` activity at every
  readout timestep (in order) followed by the central features.
Neuromodulation (`neuromod`): the synapses are split by the PRE-synaptic neuron's `graph.nt_type` into ionotropic
(`ACH/GABA/GLUT/''`) and modulatory (`DA/SER/OCT`) sub-CSRs — a fixed index partition of the same `syn_gain`, so
checkpoints and exports are independent of the flag. A graph without `nt_type` gives no modulatory synapses (with a
warning; the dynamics are then the plain ones). The loop multiplies with the modulatory CSR compressed to its non-empty
rows and columns (`mod_rows`, `mod_cols`; full brain: 37,011 modulatory synapses, 12,261 gated rows = 9.1 % of the
neurons, 2,423 columns = 1.8 %, every DA/SER/OCT neuron) and gates those rows in place.
Dynamics (batch-first `h: (B, n)` is fine; SpMM internally transposes to `(n, B)` if faster):
```
h_0 = 0
for t in range(steps):
    inp = zeros(B, n); inp[:, input_idx] += x @ w_in.T + b_in                    # if sensory_input
    inp[:, retina_idx[k]] += planes[:, :, retina_square[k]] @ w_ret[k] + b_ret[k]  # if vision, for every k
    ion = SpMM(w_ion, h_t)                      # SpMM: out[b, post] = Σ_pre w[post,pre] * h[b, pre]
    pre = ion * (1 + tanh(SpMM(w_mod, h_t))) + bias + inp    # neuromod; without it pre = SpMM(w, h_t) + bias + inp
    h_{t+1} = (1 - a) * h_t + a * act(pre)      # a = sigmoid(leak_logit)
feat = concat([h_t[:, output_idx] for t in readout_steps] + [central_proj(h_T[:, central_idx])])   # last term if central_dim
policy_logits = policy_head(feat);  value = tanh(value_head(feat))
```
`forward(x: (B,1280)) -> (policy_logits (B,4168), value (B,1), h_T (B,n) optional)`.
`FlyBrain.from_checkpoint` loads a checkpoint written before the retina existed (no `w_ret`) with `vision=False`
(warning) even on a graph that has since gained a retina — only when `sensory_input=True` (a vision-only config has no
other input path, so the strict load then fails as a config error). A vision-only model drops a checkpoint's `w_in` /
`b_in`. `input_idx` / `output_idx` are persistent buffers: the model plays with the **checkpoint's** sensory / motor
sets even when the graph on disk was rebuilt with different ones (e.g. `data/brain/full.npz` rebuilt with a retina: 5
former inputs became photoreceptors, 1,204 of the 2,048 sorted positions shifted), `from_checkpoint` warns when they
differ, and the export (§8) writes the model's sets, never the graph's. Measured on the RTX 5080 (full brain, B=384,
steps=16, satrelu, fused + RCM ordering, bf16 autocast, idle GPU): 8.8 ms/timestep fwd+bwd and 8.1 GB peak with every
feature off (sensory only); vision instead of the sensory path 8.9 ms; readout_steps / central_dim cost < 2 %;
all on (vision + sensory + readout (8,16) + neuromod + central_dim=64) 11.3 ms/timestep (+28 %, almost all of it
neuromod) and +0.6 GB.
Provide `forward(x, return_activity=True)` for visualisation. Policy loss masks illegal moves with `-inf` before
cross-entropy. The module must be exportable: `state_dict()` + `BrainConfig` + graph path fully define the network.
`model/spmm.py` implements a custom `torch.autograd.Function` (do NOT rely on `torch.sparse_csr_tensor(values_param)`
autograd: its backward densifies to a 73 GiB matrix). Forward: `torch.mm(csr, h)` with `h: (n, B)`; backward:
`dvalues = torch.sparse.sampled_addmm(Z, g, h.t()).values()` (SDDMM on the same CSR pattern, `Z` = zeros CSR) and
`dh = csr_T @ g` using a precomputed transposed CSR (`crow_t, col_t, perm_t` so `values_t = values[perm_t]`).
Measured on the RTX 5080: 9.6 ms fwd+bwd per timestep at n=140k, nnz=3M, B=256, < 1 GB extra memory.
`torch.compile` cannot trace sparse tensors — keep SpMM inside the Function; compile only dense parts if at all.
Internally keep the hidden state as `(n, B)` (neuron-major) for cuSPARSE; expose batch-first at the module boundary.

## 5. Data (`data/`)

- `fly download --games --months 2014-01,2014-02` downloads `lichess_db_standard_rated_<YYYY-MM>.pgn.zst` into `data/pgn/`.
- `data/lichess.py::build_shards(pgn_paths, out_dir, min_elo=1800, max_positions=None, skip_openings=4, workers=N, val_every=0)`
  streams games with `zstandard` + `chess.pgn`, keeps rated standard games where both players ≥ `min_elo`
  and the game has a decisive or drawn result, skips the first `skip_openings` plies, and writes shards of
  `SHARD_SIZE=262144` positions as `data/shards/<name>-NNNNN.npz` with arrays:
  `planes: uint8 (S, 20, 8, 8)` (planes 18/19 stored as uint8 0-255 / 0-1 and rescaled on load),
  `move: int16 (S,)`, `value: int8 (S,)`, `elo: int16 (S,)`, `ply: int16 (S,)`.
  Positions are shuffled globally across the month using a buffer before sharding.
- Validation hold-out (`val_every=k`, `fly build-shards --val-every 50` by default): one game in `k`, chosen by a
  stable hash (`zlib.crc32`) of its `Site` URL, is routed *whole* into a second series `<name>.val-NNNNN.npz`
  (its own shuffle pool). The val set is therefore deterministic, reproducible across builder workers and
  disjoint from training **at the game level** — a shard-level split of position-shuffled shards is not (the
  other ~70 positions of every val game, with the same value label, would sit in training shards).
  `shards.list_shards(dir, name)` only ever returns `<name>-NNNNN.npz` (never a `.val` series, also with
  `name=None`); `load_split` returns the `.val` series when present (ignoring `val_fraction`) and otherwise
  falls back to holding out whole shards with a warning.
- `data/shards.py::ShardDataset` (torch IterableDataset, shard-level shuffle, worker-sharded) and `collate`.
- Also provide `positions_from_pgn(path)` generator for tests.

## 6. Training (`train/`)

`fly train --run <name> [--stage imitation|selfplay|all] [--config cfg.yaml] [--resume]`.
`TrainConfig` (yaml): graph selection, BrainConfig, `batch_size=256`, `lr=1e-3` (AdamW, cosine schedule with
warmup), `epochs`, `value_loss_weight=1.0`, `grad_clip=1.0`, `amp=false` (bf16 autocast for the dense parts only;
SpMM stays fp32 — keep it off, see README), `eval_every`, `checkpoint_every`, the self-play settings of stage 2
(below), `seed`.

Checkpoints: `runs/<run>/ckpt-<step>.pt` + `runs/<run>/latest.pt` containing
`{'config': TrainConfig(asdict), 'brain_config': ..., 'graph_path': ..., 'model': state_dict, 'optim': ..., 'step': int, 'stage': str}`.
Stage 2 additionally keeps `runs/<run>/best.pt` (same format: the best network, see gating below).
`flychess.load_brain(path_or_run) -> (FlyBrain, BrainGraph)`.

Stage 1 (imitation): cross-entropy on the human move (label smoothing 0.0) + MSE on value. Log every 20 steps.
During imitation a quick Elo (`fly` vs random / material) runs every `elo_every` steps with `elo_games` games
(0 disables it).

Stage 2 (self-play v2, `train/selfplay.py`). The v1 stage (64 games/iteration, lr 2e-4, training on self-play
targets only) made the searching player *weaker* than the imitation checkpoint (0–5 with `superfly`): it forgot
the human data. v2 rehearses the human data, gates every candidate and generates ten times more games.
`selfplay_iters` iterations (default 50), each:

1. **Generate** `selfplay_games_per_iter=512` games with the **best** network: batched PUCT MCTS (`mcts.py`,
   `mcts_sims=100` simulations per move, all leaf evaluations of the batch in one GPU forward, no virtual loss —
   one leaf per tree per step), Dirichlet noise at the root (`dirichlet_alpha=0.3`, `dirichlet_eps=0.25`),
   temperature 1 for the first `temperature_plies=30` plies then argmax, no resignation, `max_game_plies=300` cap
   (a draw). Up to `selfplay_lockstep_games=512` games are searched in one lockstep batch (more games run as
   several consecutive batches; `selfplay_workers` threads split the games further). Progress is logged every 64
   finished games (a `status` record with the pos/s and an ETA). Every position stores `(planes, π, z, q)`: the
   shard-format planes, the root visit distribution, the game result from the mover's perspective and the root
   search value; the `ReplayBuffer` (`replay_buffer_size=200000`, a ring) keeps π sparsely (top-k visited moves)
   with the legal-move indices.
2. **Train** `selfplay_train_steps_per_iter=400` AdamW steps at `selfplay_lr=1e-4` (constant), batch
   `selfplay_batch_size=384`, the same two parameter groups and betas (0.9, 0.95) as imitation (no weight decay on
   biases, leaks, gains — `log_gain` included — and, under Dale's law, `syn_gain`). The weights and the optimizer
   state are snapshotted before training (the "best" snapshot the gate can fall back to). Every batch is mixed: `selfplay_human_mix=0.5` of it
   streams from the imitation shards (`shards_dir` / `shard_name` series and repeat factors exactly as in stage 1;
   policy target = one-hot stored move, *unmasked* cross-entropy like stage 1; value target = the stored value)
   and the rest samples the replay buffer (policy target = π with illegal moves masked; value target =
   `selfplay_value_blend · z + (1 − selfplay_value_blend) · q`, default blend 0.5). Loss = cross-entropy +
   `value_loss_weight` × MSE over all rows; `train` records carry `n_self`, `n_human`, `human_mix` and the per-half
   `policy_loss_self/human`, `value_loss_self/human`, `top1_self/human`. `selfplay_human_mix=0` reproduces v1's
   targets, `1` is pure rehearsal.
3. **Gate** (`selfplay_gate=true`): the candidate (the weights after training) plays the best weights, both as
   `superfly` (`FlyEngine`, `selfplay_gate_sims=100` simulations, deterministic), `selfplay_gate_games=40` games
   on paired distinct openings (each opening twice with the colours swapped). Openings are positions after 8–12
   plies of the iteration's own games; only when those yield too few distinct positions is the list topped up from
   a fixed set of 20 standard first moves (`fixed_openings()`: e4 e5, d4 d5, …) — the one place where moves not
   chosen by the fly enter a match (the fly still chooses every move *from* those positions; on the full brain
   with 512 games the top-up is practically never used, and `fixed_openings_used` on the record says when it was).
   The candidate is promoted to best iff its score fraction (wins + ½ draws) / games ≥
   `selfplay_gate_threshold=0.55`; otherwise it is discarded: the model reloads the best weights *and* the
   optimizer reloads the pre-training snapshot (the moments of a direction just judged worse go with it), the
   replay buffer is kept and the iteration still counts. An `elo` record with `opponent='previous-best'` logs
   `games, wins, draws, losses, elo_estimate, score, threshold, promoted, openings, fixed_openings_used, truncated,
   mean_plies, sims, iteration, seconds`, a `status` line says "candidate promoted/rejected" and the first gate
   game is logged as a `game` record (`source='eval', opponent='previous-best'`, bypassing the per-minute limit).
   **A gate that cannot run never promotes**: if the match raises (an evaluator bug, CUDA OOM, …) or returns no
   result (`FlyEngine` unimportable), the candidate is rejected exactly as above, a `status` record with
   `gate_error` (and `promoted=false, gated=false`, no `iteration`) is written, `latest.pt` / `best.pt` are saved
   with the best weights and the stage stops with `RuntimeError` so the operator notices; `--resume` redoes the
   iteration. `selfplay_gate=false` (or `selfplay_gate_games=0`) always promotes.
4. **Evaluate** every `selfplay_eval_every_iters` iterations (and after the last): `elo_games` games as `fly`
   against a random opponent (`opponent='random'`) on paired openings; without a gate also against the previous
   iteration's weights (`opponent='previous-iteration'`).
5. **Checkpoint**: `ckpt-<step>.pt` and `latest.pt` hold the best network (= the promoted candidate or the
   unchanged best), copied to `best.pt`; the `status` record `{stage: 'selfplay', iteration: n, promoted, gated}`
   marks the iteration as done.

Resume (`--resume`): finished iterations are counted from those `status` records, the optimizer state comes from
`latest.pt` when it was written by this stage at the resumed step (its moments are kept, but `selfplay_lr`, the
betas and `weight_decay` are re-applied from the config, so a `--set selfplay_lr=…` on resume takes effect), and
`best.pt` (when at least one iteration finished) is loaded as the best network — so an interrupt during training
or gating never leaks an ungated candidate into the games. Ctrl-C and any `RuntimeError` (a non-finite loss, a
gate that could not run) restore the best weights and optimizer snapshot before saving `latest.pt` / `best.pt`.
The replay buffer is not persisted.
The tiny preset (`configs/tiny.yaml`) keeps the gate at 2 games × 4 simulations.

Python API: `flychess.train(run='fly1', stage='all', **overrides)`.

## 7. Metrics (`train/metrics.py` ⟷ dashboard)

`runs/<run>/metrics.jsonl`: one JSON object per line, always with `"t": unix_time, "step": int, "kind": str`. Kinds:
- `train`: `{loss, policy_loss, value_loss, top1, top3, lr, pos_per_sec, gpu_mem_gb, epoch, stage}`
- `eval`: `{val_loss, val_top1, val_top3, val_value_mse}`
- `elo`: `{opponent, games, wins, draws, losses, elo_estimate}` (+ `score, threshold, promoted, openings, fixed_openings_used, truncated, mean_plies, sims, iteration, seconds` from the self-play gate, `opponent='previous-best'`)
- `game`: `{pgn, result, moves: int, source: 'selfplay'|'eval'}` (a sample game, ≤ 1 per minute; the self-play gate game is written regardless, with `opponent='previous-best'`)
- `activity`: `{neuron_idx: [...], values: [...]}` sampled 2048 neurons' activity for a heatmap (≤ 1 per 30 s)
- `status`: `{message, stage, total_steps, eta_s}` (a finished self-play iteration adds `iteration, promoted, gated`; a gate that could not run adds `gate_error, promoted=false, gated=false` without `iteration`)
`runs/<run>/run.json` holds the static description (config, graph meta, start time).

## 8. Web model format (`export/web.py` ⟷ `web/engine/loader.js`)

`web/model/brain.json` (header) + `web/model/brain.flyb` (little-endian binary blob):
header lists `{name: string, dtype: 'f16'|'f32'|'i8', shape: [...], offset: int, length_bytes: int, scale?: float}`
for each array, in this order: `csr_indptr (i32)`, `csr_indices (i32)`, `w (f16, already signed = sign*softplus(gain))`,
`bias (f32)`, `alpha (f32, = sigmoid(leak_logit))`, `input_idx (i32)`, `output_idx (i32)`, `w_in (f16, [n_in,1280])`,
`b_in (f32)`, `policy_w (f16, [num_moves, n_out])`, `policy_b (f32)`, `value_w (f16, [1 or hidden, n_out])`, `value_b`,
(`input_idx` / `output_idx` and `header.n_in` / `n_out` come from the MODEL's buffers — the checkpoint's sets, §4 — not
from the graph; with `sensory_input=false` `w_in` / `b_in` are written with zero rows, `[0, 1280]` / `[0]`, and
`input_idx` stays for the visualiser),
(if MLP value head, also `value_w2`, `value_b2`), plus `positions (f16, [n,3])` normalised to [0,1] for the brain visualiser,
and `super_class (u8, [n])` with a legend in the header, then `node_perm (i32, [n])`. The blob's neuron order is the
model's cache-friendly compute ordering (`header.neuron_order = 'rcm'`, reverse Cuthill-McKee; every per-neuron and
per-connection array is permuted consistently, so the blob is self-contained); `node_perm[i]` is the canonical graph
index of blob neuron `i`. Header also has `steps, activation, n, nnz, n_in, n_out, num_moves,
num_planes, run_name, train_steps, exported_at, elo_estimates, total_bytes, gzip_bytes, blob_sha256`; the loader verifies every
downloaded (and every cached) blob against `total_bytes` / `blob_sha256` and versions blob URLs by the header so a stale HTTP cache
can never pair a new header with old bytes.

After `node_perm` come the optional-feature arrays (§4), always present and in this order, with zero length when the
feature is off (2-D arrays keep their second dimension, e.g. `retina_uv [0, 2]`, `w_ret [0, 20]`), so the parser is
uniform; a blob written before they existed simply lacks them, which the engines treat as "feature off":
`retina_idx (i32, [n_ret], blob neuron index of every photoreceptor)`, `retina_square (u8, [n_ret], board square
rank*8+file, mover's perspective)`, `retina_uv (f16, [n_ret, 2], eye-map coordinates in [0,1], left eye u < 0.5)`,
`retina_eye (u8, [n_ret], 0 left / 1 right)`, `w_ret (f16, [n_ret, num_planes])`, `b_ret (f32, [n_ret])`,
`mod_indptr (i32, [n+1] or [0])`, `mod_indices (i32, [nnz_mod])`, `w_mod (f16, [nnz_mod], positive)`,
`central_idx (i32, [n_central], blob neuron indices)`, `central_w (f16, [central_dim, n_central])`,
`central_b (f32, [central_dim])`, `retina_type (u8, [n_ret], index into header.retina_type_legend)`.
The per-photoreceptor arrays are in the graph's `retina_idx` order; `retina_idx` / `central_idx` / `mod_*` are in the
blob's (compute) neuron order like everything else. Header fields for them: `vision` (bool — false when the model has
no retina path, whatever the config said), `sensory_input` (bool), `readout_steps` (list of 1-based steps, e.g.
`[8, 16]`; the head input is the concatenation of `h_t[output_idx]` at those steps followed by the central features),
`neuromod` (bool), `central_dim`, `n_ret`, `n_central`, `retina_type_legend` (sorted list of type names), `retina`
(the graph's `meta['retina']`), `nnz_mod` and `nnz_total`. `export_web` warns which of these features the blob relies
on (the browser engine must implement §8 for them). Under `neuromod` the main CSR (`csr_indptr`, `csr_indices`,
`w`, and `header.nnz`) holds the ionotropic synapses only — the loop the JS engine already runs — and the DA/SER/OCT
synapses live in `mod_*`; `nnz_total` is the graph's synapse count. Per timestep the engines compute
`pre = ion * (1 + tanh(mod))` with `ion = W @ h`, `mod = W_mod @ h` (both over all `n` rows; a row without modulatory
synapses has `mod = 0`), add `bias`, the `w_in` injection (if `sensory_input`) and the retina injection
`w_ret[k] · planes[:, retina_square[k]] + b_ret[k]` into `retina_idx[k]` (if `vision`), then apply the leaky update.
`flychess.export.numpy_forward` implements all of it and is the reference for `web/engine/flybrain.js`.
Offsets are 8-byte aligned. `csr_indices` is stored as-is (i32) — simplicity over size. Target ≤ 45 MB raw for the full brain;
`export-web` also writes `brain.flyb.gz` and the loader fetches the `.gz` when present, decoding with `DecompressionStream('gzip')`
while streaming progress via `response.body.getReader()`, then caches the decoded buffer in the Cache API / IndexedDB.
Measured in-browser cost: ~4 ms per recurrent timestep for 3M synapses in plain JS → ~35 ms per forward pass at 8 steps.
JS `FlyBrain.forward(planesFloat32) -> {policy: Float32Array(4168), value: number, activity: Float32Array(n)}` must match
Python within `1e-2` on the compared logits and the value for the vectors in `tests/vectors/model.json`, generated by
`export/testvectors.py` from the exported f16 weights (so both sides use identical rounded weights). The file has a
header (`format: 'flychess-model-vectors', version, run_name, exported_at, blob_sha256, generated_at, tolerance`) and a
`vectors` list of `{fen, moves?, n_legal, argmax, top: [[idx, uci, logit], ...], value}` where `top` holds the `TOP_K = 20`
legal moves with the highest logit; the parity test compares those logits, the value and the argmax (illegal moves and
the remaining legal logits are not compared).

## 9. CLI (`cli.py`)

```
fly download [--connectome] [--games] [--months 2014-01,...]        # defaults: both
fly build-brain [--region full|central] [--max-neurons N] [--out data/brain/<name>.npz]
fly build-shards [--months ...] [--min-elo 1800] [--workers 16] [--val-every 50]
fly train --run NAME [--stage imitation|selfplay|all] [--config cfg.yaml] [--resume] [--steps N] [--tiny]
fly dashboard [--run NAME] [--port 8765]                            # http://localhost:8765
fly play [--run NAME | --ckpt path] [--difficulty larva|fly|superfly] [--color white|black] [--gui]
fly eval --run NAME [--games 50] [--opponent random|material|stockfish|<other-run>]
fly export-web --run NAME [--out web/model] [--quant f16|i8]
fly test-vectors --run NAME
```
Difficulty: `larva` = sample policy with temperature 1.2 (no search); `fly` = argmax policy + 1-ply value check;
`superfly` = MCTS 200 simulations (the browser worker uses 100 to keep moves ≈5 s). All three use the fly brain exclusively.

## 10. Dashboard (`dashboard/`)

`fly dashboard --run NAME` serves `static/index.html` at `/`, `GET /api/runs` (list runs), `GET /api/run/<name>`
(run.json + per-kind history: the whole train / eval / Elo series, last game + activity, log tail), `WS /ws/<name>[?after=N]`
(init, then each new metrics line followed by a byte cursor; `after` resumes a reconnect without resetting the UI).
UI: run selector; live charts
(loss / policy / value / top-k / lr / throughput / Elo) using a small vendored charting lib or hand-rolled canvas;
status bar (stage, step, ETA, GPU mem); latest self-play game rendered on a board (auto-replay); neuron activity
heatmap (positions from graph → 2D projection); log tail. Dark theme, fly-themed accents. No build step.

## 11. Website (`web/`)

Static, no build step, ES modules. `index.html` loads `app.js` → creates a Web Worker (`engine/worker.js`) that
fetches `model/brain.json` + `brain.flyb` with a progress bar ("growing the fly brain… 12 MB / 31 MB"), instantiates
`FlyBrain` (`engine/flybrain.js`: pure typed-array CSR SpMV, identical math to §4) — and, when the browser has WebGPU,
`FlyBrainGPU` (`engine/flybrain-gpu.js`: the same forward pass as f32 WGSL compute shaders, one command buffer per
forward, asynchronous `forward()`; the worker falls back to `FlyBrain` when WebGPU is missing or the device is lost and
reports `backend: 'webgpu' | 'js'`) — and answers `{type:'move', fen,
difficulty}` with `{move, policyTop, value, activity(sampled), thinkMs, backend}`. `engine/mcts.js` implements the same MCTS
as Python for `superfly`. Main thread: board UI (SVG, drag & click, legal-move dots, last-move highlight, promotion
picker), fly avatar with mood driven by the value head (confident / nervous / panicking / smug), live "fly brain"
canvas showing sampled neuron activity as glowing dots at connectome positions, move commentary from the
policy/value ("the fly is 87% sure about this one"), share-result card (canvas → PNG copy), local leaderboard
(localStorage), sound effects (optional toggle), party mode (pass-and-play tournament bracket vs. the fly, multiple
humans take turns, fastest win wins). Mobile-friendly. Footer credits FlyWire (data CC BY-NC 4.0 from Codex; the
Zenodo mirror 10.5281/zenodo.10676866 is CC BY 4.0) with citations.
`scripts/deploy-pages.sh` publishes `web/` (with `web/model/`) to a `gh-pages` branch; model files ≤ 100 MB each.
