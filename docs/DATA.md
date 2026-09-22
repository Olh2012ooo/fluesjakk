# Training data

Two sources feed the imitation stage, both written as the shard format of SPEC §5 (`flychess/data/shards.py`)
into `data/shards/` and mixed at training time by `shard_name`.

## Shard format additions

* `value_scale` — optional scalar npz key (missing = 1). The float value target is `value / value_scale`.
  Outcome shards (`lichess*`) keep the old five-array layout (`value ∈ {-1, 0, +1}`); engine-evaluated shards
  store `value = round(v · 127)` as int8 with `value_scale = 127`. `read_shard` returns the scale as
  `'value_scale'`, `ShardDataset` yields it per item and `collate` divides, so a batch may mix both kinds.
* `load_split(shard_dir, val_fraction, seed, name)` — `name` may now be a list of series with repeat factors:
  `'lichess2014,lichess2015,evals:3'` (or the parsed `[('lichess2014', 1), ('lichess2015', 1), ('evals', 3)]`
  from `parse_shard_names` / `TrainConfig.shard_series()`). Each series is split on its own (its `.val` series
  when present, else the shard-level fallback) and the training lists are concatenated with a series repeated
  `k` times — `ShardDataset` shuffles shard order, so the series is sampled `k×` as often and `count_positions`
  (hence steps per epoch) counts it `k×`. Validation files are never repeated. `TrainConfig.shard_name` accepts
  the string and validates it; the trainer passes it to `load_split` unchanged.

## Lichess games (`flychess/data/lichess.py`)

`fly build-shards` — monthly PGN dumps → positions with the human move and the game outcome (`value_scale` 1),
game-level hold-out `<name>.val`. See SPEC §5.

## Lichess engine evaluations (`flychess/data/evals.py`)

`build_eval_shards(source, out_dir, name='evals', max_positions=30_000_000, min_depth=20, workers=8, ...)`
streams `https://database.lichess.org/lichess_db_eval.jsonl.zst` (CC0, 22 GB / ~270 M positions; the stream
is decompressed on the fly and closed after `max_positions`, so 30 M positions transfer ≈ 2.5 GB) or a local
`.jsonl[.zst]`, decodes the lines in worker processes and writes `<name>-NNNNN.npz` through the same shuffle
pool as the PGN pipeline. Per line the deepest eval is used (`depth < min_depth` skipped); the first pv's first
UCI move is the policy target (illegal / unparseable moves and positions without legal moves are skipped, ~50
per million); the value is Lichess's win-probability curve `v = 2 / (1 + exp(-0.00368208 · cp)) − 1` of the
White-point-of-view score, negated when Black is to move (mate → ±1), quantised to int8 with
`value_scale = 127`. `elo = 0`. The dump's FENs have four fields, so the halfmove plane is 0 and `ply` is 0
(White to move) / 1 (Black). 2 % of positions (a `crc32` hash of the FEN) go to `<name>.val-NNNNN.npz`.

```python
from flychess.data import build_eval_shards
stats = build_eval_shards(max_positions=30_000_000, workers=16)   # -> data/shards/evals-*.npz
print(stats.summary(), stats.depth_summary(), stats.value_summary())
```

Measured (RTX 5080 box, 32 cores, 16 workers, first 3 M positions of the dump, 2026-09-13): 3.76 M lines →
3.0 M kept in 25 s (≈120 k positions/s end to end, ≈250 k/s steady state; download-bound at ~12–26 MB/s
compressed), 2,939,933 train + 60,067 val, 12 + 1 shards, 107 MB on disk. 13 % of lines skipped as
`depth < 20`, 17 % of kept positions are forced mates (the head of the dump is endgame-heavy), depth of the
used eval: 55 % in 20–29, 22 % in 30–39, 12 % ≥ 60 (depth 245 = mate / tablebase). Value histogram: 42 % in
[−0.1, 0.1), 19 % at |v| ≥ 0.9. A 30 M build takes ~5 minutes.

Training on both: `fly train --set shard_name=lichess2014,lichess2015,evals:3`.
