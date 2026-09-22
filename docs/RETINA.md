# The fly's retina — how photoreceptors look at the chess board

`flychess/connectome/retina.py` · `BrainGraph.retina_*` (graph.py) · `load_column_assignment` (load.py) · tests in `tests/test_retina.py`

## 1. What the fly sees

Before v3 the board reached the brain through a dense learned projection `w_in (2048 x 1280)` into the
2048 best-connected sensory / ascending neurons — olfactory receptor neurons, Johnston's organ cells,
taste bristles… none of them visual. The retina replaces that for the visual system: every
**photoreceptor** of the connectome that can be placed on an ommatidial column looks at **exactly one
square** of the board, and the 20 board planes of that square are its only input
(`pre[retina_idx[k]] += w_ret[k] · planes[:, :, retina_square[k]] + b_ret[k]`, `w_ret: (n_ret, 20)`).
Neighbouring photoreceptors look at neighbouring squares, the two eyes split the board down the
middle, and the visual signal then flows through the real lamina → medulla → lobula → central-brain
wiring. The network is still the only thing that chooses moves; the retina only decides *where each
photoreceptor looks*.

The board is always encoded from the side to move's perspective (SPEC §3.1), so the fly sits behind
its own pieces looking up the board: rank 1 is at the bottom of its visual field (ventral), rank 8 at
the top (dorsal), file a to its left and file h to its right. With the default `retina_field='split'`
the **left eye sees files a–d** and the **right eye files e–h** (each eye's field is an 8 × 4 block of
squares); `retina_field='full'` makes both eyes see all 64 squares (a fully binocular fly), and
`retina_eyes='left' | 'right'` uses a single eye (which then always sees the whole board).

## 2. Data: the optic-lobe column assignment

`data/connectome/column_assignment.csv.gz` (FlyWire Codex, snapshot 783; Matsliah et al. 2024) has
45,528 rows `root_id, hemisphere, type, column_id, x, y, p, q`: for 31 columnar cell types
(R7, R8, L1–L5, Mi1, Mi4, Mi9, Tm1–4, Tm9, Tm20, Tm21, T1, T2, T2a, T3, T4a–d, T5a–d, C2, C3) the
ommatidial column each neuron belongs to. There are **785 columns in the left eye and 796 in the
right eye** (of ~800 ommatidia per eye); every (hemisphere, column_id) has exactly one R7 / R8 at most.

### Coordinate convention (checked on the table, `tests/test_retina.py::test_real_column_table_lattice`)

`(p, q)` are hex **axial** coordinates whose axes are 120° apart: the six neighbours of a column are
`(±1, 0)`, `(0, ±1)` and `±(1, 1)` (693 of the 796 right-eye columns have all six; with the other
convention `±(1, -1)` only 668 do). The file's `(x, y)` are the same lattice squashed to integers:
`y = p + q` and `x = floor((q - p) / 2)` hold for every row. The regular cartesian embedding used here is

    X = (q - p) · √3 / 2        Y = (p + q) / 2

which puts every column's nearest neighbour at distance exactly 1.000 (all 796 columns). Each eye spans
X ∈ [-15.6, 14.7] and Y ∈ [-14.5, 15.0] — a roughly circular field 30 columns across. Coordinates are
normalised per eye to `[0, 1]` with the min / max over that eye's *unique columns* (a crowded column
cannot bias the range; the lattice has no outliers).

**Orientation.** Correlating the column coordinates with the neurons' FAFB positions (`coordinates.csv.gz`):
`Y` correlates −0.98 / −0.99 with FAFB *y* (which grows ventrally) for R7, L1, Mi1 and T4 in both
hemispheres → **+Y is dorsal in both eyes**. `X` correlates +0.92 with FAFB *z* (posterior) for Mi1 in
both hemispheres → +X is the posterior medulla, i.e. — after the outer optic chiasm inverts the
anterior–posterior axis between lamina and medulla — the **frontal** part of the visual field, again in
both eyes. Hence

    v       = Y normalised            (0 = ventral = rank 1, 1 = dorsal = rank 8)
    u_board = X_n (left eye)          (0 = lateral = file a, 1 = frontal = file d)
            = 1 − X_n (right eye)     (0 = frontal = file e, 1 = lateral = file h)

so the frontal edges of the two eyes meet at the d|e boundary. A global flip of either axis would only
relabel which photoreceptor sees which square (the learned `w_ret` is per photoreceptor), so this
choice matters for the visualisation and the spatial-locality structure, not for what the network can
compute.

### `retina_uv` (visualisation)

The eye map drawn by the website: `u = 0.5 · u_board · 0.998` for the left eye (`u ∈ [0, 0.499]`, so it
stays below 0.5 after f16 rounding in the web export) and `u = 0.5 + 0.5 · u_board` for the right eye
(`u ∈ [0.5, 1]`); `v` as above. Both eyes side by side, lateral edges at the outside, dorsal up.

## 3. Placing photoreceptors on columns

| type  | in connectome | placed directly (in the table) | via lamina partner | unplaced | no connection ≥ 5 syn | receive-only | **in the retina** |
|-------|--------------:|-------------------------------:|-------------------:|---------:|----------------------:|-------------:|------------------:|
| R1-6  | 8,456         | 0                              | 4,141              | 4,315    | 0                     | 0            | **4,141**         |
| R7    | 1,338         | 1,253                          | 0                  | 85       | 223                   | 376          | **654**           |
| R8    | 1,357         | 1,276                          | 0                  | 81       | 235                   | 293          | **748**           |
| total | 11,151        | 2,529                          | 4,141              | 4,481    | 458                   | 669          | **5,543**         |

* **R7 / R8** have column assignments directly (`direct`); the 85 + 81 without one are not in the table.
* **R1-6** (the outer photoreceptors, `consolidated_cell_types` primary type `R1-6`) are not in the
  table. Each takes the column of its **strongest postsynaptic lamina partner** among
  `retina_partner_types = ('L1', 'L2', 'L3')` (all of which are in the table): synapse counts are summed
  per (R1-6, column) over *all* its outgoing connections in the full connectome and the column with the
  most synapses wins — a plurality (argmax), ties → smaller column id. In practice the winning column
  holds almost all of a cell's lamina synapses: median 100 %, 5th percentile 100 %, minimum 37.7 %
  (a single cell below 50 %), no exact ties. 4,141 of the 8,456 R1-6 can be placed; the other 4,315
  have no L1/L2/L3 connection at all (2,993 of them have no outgoing connection whatsoever —
  reconstruction fragments; including L4/L5 or any table type as partner would add only 6). The 4,141
  sit on 1,200 distinct columns (the whole retina covers 1,416 of the 1,581). Every partner-placed cell
  lands in the eye its `side` annotation says (`side_mismatch = 0`).
* Placement is computed on the **full connectome**, independent of the graph selection (like the
  per-neuron transmitter rule), so a photoreceptor's square never depends on `GraphConfig`.
* Photoreceptors that end up without a column are ordinary sensory neurons: they stay candidates for the
  generic `input_idx` exactly as before.
* A photoreceptor only joins the retina if it has **at least one outgoing connection of ≥ `min_syn`
  synapses** on the filtered edge set — it must be able to drive something. 458 R7 / R8 have a column
  but no connection at all (they are not in the graph either, exactly as in v2), and another 669
  R7 / R8 only *receive* (mostly from the other photoreceptor of their column): they stay in the graph
  as ordinary sensory neurons, but a `w_ret` row for them would be dead weight. This is what keeps the
  neuron set of `full.npz` identical to v2 (§7).

## 4. Columns → squares

Each eye's columns are binned onto its block of squares (4 files × 8 ranks for `split`, 8 × 8 for
`full`). Binning is nested and rank-based: first the columns are cut into file bins along `u_board`,
then, within every file bin, into rank bins along `v` (ties broken by the other coordinate, then by
column id, so the result is fully deterministic). `retina_binning`:

| binning (`split`, full brain) | columns / square left (min · median · max) | right | photoreceptors / square (min · median · max) | empty squares |
|-------------------------------|-------------------------------------------:|------:|---------------------------------------------:|--------------:|
| `uniform` (equal rectangles)  | 0 · 28 · 36                                | 0 · 28 · 36 | 0 · 90.5 · 211                         | 3             |
| `quantile` (equal column count) | 24 · 25 · 25                             | 24 · 25 · 25 | **10** · 88 · 173                     | 0             |
| **`weighted`** (default: equal photoreceptor count) | 9 · 17 · 80          | 14 · 23 · 53 | **67 · 87 · 107**                   | 0             |

Uniform binning of a round eye leaves corner squares empty, so it is only kept for reference.
Quantile binning gives every square 24–25 columns, but because R1-6 placement is incomplete and uneven
(the left eye has 2,259 retina photoreceptors, the right 3,284) some squares would receive only 10
photoreceptors and others 173. The default therefore balances the *photoreceptors that end up in the
graph* (weighted quantiles: bins are cut at equal cumulative photoreceptor count; every bin is forced to
keep at least one column). Squares are still contiguous patches of the eye — 9 to 80 columns each — and
every square gets 67–107 photoreceptors (left eye 67–77, right eye 97–107):

```
photoreceptors per square, full.npz (rank 8 at the top, files a…h)
 8 |   70   71   71   70  103  100  104  103
 7 |   72   69   68   70  104  104  101  103
 6 |   73   71   76   69  102  104  102  103
 5 |   69   72   70   70   97  107  102  102
 4 |   72   68   69   72  106   99  104  104
 3 |   73   77   68   68  104  103  104  103
 2 |   67   67   76   74   99  102  102  101
 1 |   69   70   68   70  104  104  103  101
```

Per type: R7 0–30 per square (median 9), R8 1–41 (median 10.5), R1-6 0–89 (median 70.5).

The rank bins are nested inside the file bins, so "every square gets ≥ 1 column" needs at least 8 columns
per file bin — 32 (`split`) / 64 (`full`) columns per eye; the real eyes have ~790. A smaller synthetic
table can leave squares empty; `square_stats` reports `empty_squares` and `retina_candidates` warns.

## 5. Graph format and options

`BrainGraph` gains optional fields (empty when absent from an older npz; `validate()` checks them):

| field           | dtype / shape        | meaning |
|-----------------|----------------------|---------|
| `nt_type`       | str `(n,)`           | per-neuron transmitter after the majority fallback (`''` unknown); `sign == NT_SIGN[nt_type[pre]]` |
| `retina_idx`    | int32 `(n_ret,)`     | graph index of every retina photoreceptor (unique, disjoint from `input_idx`) |
| `retina_square` | int8 `(n_ret,)`      | board square `rank * 8 + file`, mover's perspective (index into `planes.view(B, 20, 64)`) |
| `retina_uv`     | float32 `(n_ret, 2)` | eye-map coordinates in `[0, 1]`, left eye `u < 0.5`, right eye `u ≥ 0.5` |
| `retina_type`   | str `(n_ret,)`       | `'R1-6' | 'R7' | 'R8'` |
| `retina_eye`    | int8 `(n_ret,)`      | 0 = left, 1 = right |

Properties `n_ret`, `has_retina`, `has_nt_type`. `meta['retina']` records everything above: the options,
columns per eye, columns per square and photoreceptors per square (per eye and overall:
min / median / max / empty squares), the per-type placement table, `placed` (placeable photoreceptors),
`candidates` (after the connection filter), the cap, and the column table's path / size / mtime.

`GraphConfig` options (`fly build-brain` needs no new flags for the defaults):

| option                 | default                | |
|------------------------|------------------------|---|
| `retina`               | `True`                 | build the retina when `column_assignment.csv.gz` exists (otherwise a note in `meta['retina']`, no retina) |
| `retina_types`         | `('R1-6', 'R7', 'R8')` | photoreceptor cell types |
| `retina_eyes`          | `'both'`               | `'left'` / `'right'` use one eye, which then sees the whole board |
| `retina_field`         | `'split'`              | `'full'`: both eyes see all 64 squares |
| `retina_binning`       | `'weighted'`           | `'quantile'` (equal columns per square), `'uniform'` |
| `retina_partner_types` | `('L1', 'L2', 'L3')`   | lamina partners that place R1-6 |
| `retina_max`           | `None`                 | cap `n_ret`, round-robin over squares (within a square: photoreceptors that share their strongest post-synaptic partner first, then most synapses); with `max_neurons` the cap defaults to `max_neurons / 8` rounded up to a multiple of 64 (tiny: 2000 → 256, 4 per square) |
| `retina_connect`       | `True`                 | with `max_neurons`: force-keep a shortest path from every retina photoreceptor to an output neuron (§6) |

`meta['retina']` also records `hops_to_output` (synaptic hops from the photoreceptors to the nearest
output neuron in the final graph: min / median / max / unreachable) and `connect` (how many
photoreceptors were wired through and how many neurons the paths cost).

Selection changes in `build_brain_graph`: retina neurons are *roles* (kept through `max_neurons`
pruning, like inputs / outputs), and `input_idx` is chosen from the input super classes **minus the
retina** (5 of the 2048 v2 inputs were R7 / R8; they are replaced by the next sensory neurons in the
same deterministic order, so `n_in` stays 2048). `build_retina(conn, graph, cfg, columns)` computes the
same arrays for an existing graph.

## 6. Reaching the outputs

A photoreceptor is only useful if its activity can reach the neurons the heads read (`output_idx`:
descending / motor). In the full graph the retina's signal reaches an output in **2–6 synaptic hops**
(median 4; R1-6: 3–5 hops, mostly 4 — lamina → medulla → lobula / visual projection → descending;
R7 / R8: 2–6); 40 of the 5,543 photoreceptors (7 R1-6, 14 R7, 19 R8) cannot reach any output at all —
their only targets are the other photoreceptor of their column or a lamina fragment with no
downstream partner. With `BrainConfig.steps = 8` (16 for `runs/fly2`) every reachable photoreceptor
influences the heads within one forward pass.

A **pruned** graph is a different matter. `max_neurons` keeps the neurons with the most synapses, and
no lamina cell makes it into a 2000-neuron graph: in the first v3 `tiny.npz` only 12 of the 256
photoreceptors had any edge inside the graph — six closed R7 ↔ R8 pairs — so toggling `vision` changed
nothing outside the retina (max policy / value difference 0.0). `retina_connect` (default on) fixes
this in two steps, both deterministic:

1. **cap by shared partners** — `balanced_cap` still takes photoreceptors round-robin over the squares,
   but within a square it ranks them by the summed synapse count of the group sharing their strongest
   (non-photoreceptor) post-synaptic partner, then by their own synapse count. The 4 photoreceptors a
   tiny square gets therefore converge on the same L1 / L2 / Tm20 … instead of on 4 different cells;
2. **output paths** — a backward BFS from the output neurons over the filtered edge set gives every
   neuron its hop distance to the nearest output; from each retina photoreceptor (in the round-robin
   order) the builder walks to a neighbour one hop closer, preferring neurons that are kept anyway,
   then the strongest connection, then the smaller root id, and force-keeps the walked neurons as
   roles (paths merge). Path neurons displace the lowest-ranked top-k neurons; they never push `n`
   above `max_neurons` — a path that does not fit the remaining budget is skipped and its photoreceptor
   counted in `meta['retina']['connect']['unconnected']`.

`tiny.npz` now: 256 photoreceptors (R1-6 178 / R7 26 / R8 52, 4 per square) all wired through in
**3–5 hops** (54 / 200 / 2) at the cost of 239 path neurons (of a 1,488 budget), sharing 1,520 of its
2,000 neurons with `tiny-v2.npz`; their partners are L2 (42), Tm20 (22), L1 (18), Dm2, Dm9, MLt1,
Dm8b, Tm5a …. Toggling `vision` on a randomly initialised `FlyBrain(tiny)` now changes the activity of
1,013 non-retina neurons and 85 of the 128 output neurons (`tests/test_retina.py::
test_real_tiny_graph_retina_drives_the_outputs`; the synthetic version is
`test_vision_reaches_the_heads_on_a_pruned_fixture_graph`).

## 7. Files and backward compatibility

`data/brain/full.npz` and `tiny.npz` were rebuilt with `fly build-brain --force` / `--tiny --force`; the
previous files are kept as `full-v2.npz` / `tiny-v2.npz`. The new `full.npz` has **exactly the same
134,209 neurons in the same order and the same 2,700,513-entry CSR, signs and outputs** as v2 (asserted
by `tests/test_retina.py::test_real_full_graph_retina`), so the `runs/fly1` / `runs/fly2` checkpoints
still load and **reproduce bit-exactly** on it: they carry no `w_ret` and are loaded with `vision=False`
(a warning says so), and `FlyBrain` persists `input_idx` / `output_idx` as buffers, so the model keeps
playing with the checkpoint's own 2048 sensory neurons (5 of which are R7 / R8 that are now also in
`graph.retina_idx` — harmless with `vision=False`; `from_checkpoint` warns that the sets differ).
Verified: `runs/fly2/latest.pt` gives policy / value differences of 0.0 between `full.npz` and
`full-v2.npz`. `full-v2.npz` is only needed to build a *fresh* `FlyBrain(graph)` whose `w_in` rows
must line up with the old `graph.input_idx` (`input_idx` is sorted, so the 5 replaced entries shift
~1,160 rows of a freshly built model relative to the old one).

The tiny graph's neuron set changed: 256 photoreceptors and 239 path neurons took the place of the
lowest-ranked neurons (§6); a smoke run on it now really trains a vision path.

Old npz files without the new keys load with empty arrays (`has_retina == False`); old code ignores the
extra keys. `column_assignment.csv.gz` is not in `fly download`'s default file list yet (see the
report): without it `build_brain_graph` warns loudly and builds a graph without a retina, and
`FlyBrain` then warns again that `vision=True` has nothing to see.

## 8. References

* Matsliah A., Yu S., Kruk K., Bland D., Burke A. T., Gager J., Hebditch J., Silverman B., Willie K. P.,
  Willie R., Sorek M., Sterling A. R., Kind E., Garner D., Sancer G., Wernet M. F., Kim S. S.,
  Murthy M., Seung H. S. (2024). *Neuronal parts list and wiring diagram for a visual system.*
  Nature 634, 166–180. doi:10.1038/s41586-024-07981-1 — the optic-lobe cell-type catalogue and the
  column assignment used here.
* Dorkenwald S. et al. (2024). *Neuronal wiring diagram of an adult brain.* Nature 634, 124–138.
  doi:10.1038/s41586-024-07558-y — the FlyWire connectome.
* Schlegel P. et al. (2024). *Whole-brain annotation and multi-connectome cell typing of Drosophila.*
  Nature 634, 139–152. doi:10.1038/s41586-024-07686-5 — cell types and hemisphere annotations.
* Fischbach K.-F., Dittrich A. P. M. (1989). *The optic lobe of Drosophila melanogaster. I. A Golgi
  analysis of wild-type structure.* Cell Tissue Res. 258, 441–475 — lamina / medulla anatomy and the
  outer-chiasm inversion assumed for the orientation.
