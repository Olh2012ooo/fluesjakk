#!/usr/bin/env bash
# The full fly-chess pipeline: download -> build-brain -> build-shards -> train -> eval -> export-web.
#
# Usage:  scripts/pipeline.sh [run-name]              (default run name: fly1)
#
# Environment overrides (all optional):
#   MONTHS=2014-01,2014-02   Lichess months to download / shard      (default: all twelve months of 2014)
#   MIN_ELO=1800             both players at least this rating       (default 1800)
#   WORKERS=16               shard-building processes                (default: nproc, capped at 16)
#   STAGE=all                imitation | selfplay | all              (default all)
#   CONFIG=configs/default.yaml   TrainConfig yaml                   (default configs/default.yaml)
#   STEPS=                   cap the imitation stage at N steps      (default: one epoch = ~220k steps)
#   EXTRA="--set lr=5e-4"    extra `fly train` arguments
#   EVAL_GAMES=50            games per opponent for the final Elo    (0 = skip)
#   TINY=1                   smoke mode: tiny brain, 3000 games, 60 steps, 2 self-play iterations
#
# Every step is idempotent: downloads and the brain graph are skipped when present, shards are only
# built when the shard directory is empty, training resumes from runs/<run>/latest.pt when it exists.
# Start `fly dashboard --run <run>` in another terminal to watch training (http://127.0.0.1:8765).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -x .venv/bin/fly ]]; then FLY=.venv/bin/fly; else FLY=fly; fi

RUN="${1:-fly1}"
MONTHS="${MONTHS:-2014-01,2014-02,2014-03,2014-04,2014-05,2014-06,2014-07,2014-08,2014-09,2014-10,2014-11,2014-12}"
MIN_ELO="${MIN_ELO:-1800}"
WORKERS="${WORKERS:-$(( $(nproc) < 16 ? $(nproc) : 16 ))}"
STAGE="${STAGE:-all}"
CONFIG="${CONFIG:-configs/default.yaml}"
STEPS="${STEPS:-}"
EXTRA="${EXTRA:-}"
HOME_DIR="${FLYCHESS_HOME:-$ROOT}"

log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }

if [[ "${TINY:-0}" == "1" ]]; then
  MONTHS="${MONTHS%%,*}"            # one month is plenty
  SHARD_NAME=smoke
  SHARDS="$HOME_DIR/data/shards-smoke"
  SHARD_ARGS=(--max-games 3000)
  TRAIN_ARGS=(--tiny --steps "${STEPS:-60}" --set selfplay_iters=2 --set selfplay_games_per_iter=4 --set mcts_sims=8)
  BRAIN_ARGS=(--tiny)
  EVAL_GAMES="${EVAL_GAMES:-4}"
else
  SHARD_NAME=lichess
  SHARDS="$HOME_DIR/data/shards"
  SHARD_ARGS=()
  TRAIN_ARGS=(--config "$CONFIG")
  [[ -n "$STEPS" ]] && TRAIN_ARGS+=(--steps "$STEPS")
  BRAIN_ARGS=()
  EVAL_GAMES="${EVAL_GAMES:-50}"
fi

# 1. data --------------------------------------------------------------------------------------
log "download: FlyWire connectome tables + Lichess months $MONTHS"
$FLY download --connectome
$FLY download --games --months "$MONTHS"

# 2. brain graph -------------------------------------------------------------------------------
log "build-brain ${BRAIN_ARGS[*]:-(full brain: 134k neurons, 2.7M synapses)}"
$FLY build-brain "${BRAIN_ARGS[@]}"

# 3. shards ------------------------------------------------------------------------------------
if compgen -G "$SHARDS/$SHARD_NAME-*.npz" > /dev/null; then
  log "build-shards: $SHARDS already has $SHARD_NAME-*.npz shards, skipping (delete them to rebuild)"
else
  log "build-shards: months $MONTHS, min Elo $MIN_ELO, $WORKERS workers -> $SHARDS"
  $FLY build-shards --months "$MONTHS" --min-elo "$MIN_ELO" --workers "$WORKERS" --name "$SHARD_NAME" \
       --out "$SHARDS" "${SHARD_ARGS[@]}"
fi

# 4. train -------------------------------------------------------------------------------------
RESUME=()
[[ -f "$HOME_DIR/runs/$RUN/latest.pt" ]] && RESUME=(--resume)
log "train --run $RUN --stage $STAGE ${TRAIN_ARGS[*]} ${RESUME[*]:-} $EXTRA"
echo "    watch it with:  $FLY dashboard --run $RUN   (http://127.0.0.1:8765)"
# shellcheck disable=SC2086
$FLY train --run "$RUN" --stage "$STAGE" "${TRAIN_ARGS[@]}" "${RESUME[@]}" \
     --shards-dir "$SHARDS" --shard-name "$SHARD_NAME" $EXTRA

# 5. evaluate ----------------------------------------------------------------------------------
if [[ "$EVAL_GAMES" != "0" ]]; then
  log "eval --run $RUN --games $EVAL_GAMES (random + 1-ply material opponents)"
  $FLY eval --run "$RUN" --games "$EVAL_GAMES" --opponent random,material
fi

# 6. export for the website ---------------------------------------------------------------------
log "export-web --run $RUN  (web/model/ + tests/vectors/model.json)"
$FLY export-web --run "$RUN"
if command -v node > /dev/null 2>&1; then
  log "cross-language parity: node --test web/test/parity.test.mjs"
  node --test web/test/parity.test.mjs
fi

log "done. Play:  $FLY play --run $RUN [--difficulty larva|fly|superfly]   or   $FLY play --gui"
echo "    Publish the site with scripts/deploy-pages.sh (web/model/ must stay < 100 MB per file)."
