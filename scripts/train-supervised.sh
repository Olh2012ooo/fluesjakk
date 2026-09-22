#!/usr/bin/env bash
# Keep a training run alive: (re)start `fly train --resume` until it exits cleanly.
# Usage: scripts/train-supervised.sh <run> [extra fly-train args...]
# A CUDA driver abort (e.g. launch timeout on a GPU that also drives the desktop) kills the process;
# this loop resumes from runs/<run>/latest.pt, losing at most `checkpoint_every` steps.
set -u
run="$1"; shift
attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[supervisor] attempt $attempt: fly train --run $run $*"
  if [ -f "runs/$run/latest.pt" ]; then
    fly train --run "$run" --resume "$@" && break
  else
    fly train --run "$run" "$@" && break
  fi
  echo "[supervisor] fly train exited with status $? — restarting in 30 s"
  sleep 30
done
echo "[supervisor] run $run finished"
