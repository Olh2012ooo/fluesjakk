#!/usr/bin/env bash
# Publish web/ (including the exported model in web/model/) to the gh-pages branch.
#
# Usage:  scripts/deploy-pages.sh [remote] [branch]        (defaults: origin gh-pages)
#         DRY_RUN=1 scripts/deploy-pages.sh                (build the commit, do not push)
#         KEEP_HISTORY=1 scripts/deploy-pages.sh          (append to gh-pages instead of replacing it)
#         MODEL_BASE=https://huggingface.co/<user>/<repo>/resolve/main/web/ scripts/deploy-pages.sh
#                                                          (publish WITHOUT the model: the page loads the brain
#                                                           from that URL via <meta name="fly-model-base">)
#
# Only runs when invoked explicitly. Uses a temporary git worktree so the working tree is untouched.
# By default every deploy is a single orphan snapshot commit that *replaces* the branch (force
# push): the model blobs are ~75 MB per deploy, so keeping every past deploy in gh-pages history
# would bloat the repository quickly. A deploy whose tree is identical to what is already on the
# branch is skipped (BUILD.txt is derived from the source commit and the exported model, not from
# the wall clock, so unchanged content produces an identical tree).
# Commits as Carlo Esposito <96500694+cesp99@users.noreply.github.com>, co-authored by Spettro.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${1:-origin}"
BRANCH="${2:-gh-pages}"
WEB="$ROOT/web"
AUTHOR_NAME="Carlo Esposito"
AUTHOR_EMAIL="96500694+cesp99@users.noreply.github.com"
MAX_FILE_BYTES=$((100 * 1024 * 1024))   # GitHub hard limit per file

cd "$ROOT"

# ---- preconditions -------------------------------------------------------------------------
[[ -f "$WEB/index.html" ]] || { echo "error: $WEB/index.html missing" >&2; exit 1; }
MODEL_BASE="${MODEL_BASE:-}"
if [[ -z "$MODEL_BASE" && ( ! -f "$WEB/model/brain.json" || ! -f "$WEB/model/brain.flyb" ) ]]; then
  echo "error: no exported model in web/model/ — run 'fly export-web --run <name>' or set MODEL_BASE=<url>" >&2
  exit 1
fi
while IFS= read -r -d '' f; do
  size=$(stat -c %s "$f")
  if (( size > MAX_FILE_BYTES )); then
    echo "error: $f is $((size / 1048576)) MB (> 100 MB GitHub limit). Use --quant i8 or drop it." >&2
    exit 1
  fi
done < <(find "$WEB" -type f -print0)
git remote get-url "$REMOTE" >/dev/null 2>&1 || { echo "error: remote '$REMOTE' not configured" >&2; exit 1; }

SRC_COMMIT="$(git rev-parse --short HEAD)"
# run name + export timestamp come from the exported header, so BUILD.txt only changes when the
# source commit or the model changes (never from the time of day the script was run).
IFS=$'\t' read -r RUN_NAME EXPORTED_AT < <(python3 - "$WEB/model/brain.json" <<'PY' 2>/dev/null || printf '?\t?\n'
import json, sys
h = json.load(open(sys.argv[1]))
print(h.get("run_name") or "?", h.get("exported_at") or "?", sep="\t")
PY
)
[[ -n "$MODEL_BASE" ]] && { RUN_NAME="remote"; EXPORTED_AT="$MODEL_BASE"; }

# ---- temporary worktree ---------------------------------------------------------------------
TMP="$(mktemp -d "${TMPDIR:-/tmp}/flychess-pages.XXXXXX")"
WORK_BRANCH="$BRANCH"         # branch the commit is built on (a throwaway orphan in snapshot mode)
cleanup() {   # runs from any cwd (the script cd's into $TMP), hence -C "$ROOT"
  git -C "$ROOT" worktree remove --force "$TMP" >/dev/null 2>&1 || rm -rf "$TMP"
  git -C "$ROOT" worktree prune >/dev/null 2>&1 || true
  [[ "$WORK_BRANCH" != "$BRANCH" ]] && git -C "$ROOT" branch -qD "$WORK_BRANCH" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git fetch "$REMOTE" "$BRANCH" >/dev/null 2>&1 || true
PREV=""                       # commit currently published on the branch (if any)
if git show-ref --verify --quiet "refs/remotes/$REMOTE/$BRANCH"; then
  PREV="$(git rev-parse "refs/remotes/$REMOTE/$BRANCH")"
elif git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  PREV="$(git rev-parse "refs/heads/$BRANCH")"
fi

PUSH_FORCE=""
if [[ "${KEEP_HISTORY:-0}" == "1" && -n "$PREV" ]]; then
  git worktree add --detach "$TMP" "$PREV" >/dev/null
  git -C "$TMP" checkout -q -B "$BRANCH" >/dev/null 2>&1
else
  # fresh orphan: the new commit has no parent, so the published branch holds exactly one snapshot
  WORK_BRANCH="deploy-pages/tmp-$$"
  git worktree add --detach --no-checkout "$TMP" >/dev/null
  git -C "$TMP" checkout -q --orphan "$WORK_BRANCH" >/dev/null 2>&1
  git -C "$TMP" rm -rfq --cached . >/dev/null 2>&1 || true
  PUSH_FORCE="--force"
fi

# ---- copy site --------------------------------------------------------------------------------
find "$TMP" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
# everything in web/ except node_modules and tests; the model is included on purpose
if [[ -n "$MODEL_BASE" ]]; then
  # the brain is hosted elsewhere (e.g. Hugging Face): ship the site without web/model and point the page at it
  tar -C "$WEB" --exclude=node_modules --exclude=test --exclude=model --exclude=.DS_Store -cf - . | tar -C "$TMP" -xf -
  sed -i "s#<meta name=\"fly-model-base\" content=\"[^\"]*\">#<meta name=\"fly-model-base\" content=\"$MODEL_BASE\">#" "$TMP/index.html"
  grep -q "fly-model-base\" content=\"$MODEL_BASE\"" "$TMP/index.html" || { echo "error: could not set fly-model-base in index.html" >&2; exit 1; }
else
  tar -C "$WEB" --exclude=node_modules --exclude=test --exclude=.DS_Store -cf - . | tar -C "$TMP" -xf -
fi
touch "$TMP/.nojekyll"          # serve files/dirs starting with _ and keep .gz as-is
echo "flychess $SRC_COMMIT model=$RUN_NAME exported_at=$EXPORTED_AT" > "$TMP/BUILD.txt"

# ---- commit & push --------------------------------------------------------------------------
cd "$TMP"
git add -A
NEW_TREE="$(git write-tree)"
if [[ -n "$PREV" && "$NEW_TREE" == "$(git rev-parse "$PREV^{tree}")" ]]; then
  echo "nothing to deploy: $BRANCH already matches web/ (tree $NEW_TREE)"
  exit 0
fi
MSG="Deploy site from $SRC_COMMIT (model: $RUN_NAME)

Co-Authored-By: Spettro <spettro@eyed.to>"
git -c user.name="$AUTHOR_NAME" -c user.email="$AUTHOR_EMAIL" commit -q --author="$AUTHOR_NAME <$AUTHOR_EMAIL>" -m "$MSG"
git log --show-signature -1 --format='%H %G? %an <%ae>' 2>/dev/null | head -1 || true

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: built commit on '$WORK_BRANCH' in $TMP, not pushing."
  trap - EXIT
  echo "(worktree left in place; remove with: git worktree remove --force $TMP)"
  exit 0
fi
NEW_COMMIT="$(git rev-parse HEAD)"
# shellcheck disable=SC2086  # PUSH_FORCE is intentionally empty or a single flag
git push $PUSH_FORCE "$REMOTE" "$NEW_COMMIT:refs/heads/$BRANCH"
# keep a local branch of the same name in sync (best effort; skipped if it is checked out elsewhere)
[[ "$WORK_BRANCH" != "$BRANCH" ]] && git -C "$ROOT" branch -qf "$BRANCH" "$NEW_COMMIT" >/dev/null 2>&1 || true
echo "pushed $BRANCH to $REMOTE — enable GitHub Pages (branch: $BRANCH, folder: /) if not already."
