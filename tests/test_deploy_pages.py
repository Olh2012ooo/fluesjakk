"""Tests for ``scripts/deploy-pages.sh`` against a throwaway source repo and a local bare remote.

The script locates the project root relative to its own path, so it is copied into the temporary
repo; the "remote" is a bare repository on disk, which makes the real push path exercisable.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy-pages.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None or os.name != "posix",
    reason="requires git + bash on a POSIX host",
)


def _git(repo: Path, *args: str, env: dict[str, str]) -> str:
    res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env, check=True)
    return res.stdout.strip()


@pytest.fixture
def deploy_env(tmp_path: Path):
    """Source repo with the script + a fake web/ export, a bare remote, and an isolated git env."""
    home = tmp_path / "home"
    home.mkdir()
    (tmp_path / "tmp").mkdir()
    env = {
        **os.environ,
        "HOME": str(home),
        "GIT_CONFIG_GLOBAL": str(home / ".gitconfig-none"),   # ignore the developer's global config
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
        "TMPDIR": str(tmp_path / "tmp"),
    }
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True, env=env)

    src = tmp_path / "src"
    (src / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, src / "scripts" / "deploy-pages.sh")
    web = src / "web"
    (web / "model").mkdir(parents=True)
    (web / "test").mkdir()
    (web / "index.html").write_text("<h1>fly</h1>")
    (web / "test" / "x.test.mjs").write_text("// excluded from the site")
    (web / "model" / "brain.json").write_text(json.dumps({"run_name": "fly1", "exported_at": "2026-01-02T03:04:05+00:00"}))
    (web / "model" / "brain.flyb").write_bytes(b"\0" * 64)
    subprocess.run(["git", "init", "-q", "-b", "master", str(src)], check=True, env=env)
    _git(src, "remote", "add", "origin", str(remote), env=env)
    _git(src, "add", "-A", env=env)
    _git(src, "commit", "-q", "-m", "init", env=env)
    yield src, remote, env
    # DRY_RUN leaves its worktree in place on purpose; drop any leftovers so tmp_path can be removed
    for line in _git(src, "worktree", "list", "--porcelain", env=env).split("\n"):
        if line.startswith("worktree ") and "flychess-pages." in line:
            _git(src, "worktree", "remove", "--force", line.split(" ", 1)[1], env=env)


def _deploy(src: Path, env: dict[str, str], **extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(src / "scripts" / "deploy-pages.sh")], cwd=src, capture_output=True,
                          text=True, env={**env, **extra}, timeout=120, check=False)


def _published(remote: Path, env: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """(commit hashes, {path: content}) of the gh-pages branch on the bare remote (text files only)."""
    commits = _git(remote, "rev-list", "gh-pages", env=env).split()
    files = _git(remote, "ls-tree", "-r", "--name-only", "gh-pages", env=env).split("\n")
    content = {f: _git(remote, "show", f"gh-pages:{f}", env=env) for f in files if not f.endswith(".flyb")}
    return commits, content


def _blob(remote: Path, path: str, env: dict[str, str]) -> bytes:
    return subprocess.run(["git", "-C", str(remote), "show", f"gh-pages:{path}"], capture_output=True, env=env,
                          check=True).stdout


def test_first_deploy_publishes_site_without_tests(deploy_env):
    src, remote, env = deploy_env
    res = _deploy(src, env)
    assert res.returncode == 0, res.stderr
    commits, content = _published(remote, env)
    assert len(commits) == 1
    assert content["index.html"] == "<h1>fly</h1>"
    assert "model/brain.json" in content and ".nojekyll" in content
    assert _blob(remote, "model/brain.flyb", env) == b"\0" * 64
    assert not any(p.startswith("test/") for p in content)
    # BUILD.txt is keyed on the source commit + the export header, never on the wall clock
    head = _git(src, "rev-parse", "--short", "HEAD", env=env)
    assert content["BUILD.txt"] == f"flychess {head} model=fly1 exported_at=2026-01-02T03:04:05+00:00"
    log = _git(remote, "log", "-1", "--format=%s%n%b%n%an <%ae>", "gh-pages", env=env)
    assert log.startswith("Deploy site from ") and "Co-Authored-By: Spettro" in log
    assert log.endswith("Carlo Esposito <96500694+cesp99@users.noreply.github.com>")
    # temp worktree and throwaway orphan branch are cleaned up
    assert "flychess-pages." not in _git(src, "worktree", "list", env=env)
    assert _git(src, "branch", "--list", "deploy-pages/*", env=env) == ""


def test_unchanged_site_is_not_redeployed(deploy_env):
    """Regression: a wall-clock timestamp in BUILD.txt used to make every run push a fresh commit."""
    src, remote, env = deploy_env
    assert _deploy(src, env).returncode == 0
    before = _git(remote, "rev-parse", "gh-pages", env=env)
    res = _deploy(src, env)
    assert res.returncode == 0, res.stderr
    assert "nothing to deploy" in res.stdout
    assert _git(remote, "rev-parse", "gh-pages", env=env) == before


def test_redeploy_replaces_snapshot_instead_of_growing_history(deploy_env):
    """Regression: each deploy appended a commit that re-added the ~75 MB model blobs to gh-pages history."""
    src, remote, env = deploy_env
    assert _deploy(src, env).returncode == 0
    old_blob = _git(remote, "rev-parse", "gh-pages:model/brain.flyb", env=env)
    (src / "web" / "model" / "brain.flyb").write_bytes(b"\1" * 64)
    (src / "web" / "model" / "brain.json").write_text(
        json.dumps({"run_name": "fly2", "exported_at": "2026-02-02T00:00:00+00:00"}))
    res = _deploy(src, env)
    assert res.returncode == 0, res.stderr
    commits, content = _published(remote, env)
    assert len(commits) == 1, "gh-pages must hold a single snapshot"
    assert content["BUILD.txt"].endswith("model=fly2 exported_at=2026-02-02T00:00:00+00:00")
    assert _blob(remote, "model/brain.flyb", env) == b"\1" * 64
    reachable = _git(remote, "rev-list", "--objects", "gh-pages", env=env)
    assert old_blob not in reachable, "the previous model blob must no longer be reachable from gh-pages"
    # a third, unchanged deploy is still detected as a no-op against the replaced snapshot
    res = _deploy(src, env)
    assert "nothing to deploy" in res.stdout and _git(remote, "rev-list", "gh-pages", env=env) == commits[0]


def test_deploy_works_when_a_stale_local_branch_exists(deploy_env):
    """A leftover local gh-pages branch (e.g. from an older deploy) must not break the orphan checkout."""
    src, remote, env = deploy_env
    _git(src, "branch", "gh-pages", "HEAD", env=env)
    res = _deploy(src, env)
    assert res.returncode == 0, res.stderr
    assert len(_published(remote, env)[0]) == 1
    # the local branch now mirrors what was published
    assert _git(src, "rev-parse", "gh-pages", env=env) == _git(remote, "rev-parse", "gh-pages", env=env)


def test_keep_history_appends_commit(deploy_env):
    src, remote, env = deploy_env
    assert _deploy(src, env).returncode == 0
    (src / "web" / "index.html").write_text("<h1>fly v2</h1>")
    res = _deploy(src, env, KEEP_HISTORY="1")
    assert res.returncode == 0, res.stderr
    commits, content = _published(remote, env)
    assert len(commits) == 2
    assert content["index.html"] == "<h1>fly v2</h1>"
    # and an unchanged site is still a no-op in that mode
    res = _deploy(src, env, KEEP_HISTORY="1")
    assert "nothing to deploy" in res.stdout and len(_published(remote, env)[0]) == 2


def test_dry_run_does_not_push(deploy_env):
    src, remote, env = deploy_env
    res = _deploy(src, env, DRY_RUN="1")
    assert res.returncode == 0, res.stderr
    assert "not pushing" in res.stdout
    probe = subprocess.run(["git", "-C", str(remote), "rev-parse", "--verify", "gh-pages"], capture_output=True, env=env,
                           check=False)
    assert probe.returncode != 0


def test_missing_model_fails_fast(deploy_env):
    src, _, env = deploy_env
    (src / "web" / "model" / "brain.flyb").unlink()
    res = _deploy(src, env)
    assert res.returncode == 1 and "no exported model" in res.stderr
