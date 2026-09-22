"""Download the FlyWire connectome tables (Codex snapshot 783) and Lichess PGN dumps.

See docs/SPEC.md §2.1. Files are streamed with a progress bar, verified (gzip integrity for the
connectome tables, zstd frame integrity for the PGN dumps), and skipped when already present and valid.
On failure, manual instructions are printed so the user can drop the files into place by hand.
"""
from __future__ import annotations

import gzip
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import requests
from tqdm import tqdm

from flychess import paths

CODEX_BASE_URL = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783"
LICHESS_BASE_URL = "https://database.lichess.org/standard"

DEFAULT_FILES: tuple[str, ...] = (
    "connections.csv.gz",
    "neurons.csv.gz",
    "classification.csv.gz",
    "consolidated_cell_types.csv.gz",
    "coordinates.csv.gz",
)
OPTIONAL_FILES: tuple[str, ...] = ("cell_stats.csv.gz", "names.csv.gz")

_CHUNK = 1 << 20  # 1 MiB


def connectome_url(filename: str) -> str:
    return f"{CODEX_BASE_URL}/{filename}"


def games_url(month: str) -> str:
    """`month` is 'YYYY-MM'."""
    return f"{LICHESS_BASE_URL}/lichess_db_standard_rated_{month}.pgn.zst"


# ---- integrity checks ------------------------------------------------------------------------------
def verify_gzip(path: Path) -> bool:
    """True iff `path` is a complete, uncorrupted gzip stream (read through to EOF)."""
    try:
        with gzip.open(path, "rb") as f:
            while f.read(_CHUNK):
                pass
        return True
    except (OSError, EOFError, gzip.BadGzipFile):
        return False


def verify_zstd(path: Path) -> bool:
    """True iff `path` is a complete zstd stream (decompressed through to EOF, output discarded)."""
    try:
        import zstandard as zstd
    except ImportError:  # pragma: no cover - zstandard is a hard dependency, but stay graceful
        return path.stat().st_size > 0
    try:
        with open(path, "rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as reader:
            while reader.read(_CHUNK):
                pass
        return True
    except (OSError, zstd.ZstdError):
        return False


def _verify(path: Path, kind_of: Path | None = None) -> bool:
    """Integrity check chosen by the suffix of `kind_of` (default `path`; the final name for .partial)."""
    suffix = (kind_of or path).suffix
    if suffix == ".gz":
        return verify_gzip(path)
    if suffix == ".zst":
        return verify_zstd(path)
    return path.exists() and path.stat().st_size > 0


# ---- generic streaming download ------------------------------------------------------------------
def download_file(url: str, dest: Path, force: bool = False, timeout: float = 60.0) -> Path:
    """Stream `url` into `dest` with a tqdm progress bar.

    Returns `dest`. Existing valid files are skipped unless `force`. Downloads go to `dest.partial`
    and are renamed only after passing the integrity check, so an interrupted download is never
    mistaken for a finished one. Raises `RuntimeError` on HTTP / integrity failure.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if _verify(dest):
            print(f"  {dest.name}: present and valid, skipping")
            return dest
        print(f"  {dest.name}: present but corrupt, re-downloading")
    partial = dest.with_name(dest.name + ".partial")
    with requests.get(url, stream=True, timeout=timeout) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} for {url}")
        total = int(r.headers.get("Content-Length", 0)) or None
        with open(partial, "wb") as out, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name, leave=False,
            file=sys.stderr,
        ) as bar:
            for chunk in r.iter_content(chunk_size=_CHUNK):
                out.write(chunk)
                bar.update(len(chunk))
    if not _verify(partial, kind_of=dest):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"integrity check failed for {url}")
    partial.replace(dest)
    print(f"  {dest.name}: downloaded {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def _download_many(items: Iterable[tuple[str, Path]], force: bool, what: str) -> list[Path]:
    """Download every (url, dest); collect failures and print manual instructions at the end."""
    done: list[Path] = []
    failed: list[tuple[str, Path, str]] = []
    for url, dest in items:
        try:
            done.append(download_file(url, dest, force=force))
        except (requests.RequestException, RuntimeError, OSError) as e:
            print(f"  {dest.name}: FAILED ({e})")
            failed.append((url, dest, str(e)))
    if failed:
        print(f"\n{len(failed)} {what} file(s) could not be downloaded. Download them manually:")
        for url, dest, _ in failed:
            print(f"  curl -L -o {dest} {url}")
        print("and re-run the download command (existing valid files are skipped).")
        raise RuntimeError(f"failed to download {len(failed)} {what} file(s): "
                           + ", ".join(d.name for _, d, _ in failed))
    return done


# ---- public API ------------------------------------------------------------------------------------
def download_connectome(
    dest: str | Path = paths.CONNECTOME_DIR,
    files: Sequence[str] = DEFAULT_FILES,
    force: bool = False,
) -> list[Path]:
    """Download the FlyWire v783 tables listed in `files` into `dest`. Returns the local paths."""
    dest = Path(dest)
    print(f"Downloading FlyWire connectome (snapshot 783) into {dest}")
    return _download_many(((connectome_url(f), dest / f) for f in files), force, "connectome")


def download_games(
    months: Sequence[str],
    dest: str | Path = paths.PGN_DIR,
    force: bool = False,
) -> list[Path]:
    """Download Lichess standard rated dumps for `months` (e.g. ['2014-01', '2014-02']) into `dest`."""
    dest = Path(dest)
    print(f"Downloading Lichess PGN dumps into {dest}")
    items = ((games_url(m), dest / f"lichess_db_standard_rated_{m}.pgn.zst") for m in months)
    return _download_many(items, force, "PGN")
