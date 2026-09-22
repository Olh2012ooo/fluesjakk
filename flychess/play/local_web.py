"""``fly play --gui``: export the current brain to ``web/model/`` and serve ``web/`` locally.

The website (``web/``, SPEC §11) runs the exported network inside a Web Worker; this module only
makes sure ``web/model/brain.json`` + ``brain.flyb(.gz)`` are fresh (re-exported when missing or older
than the checkpoint) and serves the static directory with a threaded ``http.server``:

* correct MIME types for ``.mjs`` / ``.js`` (``text/javascript``), ``.wasm``, ``.json``, ``.svg``;
* ``brain.flyb.gz`` is sent as ``application/gzip`` **without** ``Content-Encoding`` — the loader
  (``web/engine/loader.js``) streams the raw gzip bytes through ``DecompressionStream`` itself so it can
  show download progress and cache the decoded buffer;
* ``Cache-Control: no-cache`` on everything, so a re-export is picked up on reload.
"""
from __future__ import annotations

import logging
import socket
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import torch

from flychess import paths

log = logging.getLogger(__name__)

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wasm": "application/wasm",
    ".gz": "application/gzip",
    ".flyb": "application/octet-stream",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}


class WebHandler(SimpleHTTPRequestHandler):
    """Static file handler with the MIME table above; never sets ``Content-Encoding``."""

    extensions_map: ClassVar[dict[str, str]] = {**SimpleHTTPRequestHandler.extensions_map, **MIME_TYPES}
    quiet: ClassVar[bool] = False

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        if not self.quiet:
            log.info("%s - %s", self.address_string(), fmt % args)


def _checkpoint_path(run_or_ckpt: str | Path) -> Path:
    from flychess.play.engine import resolve_checkpoint

    return resolve_checkpoint(run_or_ckpt)


def export_is_stale(run_or_ckpt: str | Path, out_dir: str | Path) -> bool:
    """True when ``out_dir/brain.json`` is missing or older than the checkpoint."""
    out_dir = Path(out_dir)
    header = out_dir / "brain.json"
    blob_ok = (out_dir / "brain.flyb").exists() or (out_dir / "brain.flyb.gz").exists()
    if not header.exists() or not blob_ok:
        return True
    ckpt = _checkpoint_path(run_or_ckpt)
    if not ckpt.exists():
        return False  # nothing newer to export from
    return ckpt.stat().st_mtime > header.stat().st_mtime


def ensure_export(run_or_ckpt: str | Path, out_dir: str | Path = paths.WEB_MODEL_DIR, quant: str = "f16",
                  device: str | torch.device | None = None, force: bool = False) -> Path:
    """Export the brain of ``run_or_ckpt`` to ``out_dir`` if missing / stale (or ``force``); returns ``out_dir``."""
    from flychess.export.web import export_web
    from flychess.play.engine import load_checkpoint

    out_dir = Path(out_dir)
    if not force and not export_is_stale(run_or_ckpt, out_dir):
        log.info("web model in %s is up to date", out_dir)
        return out_dir
    device = device or "cpu"  # the export only reads weights; CPU avoids touching a busy GPU
    model, graph, ckpt = load_checkpoint(run_or_ckpt, device)
    cfg = ckpt.get("config") or {}
    run_name = cfg.get("run", "") if isinstance(cfg, dict) else getattr(cfg, "run", "")
    if not run_name:
        p = Path(run_or_ckpt)
        run_name = p.parent.name if p.suffix == ".pt" else p.name
    meta = {"run_name": str(run_name), "train_steps": int(ckpt.get("step", 0)),
            "elo_estimates": dict(ckpt.get("elo") or {}), "stage": ckpt.get("stage", "")}
    log.info("exporting %s (step %s) -> %s", run_name, meta["train_steps"], out_dir)
    export_web(model, graph, out_dir, quant=quant, extra_meta=meta)
    return out_dir


def free_port(preferred: int = 8000, host: str = "127.0.0.1") -> int:
    """``preferred`` if it is free, otherwise an OS-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def start_server(web_dir: str | Path = paths.WEB_DIR, port: int = 8000, host: str = "127.0.0.1",
                 quiet: bool = False) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Serve ``web_dir`` in a daemon thread; returns ``(server, thread)`` — call ``server.shutdown()`` to stop."""
    handler = partial(WebHandler, directory=str(web_dir))
    WebHandler.quiet = quiet
    server = ThreadingHTTPServer((host, int(port)), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="fly-web", daemon=True)
    thread.start()
    return server, thread


def serve_local(
    run_or_ckpt: str | Path,
    port: int = 8000,
    out_dir: str | Path = paths.WEB_MODEL_DIR,
    open_browser: bool = True,
    web_dir: str | Path = paths.WEB_DIR,
    host: str = "127.0.0.1",
    block: bool = True,
    quant: str = "f16",
    quiet: bool = False,
) -> tuple[ThreadingHTTPServer, str]:
    """Export (if needed) and serve the website with the brain of ``run_or_ckpt``.

    Blocks until Ctrl-C when ``block`` is true; otherwise returns ``(server, url)`` immediately so the
    caller can ``server.shutdown()`` later (tests). ``out_dir`` must live under ``web_dir`` (default
    ``web/model``) for the page to find it.
    """
    web_dir = Path(web_dir)
    out_dir = Path(out_dir)
    ensure_export(run_or_ckpt, out_dir, quant=quant)
    if port == 0:
        port = free_port(0, host)
    server, thread = start_server(web_dir, port, host, quiet=quiet)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"🪰 serving {web_dir} at {url}  (model: {out_dir}) — Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    if not block:
        return server, url
    try:
        thread.join()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.shutdown()
        server.server_close()
    return server, url


__all__ = ["MIME_TYPES", "WebHandler", "ensure_export", "export_is_stale", "free_port", "serve_local", "start_server"]
