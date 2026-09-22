"""Play against the fly in the terminal (``fly play``), rendered with ``rich``.

The loop is fully scriptable: pass ``inputs`` (an iterable of command strings) and ``out`` (a text
stream) to run it non-interactively, which is how the tests drive it. Moves are accepted in SAN or
UCI; commands: ``undo``, ``resign``, ``hint``, ``new``, ``quit``, ``help``. After every fly move a
commentary line reports the move, how sure the policy was and the fly's mood (from the value head).
Hints show the fly's *own* top policy moves and are labelled as such — the fly is the only chess
mind in this module.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any

import chess
import chess.pgn
import torch
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from flychess.play.engine import DIFFICULTIES, FlyEngine
from flychess.train.mcts import result_string, terminal_value

PIECES = {
    (chess.PAWN, chess.WHITE): "♙", (chess.KNIGHT, chess.WHITE): "♘", (chess.BISHOP, chess.WHITE): "♗",
    (chess.ROOK, chess.WHITE): "♖", (chess.QUEEN, chess.WHITE): "♕", (chess.KING, chess.WHITE): "♔",
    (chess.PAWN, chess.BLACK): "♟", (chess.KNIGHT, chess.BLACK): "♞", (chess.BISHOP, chess.BLACK): "♝",
    (chess.ROOK, chess.BLACK): "♜", (chess.QUEEN, chess.BLACK): "♛", (chess.KING, chess.BLACK): "♚",
}
LIGHT, DARK = "#f0d9b5", "#b58863"
LIGHT_HL, DARK_HL = "#f6f669", "#baca2b"
WHITE_PIECE, BLACK_PIECE = "#ffffff", "#1a1a1a"
COMMANDS = ("undo", "resign", "hint", "new", "quit", "help")
MOOD_TEXT = {"smug": "feels smug", "confident": "feels confident", "focused": "is focused",
             "nervous": "feels nervous", "panicking": "is panicking"}


def render_board(board: chess.Board, flip: bool = False, last_move: chess.Move | None = None) -> Text:
    """Unicode board with coloured squares, coordinates and the last move highlighted."""
    text = Text()
    ranks = range(8) if flip else range(7, -1, -1)
    files = range(7, -1, -1) if flip else range(8)
    hl = {last_move.from_square, last_move.to_square} if last_move else set()
    for rank in ranks:
        text.append(f" {rank + 1} ", style="bold dim")
        for file in files:
            sq = chess.square(file, rank)
            light = (rank + file) % 2 == 1
            bg = (LIGHT_HL if light else DARK_HL) if sq in hl else (LIGHT if light else DARK)
            piece = board.piece_at(sq)
            if piece is None:
                text.append("   ", style=f"on {bg}")
            else:
                fg = WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE
                text.append(f" {PIECES[(piece.piece_type, piece.color)]} ", style=f"bold {fg} on {bg}")
        text.append("\n")
    text.append("   " + "".join(f" {chess.FILE_NAMES[f]} " for f in files), style="bold dim")
    return text


def parse_move(board: chess.Board, s: str) -> chess.Move | None:
    """SAN first, then UCI; ``None`` if neither parses to a legal move."""
    s = s.strip()
    try:
        return board.parse_san(s)
    except ValueError:
        pass
    try:
        mv = chess.Move.from_uci(s.lower())
    except ValueError:
        return None
    if mv in board.legal_moves:
        return mv
    # promotions typed without the piece (e7e8) default to a queen
    if mv.promotion is None:
        q = chess.Move(mv.from_square, mv.to_square, chess.QUEEN)
        if q in board.legal_moves:
            return q
    return None


def commentary(san: str, info: dict[str, Any]) -> str:
    """``🪰 the fly plays Nf3 (87% sure, feels confident)`` (+ search info for superfly).

    "% sure" is the policy probability of the move; for ``superfly`` it is the share of the search
    visits that went to the chosen move (``info['search_top'][0]``), which is what decided the move.
    """
    sure = round(100 * float(info.get("move_prob", 0.0)))
    if info.get("sims") and info.get("search_top"):
        sure = round(100 * float(info["search_top"][0][1]))
    mood = MOOD_TEXT.get(info.get("mood", "focused"), "is focused")
    line = f"🪰 the fly plays {san} ({sure}% sure, {mood})"
    if info.get("sims"):
        line += f" after {info['sims']} simulations"
    if "think_ms" in info:
        line += f" · {info['think_ms']:.0f} ms"
    return line


def game_pgn(board: chess.Board, result: str, human_white: bool, fly_name: str) -> str:
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "fly-chess terminal game"
    game.headers["White"] = "human" if human_white else fly_name
    game.headers["Black"] = fly_name if human_white else "human"
    game.headers["Result"] = result
    return str(game)


def _latest_run() -> str:
    from flychess import paths
    from flychess.train.metrics import list_runs

    for run in list_runs(paths.RUNS_DIR):
        if (paths.run_dir(run["name"]) / "latest.pt").exists():
            return run["name"]
    raise FileNotFoundError(f"no trained run with a latest.pt under {paths.RUNS_DIR}; train first or pass --run/--ckpt")


def play_terminal(
    run_or_ckpt: str | Path | None = None,
    difficulty: str = "fly",
    color: str = "white",
    device: str | torch.device | None = None,
    *,
    engine: FlyEngine | None = None,
    inputs: Iterable[str] | None = None,
    out: IO[str] | None = None,
    sims: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Play a game (or several, via ``new``) against the fly in the terminal.

    ``engine`` skips checkpoint loading; ``inputs`` / ``out`` make the loop non-interactive. Returns
    ``{'result', 'pgn', 'plies', 'games': [...]}`` for the last game (``result='*'`` if abandoned).
    """
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}, got {difficulty!r}")
    console = Console(file=out, force_terminal=False, width=100) if out is not None else Console()
    if engine is None:
        name = str(run_or_ckpt) if run_or_ckpt is not None else _latest_run()
        console.print(f"loading the fly brain from [bold]{name}[/] …")
        kw: dict[str, Any] = {"seed": seed}
        if sims is not None:
            kw["sims"] = sims
        engine = FlyEngine.load(name, device, **kw)
    fly_name = f"fly-{difficulty}"
    color = color.lower()
    if color == "random":
        color = "white" if engine.rng.random() < 0.5 else "black"
    human = chess.WHITE if color == "white" else chess.BLACK
    input_iter: Iterator[str] | None = iter(inputs) if inputs is not None else None

    def ask(prompt: str) -> str | None:
        if input_iter is not None:
            try:
                line = next(input_iter)
            except StopIteration:
                return None
            console.print(f"{prompt}{line}")
            return line
        try:
            return console.input(prompt)
        except (EOFError, KeyboardInterrupt):
            return None

    console.print(Panel.fit(
        f"[bold]fly-chess[/] — you play [bold]{color}[/] against [bold]{fly_name}[/]\n"
        f"brain: {engine.model.n:,} neurons · {engine.model.nnz:,} connections · {engine.model.steps} steps\n"
        "moves in SAN (Nf3) or UCI (g1f3); commands: undo · resign · hint · new · quit · help",
        title="🪰", border_style="green"))

    games: list[dict[str, Any]] = []
    board = chess.Board()
    last_move: chess.Move | None = None
    result: str | None = None
    quit_requested = False

    def announce_end(res: str, reason: str) -> None:
        nonlocal result
        result = res
        console.print(render_board(board, flip=human == chess.BLACK, last_move=last_move))
        human_score = {"1-0": human == chess.WHITE, "0-1": human == chess.BLACK}.get(res)
        verdict = "you win 🎉" if human_score else ("draw 🤝" if res == "1/2-1/2" else "the fly wins 🪰")
        console.print(Panel.fit(f"[bold]{res}[/] — {reason}: {verdict}", border_style="magenta"))
        pgn = game_pgn(board, res, human == chess.WHITE, fly_name)
        console.print(pgn)
        games.append({"result": res, "pgn": pgn, "plies": len(board.move_stack)})

    while not quit_requested:
        if result is None:
            tv = terminal_value(board)
            if tv is not None:
                res = result_string(board)
                reason = "checkmate" if board.is_checkmate() else "draw"
                announce_end(res, reason)
                continue
            if board.turn != human:
                move, info = engine.choose_move(board, difficulty)
                san = board.san(move)
                board.push(move)
                last_move = move
                console.print(commentary(san, info), style="green")
                continue
            console.print(render_board(board, flip=human == chess.BLACK, last_move=last_move))
            prompt = f"[bold]{len(board.move_stack) // 2 + 1}.{'' if board.turn else '..'}[/] your move > "
        else:
            prompt = "[dim]game over — 'new' or 'quit'[/] > "
        line = ask(prompt)
        if line is None:
            quit_requested = True
            break
        cmd = line.strip().lower()
        if not cmd:
            continue
        if cmd in ("quit", "exit", "q"):
            quit_requested = True
        elif cmd == "help":
            console.print("commands: " + " · ".join(COMMANDS) + "; moves as SAN (e4, Nf3, O-O) or UCI (e2e4)")
        elif cmd == "new":
            board = chess.Board()
            last_move = None
            result = None
            console.print("new game")
        elif result is not None:
            console.print("[yellow]the game is over — 'new' to play again or 'quit'[/]")
        elif cmd == "resign":
            announce_end("0-1" if human == chess.WHITE else "1-0", "you resigned")
        elif cmd == "undo":
            n = 0
            while board.move_stack and (n == 0 or board.turn != human):
                board.pop()
                n += 1
            last_move = board.peek() if board.move_stack else None
            console.print(f"took back {n} plies" if n else "[yellow]nothing to undo[/]")
        elif cmd == "hint":
            _, info = engine.choose_move(board, "fly")
            top = ", ".join(f"{board.san(chess.Move.from_uci(u))} ({100 * p:.0f}%)" for u, p in info["policy_top"])
            console.print(f"🪰 hint — the fly's own top policy moves for you: {top} "
                          f"(the fly {MOOD_TEXT[info['mood']]} about your position)", style="cyan")
        else:
            move = parse_move(board, line)
            if move is None:
                console.print(f"[red]'{line.strip()}' is not a legal move or command[/] (try 'help')")
                continue
            san = board.san(move)
            board.push(move)
            last_move = move
            console.print(f"you play {san}")

    if result is None and board.move_stack:
        console.print("[dim]game abandoned[/]")
        games.append({"result": "*", "pgn": game_pgn(board, "*", human == chess.WHITE, fly_name),
                      "plies": len(board.move_stack)})
    last = games[-1] if games else {"result": "*", "pgn": game_pgn(board, "*", human == chess.WHITE, fly_name),
                                    "plies": len(board.move_stack)}
    return {**last, "games": games}


__all__ = ["commentary", "parse_move", "play_terminal", "render_board"]
