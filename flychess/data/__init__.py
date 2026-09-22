"""Training data: Lichess PGN streaming/filtering (lichess.py), the Lichess engine-evaluation database
(evals.py: Stockfish value + best-move targets) and the shard format + dataset (shards.py)."""
from .evals import (  # noqa: F401
    EVAL_DB_URL,
    EvalBuildStats,
    EvalPosition,
    build_eval_shards,
    positions_from_jsonl,
)
from .lichess import (  # noqa: F401
    BuildStats,
    GameFilter,
    Position,
    build_shards,
    download_months,
    positions_from_pgn,
)
from .shards import (  # noqa: F401
    SHARD_SIZE,
    ShardDataset,
    collate,
    count_positions,
    load_split,
    parse_shard_names,
    read_shard,
    write_shard,
)
