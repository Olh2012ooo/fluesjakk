"""docs/SPEC.md must agree with the code on the contracts it documents (regression for a SPEC/code drift review)."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from flychess.connectome.graph import GraphConfig
from flychess.export import testvectors
from flychess.model.config import BrainConfig
from flychess.train.config import TrainConfig

SPEC = Path(__file__).resolve().parents[1] / "docs" / "SPEC.md"
README = Path(__file__).resolve().parents[1] / "README.md"


def _spec() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_input_super_classes_match_graph_config():
    m = re.search(r"`input_super_classes: \[([^\]]*)\]`", _spec())
    assert m, "SPEC §2.3 must list input_super_classes"
    listed = tuple(re.findall(r"'([^']+)'", m.group(1)))
    assert listed == GraphConfig().input_super_classes


def test_brain_config_fields_documented():
    spec = _spec()
    block = spec[spec.index("class BrainConfig:"):]
    block = block[:block.index("```")]
    for f in dataclasses.fields(BrainConfig):
        assert re.search(rf"^\s+{f.name}:", block, re.MULTILINE), f"BrainConfig.{f.name} missing from SPEC §4"


def test_train_config_field_names_and_dataset_class():
    spec = _spec()
    names = {f.name for f in dataclasses.fields(TrainConfig)}
    for used in ("elo_games", "selfplay_eval_every_iters", "elo_every"):
        assert used in names and f"`{used}`" in spec
    assert "eval_games" not in spec
    assert "PositionDataset" not in spec and "ShardDataset" in spec


def test_test_vector_schema_and_licences():
    spec = _spec()
    assert "top5" not in spec
    assert f"`TOP_K = {testvectors.TOP_K}`" in spec
    for key in ("n_legal", "argmax", "blob_sha256", "tolerance", "run_name", "exported_at"):
        assert key in spec
    assert "vendored (MIT)" not in spec and "BSD-2-Clause" in spec
    assert "CC-BY 4.0" not in spec and "CC BY-NC 4.0" in spec
    assert "on every logit" not in README.read_text(encoding="utf-8")
