"""The fly-brain model: sparse connectome RNN + heads (docs/SPEC.md §4)."""
from flychess.model.config import BrainConfig
from flychess.model.flybrain import (
    FlyBrain,
    inverse_softplus,
    masked_policy_log_softmax,
    metrics_to_float,
    policy_value_loss,
)
from flychess.model.spmm import SparseStructure, SpMM, spmm, spmm_dense_reference

__all__ = [
    "BrainConfig",
    "FlyBrain",
    "SpMM",
    "SparseStructure",
    "inverse_softplus",
    "masked_policy_log_softmax",
    "metrics_to_float",
    "policy_value_loss",
    "spmm",
    "spmm_dense_reference",
]
