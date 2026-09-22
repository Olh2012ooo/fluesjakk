"""BrainConfig: the hyper-parameters that, together with a graph and a state_dict, define a FlyBrain."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

ACTIVATIONS = ("relu", "gelu", "tanh", "satrelu")
DTYPES = ("float32", "bfloat16", "float16", "float64")


def _parse_steps(v: Any) -> tuple[int, ...]:
    """``()`` / ``(8, 16)`` / ``[8, 16]`` / ``'8,16'`` / ``8`` -> tuple of ints."""
    if v is None:
        return ()
    if isinstance(v, str):
        v = [p for p in v.replace(";", ",").split(",") if p.strip()]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = [v]
    try:
        return tuple(int(t) for t in v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"readout_steps must be a sequence of ints, got {v!r}") from exc


@dataclass
class BrainConfig:
    """See docs/SPEC.md §4.

    Attributes
    ----------
    graph_path : str          npz written by ``BrainGraph.save`` (``fly build-brain``).
    steps : int               recurrent timesteps per position.
    alpha : float             initial leak ``a = sigmoid(leak_logit)`` (learned per neuron).
    activation : str          'relu' | 'gelu' (tanh approximation, so JS can match it) | 'tanh'.
    input_dim : int           flattened board planes (20 × 64).
    num_moves : int           policy head outputs.
    weight_init_scale : float mean row-sum of |w| at init (see ``FlyBrain.init_syn_gain``).
    dale : bool               enforce synapse signs (``w = sign · softplus(gain)``); False = free sign.
    dtype : str               parameter dtype of the dense parts (``w_in``/``b_in`` and the heads); the recurrent
                              ``syn_gain``/``bias``/``leak_logit`` are always float32 ('float32' + autocast is the
                              recommended way to get bf16 compute; a bf16 parameter dtype has no fp32 master
                              weights, so small updates to the dense parts can round away).
    value_hidden : int        hidden width of the value MLP; 0 = plain ``Linear(n_out, 1)``.
    vision : bool             inject the board through the retina (``graph.retina_idx`` photoreceptors, each
                              looking at one board square through ``w_ret``) when the graph has one.
    sensory_input : bool      keep the dense ``w_in`` projection into ``input_idx`` (the pre-retina input path).
    readout_steps : tuple     1-based timesteps whose ``output_idx`` activity is concatenated into the head
                              input; empty = final step only. Each must lie in ``1..steps``, strictly increasing.
    neuromod : bool           synapses from DA / SER / OCT neurons gate the others multiplicatively
                              (``pre = ion * (1 + tanh(mod)) + bias + inputs``) instead of adding to them.
    homeostatic : bool        learn a per-neuron intrinsic gain ``exp(log_gain)`` on the synaptic input (Dale-safe,
                              folded into the effective weights). The trainer calibrates it on a data batch at
                              initialisation so that deep pathways (retina -> lamina -> medulla -> ... ) carry
                              signal instead of attenuating ~20x per hop.
    central_dim : int         0 = off; else a ``Linear(n_central, central_dim)`` of the final activity of the
                              graph's ``'central'`` neurons is concatenated to the head input.
    """

    graph_path: str = ""
    steps: int = 8
    alpha: float = 0.5
    activation: str = "relu"
    sat: float = 10.0            # firing-rate ceiling for 'satrelu': sat * tanh(relu(x) / sat)
    input_dim: int = 1280
    num_moves: int = 4168
    weight_init_scale: float = 1.0
    dale: bool = True
    dtype: str = "float32"
    value_hidden: int = 256
    vision: bool = True
    sensory_input: bool = True
    readout_steps: tuple[int, ...] = ()
    neuromod: bool = False
    central_dim: int = 0
    homeostatic: bool = False    # per-neuron intrinsic gain, calibrated on data at init (see FlyBrain.calibrate_gains)

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {self.activation!r}")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
        if self.sat <= 0:
            raise ValueError("sat must be > 0")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie strictly inside (0, 1)")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        self.graph_path = str(self.graph_path)
        self.readout_steps = _parse_steps(self.readout_steps)
        if any(t < 1 or t > self.steps for t in self.readout_steps):
            raise ValueError(f"readout_steps must lie in 1..steps={self.steps}, got {self.readout_steps}")
        if any(b <= a for a, b in zip(self.readout_steps, self.readout_steps[1:])):
            raise ValueError(f"readout_steps must be strictly increasing, got {self.readout_steps}")
        if self.central_dim < 0:
            raise ValueError("central_dim must be >= 0")
        if not (self.vision or self.sensory_input):
            raise ValueError("at least one of vision / sensory_input must be enabled (the board has to get in)")
        self.vision, self.sensory_input, self.neuromod = bool(self.vision), bool(self.sensory_input), bool(self.neuromod)
        self.central_dim = int(self.central_dim)

    @property
    def effective_readout_steps(self) -> tuple[int, ...]:
        """``readout_steps`` with the empty default resolved to ``(steps,)``."""
        return self.readout_steps or (self.steps,)

    # ---- (de)serialisation ---------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["readout_steps"] = list(self.readout_steps)  # YAML-safe (safe_dump has no tuple tag)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown BrainConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path

    @classmethod
    def from_yaml(cls, path: str | Path) -> BrainConfig:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    def replace(self, **overrides: Any) -> BrainConfig:
        return self.from_dict({**self.to_dict(), **overrides})
