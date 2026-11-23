"""Maximal update parametrisation (muP) for GPTmodel, with Adam, plus depth scaling.

Tune hyperparameters on a small base model, then scale it up: each scaling rule
keeps activations and per-step updates the same size as at the base, so the base's
lr, init and schedule carry over. Every rule is anchored so that at the base size
the model, its init and its optimiser are exactly the standard ones: a run tuned
without scaling is the base of a family as it stands.

Width (mup_width, mup_ff, mup_head). Tune at embed_dim = mup_width, then scale
embed_dim (and ff_dim, attn_dim) up. Each parameter falls into one of three
classes (Tensor Programs V, Adam column):

  input / vector-like   embedding, norm gains, biases. Init and lr unchanged.
  hidden matrices       Wqkv, Wo, Wug, Wd: both dims grow with width. Init variance
                        ~ 1/fan_in, Adam lr ~ base_fan_in / fan_in.
  output                the head (vocab fixed, input grows). Init variance
                        ~ 1/fan_in, lr unchanged, logits multiplied by
                        base_embed_dim / embed_dim.

Attention logits scale as sqrt(base_head_dim) / head_dim instead of 1/sqrt(head_dim),
only when mup_head is set (i.e. when width grows by widening heads). Assumes attn_dim
grows in proportion to embed_dim, and so does ff_dim unless mup_ff says otherwise.

Depth (dmup_depth, dmup_alpha), independent of width; either can be on alone.
Tune at a base depth L_base = dmup_depth, then change n_decoders = L:

  block outputs         each attention and feed-forward output is multiplied by
                        (L_base / L) ** alpha before it joins the residual stream, so
                        the blocks' combined write stays the same size at any depth.
  in-block parameters   everything inside model.decoders (the hidden matrices, their
                        biases, the norm and QK-norm gains): Adam lr x
                        (L / L_base) ** (alpha - 1). The embedding, norm_f and head are
                        unchanged.
  init                  unchanged.

alpha = 0.5 is Depth-muP (Tensor Programs VI): branch x sqrt(L_base / L), in-block lr
x sqrt(L_base / L). alpha = 1 is CompleteP (Dey et al. 2025): branch x L_base / L,
in-block lr unchanged. Pre-norm only: post-norm renormalises the stream after every
block and deepnorm has its own depth scaling (alpha, beta).

Weight decay is no rule's business: decoupled decay removes lr * wd of a weight per
step, so Parametrization divides each parameter's wd by the same factor its lr was
multiplied by, which keeps every weight's decay timescale width- and depth-free.

Structure: a ScalingRule says how it adjusts each hyperparameter -- per parameter (lr,
init) and per model (block outputs, logits, attention scale) -- with every hook
defaulting to "unchanged". Parametrization.from_config(config) holds the rules a
config switches on and combines them; it is the only object callers use.
"""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, NamedTuple

import torch
import torch.nn as nn


class Dims(NamedTuple):
    """A model's widths: the real ones, or the base ones a width rule scales from."""
    embed: float
    attn: float
    ff: float


# Each hidden matrix, by its path inside a decoder block, with its (fan_in, fan_out)
# as a function of the widths -- applied to the real widths and to the base widths.
HIDDEN_ROLES: dict[str, Callable[[Dims], tuple[float, float]]] = {
    "attention.Wqkv": lambda d: (d.embed, 3 * d.attn),
    "attention.Wo": lambda d: (d.attn, d.embed),
    "feed_forward.Wug": lambda d: (d.embed, 2 * d.ff),
    "feed_forward.Wd": lambda d: (d.ff, d.embed),
}


class Role(Enum):
    """Tensor Programs V's parameter classes."""
    VECTOR = "vector"   # embedding, norm gains, biases, and anything unrecognised
    HIDDEN = "hidden"   # the decoder blocks' weight matrices (HIDDEN_ROLES)
    OUTPUT = "output"   # the head


@dataclass(frozen=True)
class ParamSpec:
    """One parameter as the rules see it."""
    name: str
    param: nn.Parameter
    role: Role
    path: str | None    # "attention.Wqkv" etc. for hidden matrices, "projection" for the head
    fan_in: float
    fan_out: float
    in_block: bool      # inside model.decoders


def model_dims(config) -> Dims:
    return Dims(float(config.embed_dim), float(config.attn_dim), float(config.ff_dim))


def classify(model: nn.Module) -> list[ParamSpec]:
    """Every parameter of the model, with its role.

    Hidden matrices are found by role rather than by shape, so a replacement layer is
    classified the same as the nn.Linear it stands in for. A role whose module has no
    `weight` Parameter (e.g. wrapped by an adapter) is skipped and so trains as
    vector-like.
    """
    config = model.config
    dims = model_dims(config)
    known: dict[int, tuple[Role, str, float, float]] = {}
    for block in model.decoders:
        for path, fans in HIDDEN_ROLES.items():
            weight = getattr(block.get_submodule(path), "weight", None)
            if isinstance(weight, nn.Parameter):
                known[id(weight)] = (Role.HIDDEN, path, *fans(dims))
    head = getattr(model.projection.linear, "weight", None)
    if isinstance(head, nn.Parameter):
        known[id(head)] = (Role.OUTPUT, "projection", dims.embed, float(config.vocab_size))

    in_blocks = {id(p) for p in model.decoders.parameters()}
    specs = []
    for name, param in model.named_parameters():
        role, path, fan_in, fan_out = known.get(id(param), (Role.VECTOR, None, 0.0, 0.0))
        specs.append(ParamSpec(name, param, role, path, fan_in, fan_out, id(param) in in_blocks))
    return specs


@dataclass(frozen=True)
class Factor:
    """A multiplier kept as num / den and applied as x * num / den.

    A pair rather than one float so that a rule which divides (width: lr / w) and one
    which multiplies (depth: lr * d) are each exact on their own -- x * 1.0 and
    x / 1.0 are exact -- and a run with one rule off is bit-identical to a run without it.
    """
    num: float = 1.0
    den: float = 1.0

    def apply(self, x: float) -> float:
        return x * self.num / self.den

    def invert(self, x: float) -> float:
        return x * self.den / self.num


UNCHANGED = Factor()


def xavier_rescale(fan_in: float, fan_out: float, base_fan_in: float, base_fan_out: float) -> float:
    """Factor taking a xavier init at (fan_in, fan_out), variance 2 / (fan_in + fan_out),
    to the muP variance: the base's xavier variance 2 / (base_fan_in + base_fan_out)
    scaled by base_fan_in / fan_in.

    Written as one ratio so that at the base shape numerator and denominator are the
    same float products and the factor is exactly 1.0.
    """
    return math.sqrt((base_fan_in * (fan_in + fan_out)) / (fan_in * (base_fan_in + base_fan_out)))


class ScalingRule(ABC):
    """One way of scaling the base model, as its adjustment to each hyperparameter.

    Every hook defaults to "unchanged"; a rule overrides the ones it affects.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config) -> "ScalingRule | None":
        """The rule this config asks for, or None when it is off. Validates the config."""

    # Per-parameter adjustments.
    def lr(self, spec: ParamSpec) -> Factor:
        """Adam lr multiplier. Weight decay follows it inversely (see Parametrization)."""
        return UNCHANGED

    def init(self, spec: ParamSpec) -> float:
        """Multiplier on the standard (xavier) init of this parameter."""
        return 1.0

    # Model-level adjustments.
    def branch(self) -> float:
        """Multiplier on every attention / feed-forward output."""
        return 1.0

    def logits(self) -> float:
        """Multiplier on the head's logits."""
        return 1.0

    def attention_scale(self) -> float | None:
        """The SDPA `scale` for attention logits; None leaves it alone."""
        return None

    @abstractmethod
    def describe(self, specs: list[ParamSpec]) -> str:
        """A fragment for the log line."""


class WidthRule(ScalingRule):
    """muP: mup_width (base embed_dim), mup_ff (base ff_dim), mup_head (base head dim)."""

    def __init__(self, config):
        self.config = config
        ratio = config.mup_width / config.embed_dim
        self.base = Dims(float(config.mup_width), config.attn_dim * ratio,
                         float(config.mup_ff or config.ff_dim * ratio))

    @classmethod
    def from_config(cls, config) -> "WidthRule | None":
        return cls(config) if getattr(config, "mup_width", 0) > 0 else None

    def base_fans(self, spec: ParamSpec) -> tuple[float, float]:
        if spec.role is Role.HIDDEN:
            return HIDDEN_ROLES[spec.path](self.base)
        return self.base.embed, spec.fan_out  # the head: vocab doesn't scale

    def width_mult(self, spec: ParamSpec) -> float:
        """fan_in / base_fan_in: what a hidden matrix's Adam lr is divided by."""
        return spec.fan_in / self.base_fans(spec)[0]

    def lr(self, spec: ParamSpec) -> Factor:
        return Factor(den=self.width_mult(spec)) if spec.role is Role.HIDDEN else UNCHANGED

    def init(self, spec: ParamSpec) -> float:
        if spec.role is Role.VECTOR:
            return 1.0
        return xavier_rescale(spec.fan_in, spec.fan_out, *self.base_fans(spec))

    def logits(self) -> float:
        return self.config.mup_width / self.config.embed_dim

    def attention_scale(self) -> float | None:
        if not self.config.mup_head:
            return None
        return math.sqrt(self.config.mup_head) / (self.config.attn_dim // self.config.heads)

    def describe(self, specs: list[ParamSpec]) -> str:
        per_role: dict[str, float] = {}
        for spec in specs:
            if spec.role is Role.HIDDEN:
                per_role.setdefault(spec.path, self.width_mult(spec))
        roles = ", ".join(f"{k} lr/{per_role[k]:g}" for k in HIDDEN_ROLES if k in per_role)
        scale = self.attention_scale()
        attn = f"attn scale {scale:.4g}" if scale is not None else "attn scale 1/sqrt(d_head)"
        return (f"muP base {self.config.mup_width} -> {self.config.embed_dim}: {roles}; "
                f"logits x{self.logits():g}; {attn}")


class DepthRule(ScalingRule):
    """Depth-muP / CompleteP: dmup_depth (base n_decoders), dmup_alpha."""

    def __init__(self, config):
        self.base_depth: int = config.dmup_depth
        self.depth: int = config.n_decoders
        self.alpha: float = config.dmup_alpha

    @classmethod
    def from_config(cls, config) -> "DepthRule | None":
        if getattr(config, "dmup_depth", 0) <= 0:
            return None
        if not config.norm_strategy.startswith("pre"):
            raise ValueError(f"dmup_depth needs a pre-norm strategy, got {config.norm_strategy!r}")
        return cls(config)

    def block_lr_mult(self) -> float:
        """(L / L_base) ** (alpha - 1): the Adam lr multiplier inside the blocks."""
        return (self.depth / self.base_depth) ** (self.alpha - 1.0)

    def lr(self, spec: ParamSpec) -> Factor:
        return Factor(num=self.block_lr_mult()) if spec.in_block else UNCHANGED

    def branch(self) -> float:
        return (self.base_depth / self.depth) ** self.alpha

    def describe(self, specs: list[ParamSpec]) -> str:
        return (f"depth base {self.base_depth} -> {self.depth} (alpha {self.alpha:g}): "
                f"block outputs x{self.branch():.4g}, in-block lr x{self.block_lr_mult():.4g}")


class Parametrization:
    """The scaling rules a config switches on, combined. Falsy when none are on."""

    RULES: tuple[type[ScalingRule], ...] = (WidthRule, DepthRule)

    def __init__(self, rules: tuple[ScalingRule, ...]):
        self.rules = rules

    @classmethod
    def from_config(cls, config) -> "Parametrization":
        if config is None:
            return cls(())
        return cls(tuple(rule for rule in (r.from_config(config) for r in cls.RULES) if rule is not None))

    def __bool__(self) -> bool:
        return bool(self.rules)

    # Model-level adjustments: products over the rules (1.0 * x is exact).
    def branch_multiplier(self) -> float:
        return math.prod((rule.branch() for rule in self.rules), start=1.0)

    def output_multiplier(self) -> float:
        return math.prod((rule.logits() for rule in self.rules), start=1.0)

    def attention_scale(self) -> float | None:
        scales = [s for s in (rule.attention_scale() for rule in self.rules) if s is not None]
        if len(scales) > 1:
            raise ValueError("more than one scaling rule sets the attention scale")
        return scales[0] if scales else None

    # Per-parameter adjustments.
    def lr_factors(self, spec: ParamSpec) -> tuple[Factor, ...]:
        return tuple(rule.lr(spec) for rule in self.rules)

    @staticmethod
    def scale_lr(lr: float, factors: tuple[Factor, ...]) -> float:
        # Last rule first for the lr and first rule first for the weight decay (its
        # inverse): with width and depth both on that is lr * d / w and wd * w / d.
        for factor in reversed(factors):
            lr = factor.apply(lr)
        return lr

    @staticmethod
    def scale_weight_decay(weight_decay: float, factors: tuple[Factor, ...]) -> float:
        for factor in factors:
            weight_decay = factor.invert(weight_decay)
        return weight_decay

    @torch.no_grad()
    def init_(self, model: nn.Module) -> None:
        """Rescale the parameters from their standard init by every rule's init factor.

        Rescales in place rather than redrawing: no RNG is consumed, so at the base
        size (every factor exactly 1) the weights are bit-identical to the standard
        build. Called after the standard init and before anything that builds on it
        (deepnorm's beta), so those apply to scaled weights as they would to standard ones.
        """
        for spec in classify(model):
            factor = math.prod((rule.init(spec) for rule in self.rules), start=1.0)
            if factor != 1.0:
                spec.param.mul_(factor)

    def param_groups(self, model: nn.Module, lr: float, weight_decay: float,
                     no_decay: tuple[str, ...] = ()) -> list[dict]:
        """AdamW param groups, each with its own `lr` and `weight_decay`.

        Grouped by the tuple of per-rule factors, so the arithmetic is the same
        whichever other rules are on. Every group carries its own `lr` (and
        `lr_mult`, for reference), so a scheduler that scales each group's initial lr
        keeps the ratios. The lr_mult == 1 group comes first: loggers and schedulers
        that read param_groups[0] see the base lr.
        """
        groups: dict[tuple[tuple[Factor, ...], bool], list[nn.Parameter]] = {}
        for spec in classify(model):
            if not spec.param.requires_grad:
                continue
            decays = spec.name.rsplit(".", 1)[-1] not in no_decay
            groups.setdefault((self.lr_factors(spec), decays), []).append(spec.param)

        def lr_mult(factors: tuple[Factor, ...]) -> float:
            return self.scale_lr(1.0, factors)

        ordered = sorted(groups.items(),
                         key=lambda item: (lr_mult(item[0][0]) != 1.0, -lr_mult(item[0][0]), not item[0][1]))
        return [{
            "params": params,
            "lr": self.scale_lr(lr, factors),
            "lr_mult": lr_mult(factors),
            "weight_decay": self.scale_weight_decay(weight_decay, factors) if decays else 0.0,
        } for (factors, decays), params in ordered]

    def describe(self, model: nn.Module) -> str:
        """One line for the logs: the adjustments in force."""
        if not self.rules:
            return "muP off"
        specs = classify(model)
        return " | ".join(rule.describe(specs) for rule in self.rules)
