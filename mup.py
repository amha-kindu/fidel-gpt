"""Maximal update parametrisation (muP) for GPTmodel, with Adam.

Tune hyperparameters at a small base width, then scale embed_dim (and ff_dim,
attn_dim) up: under muP every layer's activations and per-step updates stay the
same size as at the base, so the base's lr, init and schedule carry over.

Each parameter falls into one of three classes (Tensor Programs V, Adam column):

  input / vector-like   embedding, norm gains, biases. Init and lr unchanged.
  hidden matrices       Wqkv, Wo, Wug, Wd: both dims grow with width. Init variance
                        ~ 1/fan_in, Adam lr ~ base_fan_in / fan_in, and decoupled
                        weight decay ~ fan_in / base_fan_in, so lr * wd -- the
                        fraction of a weight AdamW decays per step -- doesn't change.
  output                the head (vocab fixed, input grows). Init variance
                        ~ 1/fan_in, lr unchanged, logits multiplied by
                        base_embed_dim / embed_dim.

Attention logits scale as sqrt(base_head_dim) / head_dim instead of 1/sqrt(head_dim),
only when mup_head is set (i.e. when width grows by widening heads).

Every rule is anchored so that at embed_dim == mup_width the model, its
init and its optimiser are exactly the standard ones: a run tuned without muP is
the base of a muP family as it stands.

Assumes attn_dim grows in proportion to embed_dim, and so does ff_dim unless
mup_ff says otherwise.
"""
import math
from typing import NamedTuple

import torch
import torch.nn as nn


def enabled(config) -> bool:
    return getattr(config, "mup_width", 0) > 0


class BaseDims(NamedTuple):
    embed: float
    attn: float
    ff: float


def base_dims(config) -> BaseDims:
    """The base model's widths that this config is a width-scaled copy of."""
    base_embed = config.mup_width
    ratio = base_embed / config.embed_dim
    base_ff = config.mup_ff or config.ff_dim * ratio
    return BaseDims(float(base_embed), config.attn_dim * ratio, float(base_ff))


def output_multiplier(config) -> float:
    """Multiplier on the logits: base_embed_dim / embed_dim, or 1 without muP."""
    return config.mup_width / config.embed_dim if enabled(config) else 1.0


def attention_scale(config) -> float | None:
    """The SDPA `scale` for attention logits; None keeps SDPA's 1/sqrt(head_dim)."""
    if not enabled(config) or not config.mup_head:
        return None
    head_dim = config.attn_dim // config.heads
    return math.sqrt(config.mup_head) / head_dim


class HiddenMatrix(NamedTuple):
    name: str
    weight: nn.Parameter
    fan_in: float
    fan_out: float
    base_fan_in: float
    base_fan_out: float

    @property
    def width_mult(self) -> float:
        """fan_in / base_fan_in: what this matrix's Adam lr is divided by."""
        return self.fan_in / self.base_fan_in


def hidden_matrices(model: nn.Module) -> list[HiddenMatrix]:
    """Every hidden matrix in the decoder stack, with its own and its base shape.

    Found by role rather than by shape, so a replacement layer is classified the
    same as the nn.Linear it stands in for. A role whose module has no `weight`
    Parameter (e.g. wrapped by an adapter) is skipped and so trains as vector-like.
    """
    config = model.config
    base = base_dims(config)
    e, a, f = config.embed_dim, config.attn_dim, config.ff_dim
    roles = {  # attribute path: (fan_in, fan_out, base_fan_in, base_fan_out)
        "attention.Wqkv": (e, 3 * a, base.embed, 3 * base.attn),
        "attention.Wo": (a, e, base.attn, base.embed),
        "feed_forward.Wug": (e, 2 * f, base.embed, 2 * base.ff),
        "feed_forward.Wd": (f, e, base.ff, base.embed),
    }
    found = []
    for i, block in enumerate(model.decoders):
        for path, dims in roles.items():
            module = block.get_submodule(path)
            weight = getattr(module, "weight", None)
            if isinstance(weight, nn.Parameter):
                found.append(HiddenMatrix(f"decoders.{i}.{path}.weight", weight, *map(float, dims)))
    return found


def xavier_rescale(fan_in: float, fan_out: float, base_fan_in: float, base_fan_out: float) -> float:
    """Factor taking a xavier init at (fan_in, fan_out), variance 2 / (fan_in + fan_out),
    to the muP variance: the base's xavier variance 2 / (base_fan_in + base_fan_out)
    scaled by base_fan_in / fan_in.

    Written as one ratio so that at the base shape numerator and denominator are the
    same float products and the factor is exactly 1.0.
    """
    return math.sqrt((base_fan_in * (fan_in + fan_out)) / (fan_in * (base_fan_in + base_fan_out)))


@torch.no_grad()
def init_(model: nn.Module) -> None:
    """Rescale the hidden matrices and the head from their xavier init to muP variances.

    Rescales in place rather than redrawing: no RNG is consumed, so at the base
    width (every factor exactly 1) the weights are bit-identical to the standard
    build. Called after the standard init and before anything that builds on it
    (deepnorm's beta), so those apply to muP weights as they would to standard ones.
    """
    for m in hidden_matrices(model):
        m.weight.mul_(xavier_rescale(m.fan_in, m.fan_out, m.base_fan_in, m.base_fan_out))

    head = getattr(model.projection.linear, "weight", None)
    if isinstance(head, nn.Parameter):
        config = model.config
        head.mul_(xavier_rescale(config.embed_dim, config.vocab_size,
                                 base_dims(config).embed, config.vocab_size))


def param_groups(model: nn.Module, lr: float, weight_decay: float,
                 no_decay: tuple[str, ...] = ()) -> list[dict]:
    """AdamW param groups with each hidden matrix's lr and weight decay rescaled.

    Every group carries its own `lr` (and `lr_mult`, for reference), so a
    scheduler that scales each group's initial lr keeps the ratios. The lr_mult
    == 1 group comes first: loggers and schedulers that read param_groups[0]
    see the base lr.
    """
    mults = {id(m.weight): m.width_mult for m in hidden_matrices(model)}
    groups: dict[tuple[float, bool], list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        decays = name.rsplit(".", 1)[-1] not in no_decay
        groups.setdefault((mults.get(id(param), 1.0), decays), []).append(param)

    ordered = sorted(groups.items(), key=lambda item: (item[0][0] != 1.0, item[0][0], not item[0][1]))
    return [{
        "params": params,
        "lr": lr / mult,
        "lr_mult": 1.0 / mult,
        # Decoupled decay is lr * wd per step; scaling wd by the width multiplier
        # keeps that product, and so each weight's decay timescale, width-free.
        "weight_decay": weight_decay * mult if decays else 0.0,
    } for (mult, decays), params in ordered]


def describe(model: nn.Module) -> str:
    """One line for the logs: the width multipliers in force."""
    config = model.config
    if not enabled(config):
        return "muP off"
    mats = hidden_matrices(model)
    per_role = {}
    for m in mats:
        per_role.setdefault(m.name.split(".", 2)[-1].rsplit(".", 1)[0], m.width_mult)
    roles = ", ".join(f"{k} lr/{v:g}" for k, v in per_role.items())
    scale = attention_scale(config)
    attn = f"attn scale {scale:.4g}" if scale is not None else "attn scale 1/sqrt(d_head)"
    return (f"muP base {config.mup_width} -> {config.embed_dim}: {roles}; "
            f"logits x{output_multiplier(config):g}; {attn}")
