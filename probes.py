"""Diagnostic probes, registered by the modules they measure.

`compare_models.diagnose_modules` drives everything registered here but knows
nothing about what any of it measures. That inversion is the point: a probe that
reads a module's internals belongs beside that module, where it can use
`isinstance` instead of guessing at an interface, and where whoever changes the
module is looking at the probe that reads it.

This module imports only torch, so anything -- `norms.py`, `custom.py`,
`model.py` -- can register without an import cycle back through the comparison
script.

To add one:

    from probes import register

    @register("mynorm", lambda m: isinstance(m, MyNorm))
    def _mynorm(module, inputs, output, ctx):
        x = inputs[0].float()
        return {"input_r": ctx.mean(x.square().mean(-1, keepdim=True).sqrt())}

`measure` runs inside a forward hook and returns `{metric: 0-dim tensor}`. The
driver files each as `<family>/<site>/<metric>` and reduces over decoder depth,
so keep the returned names short and leave every value ON DEVICE -- the driver
batches the single host transfer, and an `.item()` here would reintroduce one
device sync per metric per site per evaluation.

These tags are variant-specific by construction: a model without the module
emits none of them. Keep them out of the internals-blind `attn/*` `ffn/*`
`collapse/*` namespace, which stays comparable across unrelated model classes
precisely because nothing in it may look inside a module.
"""
from typing import Callable, NamedTuple

import torch

# Matches compare_models.FLOOR. Clamping here rather than at finfo.tiny: dividing
# by 1e-38 produces inf, which then poisons a whole masked mean. These are
# diagnostics, so a bounded wrong answer beats an unbounded one.
FLOOR = 1e-12


class ProbeContext:
    """Masked reductions over one probe batch.

    Padding positions are excluded from every reduction here so a probe author
    cannot forget to. Padding tokens all share one embedding, so pad-pad pairs
    sit at cosine ~1 and would inflate any similarity averaged over every pair;
    self-pairs sit at exactly 1. Both are excluded once, here, rather than in
    every probe.

    It is also why these use `torch.where` rather than a multiply by zero: a
    padding position can hold inf or NaN -- a zero vector divided by its own
    clamped norm, say -- and 0 * NaN is NaN, which would take the entire
    reduction with it.

    The helpers cover token-shaped tensors, `(B, S, 1)` or `(B, S, C)`. Reduce a
    parameter-shaped tensor with plain torch instead; it has no token axis and
    nothing to mask.
    """

    def __init__(self, valid: torch.Tensor) -> None:
        self.valid = valid                               # (B, S)   bool
        self.token = valid.unsqueeze(-1)                 # (B, S, 1)
        pair = valid.unsqueeze(2) & valid.unsqueeze(1)   # (B, S, S)
        pair.diagonal(dim1=-2, dim2=-1).fill_(False)
        self.pair = pair
        self.n_valid = valid.sum().clamp_min(1)
        self.floor = FLOOR

    def _zero(self, like: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=like.dtype, device=like.device)

    def mean(self, values: torch.Tensor) -> torch.Tensor:
        """Mean over valid positions, reduced to a scalar."""
        values = values.float()
        mask = self.token.expand_as(values)
        return torch.where(mask, values, self._zero(values)).sum() / mask.sum().clamp_min(1)

    def std(self, values: torch.Tensor) -> torch.Tensor:
        """Standard deviation over valid positions, reduced to a scalar."""
        centred = values.float() - self.mean(values)
        return self.mean(centred.square()).clamp_min(0).sqrt()

    def fraction(self, predicate: torch.Tensor) -> torch.Tensor:
        """Fraction of valid positions where a bool tensor is True."""
        return self.mean(predicate.float())

    def extremes(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(min, max) over valid positions only."""
        values = values.float()
        mask = self.token.expand_as(values)
        low = torch.where(mask, values, torch.full_like(values, float("inf")))
        high = torch.where(mask, values, torch.full_like(values, float("-inf")))
        return low.amin(), high.amax()

    def channel_mean(self, values: torch.Tensor) -> torch.Tensor:
        """Mean over valid positions, KEEPING the last axis.

        For per-channel quantities -- a shell occupancy vector, a per-head score
        -- where the reduction is over tokens but each channel stays separate.
        """
        values = values.float()
        kept = torch.where(self.token, values, self._zero(values))
        return kept.flatten(0, -2).sum(0) / self.n_valid

    def collapse(self, hidden: torch.Tensor) -> torch.Tensor:
        """Mean pairwise cosine similarity of token representations.

        ~0 means tokens stay spread out; -> 1 means they have collapsed onto each
        other and depth is no longer buying anything. Pairs are taken WITHIN each
        sequence, excluding self-pairs and anything touching padding.
        """
        unit = torch.nn.functional.normalize(hidden.float(), dim=-1)
        pairwise = unit @ unit.transpose(-2, -1)
        return torch.where(self.pair, pairwise, self._zero(pairwise)).sum() / self.pair.sum().clamp_min(1)

    def isotropy(self, hidden: torch.Tensor) -> torch.Tensor:
        """How evenly the tokens spread their energy over the directions available.

        The participation ratio of the token covariance spectrum -- (sum lambda)^2
        / sum lambda^2, also called the effective rank -- divided by the most
        directions this tensor could possibly have spanned. 1 is isotropic: every
        available direction carries the same energy. -> 0 is strongly anisotropic:
        the tokens live on a handful of directions no matter how wide the feature
        axis is.

        Normalised to (0, 1] rather than left as the raw effective rank so it
        stays comparable across widths, and named for the property rather than for
        the estimator -- "rank" reads as an integer count, which this is not.

        This catches the failure that cosine collapse misses. Mean pairwise cosine
        reports how aligned tokens are with each other; a block can hold that near
        zero and still write every token into the same two-dimensional subspace,
        and an anisotropic update is a block that has stopped using the width it
        was given.

        Computed from the Gram matrix's traces -- tr(C) = ||Z||_F^2 and
        tr(C^2) = ||C||_F^2 -- so there is no eigendecomposition, and the
        covariance is (C, C) regardless of how many tokens are in the batch.
        """
        tokens = hidden.float()[self.valid]                  # (TOKENS, C)
        tokens = tokens - tokens.mean(dim=0, keepdim=True)
        covariance = tokens.transpose(0, 1) @ tokens
        trace = covariance.diagonal().sum()
        return trace.square() / (covariance.square().sum().clamp_min(FLOOR) * min(tokens.shape))


class Probe(NamedTuple):
    family: str                                          # tag prefix, e.g. "attn"
    match: Callable[[torch.nn.Module], bool]
    measure: Callable[..., dict[str, torch.Tensor]]
    per_site: bool                                       # see `register`


_REGISTRY: list[Probe] = []


def register(family: str, match: Callable[[torch.nn.Module], bool], per_site: bool = True):
    """Decorator registering `measure` for every module `match` accepts.

    `measure(module, inputs, output, ctx) -> {metric: 0-dim tensor}`, where
    `inputs` is the forward hook's positional-argument tuple.

    `per_site` decides whether the module's structural role appears in the tag:

      True  (default)  `<family>/<site>/<metric>` -- for modules that occur MORE
                       than once per block, where the tag has to say which one.
                       A block holds norm1 and norm2, Wqkv and Wo and Wug and Wd;
                       without the site segment their series would collide.
      False            `<family>/<metric>` -- for the at-most-one-per-block
                       modules, where the role is already implied by the family.
                       `attn/update_ratio` says everything `attn/attention/
                       update_ratio` would.

    Either way the driver appends the per-layer and mean/min/max reductions, and
    several probes may match one module: a family is one question asked of it,
    not one hook.
    """
    def decorate(measure):
        _REGISTRY.append(Probe(family, match, measure, per_site))
        return measure
    return decorate


def probes_for(module: torch.nn.Module) -> list[Probe]:
    """Every registered probe that accepts `module`; usually zero or one."""
    return [probe for probe in _REGISTRY if probe.match(module)]


def registered() -> tuple[Probe, ...]:
    """Everything registered so far, for introspection and tests."""
    return tuple(_REGISTRY)
