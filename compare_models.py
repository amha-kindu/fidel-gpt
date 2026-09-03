"""Short-run A/B of model architectures and attention variants on real data.

Trains one model per VARIANT from identical seeds on identical batches and
reports the three things that actually decide the question.

The batches, their order and the validation set are built once and shared by
every variant, and each variant re-verifies an order-sensitive digest of all of
it before it trains a step, so a difference in the reported numbers can only
come from the variant itself (see DataPlan). What gets reported:

  * validation loss vs step       -- quality per unit of learning
  * validation loss vs wall-clock -- quality per unit of compute, the honest
    axis when the variants differ in speed by 20-30%
  * area under the loss curve     -- each of the two above reduced to one
    number, so an arm that led for the whole run is not scored on its last
    evaluation alone (see aulc)
  * train/val generalisation gap  -- how much of an arm's win is fit to the
    batches it trained on rather than learning, read off the two loss curves
    above as loss/gap
  * per-layer attention health    -- how the sublayer is behaving in each decoder
    block, not averaged into a single scalar that hides the one bad layer
  * per-layer gradient norms      -- Gradients/Global and Gradients/<component>,
    the same decomposition train.py logs (it shares the bucketing rule, see
    utils.component_key), so a variant's gradient trace here reads directly
    against a real training run's

The first variant is the baseline every other one is reported against. It is the
model the flags describe -- model.GPTmodel, standard multi-head attention --
unless you put something else first.

A variant is a name, optionally a model class, and a set of ModelConfig
overrides, given on the command line:

    --variant base:                     # the flags as they stand, no overrides
    --variant wide:heads=16             # same class, different config
    --variant post:post_norm=true,ff_dim=2048
    --variant flat:model=model2.GPTWide           # a DIFFERENT architecture
    --variant flat6:model=model2.GPTWide,n_decoders=6

`model=` is the one key that is not a ModelConfig field: it names an importable
nn.Module subclass ('module.Class', or a bare 'Class' from model.py), and
`--model` sets the default for variants that do not name one. So the comparison
is between arbitrary architectures, not only between configurations of one -- and
neither a class name nor a config field name is hardcoded anywhere in this file.
`--config` sets the shared config base every variant starts from.

Nothing in this script knows what is inside a model beyond two structural
assumptions, both of which GPTmodel subclasses satisfy for free: the blocks live
in `model.decoders`, and each block holds exactly one attention submodule whose
attribute name says so. Everything else is read off forward hooks -- the tensor
the sublayer reads, the update it writes and the block's output -- so every tag
means the same thing for every variant, which is the only way two of them can
honestly share a chart.

Defaults are sized for a CPU smoke run. On a GPU, scale up -- the comparison is
only meaningful once the model is big enough to be data-bound rather than
noise-bound:

    python compare_models.py --steps 4000 --embed-dim 512 --n-decoders 6 \
        --seq-len 512 --batch-size 32 --training-data <path> --validation-data <path>
"""
import argparse
import hashlib
import importlib
import json
import math
import os
import random
import time
from datetime import datetime
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from config import DEVICE, ENV, LOGGER, MIXED_PRECISION_ENABLED, ModelConfig
from dataset import TextStreamDataset
from utils import component_key, get_causal_mask

# model classes are NOT imported here: every arm names its own, and resolve_model
# imports it at parse time (see DEFAULT_MODEL).

# Norms are clamped here rather than at finfo.tiny: dividing by 1e-38 produces
# inf, which then poisons an entire masked mean. These are diagnostics, so a
# bounded wrong answer beats an unbounded one.
FLOOR = 1e-12

# name:key=value,... -- the model exactly as the flags configure it, and nothing
# else. Naming a second variant here would mean naming a ModelConfig field or a
# model class, and a default that hardcodes either is a default that breaks the
# day that field is renamed or that class moves. Every comparison arm comes from
# --variant on the command line; this is only the baseline they are measured
# against.
DEFAULT_VARIANTS = ("baseline:",)

# The class every variant uses unless --model or its own `model=` says otherwise.
# It is a string resolved by import, not the imported class, so this module has
# exactly one hardcoded model reference and it is this line.
DEFAULT_MODEL = "model.GPTmodel"

# The one override key that is not a ModelConfig field.
MODEL_KEY = "model"

# Substrings that identify the attention submodule of a decoder block. Matched on
# the attribute NAME, not on the type: the whole point of comparing two models is
# that their attention classes have nothing in common but nn.Module.
ATTENTION_HINTS = ("attention", "attn")


# --------------------------------------------------------------------------- #
# device + determinism
# --------------------------------------------------------------------------- #

def sync() -> None:
    """CUDA kernels launch asynchronously, so an unsynchronised perf_counter
    measures queue time, not compute. Wall-clock is the headline number this
    script exists to produce, so every timed boundary syncs first."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def seed_everything(seed: int) -> None:
    """Every variant starts from the same RNG state -- same init, same dropout
    draw, same everything the data order does not already pin down."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mask(inputs: torch.Tensor, pad: int, causal: torch.Tensor) -> torch.Tensor:
    """Rebuild the dataset's (input != pad) & causal mask from the inputs alone.

    Materialising the loader's masks alongside the batches would cost
    batches * batch_size * seq_len^2 bytes -- gigabytes at seq_len 512 -- so the
    batch lists hold only inputs and labels and this reconstructs the mask per
    step, exactly matching what TextStreamDataset.__getitem__ produces.
    """
    return (inputs != pad).view(inputs.shape[0], 1, 1, -1) & causal


# --------------------------------------------------------------------------- #
# variant specs
# --------------------------------------------------------------------------- #

_TRUE = {"1", "true", "t", "yes", "y", "on"}
_FALSE = {"0", "false", "f", "no", "n", "off"}


def coerce(key: str, raw: str, template: ModelConfig):
    """Cast a command-line string to whatever type ModelConfig holds for `key`.

    The template's own default is the type authority, so a new config field is
    settable from the command line the moment it exists, with no table here to
    keep in sync.
    """
    if not hasattr(template, key):
        fields = ", ".join(sorted(template.to_dict()))
        raise argparse.ArgumentTypeError(f"unknown ModelConfig field '{key}'. Known fields: {fields}")
    default = getattr(template, key)
    # bool before int -- bool IS an int subclass, and int("false") is a crash.
    if isinstance(default, bool):
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise argparse.ArgumentTypeError(f"'{key}' is a flag; use true/false, got '{raw}'")
    for caster in (int, float) if isinstance(default, (int, float)) else ():
        if isinstance(default, caster):
            try:
                return caster(raw)
            except ValueError:
                raise argparse.ArgumentTypeError(f"'{key}' expects {caster.__name__}, got '{raw}'")
    return raw


def parse_overrides(text: str, template: ModelConfig) -> tuple[dict, str | None]:
    """key=value,... -> (ModelConfig overrides, model path or None).

    `model=` is pulled out rather than coerced, because it is the one key that
    names a class instead of a config field.
    """
    out, model = {}, None
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected key=value, got '{item}'")
        key, _, value = item.partition("=")
        key = key.strip()
        if key == MODEL_KEY and not hasattr(template, MODEL_KEY):
            if not value.strip():
                raise argparse.ArgumentTypeError("'model=' needs a class, e.g. 'model=model2.GPTWide'")
            model = value.strip()
            continue
        out[key] = coerce(key, value, template)
    return out, model


def resolve_model(path: str) -> type:
    """Import 'module.Class' -- or a bare 'Class' from model.py -- and return it.

    Resolved while the arguments are parsed, so a typo fails in the first second
    of the run rather than after the corpus has been tokenised and the baseline
    trained. Nothing here knows the name of any architecture: an arm is
    comparable against the baseline as soon as it exists as an importable
    nn.Module, with no edit to this file.
    """
    module_name, _, class_name = path.rpartition(".")
    module_name = module_name or "model"
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise argparse.ArgumentTypeError(f"model '{path}': cannot import '{module_name}' ({error})")
    resolved = getattr(module, class_name, None)
    if resolved is None:
        raise argparse.ArgumentTypeError(f"model '{path}': '{module_name}' has no '{class_name}'")
    if not (isinstance(resolved, type) and issubclass(resolved, torch.nn.Module)):
        raise argparse.ArgumentTypeError(f"model '{path}': '{class_name}' is not an nn.Module subclass")
    return resolved


class Variant(NamedTuple):
    """One arm of the comparison: what to build, and what to call it."""
    label: str
    model: str          # dotted path, kept for the report and summary.json
    cls: type
    overrides: dict


def parse_variant(text: str, template: ModelConfig, default_model: str) -> Variant:
    if ":" not in text:
        raise argparse.ArgumentTypeError(
            f"variant '{text}' needs a name then a colon, e.g. 'wide:heads=16' or "
            "'flat:model=model2.GPTWide' (or 'base:' for the flags as they stand)")
    name, _, spec = text.partition(":")
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"variant '{text}' has an empty name")
    overrides, model = parse_overrides(spec, template)
    model = model or default_model
    return Variant(name, model, resolve_model(model), overrides)


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #

class Recorder:
    """Per-layer scalars accumulated as 0-dim device tensors.

    Every `.item()` on a CUDA tensor is a device sync. Reading each metric off
    the device as it is produced costs one stall per metric per layer -- well
    over a hundred per evaluation -- inside the loop whose wall-clock is the
    number this script exists to report. Here the values stay on device until
    `resolve()` stacks them into a single transfer.
    """

    def __init__(self) -> None:
        self._entries: list[tuple[str, int]] = []
        self._values: list[torch.Tensor] = []

    def add(self, tag: str, layer: int, value: torch.Tensor) -> None:
        self._entries.append((tag, layer))
        self._values.append(value.detach().float().reshape(()))

    def resolve(self) -> dict[str, float]:
        """(tag, layer) -> scalars, plus a mean/min/max reduction over layers.

        Reporting only the mean once hid a real finding: mid-stack collapse at
        one layer was invisible in the average, because the deepest layer was
        fine and a six-layer mean dilutes one bad layer sixfold. Any scalar
        worth watching is worth watching per layer, and the extremes are what
        make a single-layer anomaly visible on a summary chart.
        """
        if not self._values:
            return {}
        flat = torch.stack(self._values).cpu().tolist()   # the one and only sync
        per_tag: dict[str, dict[int, float]] = {}
        for (tag, layer), value in zip(self._entries, flat):
            per_tag.setdefault(tag, {})[layer] = value

        out: dict[str, float] = {}
        for tag, layers in per_tag.items():
            values = []
            for layer in sorted(layers):
                out[f"{tag}/layer_{layer:02d}"] = layers[layer]
                values.append(layers[layer])
            out[f"{tag}/mean"] = sum(values) / len(values)
            out[f"{tag}/min"] = min(values)
            out[f"{tag}/max"] = max(values)
        return out


class Geometry:
    """Masks shared by every diagnostic, built once per diagnostic batch.

    Padding tokens all share one embedding, so pad-pad pairs sit at cosine ~1
    and inflate any similarity averaged over every pair; self-pairs sit at
    exactly 1. Both are excluded here, once, instead of every hook rebuilding a
    `torch.eye` of its own.
    """

    def __init__(self, inputs: torch.Tensor, pad: int) -> None:
        self.valid = inputs != pad                                   # (B, S)
        self.token = self.valid.unsqueeze(-1)                        # (B, S, 1)
        pair = self.valid.unsqueeze(2) & self.valid.unsqueeze(1)     # (B, S, S)
        pair.diagonal(dim1=-2, dim2=-1).fill_(False)
        self.pair = pair


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `values` over the True positions of a broadcastable bool mask.

    `torch.where` rather than multiply-by-zero: a padding position can hold inf
    or NaN -- a zero vector divided by its own clamped norm, say -- and
    0 * NaN is NaN, which would take the whole reduction with it.
    """
    mask = mask.expand_as(values)
    values = values.float()
    kept = torch.where(mask, values, torch.zeros((), dtype=values.dtype, device=values.device))
    return kept.sum() / mask.sum().clamp_min(1)


def cosine_collapse(hidden: torch.Tensor, geom: Geometry) -> torch.Tensor:
    """Mean pairwise cosine similarity of token representations.

    ~0 means tokens stay spread out; -> 1 means they have collapsed onto each
    other and depth is no longer buying anything.
    """
    unit = F.normalize(hidden.float(), dim=-1)
    return masked_mean(unit @ unit.transpose(-2, -1), geom.pair)


def isotropy(hidden: torch.Tensor, geom: Geometry) -> torch.Tensor:
    """How evenly the tokens spread their energy over the directions available.

    The participation ratio of the token covariance spectrum -- (sum lambda)^2 /
    sum lambda^2, also called the effective rank -- divided by the most
    directions this tensor could possibly have spanned. 1 is isotropic: every
    available direction carries the same energy. -> 0 is strongly anisotropic:
    the tokens live on a handful of directions no matter how wide EMBED_DIM is.

    Normalised to (0, 1] rather than left as the raw effective rank so it stays
    comparable across embed_dim, and named for the property rather than for the
    estimator -- "rank" reads as an integer count, which this is not.

    This catches the failure that cosine collapse misses. Mean pairwise cosine
    reports how aligned tokens are with each other; a block can hold that near
    zero and still write every token into the same two-dimensional subspace, and
    an anisotropic update is a block that has stopped using the width it was
    given.

    Computed from the Gram matrix's traces -- tr(C) = ||Z||_F^2 and
    tr(C^2) = ||C||_F^2 -- so there is no eigendecomposition, and the covariance
    is (EMBED_DIM, EMBED_DIM) regardless of how many tokens are in the batch.
    """
    tokens = hidden[geom.valid]                                       # (TOKENS, EMBED_DIM)
    tokens = tokens - tokens.mean(dim=0, keepdim=True)
    covariance = tokens.transpose(0, 1) @ tokens
    trace = covariance.diagonal().sum()
    return trace.square() / (covariance.square().sum().clamp_min(FLOOR) * min(tokens.shape))


def decoder_blocks(model: torch.nn.Module, label: str) -> list[torch.nn.Module]:
    blocks = getattr(model, "decoders", None)
    if blocks is None or not len(blocks):
        raise RuntimeError(
            f"variant '{label}': {type(model).__name__} has no non-empty `decoders`. The "
            "per-layer diagnostics are the point of this script, so a model whose blocks "
            "cannot be enumerated cannot be compared here.")
    return list(blocks)


def attention_modules(model: torch.nn.Module, label: str) -> list[torch.nn.Module]:
    """The attention submodule of every decoder block, found by attribute name.

    Two architectures being compared are under no obligation to agree on that
    name, and hardcoding whichever one they currently share would turn a rename
    in either into an empty attn/* series here -- which does not look like a bug,
    it looks like a real difference between the arms. Matching on ATTENTION_HINTS
    covers the plausible names, and requiring exactly one match per block means a
    model this heuristic cannot read fails loudly instead of silently.
    """
    found = []
    for layer, block in enumerate(decoder_blocks(model, label)):
        children = list(block.named_children())
        matches = [child for name, child in children
                   if any(hint in name.lower() for hint in ATTENTION_HINTS)]
        if len(matches) != 1:
            names = ", ".join(name for name, _ in children) or "no children"
            raise RuntimeError(
                f"variant '{label}': block {layer} of {type(model).__name__} has "
                f"{len(matches)} submodules whose name looks like attention "
                f"({names}). The diagnostics need exactly one; rename it to contain "
                f"one of {ATTENTION_HINTS}, or drop the extra match.")
        found.append(matches[0])
    return found


@torch.inference_mode()
def diagnose(model: torch.nn.Module, attentions: list[torch.nn.Module], inputs: torch.Tensor,
             pad: int, causal: torch.Tensor) -> dict[str, float]:
    """Per-layer attention health, from ONE forward pass.

    Everything here is read off forward hooks, from three tensors per decoder
    block: what the attention sublayer was handed, the update it wrote back, and
    what left the block. No probe reaches inside an attention module, looks up an
    attribute or recomputes a score. That is deliberate -- a diagnostic that
    knows the internals of one variant produces a tag the other variant cannot
    report, which is precisely the tag that cannot be compared. It also means
    these numbers survive any change to the attention implementations, and that
    they mean the same thing across two unrelated model classes.

    A forward hook receives its module's positional inputs alongside its output,
    so one hook per attention module sees both the tensor it attends over
    (post-norm1 under the default pre-norm block) and the update it produces.

      collapse/input      mean pairwise cosine of the tokens entering attention
      collapse/output     the same after the whole block. ~0 means tokens stay
                          spread out; -> 1 means they have collapsed onto each
                          other and depth is no longer buying anything.
      attn/input_norm     mean ||x|| entering attention -- residual-stream drift,
                          and the scale every other magnitude here is relative to
      attn/update_ratio   ||update|| / ||x||, the sublayer's gain into the
                          residual stream. Far above 1 is a block shouting over
                          the stream (pair it with that block's Gradients/Decoder<i>
                          when one variant's Gradients/Global sits an order of
                          magnitude off the other's -- the ratio says which layer is
                          loud, the gradient says whether it is also unstable); far
                          below 1 is a block that has switched off.
      attn/update_cos     mean cos(update_i, x_i). ~0 is a sublayer writing
                          genuinely new content; -> 1 means it is mostly
                          rescaling what each token already held, which is
                          attention that has stopped moving information BETWEEN
                          tokens -- the failure a loss curve hides longest.
      attn/update_isotropy
                          how evenly the update spreads over the directions
                          available to it. -> 0 is a block writing everything
                          into a handful of directions however wide EMBED_DIM
                          is, which cosine collapse cannot see (see isotropy)
    """
    rec = Recorder()
    geom = Geometry(inputs, pad)
    handles = []

    def attention_hook(layer: int):
        def hook(_module, args, output):
            x = args[0].float()
            update = (output[0] if isinstance(output, tuple) else output).float()
            x_norm = x.norm(dim=-1, keepdim=True)
            update_norm = update.norm(dim=-1, keepdim=True)
            mean_x_norm = masked_mean(x_norm, geom.token)

            rec.add("attn/input_norm", layer, mean_x_norm)
            rec.add("attn/update_ratio", layer,
                    masked_mean(update_norm, geom.token) / mean_x_norm.clamp_min(FLOOR))
            rec.add("attn/update_cos", layer,
                    masked_mean((update * x).sum(dim=-1, keepdim=True)
                                / (update_norm * x_norm).clamp_min(FLOOR), geom.token))
            rec.add("attn/update_isotropy", layer, isotropy(update, geom))
            rec.add("collapse/input", layer, cosine_collapse(x, geom))
        return hook

    def block_hook(layer: int):
        # A decoder block returns (hidden, new_kv)
        def hook(_module, _args, output):
            rec.add("collapse/output", layer, cosine_collapse(output[0], geom))
        return hook

    for layer, (block, attention) in enumerate(zip(model.decoders, attentions)):
        handles.append(attention.register_forward_hook(attention_hook(layer)))
        handles.append(block.register_forward_hook(block_hook(layer)))

    was_training = model.training
    model.eval()
    try:
        model(inputs, build_mask(inputs, pad, causal))
    finally:
        model.train(was_training)
        for handle in handles:
            handle.remove()
    return rec.resolve()


def layer_profile(diagnostics: dict[str, float], tag: str) -> list[float]:
    """Pull one tag's per-layer values back out of a resolved diagnostics dict."""
    prefix = f"{tag}/layer_"
    return [value for key, value in sorted(diagnostics.items()) if key.startswith(prefix)]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def hash_tensor(hasher, tensor: torch.Tensor) -> None:
    """Fold a tensor's exact contents into a digest.

    Normalised to CPU int64 first so the digest is a property of the token ids
    alone -- it must not change when the same batches are preloaded to device or
    pinned, only when the ids themselves do.
    """
    hasher.update(tensor.detach().to("cpu", torch.int64).contiguous().numpy().tobytes())


class BatchSet:
    """A fixed list of (inputs, labels), with its token counts precomputed.

    Fixed, because every variant must see identical data in an identical order --
    the comparison should not include loader jitter. Precomputed, because the
    token counts are constants that the old version recounted on device at every
    evaluation, one `.item()`-equivalent per batch per eval.

    Masks are deliberately not held; see build_mask.
    """

    def __init__(self, items: list[tuple[torch.Tensor, torch.Tensor]], pad: int) -> None:
        self.items = items
        self.label_tokens = [int((labels != -100).sum()) for _, labels in items]
        self.input_tokens = [int((inputs != pad).sum()) for inputs, _ in items]
        self.total_labels = sum(self.label_tokens)
        self.total_inputs = sum(self.input_tokens)

    def __len__(self) -> int:
        return len(self.items)

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for item in self.items for t in item)

    def digest(self) -> str:
        """Order-sensitive digest of every token in the set.

        Deliberately NOT cached. A cached digest would certify the batches as
        they were built; recomputing it certifies the batches as they are now,
        which is the only version that can catch an in-place write.
        """
        hasher = hashlib.blake2b(digest_size=16)
        for index, (inputs, labels) in enumerate(self.items):
            hasher.update(index.to_bytes(4, "little"))
            hash_tensor(hasher, inputs)
            hash_tensor(hasher, labels)
        return hasher.hexdigest()

    def to(self, device: torch.device) -> "BatchSet":
        self.items = [tuple(t.to(device, non_blocking=True) for t in item) for item in self.items]
        return self

    def pin(self) -> "BatchSet":
        self.items = [tuple(t.pin_memory() for t in item) for item in self.items]
        return self


def materialise(dataset: TextStreamDataset, batch_size: int, sample_cap: int,
                batch_cap: int | None, seed: int, pad: int, desc: str) -> BatchSet:
    """Draw a fixed, reproducible list of batches from a capped index pool.

    TextStreamDataset memory-maps line offsets and reads each sample lazily, so
    capping is a matter of sampling indices rather than of how much gets loaded.
    (Its first use over a new corpus builds a cached .index/.meta.json; that
    one-time pass is over the whole file, every run after it is free.)

    `batch_cap` stops materialisation early. The schedule cycles the batch list
    (see DataPlan), so any batch past `steps` is tokenised and then never touched
    -- at the default caps that is a thousand samples of startup cost for
    nothing. Capping cannot change which data is used, only how much of it is
    built; the epoch count is still computed from the full pool so the
    overfitting warning keeps its meaning.
    """
    limit = min(sample_cap, len(dataset)) if sample_cap else len(dataset)
    sampler = SubsetRandomSampler(range(limit), generator=torch.Generator().manual_seed(seed))
    total = batch_cap if batch_cap is not None else limit // batch_size
    items = []
    loader = dataset.get_loader(batch_size, sampler=sampler)
    for inputs, labels, _ in tqdm(loader, total=total, desc=desc, leave=False):
        items.append((inputs, labels))
        if len(items) >= total:
            break
    return BatchSet(items, pad)


class DataPlan:
    """The exact data every variant sees, and the exact order it sees it in.

    Built once, before the first variant, and handed to all of them unchanged.
    Sharing the batches is not sufficient on its own -- the ORDER has to be
    shared too, so the step -> batch schedule is materialised here as a list
    instead of being re-derived inside each run from a modular index. One object
    owns both, so there is no arithmetic two variants could disagree about.

    `fingerprint` is what turns that from a claim into a guarantee. It is an
    order-sensitive digest of every token a variant will be shown -- the training
    batches in schedule order, the validation set in evaluation order, and the
    diagnostic batch -- and every variant recomputes it and checks it against the
    baseline's before it trains a single step. The batches live in shared, often
    device-resident tensors for the whole comparison, so a stray in-place write
    would quietly hand variant N different data from variant 1, and the result
    would still look like a clean A/B. Recomputing per variant catches that
    instead of assuming it away, and it costs one pass over the batch data.

    What this does NOT pin is the dropout mask. Every variant starts from the
    same seed, but different attention modules draw different numbers of random
    values, so the masks diverge after the first block that differs. The default
    --dropout 0 keeps the comparison strictly like-for-like.
    """

    def __init__(self, train: BatchSet, val: BatchSet, diag: torch.Tensor, steps: int) -> None:
        self.train = train
        self.val = val
        # A clone, not a view into val.items[0]: this tensor goes through every
        # variant's forward pass, and a view would leave the validation set one
        # stray in-place write away from corruption.
        self.diag = diag.clone()
        self.schedule = [step % len(train) for step in range(steps)]

    def fingerprint(self) -> str:
        hasher = hashlib.blake2b(digest_size=16)
        # The training stream is (batches, order); digesting the batches once and
        # the schedule separately identifies it exactly, without re-hashing a
        # cycled batch once per step it is replayed.
        hasher.update(self.train.digest().encode())
        hasher.update(",".join(map(str, self.schedule)).encode())
        hasher.update(self.val.digest().encode())
        hash_tensor(hasher, self.diag)
        return hasher.hexdigest()


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

@torch.no_grad()
def validate(model: torch.nn.Module, batches: BatchSet, pad: int, causal: torch.Tensor,
             amp: bool) -> float:
    """Token-weighted mean cross-entropy over the validation batches.

    Summing per-batch means and dividing by the batch count would weight every
    batch equally regardless of how many real tokens it holds. Documents are
    padded to seq_len, so a batch of short documents contributes far fewer tokens
    than a batch of long ones while counting the same -- which biases the number
    toward whatever the heavily-padded batches happen to say. Reducing by sum and
    dividing by the true token count gives the mean the loss curves imply.

    The running total stays on device; one host read at the end instead of one
    per batch.
    """
    was_training = model.training
    model.eval()
    total = torch.zeros((), dtype=torch.float32, device=DEVICE)
    for inputs, labels in batches.items:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=amp):
            logits = model(inputs, build_mask(inputs, pad, causal))
        total += F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(),
                                 ignore_index=-100, reduction="sum")
    model.train(was_training)
    return total.item() / max(batches.total_labels, 1)


def aulc(curve: list[tuple[int, float, float]], axis: int, cutoff: float | None = None) -> float:
    """Area under the validation loss curve, normalised back to a mean loss.

    `final val` is one measurement at one step, and this script's own val set is
    sized where a 0.05 nat difference is at the edge of resolvable -- so at a few
    hundred steps an arm can take the last column on a draw of the validation
    set. This integrates every evaluation instead, by the trapezoidal rule over
    (x, val_loss), and divides by the span covered. Dividing is what makes it
    readable: the raw area is in nats*steps and compares to nothing, the mean is
    in nats and compares to the loss columns beside it.

    Two things it sees that the final loss cannot:

      * how FAST an arm got there. Two variants converging to the same place are
        a tie on the last point; the one that was lower the whole way has the
        lower area.
      * an unstable run. A curve that spikes and recovers ends wherever it ends
        -- the area carries the excursion.

    `axis` selects the x to integrate over: 0 for steps (learning per unit of
    data), 1 for elapsed seconds (learning per unit of compute). `cutoff`
    truncates there, interpolating the loss at exactly that x. That is what makes
    the seconds axis meaningful at all -- variants reach different elapsed times
    at the same step, so their areas are only comparable over a span all of them
    covered, and report() truncates at the largest such span.

    Deliberately reported ALONGSIDE the final loss and not instead of it. The
    area is weighted toward early training, where the loss is largest and falling
    fastest, so an arm that merely starts better can hold the lower AULC and
    still end up worse. The pair disagreeing is the finding, not a contradiction:
    it says the arms differ in convergence SPEED rather than in where they land.
    """
    points = [(float(point[axis]), point[2]) for point in curve]
    if cutoff is not None:
        within = [point for point in points if point[0] <= cutoff]
        # Interpolated at the cutoff rather than stopped at the last evaluation
        # before it. Evaluations land at a different elapsed time in every
        # variant, so truncating to whichever one happens to fall inside would
        # give each arm a slightly different span -- the exact thing a shared
        # cutoff exists to prevent. The next point is strictly past the cutoff
        # and this one is at or before it, so the gap below is never zero.
        if within and len(within) < len(points):
            (x0, y0), (x1, y1) = within[-1], points[len(within)]
            within.append((cutoff, y0 + (y1 - y0) * (cutoff - x0) / (x1 - x0)))
        points = within or points[:1]
    span = points[-1][0] - points[0][0]
    if span <= 0:
        # One evaluation, or a cutoff before the first: the mean over a single
        # point is that point, which is the right limit rather than a failure.
        return points[-1][1]
    area = sum(0.5 * (y0 + y1) * (x1 - x0)
               for (x0, y0), (x1, y1) in zip(points, points[1:]))
    return area / span


def gradient_norms(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Squared gradient norm per component, bucketed by utils.component_key.

    The buckets, and therefore the Gradients/* tags they become, are train.py's:
    Embedding, Projection, Decoder<i> per block, NormF for the rest. Sharing the
    rule rather than restating it is what keeps a comparison run's per-layer
    gradient series readable against a real training run's.

    Returned as 0-dim device tensors rather than floats. This has to be called
    from inside the timed window -- the snapshot is only meaningful before
    clipping, exactly where train.py takes it -- and an .item() per parameter
    would be one device sync per parameter, tens of stalls per evaluation, inside
    the region whose wall-clock this script exists to report. resolve_gradients
    turns them into numbers later, with the clock stopped.
    """
    totals: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        key = component_key(name)
        norm_sq = torch.linalg.vector_norm(param.grad.detach().float().view(-1)).square()
        totals[key] = totals[key] + norm_sq if key in totals else norm_sq
    return totals


def resolve_gradients(totals: dict[str, torch.Tensor]) -> dict[str, float]:
    """Squared sums -> Gradients/* scalars, in a single host transfer.

    Gradients/Global is the root of the summed squares, which is both what
    train.py reports and what clip_grad_norm_ measures against --grad-clip, so
    the components always add up to the number the clipping acted on.
    """
    if not totals:
        return {}
    keys = list(totals)
    values = torch.stack([totals[key] for key in keys]).cpu()      # the one and only sync
    resolved = {f"Gradients/{key}": value.sqrt().item() for key, value in zip(keys, values)}
    resolved["Gradients/Global"] = values.sum().sqrt().item()
    return resolved


def build_optimiser(model: torch.nn.Module, args) -> torch.optim.AdamW:
    # fused AdamW keeps the whole update in one kernel; it is available only on
    # CUDA, and applies identically to every variant, so it does not tilt the
    # comparison it speeds up.
    fused = DEVICE.type == "cuda" and args.fused
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                             betas=(args.beta1, args.beta2), fused=fused)


def run(variant: Variant, args, data: DataPlan, fingerprint: str,
        pad: int, causal: torch.Tensor, run_dir: str) -> dict:
    label, overrides = variant.label, variant.overrides
    # Checked BEFORE anything is built, so a variant that would train on data the
    # baseline never saw fails immediately rather than after producing a full set
    # of plausible-looking curves that are not comparable to anything.
    observed = data.fingerprint()
    if observed != fingerprint:
        raise RuntimeError(
            f"variant '{label}' would not see the same data as the baseline: data fingerprint "
            f"{observed} != {fingerprint}. The shared batch tensors were written to in place.")
    train, val = data.train, data.val

    # One writer per variant, as sibling directories under a shared parent, with
    # IDENTICAL tag names. That is what puts the variants on the same chart:
    # TensorBoard overlays equal tags across runs and colours them by run. Using
    # add_scalars() with a variant-keyed dict would instead create one nested
    # pseudo-run per variant per tag, which fragments the run list and loses the
    # per-run smoothing and visibility toggles.
    writer = SummaryWriter(os.path.join(run_dir, label.replace("/", "-")))
    seed_everything(args.seed)

    config = ModelConfig(**{**args.base_config, **overrides})
    # build() when the class offers one -- it owns weight tying, the init scheme and
    # LoRA, none of which this script should reimplement per architecture -- and the
    # plain constructor otherwise. The isinstance check catches the one way a build()
    # can lie: a staticmethod that hardcodes its own class, which would hand every
    # subclass arm a silently identical baseline model and a comparison of nothing.
    builder = getattr(variant.cls, "build", None)
    model = builder(config) if callable(builder) else variant.cls(config)
    if not isinstance(model, variant.cls):
        raise RuntimeError(
            f"variant '{label}': {variant.model}.build() returned a "
            f"{type(model).__name__}, not a {variant.cls.__name__}. A build() that "
            "hardcodes its class instead of using cls cannot be used to compare "
            "architectures -- make it a classmethod.")
    model = model.to(DEVICE)
    model.train()
    # __init__ may resolve derived config fields (a rank left at 0, say), so the
    # config that gets reported is read back off the model, not off the overrides
    # that were asked for.
    config = getattr(model, "config", config)

    # Only the training step runs through the compiled wrapper. The diagnostics
    # keep using the eager module: they install forward hooks on the decoder
    # blocks, which on a compiled module cause graph breaks (or silently never
    # fire), and they sit outside the timed region anyway, so there is nothing to
    # gain and correctness to lose. Both share parameters.
    step_model = model
    if args.compile:
        if DEVICE.type != "cuda":
            LOGGER.warning(f"--compile is enabled but DEVICE is '{DEVICE.type}', not 'cuda' -- "
                           "inductor's Triton backend is far less mature off CUDA; expect "
                           "possible failures or no speedup.")
        step_model = torch.compile(model, mode=args.compile_mode)

    # Resolved once, here rather than inside diagnose(), so a model whose blocks this
    # script cannot read fails before it trains for ten minutes and reports nothing.
    attentions = attention_modules(model, label)

    unique = {id(p): p for p in model.parameters()}
    total_params = sum(p.numel() for p in unique.values())
    # By identity, not by name: two architectures agree on neither the attribute path
    # nor the module type, and deduplicating on id() keeps a tied or shared weight
    # from being counted twice.
    attn_ids = {id(p) for attention in attentions for p in attention.parameters()}
    attn_params = sum(p.numel() for pid, p in unique.items() if pid in attn_ids)

    optimiser = build_optimiser(model, args)
    scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=args.amp)
    # linear warmup then cosine decay, hand-rolled so it stays well-defined at
    # the tiny step counts a smoke run uses
    warmup = max(1, int(args.warmup_frac * args.steps))

    def lr_scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_scale)

    def forward_backward(batch: tuple) -> torch.Tensor:
        inputs = batch[0].to(DEVICE, non_blocking=True)
        labels = batch[1].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=args.amp):
            logits = step_model(inputs, build_mask(inputs, pad, causal))
            loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), ignore_index=-100)
        scaler.scale(loss).backward()
        return loss.detach()

    # Warm up kernels, the caching allocator, cuDNN autotuning and -- when
    # --compile is set -- the whole inductor compile, before the clock starts.
    # Otherwise the first timed steps bill one variant for setup the others
    # already paid, and compilation (tens of seconds) would swamp the
    # comparison entirely. No optimiser step, so the weights are untouched.
    # Every batch is the same shape (drop_last=True), so this compiles once.
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
    warmup_started = time.perf_counter()
    for _ in range(min(args.warmup_steps, len(train))):
        forward_backward(train.items[0])
    model.zero_grad(set_to_none=True)
    sync()
    warmup_sec = time.perf_counter() - warmup_started
    if args.compile:
        LOGGER.info(f"  {label}: compile + warmup took {warmup_sec:.1f}s (excluded from timings)")

    initial = diagnose(model, attentions, data.diag, pad, causal)
    curve: list[tuple[int, float, float]] = []
    elapsed, window_start = 0.0, None
    loss_sum = torch.zeros((), dtype=torch.float32, device=DEVICE)
    loss_count, grad_totals = 0, {}
    gradients: dict[str, float] = {}
    gap = float("nan")
    diag_index = 0

    # walltime is anchored so that every variant's first point sits at the same
    # instant; TensorBoard's RELATIVE x-axis then reads directly as seconds of
    # training compute, giving loss-vs-wall-clock overlaid for free. Eval time is
    # excluded, so the axis is honest about compute rather than about this script.
    anchor = time.time()

    # tqdm advances on the generator's next(), i.e. before the body runs, so the
    # bar's own work lands outside the timed window below and does not inflate
    # the wall-clock figures this script exists to compare.
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
    progress = tqdm(range(1, args.steps + 1), total=args.steps,
                    desc=f"\033[95m{stamp}\033[0m - \033[94mINFO\033[0m - "
                         f"\033[96m{LOGGER.name}\033[0m - \033[93m{label}")

    for step in progress:
        # The clock runs per evaluation window rather than per step. A device
        # sync around every step would serialise the CPU against the GPU roughly
        # `steps` times, which both slows the run down and inflates the very
        # number being measured; one sync per window costs the same accuracy at
        # a fraction of the stalls. Evaluation happens with the clock stopped, so
        # `elapsed` stays pure training compute either way.
        if window_start is None:
            sync()
            window_start = time.perf_counter()
        evaluating = step % args.eval_every == 0 or step == args.steps

        optimiser.zero_grad(set_to_none=True)
        loss = forward_backward(train.items[data.schedule[step - 1]])
        if args.amp:
            scaler.unscale_(optimiser)
        # Snapshotted where train.py snapshots: after unscale_, so the numbers are
        # true gradients rather than loss-scaled ones, and before clipping, so they
        # describe the gradient the step produced rather than the one --grad-clip
        # allowed through. Only on steps that will be logged -- train.py pays this
        # every 100 steps, this pays it once per evaluation.
        if evaluating:
            grad_totals = gradient_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        # Accumulated on device: reading the loss every step would reintroduce
        # exactly the per-step sync the windowed clock above exists to avoid.
        loss_sum += loss
        loss_count += 1

        if evaluating:
            sync()
            elapsed += time.perf_counter() - window_start
            window_start = None

            val_loss = validate(model, val, pad, causal, args.amp)
            train_loss = (loss_sum / max(loss_count, 1)).item()
            loss_sum.zero_()
            loss_count = 0
            gap = val_loss - train_loss
            curve.append((step, elapsed, val_loss))
            walltime = anchor + elapsed

            writer.add_scalar("loss/val", val_loss, step, walltime=walltime)
            writer.add_scalar("loss/train", train_loss, step, walltime=walltime)
            writer.add_scalar("loss/gap", gap, step, walltime=walltime)
            # Running AULC over steps: at each evaluation, the mean val loss of
            # the run SO FAR. Charted rather than only summarised because the
            # step it crosses another variant's trace is the step that arm's
            # lead actually began, which a single end-of-run scalar cannot say.
            # Cheap, and the clock is stopped here (see the window above).
            writer.add_scalar("loss/aulc", aulc(curve, 0), step, walltime=walltime)
            writer.add_scalar("optim/lr", scheduler.get_last_lr()[0], step, walltime=walltime)
            # Gradients/Global + Gradients/<component>, the tags and the decomposition
            # train.py writes, so a variant's per-layer gradient trace can be read
            # against a real training run's without translating anything.
            gradients = resolve_gradients(grad_totals)
            for tag, value in gradients.items():
                writer.add_scalar(tag, value, step, walltime=walltime)
            writer.add_scalar("perf/elapsed_sec", elapsed, step, walltime=walltime)
            writer.add_scalar("perf/ms_per_step", elapsed / step * 1e3, step, walltime=walltime)
            writer.add_scalar("perf/tokens_per_sec",
                              train.total_inputs / len(train) * step / max(elapsed, 1e-9),
                              step, walltime=walltime)

            collapse = None
            # Tracked over training, not just start/end: representation collapse
            # is a trajectory, and the depth profile is the thing to compare.
            if diag_index % args.diag_every == 0 or step == args.steps:
                diagnostics = diagnose(model, attentions, data.diag, pad, causal)
                for tag, value in diagnostics.items():
                    writer.add_scalar(tag, value, step, walltime=walltime)
                collapse = diagnostics.get("collapse/output/mean")
            diag_index += 1

            postfix = {"train": f"{train_loss:6.3f}", "val": f"{val_loss:6.3f}",
                       "sec": f"{elapsed:6.1f}"}
            if collapse is not None:
                postfix["collapse"] = f"{collapse:5.3f}"
            progress.set_postfix(postfix)

    progress.close()

    final = diagnose(model, attentions, data.diag, pad, causal)
    peak_mb = (torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2) if DEVICE.type == "cuda" else 0.0
    # Over steps, which every arm shares by construction, so this one is
    # comparable as it stands. The seconds-axis area is NOT computed here: it is
    # only meaningful against a budget every variant reached, and that is not
    # known until they have all run, so report() derives it from `curve`.
    aulc_steps = aulc(curve, 0)

    # run_name="." keeps the hparams in this run's own directory; the default
    # would nest a fresh timestamped run underneath and split the variant in two.
    # The overrides go in as one JSON string rather than as columns. Variants may
    # set disjoint fields, and a column per field would leave the hparams table
    # mostly blank and would need this script to know which fields exist.
    writer.add_hparams(
        {"variant": label, "model": variant.model,
         "overrides": json.dumps(overrides, sort_keys=True),
         "attn_params": attn_params, "total_params": total_params,
         "embed_dim": config.embed_dim, "heads": config.heads,
         "n_decoders": config.n_decoders, "seq_len": config.seq_len, "lr": args.lr,
         "compile": args.compile_mode if args.compile else "off"},
        {"hparam/final_val": curve[-1][2], "hparam/aulc": aulc_steps,
         "hparam/gap": gap,
         "hparam/sec": elapsed, "hparam/warmup_sec": warmup_sec,
         "hparam/collapse_mean": final.get("collapse/output/mean", float("nan"))},
        run_name=".")
    writer.close()

    return {
        "label": label,
        "model": variant.model,
        "overrides": overrides,
        "config": config.to_dict(),
        "data_fingerprint": observed,
        "attn_params": attn_params,
        "total_params": total_params,
        "curve": curve,
        "aulc": aulc_steps,
        "gap": gap,
        "sec": elapsed,
        "warmup_sec": warmup_sec,
        "peak_mb": peak_mb,
        "tokens_per_sec": train.total_inputs / len(train) * args.steps / max(elapsed, 1e-9),
        "collapse_start": layer_profile(initial, "collapse/output"),
        "collapse_end": layer_profile(final, "collapse/output"),
        "diagnostics": final,
        "gradients": gradients,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def report(results: dict[str, dict], args, run_dir: str, fingerprint: str) -> None:
    # Both name columns are left-aligned, so each carries its own two-space gutter;
    # the numeric columns that follow are right-aligned and bring their own.
    width = max(max((len(label) for label in results), default=7), 7) + 2
    # The class each arm actually trained gets a column of its own, always -- also
    # when every arm shares one. A table of numbers that does not say what was
    # trained is a table that gets pasted somewhere and misread later.
    model_width = max(max((len(r["model"]) for r in results.values()), default=5), 5) + 2
    # Summed from the numeric column widths below rather than restated at each
    # separator: the three copies of this arithmetic had drifted 12 characters
    # short of the row they were meant to underline.
    columns = width + model_width + 13 + 12 + 11 + 9 + 12 + 9 + 9 + 14 + 8 + 8 + 10 + 10
    rule = "=" * columns
    base = args.base_config

    print("\n" + rule)
    print(f"Short-run comparison: {args.steps} steps, embed_dim={base['embed_dim']}, "
          f"heads={base['heads']}, n_decoders={base['n_decoders']}, seq_len={base['seq_len']}")
    print(rule)

    # The first variant is the reference. Absolute val loss at a few hundred
    # steps says almost nothing on its own; what the run can actually resolve is
    # the DIFFERENCE against a known-good baseline trained on the same batches
    # from the same seed, so that is what gets a column.
    baseline = results[next(iter(results))]
    # The wall-clock the SLOWEST variant needed is the only budget every variant
    # actually reached, so an iso-compute comparison has to be made there.
    budget = min(r["curve"][-1][1] for r in results.values())
    precision = 0 if budget >= 10 else 1

    print(f"\n{'variant':{width}s}{'model':{model_width}s}{'attn params':>13}{'params':>12}"
          f"{'final val':>11}{'d base':>9}{'val @ %.*fs' % (precision, budget):>12}"
          f"{'gap':>9}{'aulc':>9}{'aulc @ %.*fs' % (precision, budget):>14}{'sec':>8}"
          f"{'x base':>8}{'tok/s':>10}{'peak MB':>10}")
    print("-" * columns)
    for label, r in results.items():
        within = [point for point in r["curve"] if point[1] <= budget] or r["curve"][:1]
        delta = r["curve"][-1][2] - baseline["curve"][-1][2]
        speed = baseline["sec"] / max(r["sec"], 1e-9)
        print(f"{label:{width}s}{r['model']:{model_width}s}{r['attn_params']:>13}"
              f"{r['total_params']:>12}"
              f"{r['curve'][-1][2]:>11.4f}{delta:>+9.4f}{within[-1][2]:>12.4f}"
              f"{r['gap']:>9.4f}{r['aulc']:>9.4f}{aulc(r['curve'], 1, budget):>14.4f}"
              f"{r['sec']:>8.1f}{speed:>8.2f}{r['tokens_per_sec']:>10.0f}{r['peak_mb']:>10.0f}")
    print(f"\n  d base < 0 is better than {baseline['label']}; x base > 1 is faster than it.")
    print("  aulc is the mean val loss over the whole run, i.e. the area under the loss curve:")
    print("  lower means an arm was ahead THROUGHOUT, not only at the evaluation that happened")
    print(f"  to be last. 'aulc @' integrates against seconds rather than steps, cut at "
          f"{budget:.{precision}f}s so")
    print("  every arm is scored over a span it reached -- learning per unit of compute.")
    print("  Read both WITH final val, not instead of it: the area is weighted toward early")
    print("  training, so disagreement means the arms differ in convergence speed, not endpoint.")
    print("  gap is final val minus final train loss. Wider than the baseline's is an arm buying")
    print("  its win by fitting the batch list harder rather than by learning -- check it against")
    print("  'params' before crediting the architecture. The train side is a window mean taken in")
    print("  training mode, so compare gaps BETWEEN arms and watch loss/gap move; the absolute")
    print("  value reads low, and early in a run it can be negative.")
    print(f"  all variants verified against data fingerprint {fingerprint}: same batches,")
    print(f"  same order, same validation set.")
    if len({r["model"] for r in results.values()}) > 1:
        print("  the arms are different model CLASSES: the header describes the shared config")
        print("  base only, and 'params' is the capacity axis to read 'd base' against -- an")
        print("  architecture that wins while holding more parameters has not won yet.")

    print("\nper-layer token similarity after each block (mean pairwise cosine; -> 1 means collapsed)")
    print("-" * columns)
    for label, r in results.items():
        start = " ".join(f"{value:5.2f}" for value in r["collapse_start"])
        end = " ".join(f"{value:5.2f}" for value in r["collapse_end"])
        print(f"{label:{width}s} init [{start} ]   trained [{end} ]")

    print(rule)
    print(f"\ntensorboard --logdir {run_dir}")
    print("  loss/*, collapse/*, attn/*, Gradients/* and perf/* carry the same tag in every run,")
    print("  so each chart overlays them all. Switch the x-axis to RELATIVE for loss against")
    print("  seconds of training compute rather than steps.")

    summary = os.path.join(run_dir, "summary.json")
    with open(summary, "w", encoding="utf-8") as handle:
        # `variants` holds resolved classes; it is re-emitted below as the strings it
        # was parsed from, which is what another run can be reproduced from.
        json.dump({"env": ENV,
                   "args": {k: v for k, v in vars(args).items()
                            if k not in ("base_config", "variants")},
                   "base_config": args.base_config,
                   "variants": [{"label": v.label, "model": v.model, "overrides": v.overrides}
                                for v in args.variants],
                   "data_fingerprint": fingerprint,
                   "results": results}, handle, indent=2, default=str)
    print(f"\nmachine-readable summary: {summary}")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    template = ModelConfig()

    parser.add_argument("--tokenizer", default="tokenizers/amharic-bpe-tokenizer-25k.model")
    parser.add_argument("--training-data", default="data/pretraining/train.jsonl")
    parser.add_argument("--validation-data", default="data/pretraining/val.jsonl")
    parser.add_argument("--variant", action="append", metavar="NAME:key=value,...",
                        help="A variant to train, as a name plus ModelConfig overrides, e.g. "
                             "'wide:heads=16'. Repeatable, and the FIRST one is the baseline the "
                             "rest are reported against. Any ModelConfig field is settable, so an "
                             "ablation needs no code change; 'model=<module.Class>' additionally "
                             "trains a DIFFERENT architecture in that arm, e.g. "
                             "'flat:model=model2.GPTWide'. With none given the run trains only "
                             f"the baseline (default: {'; '.join(DEFAULT_VARIANTS)})")
    parser.add_argument("--model", default=DEFAULT_MODEL, metavar="module.Class",
                        help="Model class for variants that do not name one themselves. Any "
                             "importable nn.Module subclass taking a ModelConfig; a bare name "
                             f"is looked up in model.py (default: {DEFAULT_MODEL})")
    parser.add_argument("--config", default="", metavar="key=value,...",
                        help="ModelConfig overrides applied to EVERY variant, on top of the "
                             "flags below and underneath each variant's own overrides. Accepts "
                             "'model=' too, as an alias for --model")

    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Cap on training samples drawn from the corpus (0 = all)")
    parser.add_argument("--val-samples", type=int, default=512,
                        help="Validation set size in SAMPLES. Sizing it in batches "
                             "instead would silently rescale the val set whenever "
                             "--batch-size changed, so runs at different batch sizes "
                             "would not be measuring the same thing (default: 512)")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers used while materialising the fixed batch "
                             "lists. Only affects startup; training reads from memory")

    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--n-decoders", type=int, default=3)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-frac", type=float, default=0.1,
                        help="Fraction of --steps spent in linear LR warmup (default: 0.1)")
    parser.add_argument("--warmup-steps", type=int, default=3,
                        help="Untimed forward/backward passes before the clock starts, to pay "
                             "for kernel autotuning and compilation once (default: 3)")
    parser.add_argument("--seed", type=int, default=4321)

    parser.add_argument("--diag-samples", type=int, default=8,
                        help="Sequences used for the per-layer diagnostics, taken from the head "
                             "of the first validation batch and so capped at --batch-size. Bounds "
                             "diagnostic memory independently of the training batch (default: 8)")
    parser.add_argument("--diag-every", type=int, default=1,
                        help="Run the per-layer diagnostics every Nth evaluation. The final "
                             "step is always diagnosed (default: 1)")

    parser.add_argument("--tb-log-dir", type=str, default="logs",
                        help="TensorBoard root; each comparison gets a timestamped "
                             "subdirectory holding one run per variant")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this comparison (default: attn-compare-<timestamp>)")

    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                        help="Compile each variant with torch.compile. Compilation happens "
                             "during warmup and is excluded from the reported timings "
                             "(default: disabled)")
    parser.add_argument("--compile-mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune",
                                 "max-autotune-no-cudagraphs"],
                        help="torch.compile mode when --compile is set (default: 'default')")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="Mixed precision (default: on when the device supports it, "
                             "matching train.py)")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow TF32 matmuls on CUDA. Applies to every variant equally "
                             "(default: enabled)")
    parser.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True,
                        help="Fused AdamW on CUDA (default: enabled)")
    parser.add_argument("--preload", choices=["auto", "on", "off"], default="auto",
                        help="Hold the fixed batch lists in device memory, removing the "
                             "host-to-device copy from every timed step. 'auto' preloads when "
                             "they fit under --preload-limit-mb (default: auto)")
    parser.add_argument("--preload-limit-mb", type=int, default=2048)

    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    args.eval_every = max(1, min(args.eval_every, args.steps))
    if args.amp is None:
        args.amp = MIXED_PRECISION_ENABLED
    if args.amp and not MIXED_PRECISION_ENABLED:
        parser.error(f"--amp requested but autocast is unavailable on device '{DEVICE.type}'")

    # Precedence: flag defaults < --config < per-variant overrides. The flags
    # cover the fields a sweep changes constantly; --config reaches the rest.
    args.base_config = dict(embed_dim=args.embed_dim, heads=args.heads,
                            n_decoders=args.n_decoders, ff_dim=args.ff_dim,
                            seq_len=args.seq_len, dropout=args.dropout)
    try:
        shared, shared_model = parse_overrides(args.config, template)
        args.base_config.update(shared)
        # --config model=... is the same knob as --model, so an explicit --model wins
        # and otherwise either spelling sets the default the variants inherit.
        args.model = args.model if args.model != DEFAULT_MODEL else (shared_model or args.model)
        args.variants = [parse_variant(v, template, args.model)
                         for v in (args.variant or DEFAULT_VARIANTS)]
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))

    names = [variant.label for variant in args.variants]
    if len(set(names)) != len(names):
        parser.error(f"duplicate variant names: {names}")
    return args


def main() -> None:
    args = parse_args()

    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.LoadFromFile(args.tokenizer)
    args.base_config["vocab_size"] = tokenizer.vocab_size()

    train_data = TextStreamDataset(args.training_data, tokenizer, args.seq_len, args.workers)
    val_data = TextStreamDataset(args.validation_data, tokenizer, args.seq_len, args.workers)
    pad = train_data.pad_token

    pool = (min(args.max_samples, len(train_data)) if args.max_samples else len(train_data)) // args.batch_size
    if pool < 1:
        raise SystemExit(f"--max-samples {args.max_samples} yields no full batch at "
                         f"--batch-size {args.batch_size}")
    train = materialise(train_data, args.batch_size, args.max_samples, min(pool, args.steps),
                        args.seed, pad, "materialising train batches")
    val = materialise(val_data, args.batch_size, args.val_samples,
                      max(1, math.ceil(args.val_samples / args.batch_size)),
                      args.seed, pad, "materialising val batches")
    if not len(train) or not len(val):
        raise SystemExit("no batches materialised -- check --training-data / --validation-data")

    # Both lists are fixed and reused every step of every variant, so paying the
    # host-to-device copy once beats paying it steps * variants times. When they
    # do not fit, page-locked memory at least lets the per-step copy overlap.
    footprint = train.bytes() + val.bytes()
    preload = (args.preload == "on"
               or (args.preload == "auto" and footprint <= args.preload_limit_mb * 1024 ** 2))
    if DEVICE.type == "cuda":
        for batches in (train, val):
            batches.to(DEVICE) if preload else batches.pin()

    causal = get_causal_mask(args.seq_len).unsqueeze(0).to(DEVICE)   # (1, 1, SEQ_LEN, SEQ_LEN)
    data = DataPlan(train, val, val.items[0][0][:args.diag_samples].to(DEVICE, non_blocking=True),
                    args.steps)
    fingerprint = data.fingerprint()

    # steps * batch_size presentations drawn from a fixed pool of batches, so the
    # data is cycled. This ratio decides whether the run measures generalisation
    # or memorisation, and it is easy to set up by accident: --steps 4000 at
    # --batch-size 32 against --max-samples 10000 is 12.8 passes over the corpus,
    # deep enough into the overfitting regime that the higher-capacity variant
    # loses on val while matching on train.
    epochs = args.steps / pool
    LOGGER.info(f"{len(train)} train batches materialised of {pool} available "
                f"({args.max_samples or 'all'} sample cap, corpus has {len(train_data)}), "
                f"{len(val)} val batches, vocab {args.base_config['vocab_size']}")
    LOGGER.info(f"validation: {len(val) * args.batch_size} samples, {val.total_labels} scored "
                f"tokens -- differences much below ~0.05 nats are not resolvable at this size")
    LOGGER.info(f"{args.steps} steps x batch {args.batch_size} = {args.steps * args.batch_size} "
                f"presentations over {pool * args.batch_size} samples -> {epochs:.1f} epochs")
    if epochs > 3:
        LOGGER.warning(f"{epochs:.1f} passes over the same data -- every variant will "
                       "overfit, which flatters whichever has less effective capacity. "
                       "Raise --max-samples or lower --steps to compare generalisation, "
                       "or watch loss/gap to see which arm gets there first.")
    LOGGER.info(f"device {DEVICE} "
                f"({torch.cuda.get_device_name(DEVICE) if DEVICE.type == 'cuda' else 'cpu'}), "
                f"mixed precision {'on' if args.amp else 'off'}, "
                f"compile {args.compile_mode if args.compile else 'off'}, "
                f"batches {'preloaded' if preload else 'streamed'} ({footprint / 1024 ** 2:.0f} MiB)")
    described = []
    for variant in args.variants:
        overrides = ", ".join(f"{key}={value}" for key, value in variant.overrides.items())
        described.append(f"{variant.label} = {variant.model} ({overrides or 'no overrides'})")
    LOGGER.info(f"variants: {'; '.join(described)}")
    if len(args.variants) == 1:
        LOGGER.warning("only one variant -- this trains a baseline and compares it to itself. "
                       "Add arms with --variant NAME:field=value or "
                       "--variant NAME:model=module.Class (repeatable); "
                       f"settable fields: {', '.join(sorted(ModelConfig().to_dict()))}")
    LOGGER.info(f"data fingerprint {fingerprint} -- {len(data.schedule)} scheduled steps over "
                f"{len(train)} batches; every variant re-checks this before training")

    run_dir = os.path.join(args.tb_log_dir,
                           args.run_name or f"attn-compare-{datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)

    results = {}
    for variant in args.variants:
        results[variant.label] = run(variant, args, data, fingerprint, pad, causal, run_dir)

    report(results, args, run_dir, fingerprint)


if __name__ == "__main__":
    main()
