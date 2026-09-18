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
  * per-layer gradient norms      -- param/grad/global and param/grad/<component>,
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
from torch.utils.flop_counter import FlopCounterMode
from torch.utils.tensorboard import SummaryWriter

from config import DEVICE, ENV, LOGGER, MIXED_PRECISION_ENABLED, ModelConfig
from dataset import TextStreamDataset
import probes
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

# The same, for the feed-forward submodule. Deliberately no bare "ff": at two
# characters it matches "offset" and "buffer" as readily as a branch name, and a
# spurious second match fails the block rather than mislabelling it.
FEEDFORWARD_HINTS = ("feed_forward", "feedforward", "ffn", "mlp")


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


def decoder_blocks(model: torch.nn.Module, label: str) -> list[torch.nn.Module]:
    blocks = getattr(model, "decoders", None)
    if blocks is None or not len(blocks):
        raise RuntimeError(
            f"variant '{label}': {type(model).__name__} has no non-empty `decoders`. The "
            "per-layer diagnostics are the point of this script, so a model whose blocks "
            "cannot be enumerated cannot be compared here.")
    return list(blocks)


def sublayer_modules(model: torch.nn.Module, label: str,
                     hints: tuple[str, ...], what: str) -> list[torch.nn.Module]:
    """One named submodule per decoder block, found by attribute name.

    Two architectures being compared are under no obligation to agree on that
    name, and hardcoding whichever one they currently share would turn a rename
    in either into an empty attn/* or ffn/* series here -- which does not look
    like a bug, it looks like a real difference between the arms. Matching on
    hints covers the plausible names, and requiring exactly one match per block
    means a model this heuristic cannot read fails loudly instead of silently.

    Shared by both sublayers rather than written twice, so the two families of
    tags cannot drift apart in which models they can read.
    """
    found = []
    for layer, block in enumerate(decoder_blocks(model, label)):
        children = list(block.named_children())
        matches = [child for name, child in children
                   if any(hint in name.lower() for hint in hints)]
        if len(matches) != 1:
            names = ", ".join(name for name, _ in children) or "no children"
            raise RuntimeError(
                f"variant '{label}': block {layer} of {type(model).__name__} has "
                f"{len(matches)} submodules whose name looks like {what} "
                f"({names}). The diagnostics need exactly one; rename it to contain "
                f"one of {hints}, or drop the extra match.")
        found.append(matches[0])
    return found


def attention_modules(model: torch.nn.Module, label: str) -> list[torch.nn.Module]:
    return sublayer_modules(model, label, ATTENTION_HINTS, "attention")


def feedforward_modules(model: torch.nn.Module, label: str) -> list[torch.nn.Module]:
    return sublayer_modules(model, label, FEEDFORWARD_HINTS, "a feed-forward")


@torch.inference_mode()
def site_of(name: str) -> tuple[str, int]:
    """(role, depth) from a module path, tolerant of DDP/compile name prefixes.

    The role is the leaf attribute -- norm1, norm2, norm_f, Wqkv, Wo, Wug, Wd --
    so one tag follows the same structural position across every block, and the
    depth is the decoder index so the per-layer/mean/min/max split lands on the
    same axis as attn/* and ffn/*. Anything outside a decoder block is filed at
    layer 0. `projection.linear` is renamed for its parent, since `linear` says
    nothing about where it sits.
    """
    parts = name.split(".")
    role = "projection" if parts[-1] == "linear" else parts[-1]
    if "decoders" in parts:
        return role, int(parts[parts.index("decoders") + 1])
    return role, 0


def diagnose_modules(model: torch.nn.Module, inputs: torch.Tensor,
                     pad: int, causal: torch.Tensor) -> dict[str, float]:
    """Everything registered in `probes`, from ONE forward pass.

    Deliberately the opposite of `diagnose` on one axis: these probes DO reach
    inside their modules, read running buffers and recompute internal
    quantities. That is defensible only because the tags are variant-specific by
    construction -- a model without the module emits none of them, so nothing
    here can be mistaken for a number some other variant failed to report. They
    stay out of the attn/* ffn/* collapse/* namespace, which remains
    internals-blind precisely so it can be compared across unrelated classes.

    This function owns the mechanism and none of the meaning: the probe batch,
    the padding mask, the single host transfer, and the <family>/<site>/<metric>
    naming. What each metric IS lives beside the module that registers it, where
    `isinstance` is available and where anyone changing the module is already
    looking. See `probes.register`.
    """
    sites = [(name, module, probe)
             for name, module in model.named_modules()
             for probe in probes.probes_for(module)]
    if not sites:
        return {}                       # arms without probed modules emit nothing, not zeros

    rec = Recorder()
    ctx = probes.ProbeContext(inputs != pad)
    handles = []

    def probe_hook(probe: probes.Probe, module: torch.nn.Module, prefix: str, layer: int):
        def hook(_module, args, output):
            for metric, value in probe.measure(module, args, output, ctx).items():
                rec.add(f"{prefix}/{metric}", layer, value)
        return hook

    for name, module, probe in sites:
        site, layer = site_of(name)
        # per_site=False drops the role segment: one such module per block means
        # the family already names it, and attn/attention/update_ratio says
        # nothing attn/update_ratio does not.
        prefix = f"{probe.family}/{site}" if probe.per_site else probe.family
        handles.append(module.register_forward_hook(probe_hook(probe, module, prefix, layer)))

    was_training = model.training
    model.eval()                        # eval matters: a norm carrying running stats
    try:                                # must not fold the probe batch into them
        with torch.no_grad():
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

    The buckets, and therefore the param/grad/* tags they become, are train.py's:
    Embedding, Projection, Decoder<i> per block, NormF for the rest. Sharing the
    rule rather than restating it is what keeps a comparison run's per-layer
    gradient series readable against a real training run's.

    Returned as 0-dim device tensors rather than floats. This has to be called
    from inside the timed window -- the snapshot is only meaningful before
    clipping, exactly where train.py takes it -- and an .item() per parameter
    would be one device sync per parameter, tens of stalls per evaluation, inside
    the region whose wall-clock this script exists to report. resolve_parameters
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


def weight_norms(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Squared weight norm per component, in the same buckets as the gradients.

    The counterpart to gradient_norms, sharing utils.component_key so the two
    decompositions line up term for term. On its own a weight norm says little;
    paired with the gradient norm it gives the scale-free ratio below, which is
    the number that actually says whether a component is learning.

    Same 0-dim-device-tensor discipline: this runs inside the timed window, and
    an .item() per parameter would be tens of stalls per evaluation.
    """
    totals: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        key = component_key(name)
        norm_sq = torch.linalg.vector_norm(param.detach().float().view(-1)).square()
        totals[key] = totals[key] + norm_sq if key in totals else norm_sq
    return totals


def resolve_parameters(grads: dict[str, torch.Tensor], weights: dict[str, torch.Tensor],
                       lr: float) -> dict[str, float]:
    """Gradient, weight and update-ratio scalars, in ONE host transfer.

    Emits three families off one sync:

      param/grad/<component>    ||grad||, and param/grad/global over everything.
                                Global is the root of the summed squares, which is
                                what clip_grad_norm_ measures against --grad-clip,
                                so the components always add up to the number the
                                clipping acted on.
      param/weight/<component>  ||weight||, same buckets, plus param/weight/global.
      param/ratio/<component>   lr * ||grad|| / ||weight||: the fraction of its own
                                magnitude a component moves in one step. Roughly
                                1e-3 is healthy. Orders of magnitude below means a
                                component has stopped learning while the loss curve
                                keeps falling on the strength of the others; orders
                                above is the component about to destabilise the run.
                                Scale-free, so it is comparable BETWEEN components
                                and between variants, which neither norm is.

    Grouped under param/ rather than as three top-level families because
    TensorBoard sorts alphabetically, and the ratio is unreadable without the two
    norms it is formed from on the same screen.
    """
    if not grads and not weights:
        return {}

    grad_keys, weight_keys = list(grads), list(weights)
    stacked = torch.stack([grads[k] for k in grad_keys] + [weights[k] for k in weight_keys])
    values = stacked.sqrt().cpu().tolist()                      # the one and only sync

    grad_norm = dict(zip(grad_keys, values[:len(grad_keys)]))
    weight_norm = dict(zip(weight_keys, values[len(grad_keys):]))

    resolved = {f"param/grad/{k}": v for k, v in grad_norm.items()}
    resolved["param/grad/global"] = sum(v * v for v in grad_norm.values()) ** 0.5
    resolved.update({f"param/weight/{k}": v for k, v in weight_norm.items()})
    resolved["param/weight/global"] = sum(v * v for v in weight_norm.values()) ** 0.5
    for key, grad in grad_norm.items():
        weight = weight_norm.get(key)
        if weight:
            resolved[f"param/ratio/{key}"] = lr * grad / weight
    return resolved


def hparams(variant, config: ModelConfig, args) -> dict:
    """The variant's settings, flattened to what add_hparams accepts.

    Only bool/int/float/str survive the HPARAMS tab, so anything structured is
    rendered to a string rather than dropped -- a row missing the field that
    distinguishes it from its neighbour is worse than a stringified one.

    Every ModelConfig field is included, not just the overridden ones: TensorBoard
    only lets you filter on a column that exists, and which field mattered is the
    thing the sweep is trying to find out.
    """
    flat = {"variant": variant.label, "model": variant.model}
    for key, value in config.to_dict().items():
        flat[key] = value if isinstance(value, (bool, int, float, str)) else str(value)
    for key in ("lr", "weight_decay", "grad_clip", "steps", "batch_size", "seq_len", "amp"):
        if hasattr(args, key):
            flat[key] = getattr(args, key)
    return flat


def build_optimiser(model: torch.nn.Module, args) -> torch.optim.AdamW:
    # fused AdamW keeps the whole update in one kernel; it is available only on
    # CUDA, and applies identically to every variant, so it does not tilt the
    # comparison it speeds up.
    fused = DEVICE.type == "cuda" and args.fused
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                             betas=(args.beta1, args.beta2), fused=fused)


def fenced_json(payload: dict) -> str:
    """A payload in the fenced block TensorBoard's TEXT tab renders as code.

    default=str so a field json cannot represent -- a dtype, a device -- degrades
    to its repr rather than taking the run down at step 0, before a single batch
    has been seen.
    """
    return f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"


def parameter_census(model: torch.nn.Module) -> dict:
    """Where a variant's parameters actually sit, bucketed by component_key.

    The rule the param/grad/* tags already use, so a component's share of the
    weights reads directly against its share of the gradient norm. That pairing
    is the point: a block holding 30% of the parameters and 3% of the gradient
    has stopped learning, and neither number says so on its own.

    Counted through named_parameters(), which yields each shared tensor once
    under its first name -- so a tied embedding is counted once, matching both
    total_params in run() and the deduplication in gradient_norms.

    non_embedding is called out separately because it is usually the honest
    capacity axis, and it nets out BOTH vocabulary-sized tables -- the input
    embedding and the output projection. Each is vocab_size * embed_dim whatever
    the blocks are doing, identical across arms that share a tokeniser, and
    easily large enough to dilute a real difference in the blocks into a rounding
    error in the total. Subtracting only the input side would leave the output
    head counted as block capacity whenever the two are untied, so the figure
    would quietly mean different things depending on a config flag -- which is
    the one thing a comparison axis must never do.
    """
    census: dict[str, int] = {}
    footprint = 0
    for name, param in model.named_parameters():
        census[component_key(name)] = census.get(component_key(name), 0) + param.numel()
        footprint += param.numel() * param.element_size()
    total = sum(census.values())
    vocabulary = census.get("Embedding", 0) + census.get("Projection", 0)
    # named_parameters() deduplicates by default; the gap against the
    # undeduplicated walk is the number of parameter slots that alias another,
    # which is how weight tying shows up without asking the model about it. At 0
    # the two tables above are genuinely separate and both were counted.
    aliased = (sum(1 for _ in model.named_parameters(remove_duplicate=False))
               - sum(1 for _ in model.named_parameters()))
    return {"total": total,
            "non_embedding": total - vocabulary,
            "vocabulary": vocabulary,
            "bytes": footprint,
            "aliased_tensors": aliased,
            "by_component": dict(sorted(census.items()))}


def flop_census(model: torch.nn.Module, inputs: torch.Tensor, mask: torch.Tensor) -> dict:
    """FLOPs for one training step, measured by dispatch rather than derived.

    Counted through torch's FlopCounterMode, so this knows nothing about the
    architecture in front of it -- the same reason the activation diagnostics
    read off hooks rather than reaching inside. An arm with a different block
    structure is counted correctly with no edit here, which a hand-rolled
    2 * in * out formula per layer type could not promise.

    Normalised per POSITION, not per real token. FLOPs are spent on padding just
    the same, so dividing by the non-pad count would credit an arm with a
    throughput it did not achieve. That makes this the one place in the script
    where the denominator is deliberately not the one validate() uses.

    attention_counted is part of the result rather than a footnote. FlopCounterMode
    has formulas registered for the fused CUDA attention kernels but not for every
    backend SDPA can dispatch to -- the CPU one has none -- and where there is no
    formula the op contributes zero rather than failing. So the flag says which
    number you are holding, and it matters that the shortfall is not uniform
    across arms: it drops exactly the term an attention-heavy variant spends most
    of its time in, which would flatter it against a feed-forward-heavy one.
    """
    was_training = model.training
    model.train()
    with FlopCounterMode(display=False) as counter:
        model(inputs, mask)
    forward = counter.get_total_flops()
    with FlopCounterMode(display=False) as counter:
        model(inputs, mask).sum().backward()
    total = counter.get_total_flops()
    ops = {str(op): int(value) for op, value
           in counter.get_flop_counts().get("Global", {}).items()}
    # A throwaway loss and no optimiser step, so the weights are untouched -- but
    # that backward left gradients behind, and the caller's first real step must
    # not inherit them.
    model.zero_grad(set_to_none=True)
    model.train(was_training)

    positions = max(inputs.shape[0] * inputs.shape[1], 1)
    return {"forward": forward,
            "backward": total - forward,
            "total": total,
            "per_position": total / positions,
            "attention_counted": any("scaled_dot_product" in op for op in ops),
            "by_op": dict(sorted(ops.items(), key=lambda item: -item[1]))}


def log_configs(writer: SummaryWriter, variant: Variant, config: ModelConfig, args,
                fingerprint: str) -> None:
    """This arm's configuration, as the fenced JSON text train.py logs.

    ModelConfig and Environment carry the tag and the format they carry in a
    training run, so the TEXT tab of a comparison run reads against a real one
    without translating anything -- the same reason the param/grad/* buckets are
    shared rather than restated. ComparisonConfig is this script's answer to
    train.py's TrainingConfig: there is no TrainingConfig object here, the run
    knobs live on the parsed arguments. Variant is the one with no counterpart,
    and it is the important one -- a chart of two arms is unreadable later
    without a record of which model class and which overrides produced each.

    Written into every variant's own writer rather than once into a shared one.
    TensorBoard's TEXT tab is per-run, so a single shared writer would either
    hide the text from the runs it describes or hang a pseudo-run beside them --
    the fragmentation the add_scalars note in run() exists to avoid. Environment
    and ComparisonConfig therefore repeat per arm, which is what leaves each run
    directory independently readable.

    Emitted after the config is read back off the built model, so what lands is
    what the arm actually trained with rather than what was asked for.
    """
    writer.add_text("ModelConfig", fenced_json(config.to_dict()), 0)
    # `variants` holds resolved classes and `base_config` goes in whole beneath,
    # which is the same pair summary.json keeps out of its own args block.
    writer.add_text("ComparisonConfig", fenced_json(
        {**{key: value for key, value in vars(args).items()
            if key not in ("base_config", "variants")},
         "base_config": args.base_config}), 0)
    writer.add_text("Variant", fenced_json(
        {"label": variant.label, "model": variant.model, "overrides": variant.overrides,
         "data_fingerprint": fingerprint}), 0)
    writer.add_text("Environment", fenced_json(ENV), 0)


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
    log_configs(writer, variant, config, args, observed)

    # Measured on the eager module and on a real training batch, so the FLOPs are
    # the ones a step actually costs at this shape. Before the compile below and
    # before the clock starts: it runs a forward and a backward of its own, which
    # would otherwise be billed to this arm's wall-clock.
    probe = train.items[0][0].to(DEVICE, non_blocking=True)
    census = {"parameters": parameter_census(model),
              "flops_per_step": flop_census(model, probe, build_mask(probe, pad, causal))}
    writer.add_text("ModelCensus", fenced_json(census), 0)
    flops_per_step = census["flops_per_step"]["total"]
    if not census["flops_per_step"]["attention_counted"]:
        LOGGER.warning(
            f"  {label}: no FLOP formula is registered for the attention kernel this "
            f"device dispatches to, so perf/tflops and the ModelCensus counts exclude "
            f"attention. The shortfall is larger for arms that attend more, so treat "
            f"the number as a floor rather than as a like-for-like ratio.")

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

    # Resolved once, up front, so a model whose blocks this script cannot read
    # fails before it trains for ten minutes and reports nothing. These feed the
    # parameter census below; the sublayer DIAGNOSTICS no longer come through
    # here, since attn/* and ffn/* are registered probes now (see model.py).
    attentions = attention_modules(model, label)
    feedforwards = feedforward_modules(model, label)

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

    initial = diagnose_modules(model, data.diag, pad, causal)
    curve: list[tuple[int, float, float]] = []
    elapsed, window_start = 0.0, None
    loss_sum = torch.zeros((), dtype=torch.float32, device=DEVICE)
    loss_count, grad_totals, weight_totals = 0, {}, {}
    # Mirror loss_sum/loss_count: accumulated on device, resolved once per
    # evaluation, so per-step clipping telemetry adds no per-step sync.
    clip_sum = torch.zeros((), dtype=torch.float32, device=DEVICE)
    clip_hits = torch.zeros((), dtype=torch.float32, device=DEVICE)
    clip_count = 0
    parameters: dict[str, float] = {}
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
            weight_totals = weight_norms(model)
        # clip_grad_norm_ returns the pre-clip total norm it measured. It runs
        # every step regardless, so taking the return is free -- and it is the
        # only way to see how OFTEN clipping fires, which the every-evaluation
        # gradient snapshot above cannot show.
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        clip_sum += total_norm.detach()
        clip_hits += (total_norm.detach() > args.grad_clip).float()
        clip_count += 1
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
            writer.add_scalar("loss/ppl_val", math.exp(min(val_loss, 20.0)), step, walltime=walltime)
            writer.add_scalar("optim/lr", scheduler.get_last_lr()[0], step, walltime=walltime)
            # Accumulated on device across the window, exactly like loss_sum above,
            # so the per-step pre-clip norm costs no per-step sync. grad_norm is
            # what --grad-clip acted against and clip_frac is how often it acted:
            # at clip_frac ~ 1 the effective learning rate is not the one on
            # optim/lr, which no other tag here would show.
            writer.add_scalar("optim/grad_norm",
                              (clip_sum / max(clip_count, 1)).item(), step, walltime=walltime)
            writer.add_scalar("optim/clip_frac",
                              (clip_hits / max(clip_count, 1)).item(), step, walltime=walltime)
            clip_sum.zero_(); clip_hits.zero_(); clip_count = 0
            if args.amp:
                writer.add_scalar("optim/grad_scale", scaler.get_scale(), step, walltime=walltime)

            # param/grad/* uses train.py's buckets, so a variant's gradient trace
            # reads against a real training run's without translating anything.
            # param/weight/* is the same decomposition of the weights themselves,
            # and exists so param/ratio/* can be formed: lr * ||grad|| / ||weight||
            # is the scale-free number that says whether a component is learning
            # too fast or has stopped, which neither half tells you alone.
            parameters = resolve_parameters(grad_totals, weight_totals,
                                            scheduler.get_last_lr()[0])
            for tag, value in parameters.items():
                writer.add_scalar(tag, value, step, walltime=walltime)
            writer.add_scalar("perf/elapsed_sec", elapsed, step, walltime=walltime)
            writer.add_scalar("perf/ms_per_step", elapsed / step * 1e3, step, walltime=walltime)
            writer.add_scalar("perf/tokens_per_sec",
                              train.total_inputs / len(train) * step / max(elapsed, 1e-9),
                              step, walltime=walltime)
            # Achieved throughput, against which the device's peak is the yardstick.
            # tok/s says how fast an arm moves data; this says how much of the
            # machine it is using to do it, and the two come apart exactly where
            # the interesting answer is -- an arm can trail on tok/s because it
            # does more arithmetic per token, or because it is launch-bound and
            # leaving the device idle, and only this number tells them apart.
            writer.add_scalar("perf/tflops",
                              flops_per_step * step / max(elapsed, 1e-9) / 1e12,
                              step, walltime=walltime)
            if DEVICE.type == "cuda":
                # A curve, not just the end-of-run number in summary.json: peak
                # allocation moves with the step, and two arms can share a final
                # peak while one of them spent the run near it.
                writer.add_scalar("perf/peak_mem_mb",
                                  torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2,
                                  step, walltime=walltime)

            collapse = None
            # Tracked over training, not just start/end: representation collapse
            # is a trajectory, and the depth profile is the thing to compare.
            if diag_index % args.diag_every == 0 or step == args.steps:
                diagnostics = diagnose_modules(model, data.diag, pad, causal)
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

    final = diagnose_modules(model, data.diag, pad, causal)
    peak_mb = (torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2) if DEVICE.type == "cuda" else 0.0
    # Over steps, which every arm shares by construction, so this one is
    # comparable as it stands. The seconds-axis area is NOT computed here: it is
    # only meaningful against a budget every variant reached, and that is not
    # known until they have all run, so report() derives it from `curve`.
    aulc_steps = aulc(curve, 0)

    # HPARAMS: what a sweep is actually for. The text tags above render one
    # variant's config as a blob you can read; this renders every variant as a
    # row you can SORT by outcome and filter by setting, which is the difference
    # between reading k*spread*tau_rel charts and reading one table. run_name="."
    # writes into this variant's own directory -- the default mints a timestamped
    # subdirectory, which TensorBoard then shows as a phantom run.
    writer.add_hparams(
        hparams(variant, config, args),
        {"hparam/loss_val": curve[-1][2] if curve else float("nan"),
         "hparam/loss_aulc": aulc_steps,
         "hparam/tokens_per_sec": train.total_inputs / len(train) * args.steps / max(elapsed, 1e-9),
         "hparam/peak_mem_mb": peak_mb},
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
        "census": census,
        "tflops": flops_per_step * args.steps / max(elapsed, 1e-9) / 1e12,
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
        "param_norms": parameters,
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
    print("  loss/*, collapse/*, attn/*, ffn/*, param/* and perf/* carry the same tag in every")
    print("  run, so each chart overlays them all. attn/* and ffn/* hold the same four measurements")
    print("  per sublayer, so reading them side by side says which branch drove the block.")
    print("  Switch the x-axis to RELATIVE for loss against seconds of training compute")
    print("  rather than steps.")
    print("  TEXT holds ModelConfig, ComparisonConfig, Variant, Environment and ModelCensus per")
    print("  run, so a chart reads back to the class, overrides, commit, parameter breakdown and")
    print("  measured FLOPs that produced it. perf/tflops is achieved throughput against the")
    print("  device's peak: read it beside tok/s to tell an arm that does more arithmetic per")
    print("  token apart from one that is leaving the device idle.")

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
    parser.add_argument("--gain", type=float, default=1.0)

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
