"""Short-run A/B of the attention variants on real data.

Trains one model per VARIANT from identical seeds on identical batches and
reports the three things that actually decide the question.

The batches, their order and the validation set are built once and shared by
every variant, and each variant re-verifies an order-sensitive digest of all of
it before it trains a step, so a difference in the reported numbers can only
come from the variant itself (see DataPlan). What gets reported:

  * validation loss vs step       -- quality per unit of learning
  * validation loss vs wall-clock -- quality per unit of compute, the honest
    axis when the variants differ in speed by 20-30%
  * per-layer attention health    -- how the sublayer is behaving in each decoder
    block, not averaged into a single scalar that hides the one bad layer

The first variant is the baseline every other one is reported against. It is the
model the flags describe -- standard multi-head attention -- unless you put
something else first.

A variant is just a name plus a set of ModelConfig overrides, given on the
command line:

    --variant base:                     # the flags as they stand, no overrides
    --variant wide:heads=16
    --variant post:post_norm=true,ff_dim=2048

so any ModelConfig field can be ablated without touching this file, and no field
name is hardcoded anywhere in it. `--config` sets the shared base every variant
starts from.

Nothing in this script knows what is inside an attention module. The health
diagnostics are taken from forward hooks -- the tensor the sublayer reads, the
update it writes and the block's output -- so every tag means the same thing for
every variant, which is the only way two of them can honestly share a chart. Any
attention implementation that fits GPTmodel is measurable here as-is.

Defaults are sized for a CPU smoke run. On a GPU, scale up -- the comparison is
only meaningful once the model is big enough to be data-bound rather than
noise-bound:

    python compare_attention.py --steps 4000 --embed-dim 512 --n-decoders 6 \
        --seq-len 512 --batch-size 32 --training-data <path> --validation-data <path>
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from config import DEVICE, ENV, LOGGER, MIXED_PRECISION_ENABLED, ModelConfig
from dataset import TextStreamDataset
from model import GPTmodel
from utils import get_causal_mask

# Norms are clamped here rather than at finfo.tiny: dividing by 1e-38 produces
# inf, which then poisons an entire masked mean. These are diagnostics, so a
# bounded wrong answer beats an unbounded one.
FLOOR = 1e-12

# name:key=value,... -- the model exactly as the flags configure it, and nothing
# else. Naming a second variant here would mean naming a ModelConfig field, and a
# default that hardcodes a field is a default that breaks the day that field is
# renamed or removed. Every comparison arm comes from --variant on the command
# line; this is only the baseline they are measured against.
DEFAULT_VARIANTS = ("baseline:",)


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


def parse_overrides(text: str, template: ModelConfig) -> dict:
    out = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected key=value, got '{item}'")
        key, _, value = item.partition("=")
        key = key.strip()
        out[key] = coerce(key, value, template)
    return out


def parse_variant(text: str, template: ModelConfig) -> tuple[str, dict]:
    if ":" not in text:
        raise argparse.ArgumentTypeError(
            f"variant '{text}' needs a name then a colon, e.g. 'wide:heads=16' "
            "(or 'base:' for the flags as they stand)")
    name, _, overrides = text.partition(":")
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"variant '{text}' has an empty name")
    return name, parse_overrides(overrides, template)


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


@torch.inference_mode()
def diagnose(model: GPTmodel, inputs: torch.Tensor, pad: int,
             causal: torch.Tensor) -> dict[str, float]:
    """Per-layer attention health, from ONE forward pass.

    Everything here is read off forward hooks, from three tensors per decoder
    block: what the attention sublayer was handed, the update it wrote back, and
    what left the block. No probe reaches inside an attention module, looks up an
    attribute or recomputes a score. That is deliberate -- a diagnostic that
    knows the internals of one variant produces a tag the other variant cannot
    report, which is precisely the tag that cannot be compared. It also means
    these numbers survive any change to the attention implementations.

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
                          the stream (and the first thing to check when one
                          variant's grad_norm sits an order of magnitude off the
                          other's); far below 1 is a block that has switched off.
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
        # DecoderModule returns (hidden, new_kv)
        def hook(_module, _args, output):
            rec.add("collapse/output", layer, cosine_collapse(output[0], geom))
        return hook

    for layer, block in enumerate(model.decoders):
        handles.append(block.masked_multihead_attention.register_forward_hook(attention_hook(layer)))
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
def validate(model: GPTmodel, batches: BatchSet, pad: int, causal: torch.Tensor,
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


def build_optimiser(model: GPTmodel, args) -> torch.optim.AdamW:
    # fused AdamW keeps the whole update in one kernel; it is available only on
    # CUDA, and applies identically to every variant, so it does not tilt the
    # comparison it speeds up.
    fused = DEVICE.type == "cuda" and args.fused
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                             betas=(args.beta1, args.beta2), fused=fused)


def run(label: str, overrides: dict, args, data: DataPlan, fingerprint: str,
        pad: int, causal: torch.Tensor, run_dir: str) -> dict:
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
    model = GPTmodel.build(config).to(DEVICE)
    model.train()
    # GPTmodel.__init__ may resolve derived config fields (a rank left at 0, say),
    # so the config that gets reported is read back off the model, not off the
    # overrides that were asked for.
    config = model.config

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

    unique = {id(p): p for p in model.parameters()}
    total_params = sum(p.numel() for p in unique.values())
    attn_params = sum(p.numel() for n, p in model.named_parameters() if "masked_multihead" in n)

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

    initial = diagnose(model, data.diag, pad, causal)
    curve: list[tuple[int, float, float]] = []
    elapsed, window_start = 0.0, None
    loss_sum = torch.zeros((), dtype=torch.float32, device=DEVICE)
    loss_count, grad_norm = 0, torch.zeros((), device=DEVICE)
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

        optimiser.zero_grad(set_to_none=True)
        loss = forward_backward(train.items[data.schedule[step - 1]])
        if args.amp:
            scaler.unscale_(optimiser)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        # Accumulated on device: reading the loss every step would reintroduce
        # exactly the per-step sync the windowed clock above exists to avoid.
        loss_sum += loss
        loss_count += 1

        if step % args.eval_every == 0 or step == args.steps:
            sync()
            elapsed += time.perf_counter() - window_start
            window_start = None

            val_loss = validate(model, val, pad, causal, args.amp)
            train_loss = (loss_sum / max(loss_count, 1)).item()
            loss_sum.zero_()
            loss_count = 0
            curve.append((step, elapsed, val_loss))
            walltime = anchor + elapsed

            writer.add_scalar("loss/val", val_loss, step, walltime=walltime)
            writer.add_scalar("loss/train", train_loss, step, walltime=walltime)
            writer.add_scalar("optim/lr", scheduler.get_last_lr()[0], step, walltime=walltime)
            writer.add_scalar("optim/grad_norm", grad_norm.item(), step, walltime=walltime)
            writer.add_scalar("perf/elapsed_sec", elapsed, step, walltime=walltime)
            writer.add_scalar("perf/ms_per_step", elapsed / step * 1e3, step, walltime=walltime)
            writer.add_scalar("perf/tokens_per_sec",
                              train.total_inputs / len(train) * step / max(elapsed, 1e-9),
                              step, walltime=walltime)

            collapse = None
            # Tracked over training, not just start/end: representation collapse
            # is a trajectory, and the depth profile is the thing to compare.
            if diag_index % args.diag_every == 0 or step == args.steps:
                diagnostics = diagnose(model, data.diag, pad, causal)
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

    final = diagnose(model, data.diag, pad, causal)
    peak_mb = (torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2) if DEVICE.type == "cuda" else 0.0

    # run_name="." keeps the hparams in this run's own directory; the default
    # would nest a fresh timestamped run underneath and split the variant in two.
    # The overrides go in as one JSON string rather than as columns. Variants may
    # set disjoint fields, and a column per field would leave the hparams table
    # mostly blank and would need this script to know which fields exist.
    writer.add_hparams(
        {"variant": label, "overrides": json.dumps(overrides, sort_keys=True),
         "attn_params": attn_params, "total_params": total_params,
         "embed_dim": config.embed_dim, "heads": config.heads,
         "n_decoders": config.n_decoders, "seq_len": config.seq_len, "lr": args.lr,
         "compile": args.compile_mode if args.compile else "off"},
        {"hparam/final_val": curve[-1][2], "hparam/sec": elapsed,
         "hparam/warmup_sec": warmup_sec,
         "hparam/collapse_mean": final.get("collapse/output/mean", float("nan"))},
        run_name=".")
    writer.close()

    return {
        "label": label,
        "config": config.to_dict(),
        "data_fingerprint": observed,
        "attn_params": attn_params,
        "total_params": total_params,
        "curve": curve,
        "sec": elapsed,
        "warmup_sec": warmup_sec,
        "peak_mb": peak_mb,
        "tokens_per_sec": train.total_inputs / len(train) * args.steps / max(elapsed, 1e-9),
        "collapse_start": layer_profile(initial, "collapse/output"),
        "collapse_end": layer_profile(final, "collapse/output"),
        "diagnostics": final,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def report(results: dict[str, dict], args, run_dir: str, fingerprint: str) -> None:
    width = max(max((len(label) for label in results), default=7), 7)
    rule = "=" * (width + 81)
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

    print(f"\n{'variant':{width}s}{'attn params':>13}{'params':>12}{'final val':>11}"
          f"{'d base':>9}{'val @ %.*fs' % (precision, budget):>12}{'sec':>8}{'x base':>8}"
          f"{'tok/s':>10}{'peak MB':>10}")
    print("-" * (width + 81))
    for label, r in results.items():
        within = [point for point in r["curve"] if point[1] <= budget] or r["curve"][:1]
        delta = r["curve"][-1][2] - baseline["curve"][-1][2]
        speed = baseline["sec"] / max(r["sec"], 1e-9)
        print(f"{label:{width}s}{r['attn_params']:>13}{r['total_params']:>12}"
              f"{r['curve'][-1][2]:>11.4f}{delta:>+9.4f}{within[-1][2]:>12.4f}"
              f"{r['sec']:>8.1f}{speed:>8.2f}{r['tokens_per_sec']:>10.0f}{r['peak_mb']:>10.0f}")
    print(f"\n  d base < 0 is better than {baseline['label']}; x base > 1 is faster than it.")
    print(f"  all variants verified against data fingerprint {fingerprint}: same batches,")
    print(f"  same order, same validation set.")

    print("\nper-layer token similarity after each block (mean pairwise cosine; -> 1 means collapsed)")
    print("-" * (width + 81))
    for label, r in results.items():
        start = " ".join(f"{value:5.2f}" for value in r["collapse_start"])
        end = " ".join(f"{value:5.2f}" for value in r["collapse_end"])
        print(f"{label:{width}s} init [{start} ]   trained [{end} ]")

    print(rule)
    print(f"\ntensorboard --logdir {run_dir}")
    print("  loss/*, collapse/*, attn/* and perf/* carry the same tag in every variant's run,")
    print("  so each chart overlays them all. Switch the x-axis to RELATIVE for loss against")
    print("  seconds of training compute rather than steps.")

    summary = os.path.join(run_dir, "summary.json")
    with open(summary, "w", encoding="utf-8") as handle:
        json.dump({"env": ENV, "args": {k: v for k, v in vars(args).items() if k != "base_config"},
                   "base_config": args.base_config, "data_fingerprint": fingerprint,
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
                             "ablation needs no code change. With none given the run trains only "
                             f"the baseline (default: {'; '.join(DEFAULT_VARIANTS)})")
    parser.add_argument("--config", default="", metavar="key=value,...",
                        help="ModelConfig overrides applied to EVERY variant, on top of the "
                             "flags below and underneath each variant's own overrides")

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
        args.base_config.update(parse_overrides(args.config, template))
        args.variants = [parse_variant(v, template) for v in (args.variant or DEFAULT_VARIANTS)]
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))

    names = [name for name, _ in args.variants]
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
                       "Raise --max-samples or lower --steps to compare generalisation.")
    LOGGER.info(f"device {DEVICE} "
                f"({torch.cuda.get_device_name(DEVICE) if DEVICE.type == 'cuda' else 'cpu'}), "
                f"mixed precision {'on' if args.amp else 'off'}, "
                f"compile {args.compile_mode if args.compile else 'off'}, "
                f"batches {'preloaded' if preload else 'streamed'} ({footprint / 1024 ** 2:.0f} MiB)")
    LOGGER.info(f"variants: {', '.join(f'{n} ({o or 'no overrides'})' for n, o in args.variants)}")
    if len(args.variants) == 1:
        LOGGER.warning("only one variant -- this trains a baseline and compares it to itself. "
                       "Add arms with --variant NAME:field=value (repeatable); "
                       f"settable fields: {', '.join(sorted(ModelConfig().to_dict()))}")
    LOGGER.info(f"data fingerprint {fingerprint} -- {len(data.schedule)} scheduled steps over "
                f"{len(train)} batches; every variant re-checks this before training")

    run_dir = os.path.join(args.tb_log_dir,
                           args.run_name or f"attn-compare-{datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)

    results = {}
    for label, overrides in args.variants:
        results[label] = run(label, overrides, args, data, fingerprint, pad, causal, run_dir)

    report(results, args, run_dir, fingerprint)


if __name__ == "__main__":
    main()
