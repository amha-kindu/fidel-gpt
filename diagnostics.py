"""Training diagnostics: one implementation of every TensorBoard tag.

  loss/*  optim/*  param/*  perf/*          TrainingMonitor
  param/gsnr/*  optim/noise_scale           gradient_noise
  embedding/* attn/* ffn/* stream/* head/*  diagnose_modules, driving the probes
                                            each module registers beside itself
  TEXT: ModelCensus                         parameter_census + flop_census

A training loop drives everything through TrainingMonitor. DIAGNOSTICS.md
describes every tag.
"""
import contextlib
import json
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

import probes
from config import DEVICE
from utils import component_key


def sync() -> None:
    """CUDA kernels launch asynchronously, so an unsynchronised perf_counter
    measures queue time, not compute. Every timed boundary syncs first."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


class Recorder:
    """Per-layer scalars accumulated as 0-dim device tensors.

    Every `.item()` on a CUDA tensor is a device sync. Reading each metric off
    the device as it is produced costs one stall per metric per layer -- well
    over a hundred per probe pass. Here the values stay on device until
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

        A tag measured at one site only (head/*, norm_f, the projection) gets no
        reduction: its mean, min and max would be three more copies of layer_00.
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
            if len(values) == 1:
                continue
            out[f"{tag}/mean"] = sum(values) / len(values)
            out[f"{tag}/min"] = min(values)
            out[f"{tag}/max"] = max(values)
        return out


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
                     mask: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    """Everything registered in `probes`, from ONE forward pass in eval mode.

    This function owns the mechanism and none of the meaning: the probe batch,
    the padding mask, the single host transfer, and the <family>/<site>/<metric>
    naming. What each metric IS lives beside the module that registers it, where
    `isinstance` is available and where anyone changing the module is already
    looking. See `probes.register`.

    `mask` is what the model's forward takes; `valid` is (B, S), True at every
    real (non-pad) position, and is what every probe reduction is over.
    """
    sites = [(name, module, probe)
             for name, module in model.named_modules()
             for probe in probes.probes_for(module)]
    if not sites:
        return {}                       # a model without probed modules emits nothing, not zeros

    rec = Recorder()
    ctx = probes.ProbeContext(valid)
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

    # Each block's input is the residual stream, whatever is inside the block;
    # attn/* and ffn/* measure their updates against it (ProbeContext.residual_stream).
    def mark_stream(_block, args):
        ctx.stream = args[0]
    for block in getattr(model, "decoders", ()):
        handles.append(block.register_forward_pre_hook(mark_stream))

    was_training = model.training
    model.eval()                        # eval matters: a norm carrying running stats
    try:                                # must not fold the probe batch into them
        with torch.no_grad():
            model(inputs, mask)
    finally:
        model.train(was_training)
        for handle in handles:
            handle.remove()
    return rec.resolve()


def gradient_norms(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Squared gradient norm per component, bucketed by utils.component_key.

    The buckets -- Embedding, Projection, Decoder<i> per block, NormF for the
    rest -- are the param/grad/* groups. Taken after unscale_ and before
    clipping, so it describes the gradient the step produced rather than what
    clipping let through.

    Returned as 0-dim device tensors rather than floats: this runs inside the
    training step, and an .item() per parameter would be one device sync per
    parameter. resolve_parameters turns them into numbers in one transfer.
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
    paired with the update norm it gives the scale-free ratio below, which is
    the number that actually says whether a component is learning.

    Same 0-dim-device-tensor discipline: this runs inside the training step, and
    an .item() per parameter would be one device sync per parameter.
    """
    totals: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        key = component_key(name)
        norm_sq = torch.linalg.vector_norm(param.detach().float().view(-1)).square()
        totals[key] = totals[key] + norm_sq if key in totals else norm_sq
    return totals


def snapshot_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """A copy of every trainable parameter, taken just before an optimiser step.

    update_norms diffs against it. Measuring the step rather than predicting it
    from the gradient is the point: under AdamW each element moves by roughly lr
    whatever its gradient's size, plus the decoupled decay, so lr * ||grad||
    says next to nothing about how far a component actually moved. Costs one
    copy of the trainable weights, once per logging window.
    """
    return {name: param.detach().clone()
            for name, param in model.named_parameters() if param.requires_grad}


def update_norms(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Squared norm of what the optimiser step just did, per component.

    ||W_after - W_before|| in gradient_norms' buckets: the actual update,
    including weight decay, clipping, Adam's normalisation and the schedule's lr.
    A step the AMP scaler skipped reads as exactly 0 -- which is what happened.
    """
    totals: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name not in before:
            continue
        key = component_key(name)
        delta = param.detach().float() - before[name].float()
        norm_sq = torch.linalg.vector_norm(delta.view(-1)).square()
        totals[key] = totals[key] + norm_sq if key in totals else norm_sq
    return totals


def resolve_parameters(grads: dict[str, torch.Tensor], weights: dict[str, torch.Tensor],
                       updates: dict[str, torch.Tensor]) -> dict[str, float]:
    """Gradient, weight and update-ratio scalars, in ONE host transfer.

    Emits three families off one sync:

      param/grad/<component>    ||grad||. Their root summed square is the step's
                                pre-clip global norm; there is no param/grad/global
                                because it would be one sample of the window mean
                                optim/grad_norm already logs.
      param/norm/<component>    ||weight||, same buckets. No global: the embedding
                                and projection would dominate it.
      param/update_ratio/<component>
                                ||W_after - W_before|| / ||W_before|| across the
                                logged step's optimiser update: the fraction of
                                its own magnitude a component actually moved.
                                Roughly 1e-3 is healthy. Orders of magnitude below
                                means a component has stopped learning while the
                                loss curve keeps falling on the strength of the
                                others; orders above is the component about to
                                destabilise the run. Scale-free, so it is comparable
                                BETWEEN components and between runs, which
                                neither norm is.

    Grouped under param/ rather than as three top-level families because
    TensorBoard sorts alphabetically, and the ratio is unreadable without the
    weight norm it is formed from on the same screen.
    """
    if not grads and not weights:
        return {}

    groups = (grads, weights, updates)
    keys = [list(group) for group in groups]
    stacked = torch.stack([group[k] for group, ks in zip(groups, keys) for k in ks])
    values = stacked.sqrt().cpu().tolist()                      # the one and only sync

    grad_norm, weight_norm, update_norm, offset = {}, {}, {}, 0
    for norms, ks in zip((grad_norm, weight_norm, update_norm), keys):
        norms.update(zip(ks, values[offset:offset + len(ks)]))
        offset += len(ks)

    resolved = {f"param/grad/{k}": v for k, v in grad_norm.items()}
    resolved.update({f"param/norm/{k}": v for k, v in weight_norm.items()})
    for key, update in update_norm.items():
        weight = weight_norm.get(key)
        if weight:
            resolved[f"param/update_ratio/{key}"] = update / weight
    return resolved


def gradient_noise(model: torch.nn.Module, inputs: torch.Tensor, mask: torch.Tensor,
                   labels: torch.Tensor, chunks: int) -> dict[str, float]:
    """Gradient signal-to-noise per component, and the gradient noise scale.

    The probe batch is split into `chunks` equal chunks of b sequences, and each
    chunk's gradient g_k is taken separately. Per component (component_key):

        tr(S_b) = K/(K-1) * (mean_k ||g_k||^2 - ||g_mean||^2)    gradient noise
        ||G||^2 = ||g_mean||^2 - tr(S_b) / K                       gradient signal

    the second corrected for the noise ||g_mean||^2 still carries -- uncorrected,
    a small K overstates the signal. Both are scaled to ONE SEQUENCE (the noise of
    a b-sequence mean is 1/b of a single sequence's), so neither depends on how
    large the probe batch is or how it was chunked:

      param/gsnr/<component>   ||G||^2 / tr(S_1). A ratio of sums rather than a
                               mean of per-parameter ratios, which a few
                               near-zero-variance weights would dominate. ~1 or
                               more is a gradient one sequence already points
                               the right way with; << 1 is a component moving on
                               noise. 0 when the noise swallowed the signal
                               estimate entirely.
      optim/noise_scale        tr(S_1) / ||G||^2 over the whole model, in
                               sequences: the batch size past which a larger batch
                               stops buying a proportionally better gradient
                               (McCandlish et al., 2018, B_simple). Compare it with
                               the sequences per optimiser step. Omitted when the
                               signal estimate is not positive.

    Gradients come from torch.autograd.grad, which never touches .grad -- the
    training step's gradients are undisturbed -- and does not run the
    AccumulateGrad hooks DDP reduces through. Eval mode, so dropout's randomness
    is not counted as gradient noise; plain cross-entropy, without label
    smoothing. Costs `chunks` forward + backward passes over the probe batch and
    one fp32 copy of the trainable weights.
    """
    rows = inputs.shape[0]
    chunks = min(chunks, rows)
    if chunks < 2:
        return {}
    size = rows // chunks

    named = [(component_key(name), param) for name, param in model.named_parameters()
             if param.requires_grad]
    params = [param for _, param in named]
    total = [torch.zeros_like(param, dtype=torch.float32) for param in params]
    square: dict[str, torch.Tensor] = {}

    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            for k in range(chunks):
                rows_k = slice(k * size, (k + 1) * size)
                logits = model(inputs[rows_k], mask[rows_k])
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1).float(), labels[rows_k].flatten(), ignore_index=-100)
                grads = torch.autograd.grad(loss, params, allow_unused=True)
                for i, ((key, _), grad) in enumerate(zip(named, grads)):
                    if grad is None:
                        continue
                    grad = grad.detach().float()
                    total[i] += grad
                    norm_sq = grad.square().sum()
                    square[key] = square[key] + norm_sq if key in square else norm_sq
    finally:
        model.train(was_training)

    mean_sq: dict[str, torch.Tensor] = {}
    for (key, _), summed in zip(named, total):
        norm_sq = (summed / chunks).square().sum()
        mean_sq[key] = mean_sq[key] + norm_sq if key in mean_sq else norm_sq

    keys = list(square)
    noise = torch.stack([(square[k] / chunks - mean_sq[k]) * chunks / (chunks - 1) for k in keys])
    signal = torch.stack([mean_sq[k] for k in keys]) - noise / chunks
    noise, signal = (noise * size).cpu().tolist(), signal.cpu().tolist()   # the one sync

    out = {f"param/gsnr/{k}": max(s, 0.0) / max(n, probes.FLOOR) for k, s, n in zip(keys, signal, noise)}
    if sum(signal) > 0:
        out["optim/noise_scale"] = sum(noise) / sum(signal)
    return out


def fenced_json(payload: dict) -> str:
    """A payload in the fenced block TensorBoard's TEXT tab renders as code.

    default=str so a field json cannot represent -- a dtype, a device -- degrades
    to its repr rather than taking the run down at step 0, before a single batch
    has been seen.
    """
    return f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"


def parameter_census(model: torch.nn.Module) -> dict:
    """Where the model's parameters actually sit, bucketed by component_key.

    The rule the param/grad/* tags already use, so a component's share of the
    weights reads directly against its share of the gradient norm. That pairing
    is the point: a block holding 30% of the parameters and 3% of the gradient
    has stopped learning, and neither number says so on its own.

    Counted through named_parameters(), which yields each shared tensor once
    under its first name -- so a tied embedding is counted once, matching the
    deduplication in gradient_norms.

    non_embedding is called out separately because it is usually the honest
    capacity number, and it nets out BOTH vocabulary-sized tables -- the input
    embedding and the output projection. Each is vocab_size * embed_dim whatever
    the blocks are doing, and easily large enough to dilute a real difference in
    the blocks into a rounding error in the total. Subtracting only the input
    side would leave the output head counted as block capacity whenever the two
    are untied, so the figure would mean different things depending on a config
    flag.
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
    """FLOPs for one training batch (forward + backward), measured by dispatch.

    Counted through torch's FlopCounterMode, so this knows nothing about the
    architecture in front of it -- the same reason the probes read off hooks. A
    model with a different block structure is counted correctly with no edit
    here, which a hand-rolled 2 * in * out formula per layer type could not
    promise.

    Normalised per POSITION, not per real token: FLOPs are spent on padding just
    the same, so dividing by the non-pad count would claim a throughput the
    device did not achieve.

    attention_counted is part of the result rather than a footnote. FlopCounterMode
    has formulas registered for the fused CUDA attention kernels but not for every
    backend SDPA can dispatch to -- the CPU one has none -- and where there is no
    formula the op contributes zero rather than failing. The flag says which
    number you are holding: without it, perf/tflops under-reports exactly the term
    an attention-heavy model spends most of its time in.
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


class TrainingMonitor:
    """Every training-time tag, driven from the training loop.

    One instance per run, driven from the optimiser-step loop:

        monitor.count_tokens(inputs, pad)         every micro-batch
        monitor.before_update(step)               after unscale_, before clipping
        monitor.clip(parameters)                  in place of clip_grad_norm_
        <optimiser step>
        monitor.after_update()
        monitor.end_step(step, loss, lr, scaler)  once per optimiser step
        with monitor.paused(): <validate/save>    time left out of perf/*
        monitor.validated(step, val_loss)         after each validation

    Every `log_every` optimiser steps it closes a window and writes:

      loss/curves {train}           mean training loss over the window; val joins
                                    it on the same chart, see validated()
      optim/lr
      optim/grad_norm               window mean of the pre-clip global norm
      optim/clip_frac               share of the window's steps that clipped
      optim/grad_scale              AMP only
      param/grad|norm|update_ratio  at the window's last step (resolve_parameters)
      perf/tokens_per_sec           non-pad input tokens over the window's
                                    training time, all ranks
      perf/tflops                   flop_census's per-batch FLOPs scaled to the
                                    window, all ranks

    and at each validation: loss/curves {val}, loss/gap and, from one fixed probe
    batch, every registered probe -- embedding/* attn/* ffn/* stream/* head/* and
    whatever a model class registers itself -- plus param/gsnr/* and
    optim/noise_scale from gradient_noise, unless gsnr_chunks < 2.

    Only the coordinator writes, and only it pays for the param/* snapshot and the
    probe pass. The window accumulators stay on device on every rank, so a step
    syncs for telemetry only when it closes a window.

    perf/* is windowed rather than cumulative: a training run is long enough that
    a slowdown partway through is the thing to see, and a cumulative mean
    flattens it.
    """

    def __init__(self, logger, model: torch.nn.Module, log_every: int, max_norm: float,
                 probe: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], flops: dict,
                 batches_per_step: int, world_size: int, is_coordinator: bool,
                 gsnr_chunks: int = 8) -> None:
        self.logger = logger
        self.model = model
        self.log_every = max(1, log_every)
        self.max_norm = max_norm
        self.probe_inputs, self.probe_mask, self.probe_valid, self.probe_labels = probe
        self.gsnr_chunks = gsnr_chunks
        self.flops = flops
        self.flops_per_step = flops["total"] * batches_per_step * world_size
        self.world_size = world_size
        self.is_coordinator = is_coordinator

        self.grads: dict[str, torch.Tensor] = {}
        self.weights: dict[str, torch.Tensor] = {}
        self.updates: dict[str, torch.Tensor] = {}
        self.before: dict[str, torch.Tensor] | None = None
        self.train_loss: float | None = None
        self._reset_window()

    def _reset_window(self) -> None:
        self.loss_sum, self.steps = 0.0, 0
        self.norm_sum = torch.zeros((), dtype=torch.float32, device=DEVICE)
        self.clip_hits = torch.zeros((), dtype=torch.float32, device=DEVICE)
        self.tokens = torch.zeros((), dtype=torch.int64, device=DEVICE)
        self.excluded = 0.0
        sync()
        self.window_start = time.perf_counter()

    def logs(self, step: int) -> bool:
        """Whether this optimiser step closes a window on the coordinator."""
        return self.is_coordinator and step % self.log_every == 0

    def count_tokens(self, inputs: torch.Tensor, pad: int) -> None:
        self.tokens += (inputs != pad).sum()

    def before_update(self, step: int) -> None:
        """On a window's last step, snapshot gradients and weights -- after
        unscale_, before clipping -- and copy the weights so after_update can
        measure the step itself."""
        if self.logs(step):
            self.grads = gradient_norms(self.model)
            self.weights = weight_norms(self.model)
            self.before = snapshot_parameters(self.model)

    def clip(self, parameters) -> torch.Tensor:
        """clip_grad_norm_, accumulating the pre-clip norm for optim/*."""
        total = torch.nn.utils.clip_grad_norm_(parameters, max_norm=self.max_norm).detach()
        self.norm_sum += total
        self.clip_hits += (total > self.max_norm).float()
        return total

    def after_update(self) -> None:
        if self.before is not None:
            self.updates = update_norms(self.model, self.before)
            self.before = None

    @contextlib.contextmanager
    def paused(self):
        """Time spent inside is left out of the open window's perf/* denominator."""
        sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            sync()
            self.excluded += time.perf_counter() - started

    def end_step(self, step: int, loss: float, lr: float, scaler=None) -> None:
        """Account one optimiser step; close and write the window every log_every."""
        self.loss_sum += loss
        self.steps += 1
        if step % self.log_every != 0:
            return
        sync()
        seconds = max(time.perf_counter() - self.window_start - self.excluded, 1e-9)
        if self.is_coordinator:
            log = self.logger.log_scalar
            self.train_loss = self.loss_sum / self.steps
            self.logger.log_scalars("loss/curves", {"train": self.train_loss}, step)
            log("optim/lr", lr, step)
            log("optim/grad_norm", (self.norm_sum / self.steps).item(), step)
            log("optim/clip_frac", (self.clip_hits / self.steps).item(), step)
            if scaler is not None:
                log("optim/grad_scale", scaler.get_scale(), step)
            for tag, value in resolve_parameters(self.grads, self.weights, self.updates).items():
                log(tag, value, step)
            # This rank's non-pad count times the world size: ranks draw equally
            # sized batches, so this is the global rate to within padding noise.
            log("perf/tokens_per_sec", self.tokens.item() * self.world_size / seconds, step)
            log("perf/tflops", self.flops_per_step * self.steps / seconds / 1e12, step)
        self.grads, self.weights, self.updates = {}, {}, {}
        self._reset_window()

    def validated(self, step: int, val_loss: float) -> None:
        """loss/curves {val}, loss/gap, the probe pass and gradient_noise. Coordinator only."""
        if not self.is_coordinator:
            return
        with self.paused():
            self.logger.log_scalars("loss/curves", {"val": val_loss}, step)
            # Against the last closed train window; before the first, the open one.
            train = self.train_loss if self.train_loss is not None else self.loss_sum / max(self.steps, 1)
            self.logger.log_scalar("loss/gap", val_loss - train, step)
            diagnostics = diagnose_modules(self.model, self.probe_inputs, self.probe_mask, self.probe_valid)
            diagnostics.update(gradient_noise(self.model, self.probe_inputs, self.probe_mask,
                                              self.probe_labels, self.gsnr_chunks))
            for tag, value in diagnostics.items():
                self.logger.log_scalar(tag, value, step)

    def log_census(self) -> None:
        """TEXT/ModelCensus: parameter_census plus the flop_census of one batch."""
        if self.is_coordinator:
            self.logger.log_text("ModelCensus", fenced_json({**parameter_census(self.model),
                                                             "flops": self.flops}), step=0)


def probe_batch(inputs: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor,
                pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One fixed batch on device for every probe pass of a run: (inputs, mask,
    valid map, labels). The labels are for gradient_noise.

    Fixed so a probe tag's curve moves with the model, not with the batch."""
    inputs, mask, labels = inputs.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
    return inputs, mask, inputs != pad, labels
