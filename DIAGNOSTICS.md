# Attention Diagnostics Reference

Every scalar `compare_attention.py` writes to TensorBoard, what it measures, and what
it tells you when it moves.

```bash
python compare_attention.py --variant "h8:heads=8" --variant "h16:heads=16" ...
tensorboard --logdir logs/<run-name>
```

Each variant writes to a sibling directory under the run directory, with **identical
tag names**. That is what puts the variants on one chart: TensorBoard overlays equal
tags across runs and colours them by run.

---

## Table of Contents

- [Conventions](#conventions)
- [The measurement principle](#the-measurement-principle)
- [`collapse/*` — representation collapse](#collapse--representation-collapse)
- [`attn/*` — attention sublayer health](#attn--attention-sublayer-health)
- [`loss/*` — quality](#loss--quality)
- [`optim/*` — optimisation](#optim--optimisation)
- [`perf/*` — cost](#perf--cost)
- [hparams](#hparams)
- [Reading them together](#reading-them-together)
- [Caveats](#caveats)
- [`summary.json`](#summaryjson)

---

## Conventions

**Per layer, always.** Every diagnostic is emitted once per decoder block plus three
reductions over blocks:

```
collapse/output/layer_00   collapse/output/layer_01   ...
collapse/output/mean       collapse/output/min        collapse/output/max
```

Reporting only the mean once hid a real finding: mid-stack collapse at one layer was
invisible in the average, because the deepest layer was fine and a six-layer mean
dilutes one bad layer sixfold. `/min` matters as much as `/max` for the signed
quantities.

**The x-axis is compute, not just steps.** Every scalar is logged with
`walltime = anchor + elapsed`, where `elapsed` counts training compute only —
evaluation, warmup and `torch.compile` are excluded. Switch TensorBoard's x-axis to
**RELATIVE** and every chart becomes "against seconds of training". This is the honest
axis when variants differ in speed.

**Padding is excluded everywhere.** Pad tokens share one embedding, so pad-pad pairs
sit at cosine ~1 and would inflate any average. Self-pairs are excluded too. Pairwise
quantities are computed **within each sequence**, then averaged over the batch.

**Cadence.** `loss/*`, `optim/*` and `perf/*` are written at every evaluation
(`--eval-every`). The per-layer diagnostics are written every `--diag-every`-th
evaluation, and always at the final step.

**The probe batch is fixed.** Diagnostics run on the first `--diag-samples` sequences
of the first validation batch, cloned once and shared by every variant for the whole
comparison. Movement in these curves is the model changing, never the data.

---

## The measurement principle

Nothing in the diagnostic code knows what is inside an attention module. Everything is
read from forward hooks, from three tensors per decoder block:

| tensor | what it is |
|---|---|
| `x` | what the attention sublayer was handed — `norm1(x)` under the default pre-norm, the raw residual stream under `post_norm=true` |
| `update` | what attention wrote back, **before** the residual add |
| block output | what left the block, after attention, FFN and both residuals |

No probe reaches inside a module, looks up an attribute, or recomputes a score. That
is deliberate. A diagnostic that knows the internals of one variant produces a tag the
other variant cannot report — which is precisely the tag that cannot be compared. It
also means these numbers survive any change to the attention implementation.

---

## `collapse/*` — representation collapse

Mean pairwise cosine similarity between token representations.

| tag | measured on |
|---|---|
| `collapse/input` | the tensor attention reads |
| `collapse/output` | the decoder block's output |

```
~0    tokens stay spread out
→1    tokens have collapsed onto each other; depth is no longer buying anything
```

At the extreme every position carries the same vector, so the LM head can only emit
position-independent predictions.

**Read them as a pair per layer.** `output − input` is what the block *added* to the
collapse, and layer *n*'s output is layer *n+1*'s input. A single block driving most
of the collapse is a different problem from every block contributing evenly.

Rising monotonically with depth is normal, especially early in training. Rising
monotonically *with steps* at a fixed layer is the thing to worry about.

---

## `attn/*` — attention sublayer health

### `attn/input_norm`

Mean ‖x‖ over valid tokens — residual-stream drift, and the scale every other
magnitude here is relative to.

Under the default pre-norm, every layer reads a LayerNorm'd tensor, so all layers sit
near √`embed_dim` and this tag tracks the **LayerNorm gains** growing. Under
`post_norm=true` it tracks the raw residual stream, and block 0 reads the bare
embedding — a much smaller number than every later block.

### `attn/update_ratio`

`mean(‖update‖) / mean(‖x‖)` — the sublayer's gain into the residual stream. (A ratio
of means, not a mean of ratios.)

```
≫1     the block is shouting over the residual stream
~0.2–1 healthy
→0     the block has switched itself off; the residual path is routing around it
```

This is the first thing to check when one variant's `optim/grad_norm` sits an order of
magnitude off another's.

### `attn/update_cos`

`mean(cos(update_i, x_i))` over valid tokens — a mean of per-token cosines.

Attention's job is to write *a combination of other positions* into position *i*. If
the update is aligned with what position *i* already held, the block is not moving
information between positions; it is rescaling in place.

```
~0    writing genuinely new content
→1    mostly rescaling what each token already held — attention has stopped mixing
<0    actively subtracting each token's own direction (a de-correlating operation)
```

This is the failure a loss curve hides longest: the FFN keeps improving while
attention quietly does nothing useful.

⚠️ **Architecturally confounded — see [Caveats](#caveats).**

### `attn/update_isotropy`

How evenly the update spreads its energy over the directions available to it. The
participation ratio of the token covariance spectrum — (Σλ)² / Σλ², also called the
effective rank — divided by `min(valid_tokens, embed_dim)`.

```
1     isotropic: every available direction carries the same energy
→0    anisotropic: the update lives on a handful of directions,
      however wide embed_dim is
```

**This catches what `collapse/*` cannot.** Mean pairwise cosine reports how aligned
tokens are *with each other*; a block can hold that near zero while still writing every
token into the same two-dimensional subspace. Falling isotropy with flat collapse is
heads converging onto one operator.

Computed from the covariance traces — tr(C) = ‖Z‖²_F and tr(C²) = ‖C‖²_F — so there is
no eigendecomposition, and the covariance stays (`embed_dim`, `embed_dim`) regardless
of batch size.

---

## `loss/*` — quality

### `loss/val`

Token-weighted mean cross-entropy in nats over the whole validation set: summed over
tokens, divided by the true non-pad label count. **This is the headline number.**

Weighting matters. Documents are padded to `seq_len`, so averaging per-batch means
would let a batch of short documents count the same as a batch of long ones, biasing
the result toward whatever the heavily-padded batches happen to say.

### `loss/train`

Mean training loss over the evaluation window just closed.

Not computed identically to `loss/val`: it is an unweighted mean of per-step losses,
each already token-weighted *within* its own batch. Fine for watching the trend and
the train/val gap opening. Do not read a 0.01-nat train-vs-val difference as
meaningful.

---

## `optim/*` — optimisation

### `optim/lr`

Current learning rate from the linear-warmup-then-cosine schedule. A sanity check that
warmup and decay landed where `--warmup-frac` and `--steps` put them.

### `optim/grad_norm`

Total gradient norm **before** clipping, sampled from that single step.

If it sits above `--grad-clip` (default 1.0) for a whole run, most steps are being
clipped and you are not training at the LR you set — you are taking direction-only
steps. Correlate with `attn/update_ratio` when a variant misbehaves.

Single-step sample, not a window average, so expect it to be noisy.

---

## `perf/*` — cost

| tag | meaning |
|---|---|
| `perf/elapsed_sec` | cumulative training seconds, excluding evaluation, warmup and compilation |
| `perf/ms_per_step` | cumulative average, `elapsed / step`. Flat means steady state; a rising curve means something is growing |
| `perf/tokens_per_sec` | non-pad input tokens per second — the throughput number to quote, since it normalises away padding differences |

---

## hparams

`hparam/final_val`, `hparam/sec`, `hparam/warmup_sec`, `hparam/collapse_mean`, keyed by
variant name plus its overrides as a JSON string. Use TensorBoard's **HPARAMS** tab to
sort a sweep by final validation loss without opening each run.

Overrides go in as one JSON string rather than as columns: variants may set disjoint
fields, and a column per field would leave the table mostly blank.

---

## Reading them together

The diagnostic that identifies a *cause* is usually a combination.

| pattern | reading |
|---|---|
| `grad_norm` high **+** `update_ratio` high | the block is over-writing the residual stream — lower the LR or check the output scale |
| `collapse` rising **+** `update_isotropy` falling | genuine representational collapse |
| `collapse` flat **+** `update_cos` → 1 | attention has stopped mixing positions; the FFN is carrying the model |
| `update_ratio` decaying → 0 | the block is being routed around; its parameters are dead weight |
| `collapse` flat **+** `update_isotropy` falling | heads converging onto a single operator |
| `input_norm` growing without bound | residual-stream blow-up — check the LayerNorm gains |
| one layer's `/max` far from `/mean` | a single-layer anomaly; open the per-layer chart before concluding anything |

---

## Caveats

**`attn/update_cos` is architecturally confounded.** Standard MHA passes its update
through `W_v`/`W_o`, which rotate the output out of the span of the inputs, so ~0 is
the natural resting value. An attention variant whose value *is* the residual stream
(no `W_v`/`W_o`) is structurally confined to the span of its inputs and will sit far
higher — that is a property of the design, not a pathology. **Compare a variant against
its own trajectory, not against a different architecture at a point in time.**

**`attn/update_isotropy` normalisation.** The divisor is
`min(valid_tokens, embed_dim)`. Keep `--diag-samples × seq_len` comfortably above
`embed_dim`, or the divisor becomes the token count and the value stops reading as
"fraction of the width in use". It remains comparable across variants at the same
diagnostic size either way.

**`--diag-samples` is capped at `--batch-size`**, since the probe is taken from the
head of the first validation batch.

**Resolution.** The script prints its own floor at startup — differences much below
~0.05 nats are not resolvable at smoke-run sizes. Check the scored-token count in the
startup log before reading anything into a small `d base`.

**Dropout is not pinned across variants.** Every variant starts from the same seed, but
different attention modules draw different numbers of random values, so the masks
diverge after the first block that differs. The default `--dropout 0` keeps the
comparison strictly like-for-like. Data and data order *are* pinned and verified — see
`DataPlan` in the script.

---

## `summary.json`

Written next to the TensorBoard runs. Holds the environment (`torch` version, CUDA,
git commit), the resolved arguments, the shared data fingerprint, and per variant: the
full config, parameter counts, the `(step, elapsed, val_loss)` curve, timings, peak
memory, and the final resolved diagnostics dict.

Use it for offline plotting and for confirming after the fact that two runs saw the
same data — the fingerprint is an order-sensitive digest of every token every variant
was shown.
