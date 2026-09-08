# Attention Diagnostics Reference

Every scalar `compare_models.py` writes to TensorBoard, what it measures, and what
it tells you when it moves.

```bash
python compare_models.py --variant "h8:heads=8" --variant "h16:heads=16" ...
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
- [`attn/*` and `ffn/*` — sublayer health](#attn-and-ffn--sublayer-health)
- [`loss/*` — quality](#loss--quality)
- [`optim/*` — optimisation](#optim--optimisation)
- [`Gradients/*` — where the gradient is going](#gradients--where-the-gradient-is-going)
- [`perf/*` — cost](#perf--cost)
- [TEXT — what produced the curves](#text--what-produced-the-curves)
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

Nothing in the diagnostic code knows what is inside an attention or feed-forward module.
Everything is read from forward hooks, from five tensors per decoder block:

| tensor | what it is |
|---|---|
| attention `x` | what the attention sublayer was handed — `norm1(x)` under the `pre-*` and `rootdepth-*` strategies, the raw residual stream under `post-*` and `deepnorm` |
| attention `update` | what attention wrote back, **before** the residual add |
| feed-forward `x` | what the feed-forward sublayer was handed — `norm2(x)` under `pre-*`/`rootdepth-*`, the post-attention stream under `post-*`/`deepnorm` |
| feed-forward `update` | what the feed-forward wrote back, **before** the residual add |
| block output | what left the block, after attention, FFN and both residuals |

No probe reaches inside a module, looks up an attribute, or recomputes a score. That
is deliberate. A diagnostic that knows the internals of one variant produces a tag the
other variant cannot report — which is precisely the tag that cannot be compared. It
also means these numbers survive any change to either sublayer's implementation.

The two sublayers are found by attribute name (`ATTENTION_HINTS`, `FEEDFORWARD_HINTS`),
and a block where either match is not exactly one **fails the run** rather than emitting
an empty series — an empty series does not look like a bug, it looks like a real
difference between the arms.

---

## `collapse/*` — representation collapse

Mean pairwise cosine similarity between token representations.

| tag | measured on |
|---|---|
| `collapse/input` | the tensor attention reads |
| `collapse/mid` | the tensor the feed-forward reads, i.e. after attention has written back |
| `collapse/output` | the decoder block's output |

```
~0    tokens stay spread out
→1    tokens have collapsed onto each other; depth is no longer buying anything
```

At the extreme every position carries the same vector, so the LM head can only emit
position-independent predictions.

**Read them as a trace per layer.** `mid − input` is what *attention* added to the
collapse and `output − mid` is what the *feed-forward* added, so the three together
attribute a block's collapse to the branch that caused it. `output − input` is the
block's total, and layer *n*'s output is layer *n+1*'s input. A single block driving
most of the collapse is a different problem from every block contributing evenly — and
one *branch* driving it is different again.

At init on a 6-layer model, attention adds roughly +0.002–0.003 per layer and the
feed-forward adds ~0, so the collapse that shows up in `collapse/output` starts out
almost entirely attention's doing.

Rising monotonically with depth is normal, especially early in training. Rising
monotonically *with steps* at a fixed layer is the thing to worry about.

---

## `attn/*` and `ffn/*` — sublayer health

Both residual branches carry the **same four measurements under the same definitions**,
computed by one shared hook body so they cannot drift apart:

| tag | attention | feed-forward |
|---|---|---|
| `input_norm` | `attn/input_norm` | `ffn/input_norm` |
| `update_ratio` | `attn/update_ratio` | `ffn/update_ratio` |
| `update_cos` | `attn/update_cos` | `ffn/update_cos` |
| `update_isotropy` | `attn/update_isotropy` | `ffn/update_isotropy` |

That is the point of having both: a block-level number says something changed, and the
pair says **which branch did it**. Pair them with `collapse/input → mid → output` for
the same attribution on collapse.

⚠️ **The two families share definitions but not resting values.** See
[Caveats](#caveats) before reading a difference between `attn/x` and `ffn/x` as a
finding.

Below, `<s>` stands for either prefix.

### `<s>/input_norm`

Mean ‖x‖ over valid tokens — residual-stream drift, and the scale every other
magnitude here is relative to.

Under `pre-*` and `rootdepth-*`, every layer reads a normalized tensor, so all layers sit
near √`embed_dim` and this tag tracks the **norm gains** growing rather than the residual
stream itself. Under `post-*` and `deepnorm` it tracks the raw residual stream, and block 0
reads the bare embedding — a much smaller number than every later block.

`attn/input_norm` and `ffn/input_norm` read the same value under `pre-*`/`rootdepth-*`
(both are normalized), so a gap between them there means the norm gains of `norm1` and
`norm2` have diverged. Under `post-*` they legitimately differ: the feed-forward reads
the stream after attention has already written to it.

### `<s>/update_ratio`

`mean(‖update‖) / mean(‖x‖)` — the sublayer's gain into the residual stream. (A ratio
of means, not a mean of ratios.)

```
≫1     the block is shouting over the residual stream
~0.2–1 healthy
→0     the block has switched itself off; the residual path is routing around it
```

This is the first thing to check when one variant's `Gradients/Global` sits an order of
magnitude off another's — pair it with that block's `Gradients/Decoder<i>` to tell a loud
block from an unstable one.

### `<s>/update_cos`

`mean(cos(update_i, x_i))` over valid tokens — a mean of per-token cosines.

```
~0    writing genuinely new content
→1    mostly rescaling what each token already held
<0    actively subtracting each token's own direction (a de-correlating operation)
```

Both branches end in an output projection that rotates the update out of the span of its
input, so **both rest near 0 at init** — measured 0.0002 and −0.0038 on a 6-layer model.
What a *rise* means is where they part:

- **`attn/update_cos` → 1** is the serious one. Attention's job is to write *a combination
  of other positions* into position *i*; if the update aligns with what *i* already held,
  it has stopped moving information between positions. This is the failure a loss curve
  hides longest — the FFN keeps improving while attention quietly does nothing useful.
- **`ffn/update_cos` → 1** is weaker. The feed-forward is position-wise and never moved
  information between tokens, so a rise only says the branch has decayed toward rescaling
  in place rather than transforming.

⚠️ **Architecturally confounded — see [Caveats](#caveats).**

### `<s>/update_isotropy`

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

⚠️ This is the pair with the **largest** structural gap between the two branches. At init
on a 6-layer model, `attn/update_isotropy` measures 0.19 and `ffn/update_isotropy` 0.67:
attention averages over positions with near-uniform weights, which is close to rank one,
while the feed-forward's update arrives through a random down-projection. That 3.5×
difference is the two designs, not a fault in either.

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

### `loss/gap`

`loss/val − loss/train` at the same evaluation. A variant whose gap opens wider than the
baseline's is buying its win by fitting the batch list harder rather than by learning —
read it against `params` before crediting the architecture. Because the train side is a
window mean taken in training mode, compare gaps **between** arms and watch the trend;
the absolute value reads low and can be negative early in a run.

### `loss/aulc`

Running area under the validation curve over steps: at each evaluation, the mean `loss/val`
of the run **so far**. Lower means an arm was ahead throughout, not only at the last
evaluation. Charted rather than reported once at the end because the step where one arm's
trace crosses another's is the step its lead actually began.

Read it **with** the final loss, not instead of it — the area is weighted toward early
training, so an arm that merely starts better can hold the lower AULC and still end up
worse. The two disagreeing is the finding: it says the arms differ in convergence *speed*
rather than in where they land. `report()` also prints a seconds-axis AULC truncated to a
budget every arm reached; that one is not a scalar tag.

---

## `optim/*` — optimisation

### `optim/lr`

Current learning rate from the linear-warmup-then-cosine schedule. A sanity check that
warmup and decay landed where `--warmup-frac` and `--steps` put them.

---

## `Gradients/*` — where the gradient is going

Gradient norms taken **before** clipping, at the same step as each evaluation, bucketed by
`utils.component_key` — the same rule `train.py` uses, so a comparison run's gradient series
reads directly against a real training run's.

| tag | covers |
|---|---|
| `Gradients/Global` | root of the summed squares over every component — what `clip_grad_norm_` measures against `--grad-clip` |
| `Gradients/Embedding` | the input embedding table |
| `Gradients/Decoder<i>` | every parameter in decoder block *i* |
| `Gradients/Projection` | the output head |
| `Gradients/NormF` | everything else — `norm_f` alone in `GPTmodel`, plus any top-level layers a subclass adds |

If `Gradients/Global` sits above `--grad-clip` (default 1.0) for a whole run, most steps are
being clipped and you are not training at the LR you set — you are taking direction-only
steps. Correlate with `attn/update_ratio` and `ffn/update_ratio` when a variant misbehaves:
the ratios say which block *and which branch* is loud, the gradient says whether it is also
unstable.

The components add up to the number clipping acted on, so a component's share is directly
comparable to its share of the parameters in `ModelCensus` — a block holding 30% of the
weights and 3% of the gradient has stopped learning, and neither number says so alone.

Single-step samples, not window averages, so expect them to be noisy.

---

## `perf/*` — cost

| tag | meaning |
|---|---|
| `perf/elapsed_sec` | cumulative training seconds, excluding evaluation, warmup and compilation |
| `perf/ms_per_step` | cumulative average, `elapsed / step`. Flat means steady state; a rising curve means something is growing |
| `perf/tokens_per_sec` | non-pad input tokens per second — the throughput number to quote, since it normalises away padding differences |
| `perf/tflops` | achieved TFLOP/s, from the dispatch-measured FLOPs of one training step |

Read `perf/tflops` beside `perf/tokens_per_sec`: they come apart exactly where the
interesting answer is. An arm can trail on tok/s because it does more arithmetic per token,
or because it is launch-bound and leaving the device idle, and only the achieved-FLOPs
number tells those apart.

---

## TEXT — what produced the curves

Each run writes five payloads to the **TEXT** tab, all at step 0, as fenced JSON:

| tag | holds |
|---|---|
| `ModelConfig` | the variant's fully resolved `ModelConfig` — every field, after overrides |
| `ComparisonConfig` | the whole invocation: every CLI argument, the variant list, and `base_config` |
| `Variant` | this arm's label, model class, its overrides alone, and the data fingerprint |
| `Environment` | python, platform, torch, CUDA, cuDNN, and the git commit |
| `ModelCensus` | parameter counts bucketed by `component_key`, plus measured FLOPs per step broken down by op |

This is what lets a chart read back to the class, overrides, commit, parameter breakdown and
measured FLOPs that produced it.

⚠️ `Environment.git_commit` is the commit hash only — it does **not** record whether the
working tree was dirty. A run produced by uncommitted code reports the hash of the commit it
was based on, so two runs with different behaviour can carry the same hash.

There is no **HPARAMS** tab: `compare_models.py` writes no `add_hparams` call. To sort a
sweep by final validation loss, use the table `report()` prints, or `summary.json`.

---

## Reading them together

The diagnostic that identifies a *cause* is usually a combination.

| pattern | reading |
|---|---|
| `Gradients/Global` high **+** either `update_ratio` high | that branch is over-writing the residual stream — lower the LR or check its output scale |
| `collapse` rising **+** `update_isotropy` falling | genuine representational collapse |
| `collapse` flat **+** `attn/update_cos` → 1 | attention has stopped mixing positions; the FFN is carrying the model — confirm with `ffn/update_ratio` holding up while `attn/update_ratio` decays |
| either `update_ratio` decaying → 0 | that branch is being routed around; its parameters are dead weight |
| `collapse` flat **+** `attn/update_isotropy` falling | heads converging onto a single operator |
| `attn/update_ratio` ≫ `ffn/update_ratio` (or the reverse) | the block is effectively one branch; the other is along for the ride |
| `mid − input` ≫ `output − mid` (or the reverse) | one branch is driving the block's collapse — read that branch's `update_*` next |
| `input_norm` growing without bound | residual-stream blow-up — check the norm gains |
| `attn/input_norm` ≠ `ffn/input_norm` under `pre-*` | `norm1` and `norm2` gains have diverged (they read the same scale otherwise) |
| one layer's `/max` far from `/mean` | a single-layer anomaly; open the per-layer chart before concluding anything |

---

## Caveats

**`attn/update_cos` is architecturally confounded.** Standard MHA passes its update
through `W_v`/`W_o`, which rotate the output out of the span of the inputs, so ~0 is
the natural resting value. An attention variant whose value *is* the residual stream
(no `W_v`/`W_o`) is structurally confined to the span of its inputs and will sit far
higher — that is a property of the design, not a pathology. **Compare a variant against
its own trajectory, not against a different architecture at a point in time.**

**`attn/*` and `ffn/*` are not directly comparable to each other.** They share
definitions, which is what makes each readable against its own history and against the
other arms — it is *not* a licence to read `ffn/update_isotropy > attn/update_isotropy`
as the feed-forward being healthier. The branches have different resting values by
construction (0.19 vs 0.67 for isotropy at init on a 6-layer model). Use the pair to
attribute a *change* to a branch; do not rank the branches against each other at a point
in time.

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
