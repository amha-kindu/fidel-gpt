# Training Diagnostics Reference

Every tag the training monitor and the registered probes write to TensorBoard: what it
measures, how it is computed, what it reads at initialisation, and what it means when it
moves.

```bash
tensorboard --logdir <tb_log_dir>
```

---

## Table of Contents

- [Training Diagnostics Reference](#training-diagnostics-reference)
  - [Table of Contents](#table-of-contents)
  - [The tree at a glance](#the-tree-at-a-glance)
  - [When things are written](#when-things-are-written)
  - [Conventions](#conventions)
  - [`loss/*` — quality](#loss--quality)
    - [`loss/curves` `{train, val}`](#losscurves-train-val)
    - [`loss/gap`](#lossgap)
  - [`optim/*` — the optimiser](#optim--the-optimiser)
  - [`param/*` — where the gradient goes and what it moves](#param--where-the-gradient-goes-and-what-it-moves)
    - [`param/grad/<group>`](#paramgradgroup)
    - [`param/norm/<group>`](#paramnormgroup)
    - [`param/update_ratio/<group>`](#paramupdate_ratiogroup)
  - [`perf/*` — cost](#perf--cost)
  - [Gradient noise — `param/gsnr/*` and `optim/noise_scale`](#gradient-noise--paramgsnr-and-optimnoise_scale)
    - [`param/gsnr/<group>`](#paramgsnrgroup)
    - [`optim/noise_scale`](#optimnoise_scale)
    - [Cost and settings](#cost-and-settings)
  - [The probes](#the-probes)
    - [How a probe pass works](#how-a-probe-pass-works)
    - [The shared measurements](#the-shared-measurements)
    - [`embedding/*` — what the embedding table hands the stack](#embedding--what-the-embedding-table-hands-the-stack)
    - [`attn/*` and `ffn/*` — sublayer health](#attn-and-ffn--sublayer-health)
      - [`<s>/input_norm`](#sinput_norm)
      - [`<s>/update_ratio`](#supdate_ratio)
      - [`<s>/update_cos`](#supdate_cos)
      - [`<s>/update_isotropy`](#supdate_isotropy)
    - [`stream/*` — the residual stream between blocks](#stream--the-residual-stream-between-blocks)
      - [`stream/norm`](#streamnorm)
      - [`stream/collapse`](#streamcollapse)
    - [`head/*` — what the output head reads and how sharply it answers](#head--what-the-output-head-reads-and-how-sharply-it-answers)
  - [TEXT — `ModelCensus`](#text--modelcensus)
  - [Reading them together](#reading-them-together)
  - [Caveats](#caveats)
  - [Adding a probe](#adding-a-probe)

---

## The tree at a glance

`<group>` is one of `Embedding`, `Decoder<i>` (one per block), `Projection`, `NormF`.
Every probe tag ends in a layer suffix, described under [Conventions](#conventions).

```
loss/
├── curves {train, val}         train and val loss on ONE chart
└── gap                         val − train
optim/
├── lr                          learning rate
├── grad_norm                   mean pre-clip global gradient norm over the window
├── clip_frac                   share of the window's steps that were clipped
├── grad_scale                  AMP loss scale (mixed precision only)
└── noise_scale                 gradient noise scale ≈ critical batch size, in sequences
param/
├── grad/<group>                ‖grad‖, before clipping
├── norm/<group>                ‖W‖, before the optimiser step
├── update_ratio/<group>        ‖W_after − W_before‖ / ‖W_before‖ across one step
└── gsnr/<group>                gradient signal-to-noise ratio, per sequence
perf/
├── tokens_per_sec              non-pad input tokens per second of training
└── tflops                      achieved TFLOP/s
embedding/                      the embedding table's output             (layer_00)
├── embed_std                   per-token spread across channels
├── isotropy                    share of the available directions in use
└── collapse                    mean pairwise cosine between distinct token ids
attn/                           the attention sublayer, per block
├── input_norm                  ‖input‖ — norm1's gains under pre-norm
├── update_ratio                ‖update‖ / ‖stream‖
├── update_cos                  cos(update, stream)
└── update_isotropy             share of the available directions the update uses
ffn/                            the feed-forward sublayer, per block — same four
├── input_norm                  ‖input‖ — norm2's gains under pre-norm
├── update_ratio
├── update_cos
└── update_isotropy
stream/                         each block's output, per block
├── norm                        ‖x‖ of the residual stream
└── collapse                    mean pairwise cosine between tokens
head/                           the output projection                    (layer_00)
├── input_norm                  ‖input‖ — norm_f's gains under pre-norm
├── isotropy                    of what the head reads
├── collapse                    of what the head reads
└── logit_std                   per-token spread of the logits
TEXT
└── ModelCensus                 parameter counts per group, FLOPs per batch by op
```

Read in forward-pass order, the probe families trace one token's path through the model:
`embedding/` → for each block `attn/` → `ffn/` → `stream/` → `head/`.

---

## When things are written

| what | when | measured on |
|---|---|---|
| `loss/curves {train}`, `optim/*`, `param/*`, `perf/*` — except the two below | once per **window** of `log_every` optimiser steps (default 100) | the training batches |
| `loss/curves {val}`, `loss/gap` | at every validation | the validation set |
| `embedding/*` `attn/*` `ffn/*` `stream/*` `head/*` | at every validation | one **fixed probe batch**, drawn once from the validation data at the start of the run, with its labels |
| `param/gsnr/*`, `optim/noise_scale` | at every validation | the same probe batch, split into `gsnr_chunks` chunks |
| `ModelCensus` | once, at step 0 | — |

Every step is an **optimiser step**: under gradient accumulation, one step spans
`grad_accum_steps` micro-batches.

Only the coordinator rank writes. Window accumulators live on the device and are read
back once per window, so no training step pays a device sync for telemetry except the one
that closes a window.

---

## Conventions

**Per layer, always.** A probe measured in more than one decoder block is written once per
block, plus three reductions over blocks:

```
stream/collapse/layer_00   stream/collapse/layer_01   ...
stream/collapse/mean       stream/collapse/min        stream/collapse/max
```

A mean alone hides a single bad layer: one collapsing block in a six-block stack moves the
mean by a sixth of its excursion. `/min` and `/max` make a single-layer anomaly visible on a
summary chart; when they part from `/mean`, open the per-layer chart.

A probe measured in **one place** — `embedding/*`, `head/*` — has only `layer_00`. Its mean,
min and max would be three more copies of the same number.

**Masked reductions.** Every probe reduction runs over valid (non-pad) positions only.
Padding tokens share one embedding, so pad–pad pairs would sit at cosine ≈ 1 and inflate
every similarity; padding positions can also hold inf or NaN, which a masked mean must not
touch.

**Floors.** Ratios divide by a quantity clamped at `1e-12`, so a dead component reads as a
bounded wrong answer rather than poisoning a mean with inf.

---

## `loss/*` — quality

### `loss/curves` `{train, val}`

Train and validation loss **on the same chart**, as two lines.

| line | value |
|---|---|
| `train` | mean training loss over the window: the average, over the window's optimiser steps, of each step's loss (itself the mean over its micro-batches). Measured in training mode, with dropout on |
| `val` | the validation loss at that validation, in eval mode |

The two lines are written at different steps — `train` every `log_every` steps, `val` at
every validation — and TensorBoard draws each at its own steps.

### `loss/gap`

`val − train`, where `train` is the most recent **closed** window's mean (or, before the
first window closes, the open one's).

A widening gap is the model fitting the training batches faster than it is learning.
Because `train` is a window mean taken with dropout on, the absolute value reads low and
can be negative early in a run; watch the trend.

---

## `optim/*` — the optimiser

| tag | value | read it as |
|---|---|---|
| `optim/lr` | the scheduler's learning rate after the window's last step — the rate the **next** step will use | a sanity check that warmup and decay land where the schedule puts them |
| `optim/grad_norm` | mean, over the window's steps, of the global gradient norm **before** clipping (after unscaling under AMP) | the gradient's size the optimiser was fed; spikes are the instability to look for |
| `optim/clip_frac` | fraction of the window's steps whose pre-clip norm exceeded `max_norm` | whether `optim/lr` is telling the truth |
| `optim/grad_scale` | the AMP loss scale at the window's end (mixed precision only) | repeated halving means steps are producing inf/NaN and being skipped |

**`optim/clip_frac` near 1** means almost every step is rescaled to the clip threshold: the
step size is set by `max_norm`, not the learning rate, and the effective learning rate is
not the one on `optim/lr`. Nothing else in the tree shows that.

**`optim/grad_scale` falling** means the run is training on fewer steps than its x-axis
says. A skipped step also reads as `param/update_ratio` exactly 0 if it lands on a window's
last step.

---

## `param/*` — where the gradient goes and what it moves

Three families over one decomposition of the parameters, all taken at the **last step of
each window**:

| group | covers |
|---|---|
| `Embedding` | the input embedding table |
| `Decoder<i>` | every parameter in decoder block *i* |
| `Projection` | the output head |
| `NormF` | everything else — `norm_f` in the standard model |

### `param/grad/<group>`

‖grad‖ per group, after unscaling and **before** clipping, so it describes the gradient the
step produced rather than what clipping let through. The groups' root summed square is that
step's global pre-clip norm — one sample of what `optim/grad_norm` averages.

Compare a group's share of the gradient with its share of the parameters in
`ModelCensus`: a block holding 30% of the weights and 3% of the gradient has stopped
learning, and neither number says so alone.

### `param/norm/<group>`

‖W‖ per group, before the optimiser step — named to mirror `stream/norm`. On its own it
mostly tracks initialisation and weight decay; its job is to be the denominator of the
ratio below.

### `param/update_ratio/<group>`

`‖W_after − W_before‖ / ‖W_before‖` across one optimiser step: the fraction of its own
magnitude a group **actually moved**.

```
~1e-3        healthy
≪1e-3        this group has stopped learning while the loss falls on the strength of the others
≫1e-3        this group is about to destabilise the run
```

It is **measured, not predicted from the gradient**. Under AdamW each element moves by
roughly `lr` whatever its gradient's size, plus the decoupled weight decay, so
`lr · ‖grad‖ / ‖W‖` would track the gradient's scale rather than the step. The measured
update already includes clipping, Adam's normalisation, weight decay and the scheduled
learning rate.

The only **scale-free** number in this section, and therefore the only one comparable
*between groups* and *between runs* of different widths or initialisations.

Only trainable parameters contribute to the update; `param/norm` counts every parameter.
With part of a group frozen (LoRA, partial fine-tuning), the ratio is the trainable part's
movement relative to the whole group.

Costs one copy of the trainable weights per window.

All three are single-step samples, so expect them to be noisier than the window means in
`optim/*`.

---

## `perf/*` — cost

| tag | value |
|---|---|
| `perf/tokens_per_sec` | non-pad input tokens processed in the window, divided by the window's training seconds. This rank's count times the world size |
| `perf/tflops` | achieved TFLOP/s: the measured FLOPs of one batch (see `ModelCensus`) × micro-batches per step × world size × steps in the window, over the window's training seconds |

Both are **per window**, not cumulative, so a slowdown partway through a long run shows up
rather than being averaged away. Time spent validating, running the probes and
snapshotting checkpoints is excluded; the clock syncs the device at every boundary, so it
measures compute, not kernel-launch queue time.

Read them together. `tokens_per_sec` says how fast data moves; `tflops` says how much of
the machine is in use. A model can trail on tokens/s because it does more arithmetic per
token, or because it is launch-bound and leaving the device idle — only the achieved-FLOPs
number tells those apart.

FLOPs are counted on every position, padding included, since padding costs the same
compute. Tokens count non-pad inputs only.

---

## Gradient noise — `param/gsnr/*` and `optim/noise_scale`

How much of the gradient is **signal** — the direction the whole data distribution agrees
on — and how much is **noise** from which sequences happened to be sampled. Nothing else in
the tree separates the two: `param/grad/*` and `optim/grad_norm` measure the gradient's
size, which signal and noise both contribute to.

Measured at every validation, on the fixed probe batch, as a step of its own:

1. The probe batch is split into `K = gsnr_chunks` equal chunks of `b` sequences (default
   `K = 8`).
2. Each chunk's gradient `g_k` of the plain cross-entropy loss is taken separately, in eval
   mode, without touching the training step's gradients.
3. Per group, from the chunk gradients:

```
tr(Σ_b)  =  K/(K−1) · ( mean_k ‖g_k‖²  −  ‖ḡ‖² )        noise:  unbiased trace of the chunk covariance
‖G‖²     =  ‖ḡ‖²  −  tr(Σ_b) / K                          signal: ‖ḡ‖² with its own noise removed
tr(Σ₁)   =  b · tr(Σ_b)                                   noise of ONE sequence
```

The correction on `‖G‖²` matters: the mean of K noisy gradients still carries `tr(Σ_b)/K`
of noise, and left in, a small K overstates the signal. Scaling the noise to one sequence
makes both tags independent of the probe batch's size and of K.

### `param/gsnr/<group>`

`‖G‖² / tr(Σ₁)` — the gradient signal-to-noise ratio of one group, per sequence.

```
≳ 1      a single sequence's gradient already points the right way
≪ 1      the group is moving mostly on sampling noise
0        the noise swallowed the signal estimate entirely (it came out ≤ 0)
```

A **ratio of sums** over the group, not a mean of per-parameter ratios: per-parameter GSNR
is heavy-tailed, and a few near-zero-variance weights would dominate a mean. Groups are the
`param/*` groups.

GSNR falls as training proceeds — the easy, shared directions are learned first — and a
sustained fall alongside a widening `loss/gap` is the gradient turning into noise the model
then fits (Liu et al., 2020). Groups differ by orders of magnitude by design: a final norm's
few gains see a gradient most sequences agree on; an embedding row sees a gradient only
from the sequences containing its token.

### `optim/noise_scale`

`tr(Σ₁) / ‖G‖²` over the whole model, **in sequences** — the gradient noise scale
`B_simple` (McCandlish et al., 2018). It estimates the **critical batch size**: below it, a
larger batch buys a proportionally better gradient and so fewer steps; above it, extra
sequences per step are mostly wasted compute.

Compare it with the sequences per optimiser step (`batch_size × grad_accum_steps ×` world
size). It typically **grows** as the loss falls, so a batch size that was right early can
become too small late. Omitted at a validation where the signal estimate is not positive —
common at initialisation and on data with no learnable structure.

On a small two-block model on real data, it read ≈ 5.4, 6.3 and 6.7 sequences at three
successive validations, while per-group GSNR fell from its first reading.

### Cost and settings

`K` forward + backward passes over the probe batch per validation, plus one fp32 copy of the
trainable weights. `gsnr_chunks < 2` disables both tags.

Each chunk is `probe batch ÷ K` sequences, so a small probe batch gives few, small chunks
and a noisy estimate. A larger probe batch, or reading the trend over several validations
rather than one point, steadies it.

---

## The probes

### How a probe pass works

At every validation, one forward pass runs on the **fixed probe batch** in eval mode
(dropout off) with gradients off. Forward hooks on the probed modules measure what each
module was handed and what it produced, then everything is read back to the host in a
single transfer.

The batch is fixed for the whole run, so a probe's curve moves with the model rather than
with the data.

**What each probe reads**

| tensor | what it is |
|---|---|
| embedding output | the table lookup, before dropout |
| block input | the residual stream entering the block — by definition, whatever is inside it |
| attention input | what attention was handed: `norm1(x)` under `pre-*`, the stream itself under `post-*` and `deepnorm` |
| attention update | what attention wrote back, **before** the residual add |
| feed-forward input | `norm2(x)` under `pre-*`; the post-attention stream under `post-*` and `deepnorm` |
| feed-forward update | what the feed-forward wrote back, before the residual add |
| block output | what left the block, after both residual adds |
| head input | `norm_f(x)` under `pre-*`; the last block's output under `post-*` and `deepnorm` |
| logits | the head's output |

**Finding the stream.** `attn/*` and `ffn/*` measure each update against the **residual
stream it is added to**, not against the sublayer's input. The stream at a block's input is
recorded before the block runs. If attention was handed a different tensor — a normalised
copy, i.e. pre-norm — its update lands on that stream, and the feed-forward's stream is
block input + attention update (exact in eval mode, where dropout is the identity). If
attention was handed the stream itself (post-norm, DeepNorm), the feed-forward's own input
is the stream too. Only tensor identity is inspected, never the norm strategy, so the same
code reads every placement.

### The shared measurements

Four quantities recur across families, each defined once.

**norm** — mean ‖x‖ over valid tokens.

**collapse** — mean pairwise cosine similarity between token representations, over pairs
**within each sequence**, excluding self-pairs and any pair touching padding.

```
~0    tokens stay spread out
→1    tokens have collapsed onto each other; depth is no longer buying anything
<0    tokens actively pushed apart
```

At the extreme every position carries the same vector, and the head can only emit
position-independent predictions.

**isotropy** — how evenly the tokens spread their energy over the available directions:
the participation ratio of the centred token covariance spectrum, (Σλ)² / Σλ² (the
effective rank), divided by the most directions the tensor could span,
`min(valid tokens, width)`.

```
1     isotropic: every available direction carries the same energy
→0    anisotropic: the tokens live on a handful of directions, however wide the model is
```

This catches what collapse cannot. Mean pairwise cosine says how aligned tokens are with
*each other*; a tensor can hold that near zero while putting every token in the same
two-dimensional subspace. Computed from covariance traces, with no eigendecomposition.

**spread** (`embed_std`, `logit_std`) — the per-token standard deviation across the last
axis, averaged over valid tokens (population std, no Bessel correction).

---

### `embedding/*` — what the embedding table hands the stack

The head's mirror image, read off the embedding table's output. Measured once (`layer_00`).

| tag | measures | at init |
|---|---|---|
| `embedding/embed_std` | per-token spread of the embedding across channels | ≈ the init std (`embed_std`, 0.02 by default) |
| `embedding/isotropy` | share of the available directions the embedded tokens use, weighted by token frequency — the distribution block 0 actually sees | ≈ 0.9: random rows |
| `embedding/collapse` | mean pairwise cosine between **distinct** token ids in the probe batch | ≈ 0 |

**`embed_std`** is the scale block 0's residual stream starts at, and so what block 0's
`update_ratio`s are measured against. Watch it against `stream/norm` at layer 0: under
pre-norm the embedding is a small fraction of the stream from the first block on.

**`collapse` uses distinct ids**, one vector per id present in the batch, with pairs taken
across the whole batch. Repeats of one id embed identically, so the per-sequence pairs used
elsewhere would mostly measure how often the batch repeats itself, not how far apart the
table keeps its words. Rising means the table is pulling its words together.

There is no `input_norm` here: the table's input is token ids.

---

### `attn/*` and `ffn/*` — sublayer health

Both residual branches carry the **same four measurements under the same definitions**,
computed by one shared function so they cannot drift apart. That is what makes the pair
useful: a block-level number says something changed, and the pair says **which branch did
it**.

| tag | attention | feed-forward |
|---|---|---|
| `input_norm` | `attn/input_norm` | `ffn/input_norm` |
| `update_ratio` | `attn/update_ratio` | `ffn/update_ratio` |
| `update_cos` | `attn/update_cos` | `ffn/update_cos` |
| `update_isotropy` | `attn/update_isotropy` | `ffn/update_isotropy` |

Below, `<s>` stands for either prefix.

#### `<s>/input_norm`

Mean ‖input‖ of what the sublayer was **handed**.

Under `pre-*` that is `norm1(x)` for attention and `norm2(x)` for the feed-forward, whose
size is set entirely by the norm's learned gains: the stream's own scale is divided out.
So this pair, with `head/input_norm` for `norm_f`, **tracks every norm in the model**. It
reads √`embed_dim` at init (unit gains), and moves exactly as the gains learn.

Under `post-*` and `deepnorm` the sublayer reads the stream itself, so this repeats a
`stream/norm` value rather than reading any gains.

#### `<s>/update_ratio`

`mean(‖update‖) / mean(‖stream‖)` — the sublayer's gain into the residual stream (a ratio
of means, not a mean of ratios).

```
≫1     the branch is shouting over the stream
→0     the branch has switched itself off; the residual path is routing around it
```

Under pre-norm, block 0's stream is the bare embedding — about `embed_std · √embed_dim`,
small — so block 0's ratios read well above 1 at init. That is real: early in training the
branches swamp the embedding. On a six-block, 64-wide pre-norm model at init,
`attn/update_ratio` measured ≈ 9 at layer 0 and ≈ 0.5 at layer 5, as the stream grew from
≈ 1.9 to ≈ 12. Read each block against its own trajectory before reading an absolute value
as a fault.

#### `<s>/update_cos`

`mean(cos(update_i, stream_i))` over valid tokens — a mean of per-token cosines.

```
~0    writing genuinely new content
→1    mostly reinforcing what each token already held
<0    subtracting each token's own direction
```

Both branches end in an output projection that rotates the update out of the span of its
input, so both rest near 0 at init (within ±0.03 on a six-block model). What a *rise* means
differs:

- **`attn/update_cos` → 1** is the serious one. Attention's job is to write a combination of
  *other* positions into position *i*; an update aligned with what *i* already held means it
  has stopped moving information between positions. It is the failure the loss curve hides
  longest — the feed-forward keeps improving while attention quietly does nothing.
- **`ffn/update_cos` → 1** is weaker. The feed-forward is position-wise and never moved
  information between tokens, so a rise only says it has decayed toward rescaling in place.

#### `<s>/update_isotropy`

[Isotropy](#the-shared-measurements) of the update. Falling means the branch is writing
everything into a handful of directions — a branch that has stopped using the width it was
given, which collapse cannot see.

⚠️ The two branches rest at **different values by construction**. At init on a six-block,
64-wide model, `attn/update_isotropy` measured 0.12–0.14 and `ffn/update_isotropy` 0.29:
attention averages over positions with near-uniform weights, close to rank one, while the
feed-forward's update comes through a random down-projection. That gap is the two designs,
not a fault in either.

---

### `stream/*` — the residual stream between blocks

Read off each decoder block's **output** — every block, including the last. Block *l*'s
output is block *l + 1*'s input, so one reading per block covers the stream between every
pair of blocks, and the last block's output is what `norm_f` and the head read.

| tag | measures |
|---|---|
| `stream/norm` | mean ‖x‖ of the residual stream — the scale every `attn/*` and `ffn/*` update is measured against |
| `stream/collapse` | [collapse](#the-shared-measurements) of the stream, within each sequence |

#### `stream/norm`

Under `pre-*` the stream is never normalised, so it grows with depth as each branch adds to
it; that growth is what pre-norm does not bound, and the thing to watch. Under `post-*` and
`deepnorm` every block ends in a norm, so it sits near √`embed_dim` times that norm's gains.

#### `stream/collapse`

Read it as a **trace over depth**. The step from layer *l − 1* to layer *l* is what block *l*
added to the collapse; a single block driving most of it is a different problem from every
block contributing evenly. To tell which branch inside a block did it, read that block's
`attn/update_cos` and `update_isotropy` alongside `ffn/`'s.

Rising with depth is normal, especially early in training. Rising **with steps** at a fixed
layer is the thing to worry about.

---

### `head/*` — what the output head reads and how sharply it answers

Measured once, at the output projection (`layer_00`). A forward hook sees no labels, so
nothing target-relative — accuracy, calibration, per-token loss — is here.

| tag | measures |
|---|---|
| `head/input_norm` | mean ‖input‖ of what the head reads. Under `pre-*` that is `norm_f`'s output, so it tracks `norm_f`'s learned gains (√`embed_dim` at init); under `post-*` and `deepnorm` it is the last block's output |
| `head/isotropy` | [isotropy](#the-shared-measurements) of the head's input |
| `head/collapse` | [collapse](#the-shared-measurements) of the head's input |
| `head/logit_std` | per-token spread of the logits across the vocabulary |

**`head/isotropy`** is the head's raw material: a rank-starved input caps what any head can
discriminate, however well trained. Read it against the last block's
`ffn/update_isotropy` to see whether the narrowing happened in the stack or at `norm_f`.

**`head/collapse` against the last layer of `stream/collapse`** is what `norm_f` does to
collapse. Under `-rms`, `norm_f` only rescales each token and weights channels by its gains,
so the two start equal and part only as those gains learn. Under `-ln` the per-token
centring moves it as well.

**`head/logit_std`** is the temperature the head has taught itself. Rising while
`loss/curves {val}` is flat is a head sharpening rather than learning.

---

## TEXT — `ModelCensus`

Written once, at step 0, as fenced JSON in the TEXT tab.

| field | holds |
|---|---|
| `total` | parameter count, each shared tensor counted once |
| `non_embedding` | `total` minus both vocabulary-sized tables — usually the honest capacity number |
| `vocabulary` | the embedding table plus the output projection |
| `bytes` | parameter memory |
| `aliased_tensors` | parameter slots that alias another — how weight tying shows up; 0 means the tables are separate |
| `by_component` | parameter counts per `param/*` group |
| `flops.forward`, `flops.backward`, `flops.total` | FLOPs for one training batch, measured by dispatch on the probe batch |
| `flops.per_position` | `flops.total` per sequence position |
| `flops.attention_counted` | whether the attention kernel's FLOPs were counted (see [Caveats](#caveats)) |
| `flops.by_op` | FLOPs per operator, largest first |

`by_component` uses the same groups as `param/*`, so a group's share of the parameters reads
directly against its share of the gradient.

---

## Reading them together

The diagnostic that identifies a *cause* is usually a combination.

| pattern | reading |
|---|---|
| `optim/grad_norm` spiking **+** one branch's `update_ratio` high | that branch is over-writing the stream — lower the learning rate or check its output scale |
| `optim/clip_frac` → 1 | the step size is set by `max_norm`, not `optim/lr` |
| `param/update_ratio/<group>` ≪ others **+** `param/grad/<group>` small | that group has stopped learning |
| `param/gsnr/<group>` ≪ 1 **+** `param/update_ratio/<group>` healthy | that group is moving, but mostly on noise |
| `param/gsnr/*` falling **+** `loss/gap` widening | the gradient is turning into noise the model is fitting |
| `optim/noise_scale` ≫ sequences per step | a larger batch would buy fewer steps; the batch is below the critical size |
| `optim/noise_scale` ≪ sequences per step | extra sequences per step are mostly wasted compute |
| `stream/collapse` rising **+** `update_isotropy` falling | genuine representational collapse |
| `stream/collapse` flat **+** `attn/update_cos` → 1 | attention has stopped mixing positions; the feed-forward is carrying the model — confirm with `ffn/update_ratio` holding while `attn/update_ratio` decays |
| `stream/collapse` flat **+** `attn/update_isotropy` falling | heads converging onto a single operator |
| either `update_ratio` → 0 | that branch is being routed around; its parameters are dead weight |
| `attn/update_ratio` ≫ `ffn/update_ratio` (or the reverse) | the block is effectively one branch |
| `stream/norm` growing without bound | residual-stream blow-up — find the branch whose `update_ratio` stays high while it grows |
| `stream/norm` growing **+** `update_ratio` falling at a steady ‖update‖ | the stream is outgrowing the branches; deeper blocks are losing their say |
| `attn/input_norm` and `ffn/input_norm` diverging under `pre-*` | `norm1` and `norm2` gains have learned different scales |
| `head/isotropy` ≪ last block's `ffn/update_isotropy` | the narrowing happened at `norm_f`, not in the stack |
| `embedding/collapse` rising | the table is pulling its words together |
| `head/logit_std` rising **+** `loss/curves {val}` flat | the head is sharpening, not learning |
| `perf/tokens_per_sec` falling **+** `perf/tflops` steady | more arithmetic per token, not a slower device |
| one layer's `/max` far from `/mean` | a single-layer anomaly; open the per-layer chart before concluding anything |

---

## Caveats

**`attn/update_cos` is architecturally confounded.** Standard attention passes its update
through `W_v`/`W_o`, which rotate the output out of the span of the inputs, so ~0 is the
natural resting value. An attention variant whose value *is* the residual stream would sit
far higher by construction. Read it against its own trajectory.

**`attn/*` and `ffn/*` are not comparable to each other at a point in time.** They share
definitions, not resting values. Use the pair to attribute a *change* to a branch; do not
rank the branches against each other.

**Isotropy needs enough tokens.** The divisor is `min(valid tokens, width)`. Keep the probe
batch's valid-token count comfortably above `embed_dim`, or the divisor becomes the token
count and the value stops reading as "fraction of the width in use".

**The probe batch is one batch.** Every probe number is measured on it alone. It keeps the
curves comparable across the run, but a batch unrepresentative of the data gives numbers
unrepresentative of the model.

**`flops.attention_counted` can be false.** FLOP formulas exist for the fused CUDA attention
kernels but not for every backend (the CPU one has none); there, attention contributes zero
and `perf/tflops` under-reports exactly the term an attention-heavy model spends most of its
time in.

**Multi-device runs.** Only the coordinator writes. `perf/*` scales the coordinator's own
token count and FLOPs by the world size, which is the global figure to within padding noise
across ranks.

**`param/*` and the probes are snapshots**, taken at one step and on one batch respectively;
`optim/*` and `perf/*` are window means. Expect the former to be noisier.

**Gradient noise is measured on validation data, in eval mode, with plain cross-entropy.**
The training gradient also carries dropout's noise and, when set, label smoothing, so
`param/gsnr/*` reads the data's noise alone and `optim/noise_scale` the batch size that noise
calls for.

---

## Adding a probe

A probe is registered beside the module it measures, and the probe pass picks it up with no
other change:

```python
from probes import register

@register("mynorm", lambda m: isinstance(m, MyNorm))
def _mynorm(module, inputs, output, ctx):
    x = inputs[0].float()
    return {"input_r": ctx.mean(x.square().mean(-1, keepdim=True).sqrt())}
```

It runs inside a forward hook and returns `{metric: 0-dim tensor}`, left on the device. Tags
are filed as `<family>/<site>/<metric>`, where the site is the module's attribute name
(`norm1`, `Wqkv`, …) — or as `<family>/<metric>` with `per_site=False`, for a module that
occurs at most once per block — and get the per-layer and `/mean` `/min` `/max` suffixes
like every other probe.

`ctx` provides the masked reductions every probe here uses — `mean`, `std`, `fraction`,
`collapse`, `unique_collapse`, `isotropy` — so padding is excluded without the probe having
to remember to.
