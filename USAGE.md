# CLI Usage Guide

All scripts are run from the project root. Checkpoints are saved to the `checkpoints/` directory by default.

---

## Table of Contents
1. [train_tokenizer.py — Train a tokenizer](#1-train_tokenizerpy)
2. [stats.py — Analyse dataset token lengths](#2-statspy)
3. [train.py — Pretrain a model](#3-trainpy)
4. [finetune.py — Finetune a pretrained model](#4-finetunepy)
5. [package_model.py — Package a checkpoint for deployment](#5-package_modelpy)
6. [inference.py — Text-completion REPL](#6-inferencepy)
7. [chatbot.py — Conversational chatbot REPL](#7-chatbotpy)

---

## 1. `train_tokenizer.py`

Trains a SentencePiece tokenizer on one or more JSONL datasets.

```
python train_tokenizer.py \
  --data <path1.jsonl>[,<path2.jsonl>,...] \
  --model-prefix <output-prefix> \
  [--vocab-size 25000] \
  [--model-type bpe] \
  [--max-sentence-length 1000]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data` | Yes | — | Comma-separated paths to JSONL dataset files |
| `--model-prefix` | No | `tokenizer` | Output path prefix; produces `<prefix>.model` and `<prefix>.vocab` |
| `--vocab-size` | No | `25000` | Number of vocabulary tokens |
| `--model-type` | No | `unigram` | Algorithm: `bpe`, `unigram`, `char`, or `word` |
| `--max-sentence-length` | No | `1000` | Lines longer than this (in bytes) are skipped during training |

**Example**
```
python train_tokenizer.py \
  --data data/pretraining/merged/train.jsonl \
  --model-prefix tokenizers/amharic-bpe-tokenizer-25k \
  --vocab-size 25000 \
  --model-type bpe
```

---

## 2. `stats.py`

Analyses the token-length distribution of a JSONL file and writes a TensorBoard histogram and a PNG plot.

```
python stats.py \
  --file <dataset.jsonl> \
  --tokenizer <tokenizer.model> \
  [--max-len 256] \
  [--output-dir ./token_stats] \
  [--max-samples N]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--file` | Yes | — | Path to the JSONL dataset |
| `--tokenizer` | Yes | — | Path to the SentencePiece `.model` file |
| `--max-len` | No | `256` | Sequence length threshold (truncation line on the plot) |
| `--output-dir` | No | `./token_stats` | Directory for TensorBoard logs and PNG output |
| `--max-samples` | No | all | Process only the first N samples |

**Example**
```
python stats.py \
  --file data/pretraining/merged/train.jsonl \
  --tokenizer tokenizers/amharic-bpe-tokenizer-25k.model \
  --max-len 1024

tensorboard --logdir=token_stats
```

---

## 3. `train.py`

Pretrains a GPT model from scratch, or from weight initialisation taken from an existing checkpoint.

### Modes

| Mode | Key flags | What is restored from checkpoint |
|---|---|---|
| **Fresh training** | *(none)* | Nothing — model is randomly initialised |
| **Weight init** | `--init-weights <path>` | Model weights + model architecture; training starts from step 0 |
| **Resume** | `--resume` | Model weights, architecture, all training hyperparameters, optimiser state, scheduler state, step counter |

### Full argument reference

```
python train.py \
  --checkpoint <path>           # where to save checkpoints (base path)
  --tokenizer <path>            # SentencePiece .model file
  --training-data <path>        # JSONL training file  (required unless --resume)
  --validation-data <path>      # JSONL validation file (required unless --resume)
  [--init-weights <path>]       # checkpoint to borrow architecture + weights from
  [--resume]                    # resume from --checkpoint
  [OPTIONS...]
```

#### Data & I/O

| Argument | Default | Description |
|---|---|---|
| `--training-data` | — | Path to training JSONL. Required unless `--resume`. |
| `--validation-data` | — | Path to validation JSONL. Required unless `--resume`. |
| `--tokenizer` | — | Path to SentencePiece `.model` file. Always required. |
| `--checkpoint` | `checkpoints/model.pt` | Base path for saving checkpoints. Step suffix is appended automatically (e.g. `-7.00K.pt`). |
| `--init-weights` | — | Load model weights and architecture from this checkpoint; begin training from step 0. All other model args are ignored. |
| `--resume` | `False` | Resume training from `--checkpoint`, restoring all state. |
| `--stream` | `False` | Stream data from disk. Forced on automatically for files > 200 MB. Saved in the checkpoint and restored on resume. |
| `--pack-sequences` | `True` | Pack multiple documents per sequence to eliminate padding waste. |
| `--dl-workers` | `0` | Dataloader subprocess count. |

#### Model architecture
*(Ignored when `--init-weights` or `--resume` is used — architecture is loaded from the checkpoint.)*

| Argument | Default | Description |
|---|---|---|
| `--embed-dim` | `512` | Embedding / hidden dimension |
| `--n-blocks` | `6` | Number of decoder blocks |
| `--heads` | `8` | Number of attention heads |
| `--vocab-size` | `25000` | Vocabulary size (must match tokenizer) |
| `--seq-len` | `50` | Context window length |
| `--ff-dim` | `2048` | Feed-forward inner dimension |
| `--dropout` | `0.1` | Dropout probability |
| `--post-norm` | `False` | Use post-norm (vs pre-norm) residual connections |
| `--tie-weights` / `--no-tie-weights` | tied | Share embedding and projection weights |

#### Training hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `10` | Total number of epochs |
| `--batch-size` | `64` | Per-device batch size |
| `--grad-accum-steps` | `1` | Gradient accumulation steps (effective batch = batch × accum × world-size) |
| `--init-lr` | `2e-4` | Peak learning rate |
| `--min-lr` | `8e-5` | Minimum learning rate (floor for cosine/linear decay) |
| `--lr-scheduler` | `warmup_linear` | Scheduler type: `warmup_constant`, `warmup_linear`, `warmup_cosine`, `inverse_sqrt` |
| `--warmup-steps` | `1000` | Number of linear warmup steps |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--beta1` | `0.9` | AdamW β₁ |
| `--beta2` | `0.999` | AdamW β₂ |
| `--epsilon` | `1e-8` | AdamW ε |
| `--max-norm` | `1.0` | Gradient clipping threshold |
| `--label-smoothing` | `0.0` | Cross-entropy label smoothing factor |

#### Checkpointing & validation

| Argument | Default | Description |
|---|---|---|
| `--save-every` | `1000` | Save a checkpoint every N weight updates |
| `--max-checkpoints-to-keep` | `5` | Delete the oldest checkpoint once this many exist |
| `--validate-every` | `100` | Run validation every N weight updates |
| `--vt-ratio` | `1.0` | Fraction of steps-since-last-validation to use as validation batch count |
| `--es-patience` | `10000` | Early-stopping patience (in weight-update steps) |
| `--es-min-delta` | `0.01` | Minimum validation-loss improvement to reset patience |
| `--tb-log-dir` | `logs` | TensorBoard log directory |

#### Distributed / hardware

| Argument | Default | Description |
|---|---|---|
| `--is-distributed` | `False` | Enable multi-GPU DDP training |
| `--dist-backend` | `nccl` | DDP backend: `nccl`, `gloo`, `mpi`, or `ucc` |
| `--sdp-kernel` | auto | Force a specific SDPA kernel: `MATH`, `EFFICIENT_ATTENTION`, `CUDNN_ATTENTION`, or `FLASH_ATTENTION` |

### Examples

**Fresh training**
```
python train.py \
  --training-data data/pretraining/merged/train.jsonl \
  --validation-data data/pretraining/merged/val.jsonl \
  --tokenizer tokenizers/amharic-bpe-tokenizer-25k.model \
  --checkpoint checkpoints/amharic-gpt-small.pt \
  --stream --seq-len 1024 --embed-dim 512 --n-blocks 12 \
  --vocab-size 25000 --ff-dim 2048 --heads 8 --dropout 0.1 \
  --init-lr 5e-4 --min-lr 5e-5 --lr-scheduler warmup_cosine \
  --epochs 4 --batch-size 32 --grad-accum-steps 8 \
  --warmup-steps 2000 --weight-decay 0.01 \
  --beta1 0.9 --beta2 0.98 --epsilon 1e-9 \
  --validate-every 500 --vt-ratio 0.05 \
  --save-every 7000 --dl-workers 4
```

**Weight initialisation from an existing checkpoint**
```
python train.py \
  --training-data data/pretraining/merged/train.jsonl \
  --validation-data data/pretraining/merged/val.jsonl \
  --tokenizer tokenizers/amharic-bpe-tokenizer-25k.model \
  --checkpoint checkpoints/amharic-gpt-small-v2.pt \
  --init-weights checkpoints/amharic-gpt-small-21.00K.pt \
  --stream --dropout 0.1 \
  --init-lr 5e-4 --min-lr 5e-5 --lr-scheduler warmup_cosine \
  --epochs 4 --batch-size 32 --grad-accum-steps 8 \
  --warmup-steps 2000 --weight-decay 0.01 \
  --beta1 0.9 --beta2 0.98 --epsilon 1e-9 \
  --validate-every 500 --vt-ratio 0.05 \
  --save-every 7000 --dl-workers 4
```
Architecture flags (`--embed-dim`, `--n-blocks`, etc.) are read from the checkpoint and must not be specified.

**Resume**
```
python train.py --resume \
  --checkpoint checkpoints/amharic-gpt-small-21.00K.pt \
  --tokenizer tokenizers/amharic-bpe-tokenizer-25k.model
```
All hyperparameters are restored from the checkpoint. Pass any hyperparameter explicitly to override the saved value (e.g. `--epochs 8` to extend training). `--training-data` and `--validation-data` are optional: omit them to use the saved paths, or provide them to override.

**Multi-GPU DDP**
```
torchrun --nproc-per-node=4 train.py --is-distributed \
  --training-data data/pretraining/merged/train.jsonl \
  ...
```

---

## 4. `finetune.py`

Finetunes a pretrained model on instruction/conversation data. Supports full-parameter finetuning, selective layer finetuning, and LoRA.

### Modes

| Mode | Key flags |
|---|---|
| **Full finetuning** | *(none)* — all parameters are trained |
| **Selective layers** | `--trainable-params <json>` — only listed layers are trained |
| **LoRA** | `--lora --lora-targets <json>` — low-rank adapters are injected and trained |
| **Resume** | `--resume` with `--lora-checkpoint` or `--finetuned-checkpoint` |

### Full argument reference

```
python finetune.py \
  --checkpoint <pretrained.pt> \
  --tokenizer <path> \
  --training-data <path>    # required unless --resume
  --validation-data <path>  # required unless --resume
  [OPTIONS...]
```

#### Data & I/O

| Argument | Default | Description |
|---|---|---|
| `--training-data` | — | Path to training JSONL. Required unless `--resume`. |
| `--validation-data` | — | Path to validation JSONL. Required unless `--resume`. |
| `--tokenizer` | — | Path to SentencePiece `.model` file. Always required. |
| `--checkpoint` | `checkpoints/model.pt` | Path to the pretrained base checkpoint to finetune from. Also used as the base path for saving finetuned checkpoints (suffixed with `-finetuned.pt` or `-lora-adapters-{R}R-{α}SF.pt`). |
| `--lora-checkpoint` | — | When `--resume`: path to the LoRA adapter checkpoint to resume from. |
| `--finetuned-checkpoint` | — | When `--resume`: path to the full finetuned checkpoint to resume from. |
| `--trainable-params` | — | Path to a JSON file specifying which layers to train (all others are frozen). |
| `--resume` | `False` | Resume finetuning from `--lora-checkpoint` or `--finetuned-checkpoint`. |
| `--dl-workers` | `0` | Dataloader subprocess count. |
| `--pack-sequences` / `--no-pack-sequences` | `True` | Pack multiple conversations per row behind a block-diagonal causal mask, instead of padding each conversation out to `--seq-len` individually. Eliminates padding waste and increases real tokens seen per step. |

#### LoRA

| Argument | Default | Description |
|---|---|---|
| `--lora` | `False` | Enable LoRA finetuning |
| `--lora-targets` | — | Path to a JSON file specifying which modules to inject LoRA into. Required with `--lora`. |
| `--lora-rank` | `16` | LoRA rank r (size of the low-rank matrices) |
| `--lora-alpha` | `32` | LoRA scaling factor α |
| `--lora-dropout` | `0.05` | Dropout applied to LoRA inputs |

#### Model overrides
*(Architecture is always loaded from `--checkpoint`. Only these two can be overridden.)*

| Argument | Default | Description |
|---|---|---|
| `--seq-len` | from checkpoint | Override context window length |
| `--dropout` | from checkpoint | Override dropout rate |

#### Training hyperparameters
Same as `train.py`. All have the same defaults and meanings.
`--sampler-alpha` (default `0.5`) controls temperature mixing across task datasets.

### Examples

**Full finetuning**
```
python finetune.py \
  --checkpoint checkpoints/amharic-gpt-small-21.00K.pt \
  --tokenizer tokenizers/amharic-bpe-tokenizer-25k.model \
  --training-data data/finetuning/train.jsonl \
  --validation-data data/finetuning/val.jsonl \
  --init-lr 1e-4 --min-lr 1e-5 --lr-scheduler warmup_cosine \
  --epochs 3 --batch-size 16 --grad-accum-steps 4 \
  --warmup-steps 500 --save-every 2000 --validate-every 200
```
Saves to `checkpoints/amharic-gpt-small-21.00K-finetuned.pt`.

**LoRA finetuning**
```
python finetune.py \
  --checkpoint checkpoints/amharic-gpt-small-21.00K.pt \
  --tokenizer tokenizers/amharic-bpe-tokenizer-25k.model \
  --training-data data/finetuning/train.jsonl \
  --validation-data data/finetuning/val.jsonl \
  --lora --lora-targets configs/lora_targets.json \
  --lora-rank 16 --lora-alpha 32 \
  --init-lr 2e-4 --epochs 3 --batch-size 16 --grad-accum-steps 4 \
  --save-every 2000 --validate-every 200
```
Saves to `checkpoints/amharic-gpt-small-21.00K-lora-adapters-16R-32SF.pt`.

**Resume LoRA finetuning**
```
python finetune.py --resume \
  --checkpoint checkpoints/amharic-gpt-small-21.00K.pt \
  --tokenizer tokenizers/amharic-bpe-tokenizer-25k.model \
  --lora-checkpoint checkpoints/amharic-gpt-small-21.00K-lora-adapters-16R-32SF-3.50K.pt
```

---

## 5. `package_model.py`

Packages a trained checkpoint into a clean, self-contained deployment directory. Strips optimizer state and training metadata. Copies `model.py`, `lora.py` (if LoRA), and a trimmed `config.py` containing `ModelConfig` & `ModelWithLoRAConfig`.

The latest versioned sibling is selected automatically for every checkpoint argument (e.g. passing `checkpoints/my-model.pt` picks up `my-model-21.00K.pt`).

### Why `--checkpoint` is always required

Finetuning checkpoints (both LoRA and selective-param) only save weights where `requires_grad=True` — they are always a partial delta, never a complete model. The base pretrained checkpoint provides the frozen weights needed to reconstruct the full model. `--checkpoint` is therefore required in every combination.

### Valid combinations

Exactly one of the three combinations below must be used — all other argument combinations are rejected.

| Combination | Arguments | Use case |
|---|---|---|
| 1 | `--checkpoint` + `--lora-checkpoint` | LoRA finetuning — base weights served alongside LoRA adapters |
| 2 | `--checkpoint` + `--finetuned-checkpoint` | Selective-param finetuning — finetuned delta merged into base |
| 3 | `--checkpoint` + `--finetuned-checkpoint` + `--lora-checkpoint` | Selective-param finetuning + LoRA |

```
python package_model.py \
  --checkpoint <base-path> \
  --model-name <directory-name> \
  [--output-dir models/] \
  [--lora-checkpoint <lora-base-path>] \
  [--finetuned-checkpoint <finetuned-base-path>]
```

| Argument | Required | Description |
|---|---|---|
| `--checkpoint` | Yes | Base pretrained checkpoint path |
| `--model-name` | Yes | Output subdirectory name under `--output-dir` |
| `--output-dir` | No | Root output directory (default: `models/`) |
| `--lora-checkpoint` | No | Base path for LoRA adapter checkpoints. Used in combinations 1 and 3. |
| `--finetuned-checkpoint` | No | Base path for selective-param finetuned checkpoints. Used in combinations 2 and 3. |

### Output layout

**Combination 1** — `--checkpoint` + `--lora-checkpoint`
```
models/<model-name>/
├── checkpoint.pt          ← base weights only
├── metadata.json          ← ModelConfig values as JSON
├── checkpoint-lora.pt     ← LoRA adapter weights only
├── metadata-lora.json     ← ModelWithLoRAConfig values + lora targets as JSON
├── model.py
├── lora.py
└── config.py              ← ModelConfig, ModelWithLoRAConfig definitions
```

**Combination 2** — `--checkpoint` + `--finetuned-checkpoint`
```
models/<model-name>/
├── checkpoint.pt          ← base + finetuned delta merged, weights only
├── metadata.json          ← ModelConfig values as JSON
├── model.py
└── config.py              ← ModelConfig definition
```

**Combination 3** — `--checkpoint` + `--finetuned-checkpoint` + `--lora-checkpoint`
```
models/<model-name>/
├── checkpoint.pt          ← base + finetuned delta merged, weights only
├── metadata.json          ← ModelConfig values as JSON
├── checkpoint-lora.pt     ← LoRA adapter weights only
├── metadata-lora.json     ← ModelWithLoRAConfig values as JSON
├── model.py
├── lora.py
└── config.py              ← ModelConfig, ModelWithLoRAConfig definitions
```

### Examples

**Combination 1** — LoRA only
```
python package_model.py \
  --checkpoint checkpoints/amharic-gpt-small.pt \
  --lora-checkpoint checkpoints/amharic-gpt-small-21.00K-lora-adapters-16R-32SF.pt \
  --model-name amharic-gpt-small-chat
```

**Combination 2** — Selective-param finetuning
```
python package_model.py \
  --checkpoint checkpoints/amharic-gpt-small.pt \
  --finetuned-checkpoint checkpoints/amharic-gpt-small-21.00K-finetuned.pt \
  --model-name amharic-gpt-small-chat
```

**Combination 3** — Selective-param finetuning + LoRA
```
python package_model.py \
  --checkpoint checkpoints/amharic-gpt-small.pt \
  --finetuned-checkpoint checkpoints/amharic-gpt-small-21.00K-finetuned.pt \
  --lora-checkpoint checkpoints/amharic-gpt-small-21.00K-finetuned-lora-adapters-16R-32SF.pt \
  --model-name amharic-gpt-small-chat
```

Package to the inference server:
```
python package_model.py \
  --checkpoint checkpoints/amharic-gpt-small.pt \
  --lora-checkpoint checkpoints/amharic-gpt-small-21.00K-lora-adapters-16R-32SF.pt \
  --model-name amharic-gpt-small-chat \
  --output-dir L:/dev/Fidel/fidel-inference/app/models
```

---

## 6. `inference.py`

Interactive text-completion REPL using a base pretrained model.

```
python inference.py \
  --checkpoint <path> \
  --tokenizer <path> \
  [SAMPLING OPTIONS...]
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | — | Path to a trained checkpoint |
| `--tokenizer` | — | Path to SentencePiece `.model` file |
| `--top-k` | `0` | Keep only the top-k tokens before sampling. `0` disables. |
| `--top-p` | `1.0` | Nucleus sampling threshold. `1.0` disables. |
| `--temperature` | `1.0` | Sampling temperature. Lower = more focused; higher = more random. |
| `--repetition-penalty` | `1.15` | Multiplicative penalty for previously seen tokens (HuggingFace-style) |
| `--presence-penalty` | `0.0` | Flat penalty per token that has appeared (OpenAI-style) |
| `--freq-penalty` | `0.3` | Penalty scaled by token count (OpenAI-style) |
| `--no-repeat-ngram-size` | `3` | Ban any n-gram that has already appeared |
| `--rep-window` | `200` | Token history window for repetition penalties |
| `--kv-cache-size` | `0` | Sliding KV cache window size. `0` (default) uses the model's full context window (`max_len`); a smaller positive value trades long-range context for a smaller cache. |

**Example**
```
python inference.py \
  --checkpoint checkpoints/amharic-gpt-small-21.00K.pt \
  --tokenizer tokenizers/amharic-bpe-tokenizer-25k.model \
  --top-k 50 --top-p 0.9 --temperature 0.8
```
Type `exit` to quit.

---

## 7. `chatbot.py`

Multi-turn conversational chatbot using a finetuned (or LoRA-adapted) model. Maintains conversation history across turns and applies a system prompt.

```
python chatbot.py \
  --checkpoint <base-pretrained.pt> \
  --tokenizer <path> \
  (--lora-checkpoint <path> | --finetuned-checkpoint <path>) \
  [SAMPLING OPTIONS...] \
  [--sdp-kernel ...]
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | — | Path to the base pretrained checkpoint. Always required. |
| `--tokenizer` | — | Path to SentencePiece `.model` file. |
| `--lora-checkpoint` | — | Path to a LoRA adapter checkpoint. Adapters are merged into the base weights before inference. |
| `--finetuned-checkpoint` | — | Path to a full finetuned checkpoint. |
| `--sdp-kernel` | auto | Force a specific SDPA kernel (see `train.py` for choices). |
| Sampling options | — | Same as `inference.py` (`--top-k`, `--top-p`, `--temperature`, etc.) but all optional (defaults from `InferenceConfig`). |

At least one of `--lora-checkpoint` or `--finetuned-checkpoint` must be provided.

**Example — LoRA chatbot**
```
python chatbot.py \
  --checkpoint checkpoints/amharic-gpt-small-21.00K.pt \
  --lora-checkpoint checkpoints/amharic-gpt-small-21.00K-lora-adapters-16R-32SF-3.50K.pt \
  --tokenizer tokenizers/amharic-bpe-tokenizer-25k.model \
  --top-k 50 --temperature 0.9
```

**Example — Full finetuned chatbot**
```
python chatbot.py \
  --checkpoint checkpoints/amharic-gpt-small-21.00K.pt \
  --finetuned-checkpoint checkpoints/amharic-gpt-small-21.00K-finetuned-3.50K.pt \
  --tokenizer tokenizers/amharic-bpe-tokenizer-25k.model
```
Type `exit` to quit.

---

## Common patterns

### Typical workflow

```
# 1. Train a tokenizer
python train_tokenizer.py --data data/corpus.jsonl --model-prefix tokenizers/my-tokenizer --vocab-size 25000

# 2. Check the data fits the sequence length
python stats.py --file data/train.jsonl --tokenizer tokenizers/my-tokenizer.model --max-len 1024

# 3. Pretrain
python train.py --training-data data/train.jsonl --validation-data data/val.jsonl \
  --tokenizer tokenizers/my-tokenizer.model --checkpoint checkpoints/my-model.pt \
  --stream --seq-len 1024 --embed-dim 512 --n-blocks 12 ...

# 4. Finetune
python finetune.py --checkpoint checkpoints/my-model-21.00K.pt \
  --tokenizer tokenizers/my-tokenizer.model \
  --training-data data/finetune-train.jsonl --validation-data data/finetune-val.jsonl \
  --lora --lora-targets configs/lora_targets.json ...

# 5. Chat
python chatbot.py \
  --checkpoint checkpoints/my-model-21.00K.pt \
  --lora-checkpoint checkpoints/my-model-21.00K-lora-adapters-16R-32SF-3.50K.pt \
  --tokenizer tokenizers/my-tokenizer.model

# 6. Package for deployment
python package_model.py \
  --checkpoint checkpoints/my-model.pt \
  --model-name my-model \
  --output-dir ../fidel-inference/app/models
```

### Checkpoint naming convention

Training checkpoints are saved as:
```
{base-path}-{step/1000:.2f}K.pt
```
e.g. `checkpoints/amharic-gpt-small-7.00K.pt`, `...-14.00K.pt`, etc.

Finetuning checkpoints append a suffix before the step:
```
{base-path}-finetuned-{step}K.pt
{base-path}-lora-adapters-{rank}R-{alpha}SF-{step}K.pt
```

`package_model.py` identifies versioned siblings automatically by scanning for this pattern.
