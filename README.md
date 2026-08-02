# Amharic Text Generation – Custom GPT Training

This project provides the full training and fine-tuning pipeline for a **custom GPT-style language model** specialized for the **Amharic language**. Unlike most multilingual models where Amharic is a low-resource side case, this model is trained **from scratch on 12+ GB of Amharic text** and further adapted on **smaller supervised datasets** for task-specific improvements (dialogue, summarization, classification).

### 🔎 Why this project?
- **Amharic is underrepresented** in large multilingual LLMs. Most perform poorly in fluency and grammar for Amharic text generation.
- This repository builds a **dedicated model** with a vocabulary and training process tailored for the script, morphology, and punctuation rules of Amharic.
- It enables downstream applications such as **chatbots, virtual assistants, educational tools, translation helpers, summarization systems, and Amharic NLP research**.

### 📌 What’s included
- Scripts to **preprocess large Amharic corpora** (cleaning, tokenization, dataset sharding).
- **Pretraining pipeline** (`train.py`) for building GPT models from scratch.
- **Fine-tuning pipeline** (`finetune.py`) with support for LoRA adapters and parameter-efficient training.
- **Tokenizer training** (`tokenizers/train_tokenizer.py`) to create a SentencePiece model with custom symbols and Amharic script coverage.
- **Inference and chatbot scripts** (`inference.py`, `chatbot.py`) for interactive or batch text generation.
- Utilities for **learning rate schedules**, **logging to TensorBoard**, and **LoRA parameter configuration**.

### 📊 Corpus & Training setup
- **Corpus size**: 12+ GB raw Amharic text from multiple sources (books, Wikipedia, news, etc.).
- **Vocabulary**: ~25,000 tokens trained with SentencePiece BPE.
- **Architecture** (example):  
  - Embedding dimension: 1024  
  - Number of blocks: 6  
  - Attention heads: 16  
  - Feed-forward dim: 4096  
  - Context window: 1024–2048 tokens  
- **Optimization**: AdamW with warmup schedulers (`inverse_sqrt`, `warmup_cosine`), gradient accumulation, label smoothing.
- **Fine-tuning**: on curated Amharic supervised datasets for specific tasks, with optional **LoRA adapters** for efficiency.

---

## 🔧 Technical Overview

### Model Architecture
- **Type**: Decoder-only Transformer (GPT-style)
- **Embedding dimension**: configurable (`d_model`)
- **Feed-forward layer size**: configurable (`d_ff`)
- **Attention**: multi-head self-attention (`n_heads`)
- **Depth**: configurable (`n_layers`)
- **Sequence length**: configurable (`max_seq_len`)
- **Vocabulary**: trained with SentencePiece (default ~25k tokens)
- **Special tokens**: `[USER]`, `[BOT]`, `[SYSTEM]`, `[STOP]`, `[UNK]`, `[PAD]`, `[SOS]`

### Training Pipeline
1. **Preprocessing**
   - Text normalization, cleaning, and deduplication.
   - Tokenization via SentencePiece (`tokenizers/train_tokenizer.py`).
   - Data prepared into indexed/streamable format (`dataset.py`, `preprocessor.py`).

2. **Pretraining**
   - Run with `train.py`.
   - Optimized with AdamW + custom LR schedulers (`lr_schedulers.py`).
   - Supports gradient accumulation and mixed precision.
   - TensorBoard logging via `tensorboard_logger.py`.

3. **Fine-tuning**
   - Run with `finetune.py`.
   - Continues training from a checkpoint on smaller, curated datasets.
   - Can target subsets of parameters via `trainable_params.json` or apply LoRA adapters (`lora.py`, `lora_targets.json`).

4. **Monitoring**
   - Training/validation loss
   - Perplexity
   - Confidence metrics (entropy)

5. **Inference**
   - Greedy or sampling-based decoding in `inference.py`.
   - Interactive chatbot loop in `chatbot.py`.
   - Configurable sampling params: `temperature`, `top-p`, `top-k`.

---

## ⚙️ Installation

```bash
git clone https://github.com/amha-kindu/text-generation.git
cd text-generation
pip install -r requirements.txt
````

Ensure you install a **CUDA-enabled PyTorch** if training on GPU.

---

# 🚀 Usage

## 1. Train Tokenizer

```bash
python train_tokenizer.py \
  --training-data data/pretraining/train.jsonl \
  --vocab-size 25000 --max-sentence-length 50000
```
This produces `.model` and `.vocab` files used by the training scripts.

---
## 2. Pretraining

```bash
python train.py \
  --stream \
  --seq-len 2048 \
  --embed-dim 1024 \
  --n-decoders 6 \
  --vocab-size 25000 \
  --ff-dim 4096 \
  --heads 16 \
  --dropout 0.2 \
  --init-lr 2e-04 \
  --min-lr 8e-05 \
  --epochs 10 \
  --batch-size 12 \
  --grad-accum-steps 4 \
  --validate-every 10 \
  --checkpoint weights/ck_pnt.pt \
  --save-every 10000 \
  --training-data data/pretraining/train.jsonl \
  --validation-data data/pretraining/val.jsonl \
  --tokenizer tokenizers/tokenizer.model \
  --dl-workers 4

```
### Key options

* `--stream` : enables memory-efficient streaming dataset loader.
* `--seq-len` : max sequence length.
* `--embed-dim` : embedding dimension.
* `--n-decoders` : number of Transformer layers.
* `--heads` : number of attention heads.
* `--ff-dim` : feed-forward layer size.
* `--lr-scheduler` : learning rate schedule (`inverse_sqrt` in pretraining).
* `--save-every` / `--validate-every` : checkpointing and validation intervals.
---
## 3. Fine-tune

```bash
python finetune.py \
  --trainable-params trainable_params.json \
  --lora \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-targets lora_targets.json \
  --lora-dropout 0.1 \
  --dropout 0.1 \
  --init-lr 5e-04 \
  --min-lr 2e-04 \
  --epochs 10 \
  --batch-size 24 \
  --grad-accum-steps 4 \
  --checkpoint weights/ck_pnt.pt \
  --validate-every 5 \
  --save-every 1000 \
  --dl-workers 4 \
  --vt-ratio 1.5 \
  --training-data data/finetuning/train.json \
  --validation-data data/finetuning/val.json \
  --tokenizer tokenizers/tokenizer.model

```
### Key options

* `--lora` / `--lora-*` : enables LoRA adapters (rank, alpha, dropout, targets).
* `--trainable-params` : JSON file specifying which parameters to update.
* `--label-smoothing` : applies label smoothing during training.
* `--vt-ratio` : validation-to-training ratio control for logging.
* `--lr-scheduler` : e.g., `warmup_cosine` for fine-tuning.

---
## 4. Inference

```bash
python inference.py --checkpoint weights/last_ckpnt.pt\
  --tokenizer tokenizers/tokenizer.model
```

## 5. Chat Mode

```bash
python chatbot.py \
  --top-k 50 --top-p 0.7 --temperature 0.9 \
  --checkpoint weights/last_ckpnt.pt \
  --lora-checkpoint weights/lora.pt \
  --tokenizer tokenizers/tokenizer.model
```
---

# 📊 Logging & Visualization

Start TensorBoard to monitor training:

```bash
tensorboard --logdir runs
```

Logged metrics:

* Training/validation loss
* Perplexity
* Entropy-based confidence

---

# 🧩 LoRA Support

* `lora.py` defines low-rank adapters for efficient fine-tuning.
* Target modules are listed in `lora_targets.json`.
* You can restrict training to LoRA adapters or specific parameter subsets using `trainable_params.json`.

---

# 🔮 Next Steps

* Evaluate on downstream Amharic NLP tasks (QA, summarization, dialogue).
* Deploy as API (FastAPI/Gradio).
* Longer context window support (beyond current max sequence length).
* Explore retrieval-augmented generation (RAG).

---

# 📜 License

MIT License (see [LICENSE](LICENSE)) if included.
