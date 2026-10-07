import os
import json
import math
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
import sentencepiece as spm
from datetime import datetime
from torch.nn.attention import SDPBackend

from config import *
from model import GPTmodel
from tensorboard_logger import TensorboardLogger
from lr_schedulers import LRScheduler, get_lr_scheduler
from torch.utils.data import RandomSampler
from dataset import FineTuningDataset, MultiTaskDataset, PackedFineTuningDataset
from diagnostics import TrainingMonitor, fenced_json, flop_census, probe_batch
from utils import EarlyStopping, build_param_groups, init_sdp_backend, save_checkpoint, set_trainable_params
from train import validate


def finetune(config: TrainingConfig, model: GPTmodel, finetune_dataset: MultiTaskDataset, val_dataset: MultiTaskDataset, pad: int, training_state: TrainingState | None = None) -> None:
    tb_logger = TensorboardLogger(config.tb_log_dir)

    tb_logger.log_text("TrainingConfig", fenced_json(config.to_dict()), step=0)
    tb_logger.log_text("ModelConfig", fenced_json(model.config.to_dict()), step=0)
    tb_logger.log_text("Environment", fenced_json(ENV), step=0)
    
    scaler = torch.GradScaler(init_scale=config.grad_scaler_init, device=DEVICE.type) if MIXED_PRECISION_ENABLED else None

    early_stopping = EarlyStopping(patience=config.es_patience, min_delta=config.es_min_delta)

    optimizer = torch.optim.AdamW(
        params=build_param_groups(model, config.weight_decay),
        lr=config.init_lr,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon
    )
    scheduler = get_lr_scheduler(optimizer, config, model.config.embed_dim)
        
    global_step = 0
    initial_epoch = 0
    training_loss = 0
    val_loss = 0
    should_early_stop = False
    if training_state:
        global_step = training_state.global_step + 1
        initial_epoch = int(training_state.global_step / config.steps_per_epoch)
        training_loss = training_state.training_loss
        val_loss = training_state.validation_loss
        early_stopping.best_loss = training_state.best_val_loss
        optimizer.load_state_dict(training_state.optimizer_state)
        scheduler.load_state_dict(training_state.lr_scheduler_state)
        if scaler and getattr(training_state, 'scaler_state', None):
            scaler.load_state_dict(training_state.scaler_state)

    loss_func = nn.CrossEntropyLoss(ignore_index=finetune_dataset.ignore_index, label_smoothing=config.label_smoothing).to(DEVICE)

    raw_data_loader = finetune_dataset.get_loader(config.batch_size)

    val_batches = int(config.vt_ratio * config.validate_every * config.grad_accum_steps)
    intent_val_batches = max(2, val_batches // len(val_dataset.task_names))
    # Per-intent validation: each member gets its own loader so a small,
    # heavily-oversampled bucket can't hide inside an aggregate loss.
    intent_val_loaders = {}
    for intent, ds in val_dataset.items():
        if isinstance(ds, PackedFineTuningDataset):
            ds.samples_per_epoch = config.batch_size * intent_val_batches
            intent_val_loaders[intent] = (ds, ds.get_loader(config.batch_size))
        else:
            sampler = RandomSampler(ds, replacement=True, num_samples=config.batch_size * intent_val_batches)
            intent_val_loaders[intent] = (None, ds.get_loader(config.batch_size, sampler=sampler))

    # Main val loss follows the train sampling mix so it stays comparable to
    # the training loss curve; intents absent from training get zero weight.
    train_probs = dict(zip(finetune_dataset.task_names, finetune_dataset.task_probs(config.sampler_alpha)))
    total_weight = sum(train_probs.get(intent, 0.0) for intent in val_dataset.task_names) or 1.0
    val_weights = {intent: train_probs.get(intent, 0.0) / total_weight for intent in val_dataset.task_names}

    # One fixed probe batch for the whole run, an equal share of rows from every
    # intent's validation set so the probes see the task mix, not one task.
    share = math.ceil(config.batch_size / len(intent_val_loaders))
    firsts = [next(iter(loader)) for _, loader in intent_val_loaders.values()]
    probe = probe_batch(torch.cat([b[0][:share] for b in firsts])[:config.batch_size],
                        torch.cat([b[2][:share] for b in firsts])[:config.batch_size],
                        torch.cat([b[1][:share] for b in firsts])[:config.batch_size], pad)
    flops = flop_census(model, probe[0], probe[1]) if GLOBAL_RANK == COORDINATOR_RANK else {"total": 0}
    monitor = TrainingMonitor(tb_logger, model, config.log_every, config.max_norm, probe, flops,
                              batches_per_step=config.grad_accum_steps, world_size=1,
                              is_coordinator=GLOBAL_RANK == COORDINATOR_RANK,
                              gsnr_chunks=config.gsnr_chunks)
    monitor.log_census()

    for epoch in range(initial_epoch, config.epochs):
        finetune_dataset.set_epoch(epoch)
        data_loader = tqdm(raw_data_loader, desc=f"\033[95m{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}\033[0m - \033[94mINFO\033[0m - \033[96m{LOGGER.name}\033[0m - \033[93mEpoch {epoch+1}/{config.epochs}", disable = GLOBAL_RANK != COORDINATOR_RANK, total=config.batches_per_epoch)
        for i, batch in enumerate(data_loader):
            # (N_BATCHES, SEQ_LEN)
            decoder_input: torch.Tensor = batch[0].to(DEVICE, non_blocking=True)
            label: torch.Tensor         = batch[1].to(DEVICE, non_blocking=True)

            # (N_BATCHES, 1, SEQ_LEN, SEQ_LEN)
            decoder_mask: torch.Tensor  = batch[2].to(DEVICE, non_blocking=True)
            monitor.count_tokens(decoder_input, pad)

            with torch.autocast(device_type=DEVICE.type, enabled=MIXED_PRECISION_ENABLED):
                # (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
                logits: torch.Tensor = model(decoder_input, decoder_mask)

                # Compute the cross-entropy loss
                batch_loss: torch.Tensor = loss_func(
                    # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
                    logits.view(-1, model.config.vocab_size),

                    # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
                    label.view(-1)
                )
            
            training_loss += batch_loss.detach().item() / config.grad_accum_steps
            update_weights = ((i + 1) % config.grad_accum_steps) == 0

            avg_loss = batch_loss / config.grad_accum_steps
            if MIXED_PRECISION_ENABLED:
                scaler.scale(avg_loss).backward()
                if update_weights:
                    scaler.unscale_(optimizer)
                    monitor.before_update(global_step)
                    monitor.clip(model.parameters())
                    scaler.step(optimizer)
                    monitor.after_update()
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                avg_loss.backward()
                if update_weights:
                    monitor.before_update(global_step)
                    monitor.clip(model.parameters())
                    optimizer.step()
                    monitor.after_update()
                    scheduler.step()
                    optimizer.zero_grad()

            if update_weights:
                monitor.end_step(global_step, training_loss, scheduler.get_last_lr()[0], scaler)

                if GLOBAL_RANK == COORDINATOR_RANK and global_step % config.validate_every == 0:
                    with monitor.paused():
                        model.eval()
                        intent_losses = {}
                        for intent, (packed_ds, intent_loader) in intent_val_loaders.items():
                            if packed_ds is not None:
                                # fresh draws each validation, like the train-side epoch reshuffle
                                packed_ds.set_epoch(global_step // config.validate_every)
                            intent_losses[intent] = validate(model, intent_loader, loss_func)
                        model.train()

                    val_loss = sum(val_weights[intent] * loss for intent, loss in intent_losses.items())
                    # Every intent on one chart; loss/curves {val} is their train-mix weighting.
                    tb_logger.log_scalars("loss/tasks", intent_losses, global_step)
                    monitor.validated(global_step, val_loss)

                    if early_stopping(val_loss):
                        LOGGER.info(f"Early stopping triggered at epoch {epoch + 1}; avg val loss {early_stopping.best_loss:.4f} did not decrease significantly for {early_stopping.patience} consecutive weight updates")
                        should_early_stop = True
                        break

                data_loader.set_postfix({
                    "train_loss": f"{training_loss:6.3f}",
                    "val_loss": f"{val_loss:6.3f}"
                })
                
                if GLOBAL_RANK == COORDINATOR_RANK and global_step and global_step % config.save_every == 0:
                    # Snapshot trainable weights to CPU synchronously so the async
                    # thread-pool write cannot race with optimizer.step() next batch.
                    # remove_duplicate=False keeps both names of tied params (embedding/
                    # projection) so the checkpoint stays complete on its own.
                    with monitor.paused():
                        weights_snapshot = {k: v.detach().cpu() for k, v in model.named_parameters(remove_duplicate=False) if v.requires_grad}
                    save_checkpoint(
                        weights=weights_snapshot,
                        model_config=model.config,
                        global_step=global_step,
                        config=config,
                        training_state=TrainingState(
                            epoch=epoch,
                            global_step=global_step,
                            training_loss=training_loss,
                            validation_loss=val_loss,
                            best_val_loss=early_stopping.best_loss,
                            optimizer_state=optimizer.state_dict(),
                            lr_scheduler_state=scheduler.state_dict(),
                            scaler_state=scaler.state_dict() if scaler else None,
                        )
                    )

                training_loss = 0.0
                global_step += 1

        # Discard gradients from any partial accumulation window at the epoch boundary.
        if len(data_loader) > 0 and (i + 1) % config.grad_accum_steps != 0:
            optimizer.zero_grad()

        if should_early_stop:
            break

    tb_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a pretrained GPT model")
    parser.add_argument("--training-data", required=False, type=str, help="Path to the training dataset (comma-separated for multiple files)")
    parser.add_argument("--validation-data", required=False, type=str, help="Path to the validation dataset (comma-separated for multiple files)")
    parser.add_argument("--tokenizer", type=str, required=True, help="The path to the trained tokenizer model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--grad-accum-steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, help="Number of warmup steps")
    parser.add_argument("--save-every", type=int, help="Number of weight updates between checkpoints")
    parser.add_argument("--validate-every", type=int, help="Number of weight updates between validations")
    parser.add_argument("--gsnr-chunks", type=int, help="Chunks the probe batch is split into for param/gsnr/* and optim/noise_scale at each validation; < 2 disables them (default: 8)")
    parser.add_argument("--log-every", type=int, help="Number of weight updates per diagnostics window: loss/curves train, optim/*, param/*, perf/* (default: 100)")
    parser.add_argument("--vt-ratio", type=float, help="The ratio between the number of samples to validate the model on and the number of samples it has seen, since the last validation")
    parser.add_argument("--init-lr", type=float, help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, help="Minimum learning rate")
    parser.add_argument("--lr-scheduler", type=str, choices=[LRScheduler.WARMUP_CONSTANT.value, LRScheduler.WARMUP_LINEAR.value, LRScheduler.WARMUP_COSINE.value, LRScheduler.INVERSE_SQRT.value], help="Learning rate scheduler(default: warmup_linear)")
    parser.add_argument("--weight-decay", type=float, help="L2 regularization coefficient")
    parser.add_argument("--beta1", type=float, help="Adam optimizer beta1")
    parser.add_argument("--beta2", type=float, help="Adam optimizer beta2")
    parser.add_argument("--epsilon", type=float, help="Adam optimizer epsilon")
    parser.add_argument("--max-norm", type=float, help="Gradient clipping threshold")
    parser.add_argument("--label-smoothing", type=float, help="Label smoothing factor")
    parser.add_argument("--es-patience", type=int, help="Early stopping patience(number of steps)")
    parser.add_argument("--es-min-delta", type=float, help="Early stopping min delta")
    parser.add_argument("--tb-log-dir", type=str, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train the model")
    parser.add_argument("--seq-len", type=int, help="Sequence length of the input")
    parser.add_argument("--dropout", type=float, help="Dropout probability")
    parser.add_argument("--resume", default=False, action="store_true", help="Resume finetuning from checkpoint")
    parser.add_argument("--max-checkpoints-to-keep", type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument("--dl-workers", type=int, help="Number of subprocesses to use for data loading")
    parser.add_argument("--sampler-alpha", type=float, help="The alpha parameter to use for temperature mix sampler")
    parser.add_argument("--trainable-params", type=str, help="Path to a json file containing layers to train during finetuning")
    parser.add_argument("--lora", default=False, action="store_true", help="Flag to use LoRA during finetuning")
    parser.add_argument("--lora-rank", type=int, help="Size of the low-rank matrices when finetuning with LoRA")
    parser.add_argument("--lora-alpha", type=int, help="Parameter that scales the LoRA updates when finetuning with LoRA")
    parser.add_argument("--lora-dropout", type=float, help="The Dropout applied to LoRA's input")
    parser.add_argument("--lora-targets", type=str, help="Path to a json file containing layers to apply LoRA on")
    parser.add_argument("--lora-checkpoint", default="", type=str, help="Path to LoRA adapters")
    parser.add_argument("--finetuned-checkpoint", default="", type=str, help="Path to finetuning checkpoint")
    parser.add_argument("--sdp-kernel", default=None, type=str, choices=[SDPBackend.MATH.name, SDPBackend.EFFICIENT_ATTENTION.name, SDPBackend.CUDNN_ATTENTION.name, SDPBackend.FLASH_ATTENTION.name], help="SDPA kernel to use for attention calculation")
    parser.add_argument("--pack-sequences", action=argparse.BooleanOptionalAction, default=None, help="Pack multiple conversations per sequence to eliminate padding waste (default: enabled)")
    parser.add_argument("--activation-ckpt", action=argparse.BooleanOptionalAction, default=None, help="Trade compute for memory by recomputing decoder activations during backward instead of storing them (default: disabled)")
    parser.add_argument("--reinit-special-tokens", default=False, action="store_true", help="Re-initialize the [USER]/[BOT]/[SYSTEM]/[CONTEXT]/[STOP] embedding rows before training; use on the first finetuning run of a checkpoint that never saw these tokens during pretraining")

    args = parser.parse_args()

    if not args.resume:
        if not args.training_data:
            parser.error("--training-data is required")
        if not args.validation_data:
            parser.error("--validation-data is required")

    init_sdp_backend(args.sdp_kernel)
    
    if args.lora:
        assert args.lora_targets, "If you want to use LoRA, please provide a path to the LoRA targets"

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"File {args.checkpoint} does not exist")
    LOGGER.info(f"Loading checkpoint from '{args.checkpoint}'...")
    pretraining_checkpoint: dict = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    weights: dict = pretraining_checkpoint["weights"]
    
    training_config = TrainingConfig()
    training_config.update(**args.__dict__, finetuning=True)
    
    model_config: ModelConfig = pretraining_checkpoint["model_config"]
    model_config.update(dropout=args.dropout)
    
    if args.lora:
        assert os.path.exists(args.lora_targets) and os.path.isfile(args.lora_targets), f"File {args.lora_targets} does not exist"
        with open(args.lora_targets, 'r') as f:
            args.lora_targets = json.load(f)
        
        model_config = ModelWithLoRAConfig(**model_config.to_dict())
        model_config.update(**args.__dict__)
        
        training_config.checkpoint = training_config.checkpoint.replace(".pt", f"-lora-adapters-{model_config.lora_rank}R-{model_config.lora_alpha}SF.pt")
    else:
        training_config.checkpoint = training_config.checkpoint.replace(".pt", f"-finetuned.pt")
        
    training_state = None
    if args.resume:
        if args.lora_checkpoint:
            if not os.path.isfile(args.lora_checkpoint):
                raise FileNotFoundError(f"File {args.lora_checkpoint} does not exist")
            LOGGER.info(f"Loading lora checkpoint from '{args.lora_checkpoint}'...")
            checkpoint: dict = torch.load(args.lora_checkpoint, map_location=DEVICE, weights_only=False)
        else:
            if not os.path.isfile(args.finetuned_checkpoint):
                raise FileNotFoundError(f"File {args.finetuned_checkpoint} does not exist")
            LOGGER.info(f"Loading finetuning checkpoint from '{args.finetuned_checkpoint}'...")
            checkpoint: dict = torch.load(args.finetuned_checkpoint, map_location=DEVICE, weights_only=False)
        
        weights.update(checkpoint["weights"])
        
        training_config: TrainingConfig = checkpoint["training_config"]
        training_config.update(skip=['checkpoint'], **args.__dict__)
        
        model_config: ModelConfig | ModelWithLoRAConfig = checkpoint["model_config"]
        model_config.update(dropout=args.dropout, lora_dropout=args.lora_dropout)
        
        training_state = checkpoint["training_state"]
    
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.LoadFromFile(args.tokenizer)
    
    intents = [
        "qa",
        "sentiment_analysis",
        "dialogue",
        "story_generation",
        "spellcheck", "other",
        "sentence_classification", "ner", "summarization",
        "reverse_summarization", "title_generation",
    ]
    train_datasets = {k: v for k, v in FineTuningDataset.load_many(intents, training_config.training_data, tokenizer, model_config.seq_len).items() if len(v) > 0}
    samples = sum(len(ds) for ds in train_datasets.values())
    tokens = sum(ds.tokens for ds in train_datasets.values())

    if training_config.pack_sequences:
        # Packed rows are measured in real tokens, not raw conversation count, since
        # each row holds several conversations back-to-back instead of one padded
        # out to seq_len.
        training_config.batches_per_epoch = int(tokens / model_config.seq_len / (training_config.batch_size * WORLD_SIZE))
        train_datasets = {k: PackedFineTuningDataset(v, model_config.seq_len) for k, v in train_datasets.items()}
    else:
        training_config.batches_per_epoch = int(samples / (training_config.batch_size * WORLD_SIZE))
    training_config.steps_per_epoch = int(training_config.batches_per_epoch / training_config.grad_accum_steps)

    finetune_dataset = MultiTaskDataset(
        train_datasets,
        alpha=training_config.sampler_alpha,
        samples_per_epoch=training_config.batch_size * training_config.batches_per_epoch,
        workers=training_config.dl_workers,
    )

    val_datasets = {k: v for k, v in FineTuningDataset.load_many(intents, training_config.validation_data, tokenizer, model_config.seq_len).items() if len(v) > 0}
    if training_config.pack_sequences:
        val_datasets = {k: PackedFineTuningDataset(v, model_config.seq_len) for k, v in val_datasets.items()}
    val_dataset = MultiTaskDataset(val_datasets)
    
    model = GPTmodel.build(model_config, weights).to(DEVICE)
    model.activation_ckpt = training_config.activation_ckpt

    if args.reinit_special_tokens:
        # None of these appear in raw-text pretraining data, yet tied embedding/projection weights
        # (model.py) put every one of them in every pretraining step's softmax normalization without
        # ever being a positive target -- likely biased against being generated, not neutral.
        special_token_ids = [tokenizer.PieceToId(piece) for piece in ("[USER]", "[BOT]", "[SYSTEM]", "[CONTEXT]", "[STOP]")]
        for token_id in special_token_ids:
            nn.init.normal_(model.embedding.embedding.weight[token_id], mean=0.0, std=0.02)
        LOGGER.info(f"Re-initialized embedding rows for special tokens {special_token_ids}")

    trainable_params = model_config.lora_targets if args.lora else None
    if args.trainable_params:
        assert os.path.exists(args.trainable_params), f"File {args.trainable_params} does not exist"
        with open(args.trainable_params, 'r') as f:
            trainable_params = json.load(f)
    
    set_trainable_params(model, trainable_params)
    
    if GLOBAL_RANK == COORDINATOR_RANK:
        numerical_configs = {k: v for k, v in training_config.to_dict().items() if not isinstance(v, str)}
        LOGGER.info(f"Total training samples: {samples}")
        LOGGER.info(f"Using training config: {numerical_configs}")
        LOGGER.info(f"Initiating training with {'mixed-precision' if MIXED_PRECISION_ENABLED else 'single-precision'}...")
        LOGGER.info(f"Using model config: {model_config}")
        if args.lora:
            LOGGER.info("Using LoRA for finetuning")
        LOGGER.info(f"Using training config: {training_config}")
        LOGGER.info(f"Unfrozen Model size: {sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024 ** 2):.2f}MB")
        LOGGER.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    finetune(training_config, model, finetune_dataset, val_dataset, tokenizer.pad_id(), training_state)
    
    THREAD_POOL.shutdown()
