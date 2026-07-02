import os
import json
import math
import time
import torch
import argparse
import contextlib
import torch.nn as nn
from tqdm import tqdm
import sentencepiece as spm
from datetime import datetime
import torch.distributed as dist
from torch.nn.attention import SDPBackend

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, RandomSampler

from config import *
from model import GPTmodel
from tensorboard_logger import TensorboardLogger
from lr_schedulers import LRScheduler, get_lr_scheduler
from dataset import NLPDataset, TextDataset, TextStreamDataset, PackedTextStreamDataset
from utils import EarlyStopping, init_sdp_backend, log_gradients, log_weight_norms, log_confidence_metrics, save_checkpoint, validate


def train(config: TrainingConfig, model: GPTmodel, train_dataset: NLPDataset, val_dataset: NLPDataset, is_distributed: bool = False, training_state: TrainingState | None = None) -> None:
    tb_logger = TensorboardLogger(config.tb_log_dir)
    
    tb_logger.log_text("TrainingConfig", f"```json\n{json.dumps(config.__dict__, indent=2)}\n```", step=0)
    tb_logger.log_text("ModelConfig", f"```json\n{json.dumps(model.config.__dict__, indent=2)}\n```", step=0)
    tb_logger.log_text("Environment", f"```json\n{json.dumps(ENV, indent=2)}\n```", step=0)

    base_model = model
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[LOCAL_RANK])

    scaler = torch.GradScaler(init_scale=config.grad_scaler_init, device=DEVICE.type) if MIXED_PRECISION_ENABLED else None

    early_stopping = EarlyStopping(patience=config.es_patience, min_delta=config.es_min_delta)

    optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=config.init_lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon
    )

    scheduler = get_lr_scheduler(optimizer, config, base_model.config.embed_dim)
    
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

    loss_func = nn.CrossEntropyLoss(ignore_index=train_dataset.ignore_index, label_smoothing=config.label_smoothing).to(DEVICE)
    
    # PackedTextStreamDataset handles DDP sharding internally via set_epoch;
    # an external DistributedSampler is only needed for map-style datasets.
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK, shuffle=True, drop_last=True)
        if is_distributed and not isinstance(train_dataset, PackedTextStreamDataset)
        else None
    )
    raw_data_loader = train_dataset.get_loader(config.batch_size, sampler=train_sampler)

    val_batches = int(config.vt_ratio * config.validate_every * config.grad_accum_steps)
    if isinstance(val_dataset, PackedTextStreamDataset):
        val_loader = val_dataset.get_loader(config.batch_size)
    else:
        val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=config.batch_size * val_batches)
        val_loader = val_dataset.get_loader(config.batch_size, sampler=val_sampler)

    last_step_time = time.monotonic()
    for epoch in range(initial_epoch, config.epochs):
        if isinstance(train_dataset, PackedTextStreamDataset):
            train_dataset.set_epoch(epoch)
        elif train_sampler is not None:
            train_sampler.set_epoch(epoch)
        data_loader = tqdm(raw_data_loader, desc=f"\033[95m{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}\033[0m - \033[94mINFO\033[0m - \033[96m{LOGGER.name}\033[0m - \033[93mEpoch {epoch+1}/{config.epochs}", disable = GLOBAL_RANK != COORDINATOR_RANK, total=config.batches_per_epoch)

        for i, batch in enumerate(data_loader):
            
            # (N_BATCHES, SEQ_LEN)
            decoder_input: torch.Tensor = batch[0].to(DEVICE, non_blocking=True)
            label: torch.Tensor         = batch[1].to(DEVICE, non_blocking=True)

            # (N_BATCHES, 1, SEQ_LEN, SEQ_LEN)
            decoder_mask: torch.Tensor  = batch[2].to(DEVICE, non_blocking=True)

            with torch.autocast(device_type=DEVICE.type, enabled=MIXED_PRECISION_ENABLED):
                # (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
                logits: torch.Tensor = model(decoder_input, decoder_mask)

                # Compute the cross-entropy loss
                batch_loss: torch.Tensor = loss_func(
                    # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
                    logits.view(-1, base_model.config.vocab_size),

                    # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
                    label.view(-1)
                )
            
            training_loss += batch_loss.detach().item() / config.grad_accum_steps
            update_weights = ((i + 1) % config.grad_accum_steps) == 0
            
            avg_loss = batch_loss / config.grad_accum_steps
            # Skip gradient sync on accumulation steps; only sync on the last step.
            sync_ctx = model.no_sync() if is_distributed and not update_weights else contextlib.nullcontext()
            if MIXED_PRECISION_ENABLED:
                with sync_ctx:
                    scaler.scale(avg_loss).backward()
                if update_weights:
                    scaler.unscale_(optimizer)
                    if GLOBAL_RANK == COORDINATOR_RANK and global_step % 100 == 0:
                        grad_snapshot, weight_snapshot = {}, {}
                        for name, param in base_model.named_parameters():
                            weight_snapshot[name] = param.detach().cpu()
                            if param.grad is not None:
                                grad_snapshot[name] = param.grad.detach().cpu()
                        
                        log_gradients(tb_logger, grad_snapshot, global_step)
                        log_weight_norms(tb_logger, weight_snapshot, global_step)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                with sync_ctx:
                    avg_loss.backward()
                if update_weights:
                    if GLOBAL_RANK == COORDINATOR_RANK and global_step % 100 == 0:
                        grad_snapshot, weight_snapshot = {}, {}
                        for name, param in base_model.named_parameters():
                            weight_snapshot[name] = param.detach().cpu()
                            if param.grad is not None:
                                grad_snapshot[name] = param.grad.detach().cpu()
                        
                        log_gradients(tb_logger, grad_snapshot, global_step)
                        log_weight_norms(tb_logger, weight_snapshot, global_step)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if update_weights:
                now = time.monotonic()
                tb_logger.log_scalar("Training/TokensPerSec", WORLD_SIZE * config.batch_size * base_model.config.seq_len * config.grad_accum_steps / (now - last_step_time), global_step)
                last_step_time = now
                if MIXED_PRECISION_ENABLED and GLOBAL_RANK == COORDINATOR_RANK:
                    tb_logger.log_scalar("Training/ScalerScale", scaler.get_scale(), global_step)

                tb_logger.log_scalars("Loss/Curves", {"Train": training_loss}, global_step)
                tb_logger.log_scalar("Training/LearningRate", scheduler.get_last_lr()[0], global_step)
                
                if GLOBAL_RANK == COORDINATOR_RANK and global_step % config.validate_every == 0:
                    if isinstance(val_dataset, PackedTextStreamDataset):
                        val_dataset.set_epoch(global_step // config.validate_every)
                    model.eval()
                    val_loss = validate(
                        model=base_model,
                        data_loader=val_loader,
                        loss_func=loss_func,
                        max_batches=val_batches,
                    )
                    model.train()

                    if early_stopping(val_loss):
                        LOGGER.info(f"Early stopping triggered at epoch {epoch + 1}; avg val loss {early_stopping.best_loss:.4f} did not decrease significantly for {early_stopping.patience} consecutive weight updates")
                        should_early_stop = True

                if is_distributed and global_step % config.validate_every == 0:
                    stop_tensor = torch.tensor(int(should_early_stop), device=DEVICE)
                    dist.broadcast(stop_tensor, src=COORDINATOR_RANK)
                    should_early_stop = bool(stop_tensor.item())

                if should_early_stop:
                    break

                if global_step % config.validate_every == 0:
                    tb_logger.log_scalars("Loss/Curves", {"Val": val_loss}, global_step)
                    tb_logger.log_scalar("Perplexity/Val", math.exp(min(val_loss, 20)), global_step)
                    tb_logger.log_scalar("Loss/Gap", val_loss - training_loss, global_step)

                data_loader.set_postfix({
                    "train_loss": f"{training_loss:6.3f}",
                    "val_loss": f"{val_loss:6.3f}"
                })
                
                if GLOBAL_RANK == COORDINATOR_RANK and global_step % 100 == 0:
                    log_confidence_metrics(tb_logger, logits.detach().cpu(), global_step)
                
                if GLOBAL_RANK == COORDINATOR_RANK and global_step and global_step % config.save_every == 0:
                    weights_snapshot = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
                    save_checkpoint(
                        weights=weights_snapshot,
                        model_config=base_model.config,
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

    if is_distributed:
        dist.barrier()
    tb_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument("--is-distributed", action="store_true", help="Device to train the model on")
    parser.add_argument("--training-data", required=True, type=str, help="Path to the training dataset")
    parser.add_argument("--validation-data", required=True, type=str, help="Path to the validation dataset")
    parser.add_argument("--tokenizer", type=str, required=True, help="The path to the trained tokenizer model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--init-weights", type=str, help="Path to checkpoint with weights for initialization")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--grad-accum-steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, help="Number of warmup steps")
    parser.add_argument("--save-every", type=int, help="Number of weight updates between checkpoints")
    parser.add_argument("--validate-every", type=int, help="Number of weight updates between validations")
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
    parser.add_argument("--embed-dim", type=int, help="Dimensionality of the model")
    parser.add_argument("--n-blocks", type=int, help="Number of decoder blocks")
    parser.add_argument("--heads", type=int, help="Number of attention heads")
    parser.add_argument("--vocab-size", type=int, help="Vocabulary size to use")
    parser.add_argument("--dropout", type=float, help="Dropout probability")
    parser.add_argument("--ff-dim", type=int, help="Dimensionality of the feed forward layer")
    parser.add_argument("--post-norm", action="store_true", help="Apply layer normalization after each residual block (post-norm Transformer style)")
    parser.add_argument("--tie-weights", action=argparse.BooleanOptionalAction, default=None, help="Tie embedding and projection weights (default: enabled)")
    parser.add_argument("--dist-backend", type=str, default="nccl", help="Distributed backend")
    parser.add_argument("--resume", default=False, action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--max-checkpoints-to-keep", type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument("--dl-workers", type=int, help="Number of subprocesses to use for data loading")
    parser.add_argument("--stream", default=False, action="store_true", help="Stream data from disk")
    parser.add_argument("--pack-sequences", action=argparse.BooleanOptionalAction, default=None, help="Pack multiple documents per sequence to eliminate padding waste (default: enabled)")
    parser.add_argument("--sdp-kernel", default=None, type=str, choices=[SDPBackend.MATH.name, SDPBackend.EFFICIENT_ATTENTION.name, SDPBackend.CUDNN_ATTENTION.name, SDPBackend.FLASH_ATTENTION.name], help="SDPA kernel to use for attention calculation")

    args = parser.parse_args()
    
    init_sdp_backend(args.sdp_kernel)
    
    if args.is_distributed:
        assert torch.cuda.device_count() > 1, "Must have more than one CUDA supporting GPUs to initiate distributed training"
        assert args.dist_backend in ["nccl", "gloo", "mpi", "ucc"], "Distributed backend must be one of the following: nccl, gloo, mpi or ucc"

        dist.init_process_group(backend=args.dist_backend)

    if args.embed_dim and args.heads:
        assert args.embed_dim % args.heads == 0, "embed_dim must be divisible by heads"

    model_config = ModelConfig()
    model_config.update(**args.__dict__)

    training_config = TrainingConfig()
    training_config.update(**args.__dict__)

    training_state, weights = None, {}
    if args.init_weights:
        if not os.path.isfile(args.init_weights):
            raise FileNotFoundError(f"File {args.init_weights} does not exist")
        LOGGER.info(f"Loading initial weights from '{args.init_weights}'...")
        checkpoint = torch.load(args.init_weights, map_location=DEVICE, weights_only=False)
        weights = checkpoint['weights']
        model_config: ModelConfig = checkpoint['model_config']
        model_config.update(dropout=args.dropout)
    elif args.resume:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"File {args.checkpoint} does not exist")
        LOGGER.info(f"Loading checkpoint from '{args.checkpoint}'...")
        checkpoint: dict = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        weights = checkpoint["weights"]
        
        model_config: ModelConfig = checkpoint["model_config"]
        model_config.update(dropout=args.dropout)
        
        training_config: TrainingConfig = checkpoint["training_config"]
        training_config.update(skip=['checkpoint', 'training_data', 'validation_data'], **args.__dict__)
        
        training_state = checkpoint["training_state"]
    
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.LoadFromFile(args.tokenizer)
    
    if args.stream or os.path.getsize(training_config.training_data) > 200 * 1024 * 1024:
        if GLOBAL_RANK == COORDINATOR_RANK:
            LOGGER.info(f"File '{os.path.basename(training_config.training_data)}' too large! streaming file...")
        if training_config.pack_sequences:
            train_dataset = PackedTextStreamDataset(training_config.training_data, tokenizer, model_config.seq_len, training_config.dl_workers)
        else:
            train_dataset = TextStreamDataset(training_config.training_data, tokenizer, model_config.seq_len, training_config.dl_workers)
    else:
        train_dataset = TextDataset(training_config.training_data, tokenizer, model_config.seq_len, training_config.dl_workers)
            
    if args.stream or os.path.getsize(training_config.validation_data) > 200 * 1024 * 1024:
        if GLOBAL_RANK == COORDINATOR_RANK:
            LOGGER.info(f"File '{os.path.basename(training_config.validation_data)}' too large! streaming file...")
        if training_config.pack_sequences:
            val_dataset = PackedTextStreamDataset(training_config.validation_data, tokenizer, model_config.seq_len)
        else:
            val_dataset = TextStreamDataset(training_config.validation_data, tokenizer, model_config.seq_len)
    else:
        val_dataset = TextDataset(training_config.validation_data, tokenizer, model_config.seq_len)
    
    training_config.batches_per_epoch = int(len(train_dataset) / (training_config.batch_size * WORLD_SIZE))
    training_config.steps_per_epoch = int(training_config.batches_per_epoch / training_config.grad_accum_steps)
    
    model = GPTmodel.build(model_config, weights).to(DEVICE)
    
    if GLOBAL_RANK == COORDINATOR_RANK:
        numerical_configs = {k: v for k, v in training_config.to_dict().items() if not isinstance(v, str)}
        LOGGER.info(f"Total training samples: {len(train_dataset)}")
        if train_dataset.tokens > 0:
            LOGGER.info(f"Total training tokens: {train_dataset.tokens}")
            LOGGER.info(f"Average tokens per sample: {train_dataset.tokens / len(train_dataset):.2f}")
        LOGGER.info(f"Using training config: {numerical_configs}")
        LOGGER.info(f"Initiating training with {'mixed-precision' if MIXED_PRECISION_ENABLED else 'single-precision'}...")
        LOGGER.info(f"Using model config: {model_config}")
        LOGGER.info(f"Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2):.2f}MB")
        LOGGER.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train(training_config, model, train_dataset, val_dataset, args.is_distributed, training_state)
    
    THREAD_POOL.shutdown()
    if args.is_distributed:
        dist.destroy_process_group()