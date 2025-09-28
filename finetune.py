import os
import json
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
import sentencepiece as spm
from datetime import datetime

from config import *
from model import GPTmodel
from tensorboard_logger import TensorboardLogger
from lr_schedulers import LRScheduler, get_lr_scheduler
from dataset import MultiTaskDataset, TemperatureSampler, FineTuningDataset
from utils import EarlyStopping, log_confidence_metrics, log_gradients, save_checkpoint, set_trainable_params, validate


def finetune(config: TrainingConfig, model: GPTmodel, finetune_dataset: MultiTaskDataset, val_dataset: MultiTaskDataset, training_state: TrainingState | None = None) -> None:
    tb_logger = TensorboardLogger(config.tb_log_dir)
    
    tb_logger.log_text("TrainingConfig", f"```json\n{json.dumps(config.__dict__, indent=2)}\n```", step=0)
    tb_logger.log_text("ModelConfig", f"```json\n{json.dumps(model.config.__dict__, indent=2)}\n```", step=0)
    tb_logger.log_text("Environment", f"```json\n{json.dumps(ENV, indent=2)}\n```", step=0)
    
    scaler = torch.GradScaler(init_scale=config.grad_scaler_init, device=DEVICE.type) if MIXED_PRECISION_ENABLED else None

    early_stopping = EarlyStopping(patience=config.es_patience, min_delta=config.es_min_delta)

    optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=config.init_lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon
    )
    scheduler = get_lr_scheduler(optimizer, config, model.config.embed_dim)
        
    accum_loss = 0
    global_step = 0
    initial_epoch = 0
    training_loss = 0
    validation_loss = 0
    if training_state:
        global_step = training_state.global_step + 1
        initial_epoch = int(training_state.global_step / config.steps_per_epoch)
        training_loss = training_state.training_loss
        validation_loss = training_state.validation_loss
        early_stopping.best_loss = training_state.best_val_loss
        optimizer.load_state_dict(training_state.optimizer_state)
        scheduler.load_state_dict(training_state.lr_scheduler_state)

    loss_func = nn.CrossEntropyLoss(ignore_index=finetune_dataset.ignore_index, label_smoothing=config.label_smoothing).to(DEVICE)
        
    train_sampler = TemperatureSampler(
        finetune_dataset,
        alpha=training_config.sampler_alpha,
        iter_size=config.batch_size * config.batches_per_epoch,
    )
    data_loader = finetune_dataset.get_loader(config.batch_size, sampler=train_sampler)    
    
    val_sampler = TemperatureSampler(
        val_dataset,
        alpha=training_config.sampler_alpha,
        iter_size=config.batch_size * int(config.vt_ratio * config.validate_every * config.grad_accum_steps),
    )
    val_data_loader = val_dataset.get_loader(config.batch_size, sampler=val_sampler)

    for epoch in range(initial_epoch, config.epochs):
        data_loader = tqdm(data_loader, desc=f"\033[95m{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}\033[0m - \033[94mINFO\033[0m - \033[96m{LOGGER.name}\033[0m - \033[93mEpoch {epoch+1}/{config.epochs}", disable = GLOBAL_RANK != COORDINATOR_RANK, total=config.batches_per_epoch)
        for i, batch in enumerate(data_loader):
            model.train()
            
            # (N_BATCHES, SEQ_LEN)
            decoder_input: torch.Tensor = batch[0].to(DEVICE)
            label: torch.Tensor         = batch[1].to(DEVICE)
            
            # (N_BATCHES, 1, SEQ_LEN, SEQ_LEN)
            decoder_mask: torch.Tensor  = batch[2].to(DEVICE)
            
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
            
            accum_loss += batch_loss.item()

            accum_grad = (i + 1) % config.grad_accum_steps != 0 and i + 1 != config.batches_per_epoch

            if MIXED_PRECISION_ENABLED:
                scaler.scale(batch_loss).backward()
                if not accum_grad:
                    scaler.unscale_(optimizer)
                    log_gradients(tb_logger, {name: param.grad for name, param in model.named_parameters()}, global_step)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                batch_loss.backward()
                if not accum_grad:
                    log_gradients(tb_logger, {name: param.grad for name, param in model.named_parameters()}, global_step)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if not accum_grad:
                accum_steps = config.grad_accum_steps if (i + 1) % config.grad_accum_steps == 0 else (i + 1) % config.grad_accum_steps
                if training_loss == 0:
                    training_loss = (accum_loss / accum_steps)
                else:
                    training_loss = config.ema_alpha * training_loss + (1 - config.ema_alpha) * (accum_loss / accum_steps)
                accum_loss = 0.0
                
                if global_step % config.validate_every == 0:
                    val_loss = validate(
                        model=model,
                        data_loader=val_data_loader,
                        loss_func=loss_func
                    )
                    avg_val_loss = val_loss.item()
                    
                    if validation_loss == 0:
                        validation_loss = avg_val_loss
                    else:
                        validation_loss = config.ema_alpha * validation_loss + (1 - config.ema_alpha) * avg_val_loss
                    
                    if early_stopping(validation_loss):
                        LOGGER.info(f"Early stopping triggered at epoch {epoch + 1}; avg val loss {early_stopping.best_loss:.4f} did not decrease significantly for {early_stopping.patience} consecutive weight updates")
                        break
                
                if global_step % config.validate_every == 0:
                    tb_logger.log_scalars("Loss", {"Validation": validation_loss}, global_step)
                    tb_logger.log_scalar('Loss Gap', validation_loss - training_loss, global_step)
                
                tb_logger.log_scalars("Loss", {"Training": training_loss}, global_step)
                tb_logger.log_scalar("Learning Rate", scheduler.get_last_lr()[0], global_step)
                
                data_loader.set_postfix({
                    "train_loss": f"{training_loss:6.3f}",
                    "val_loss": f"{validation_loss:6.3f}"
                })
                
                log_confidence_metrics(tb_logger, logits.detach(), global_step)
                
                if GLOBAL_RANK == COORDINATOR_RANK and global_step and global_step % config.save_every == 0:
                    save_checkpoint(
                        model=model,
                        global_step=global_step,
                        config=config,
                        training_state=TrainingState(
                            epoch=epoch,
                            global_step=global_step,
                            training_loss=training_loss,
                            validation_loss=validation_loss,
                            best_val_loss=early_stopping.best_loss,
                            optimizer_state=optimizer.state_dict(),
                            lr_scheduler_state=scheduler.state_dict()
                        )
                    )

                global_step += 1
    
    tb_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a pretrained GPT model")
    parser.add_argument("--training-data", required=True, type=str, help="Path to the training dataset")
    parser.add_argument("--validation-data", required=True, type=str, help="Path to the validation dataset")
    parser.add_argument("--tokenizer", type=str, required=True, help="The path to the trained tokenizer model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
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
    parser.add_argument("--ema-alpha", type=float, help="Exponential moving average parameter")
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

    args = parser.parse_args()
    
    if args.lora:
        assert args.lora_targets, "If you want to use LoRA, please provide a path to the LoRA targets"

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"File {args.checkpoint} does not exist")
    LOGGER.info(f"Loading checkpoint from '{args.checkpoint}'...")
    pretraining_checkpoint: dict = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    weights: dict = pretraining_checkpoint["weights"]
    
    training_config = DEFAULT_TRAINING_CONFIG
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
            if not os.path.exists(args.lora_checkpoint):
                raise FileNotFoundError(f"File {args.lora_checkpoint} does not exist")
            LOGGER.info(f"Loading lora checkpoint from '{args.lora_checkpoint}'...")
            checkpoint: dict = torch.load(args.lora_checkpoint, map_location=DEVICE, weights_only=False)
        else:
            if not os.path.exists(args.finetuned_checkpoint):
                raise FileNotFoundError(f"File {args.finetuned_checkpoint} does not exist")
            LOGGER.info(f"Loading finetuning checkpoint from '{args.finetuned_checkpoint}'...")
            checkpoint: dict = torch.load(args.finetuned_checkpoint, map_location=DEVICE, weights_only=False)
        
        weights.update(checkpoint["weights"])
        
        training_config: TrainingConfig = checkpoint["training_config"]
        training_config.update(skip=['checkpoint', 'training_data', 'validation_data'], **args.__dict__)
        
        model_config: ModelConfig | ModelWithLoRAConfig = checkpoint["model_config"]
        model_config.update(dropout=args.dropout, lora_dropout=args.lora_dropout)
        
        training_state = checkpoint["training_state"]
    
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.LoadFromFile(args.tokenizer)
    
    intents = [
        "qa",
        "afrisent",
        "multiturn-dialogue",
        "amharic_story_generation",
        "amharic_spellcheck", "other",
        "masakhanews", "masakhaner", "xlsum_summarization",
        "xlsum_reverse_summarization", "amharic_title_generation",
    ]
    finetune_dataset = MultiTaskDataset(
        datasets={
            intent: FineTuningDataset(intent, training_config.training_data, tokenizer, model_config.seq_len) for intent in intents
        },
        workers=training_config.dl_workers
    )
    samples = len(finetune_dataset)
    
    training_config.batches_per_epoch = int(samples / (training_config.batch_size * WORLD_SIZE))
    training_config.steps_per_epoch = int(training_config.batches_per_epoch / training_config.grad_accum_steps)
    
    val_dataset = MultiTaskDataset(
        datasets={
            intent: FineTuningDataset(intent, training_config.validation_data, tokenizer, model_config.seq_len) for intent in intents
        },
        workers=training_config.dl_workers,
    )
    
    model = GPTmodel.build(model_config, weights).to(DEVICE)
    
    trainable_params = model_config.lora_targets
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
        LOGGER.info(f"Model size: {sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2):.2f}MB")
        LOGGER.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    finetune(training_config, model, finetune_dataset, val_dataset, training_state)
    
    THREAD_POOL.shutdown()
