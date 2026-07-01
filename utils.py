import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import *
from model import GPTmodel
from tensorboard_logger import TensorboardLogger


class Conversation:
    def __init__(self, type: str, system_text=None) -> None:
        self.type = type
        self.exchanges = []
        self.system_text = system_text
    
    def add_exchange(self, input_text: str, output_text: str):
        self.exchanges.append({
            "input": input_text,
            "output": output_text
        })

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.counter = 0
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

def init_sdp_backend(name: str | None) -> None:
    if name is None:
        return
    
    from torch.backends.cuda import (
        enable_math_sdp,
        enable_mem_efficient_sdp,
        enable_flash_sdp,
        enable_cudnn_sdp,
    )

    name = name.upper()
    if name == "MATH":
        enable_math_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_flash_sdp(False)
        enable_cudnn_sdp(False)
    elif name == "EFFICIENT_ATTENTION":
        enable_math_sdp(False)
        enable_mem_efficient_sdp(True)
        enable_flash_sdp(False)
        enable_cudnn_sdp(False)
    elif name == "FLASH_ATTENTION":
        enable_math_sdp(False)
        enable_mem_efficient_sdp(False)
        enable_flash_sdp(True)
        enable_cudnn_sdp(False)
    elif name == "CUDNN_ATTENTION":
        enable_math_sdp(False)
        enable_mem_efficient_sdp(False)
        enable_flash_sdp(False)
        enable_cudnn_sdp(True)
    else:
        raise ValueError("Use one of: MATH, EFFICIENT_ATTENTION, FLASH_ATTENTION, CUDNN_ATTENTION")


@torch.no_grad() 
def get_causal_mask(size: int) -> torch.Tensor:
    """
        Strictly upper triangular matrix, where False denotes a masked position (no attention).
            mask[i, j] = False if i < j, else True.
    """
    # [[
    #     [True, False, False, False, False],
    #     [True, True,  False, False, False],
    #     [True, True,  True,  False, False],
    #     [True, True,  True,  True,  False],
    #     [True, True,  True,  True,  True ]
    # ]]
    
    return torch.ones(1, size, size, dtype=torch.bool).tril(diagonal=0)

def _non_blocking():
    def decorator(func):
        def wrapper(*args, **kwargs):
            def _on_done(future):
                exc = future.exception()
                if exc:
                    LOGGER.error(f"Background task '{func.__name__}' failed: {exc}", exc_info=exc)
            THREAD_POOL.submit(func, *args, **kwargs).add_done_callback(_on_done)
        return wrapper
    return decorator

@_non_blocking()
def log_confidence_metrics(tb_logger: TensorboardLogger, logits: torch.Tensor, global_step: int):
    with torch.no_grad():
        # Cast to fp32: under fp16 autocast, 1e-9 underflows to 0.0 making clamp a no-op.
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-9)), dim=-1).mean().item()
        tb_logger.log_scalar("Confidence/Entropy", entropy, global_step)

@_non_blocking()
def log_gradients(tb_logger: TensorboardLogger, grads: dict[str, torch.Tensor], global_step: int):
    with torch.no_grad():
        global_norm_sq = 0.0
        for name, grad in grads.items():
            if grad is not None:
                param_norm = torch.linalg.vector_norm(grad.view(-1)).item()
                global_norm_sq += param_norm ** 2
                tb_logger.log_scalar(f"GradNorm/{name}", param_norm, global_step)
        tb_logger.log_scalar(f"GradNorm/Global", global_norm_sq ** 0.5, global_step)


@torch.no_grad()
def validate(model: GPTmodel, data_loader: DataLoader, loss_func: nn.CrossEntropyLoss):
    val_loss = 0.0
    for batch in data_loader:
        # (N_BATCHES, SEQ_LEN)
        decoder_input: torch.Tensor = batch[0].to(DEVICE, non_blocking=True)
        label: torch.Tensor         = batch[1].to(DEVICE, non_blocking=True)

        # (N_BATCHES, 1, SEQ_LEN, SEQ_LEN)
        decoder_mask: torch.Tensor  = batch[2].to(DEVICE, non_blocking=True)
        
        with torch.autocast(DEVICE.type, enabled=MIXED_PRECISION_ENABLED):
            # (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
            logits: torch.Tensor = model(decoder_input, decoder_mask)

            loss: torch.Tensor = loss_func(
                # (N_BATCHES, SEQ_LEN, VOCAB_SIZE) --> (N_BATCHES * SEQ_LEN, VOCAB_SIZE)
                logits.view(-1, model.config.vocab_size),

                # (N_BATCHES, SEQ_LEN) --> (N_BATCHES * SEQ_LEN, )
                label.view(-1)
            ) 

        val_loss += loss.item()

    return val_loss / len(data_loader) if len(data_loader) > 0 else 0.0

@_non_blocking()
def save_checkpoint(weights: dict, model_config: ModelConfig, global_step: int, config: TrainingConfig, training_state: TrainingState):
    pattern = re.compile(r"(-(?:\d+\.\d{2})K)?\.pt$")
    oldest_checkpoint = pattern.sub(f"-{(global_step - config.max_checkpoints_to_keep * config.save_every) / 1000:.2f}K.pt", config.checkpoint)

    if global_step > config.max_checkpoints_to_keep * config.save_every and os.path.exists(oldest_checkpoint):
        os.remove(oldest_checkpoint)

    checkpoint = {
        "weights": weights,
        "model_config": model_config,
        "training_state": training_state,
        "training_config": config
    }

    torch.save(
        checkpoint,
        pattern.sub(f"-{global_step / 1000:.2f}K.pt", config.checkpoint)
    )


def set_trainable_params(model: GPTmodel, trainable_modules: dict, for_inference: bool = False):
    if trainable_modules is None and not for_inference:
        return  # leave all parameters trainable (full-model finetuning)
    trainables_params = set()
    if trainable_modules and not for_inference:
        for submodule_name, data in trainable_modules.items():
            if data["type"] == 'ModuleList':
                for idx in data['indices']:
                    if len(data['submodules']) == 0:
                        trainables_params.add(f"{submodule_name}.{idx}")
                    for target in data['submodules']:
                        temp = target.split(".")
                        if len(temp) > 1:
                            layer_name, layer_parent = temp[-1], ".".join(temp[:-1])
                            trainables_params.add(f"{submodule_name}.{idx}.{layer_parent}.{layer_name}")
                        else:
                            trainables_params.add(f"{submodule_name}.{idx}.{temp[0]}")
            elif data["type"] == 'Module':
                if len(data['submodules']) == 0:
                    trainables_params.add(f"{submodule_name}")
                for target in data['submodules']:
                    temp = target.split(".")
                    if len(temp) > 1:
                        layer_name, layer_parent = temp[-1], ".".join(temp[:-1])
                        trainables_params.add(f"{submodule_name}.{layer_parent}.{layer_name}")
                    else:
                        trainables_params.add(f"{submodule_name}.{temp[0]}")
            else:
                raise ValueError(f"Unknown type: {data['type']}")
    
    for param_name, param in model.named_parameters():
        param.requires_grad = any(
            param_name == p or param_name.startswith(p + ".") for p in trainables_params
        )
   