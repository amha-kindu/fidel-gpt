import re
import torch
import torch.nn as nn

from config import *


class Conversation:
    def __init__(self, type: str, system_text=None, context_text=None) -> None:
        self.type = type
        self.exchanges = []
        self.system_text = system_text
        self.context_text = context_text

    def add_exchange(self, input_text: str, output_text: str, context_text: str | None = None):
        self.exchanges.append({
            "input": input_text,
            "output": output_text,
            "context": context_text
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

def component_key(name: str) -> str:
    """Parameter name -> the component its norms are reported under.

    One rule, used by diagnostics.py for every param/* tag and by the census, so
    a bucket covers the same parameters in a training run and in a comparison run.

    Anything that is not the embedding, the projection or a decoder block falls
    into NormF -- that is norm_f alone in GPTmodel, but a subclass with its own
    top-level layers lands there too.
    """
    if name.startswith("embedding"):
        return "Embedding"
    if name.startswith("decoders."):
        return f"Decoder{name.split('.')[1]}"
    if name.startswith("projection"):
        return "Projection"
    return "NormF"

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


NO_DECAY_PARAMS: tuple[str, ...] = ()


def build_param_groups(model: nn.Module, weight_decay: float, no_decay: tuple[str, ...] = NO_DECAY_PARAMS) -> list[dict]:
    decayed: list[nn.Parameter] = []
    undecayed: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.rsplit(".", 1)[-1] in no_decay:
            undecayed.append(param)
        else:
            decayed.append(param)

    groups = [{"params": decayed, "weight_decay": weight_decay}]
    if undecayed:
        groups.append({"params": undecayed, "weight_decay": 0.0})

    return groups


def set_trainable_params(model: nn.Module, trainable_modules: dict, for_inference: bool = False):
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
   