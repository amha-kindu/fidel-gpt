import os
import sys
import torch
import numpy
import random
import atexit
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor


random.seed(4321)
torch.manual_seed(4321)
numpy.random.seed(4321)

class PartialLineHandler(logging.StreamHandler):
    def emit(self, record):
        # Determine whether this record is 'partial'
        partial = getattr(record, 'partial', False)

        try:
            msg = self.format(record)
            if partial:
                # Write the message without a newline
                self.stream.write(msg)
            else:
                # Normal behavior: newline at the end
                self.stream.write(msg + "\n")
            self.flush()
        except Exception:
            self.handleError(record)

class PartialFormatter(logging.Formatter):
    def format(self, record):
        if getattr(record, 'partial', False):
            return record.getMessage()
        else:
            return super().format(record)
        
COORDINATOR_RANK = 0
GLOBAL_RANK = int(os.getenv("RANK", "0"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
CHECKPOINTS_DIR = os.path.join(WORKING_DIR, "checkpoints")

DEVICE = torch.device('cpu')
MIXED_PRECISION_ENABLED = False
LOGGER = logging.getLogger(f"CPU {GLOBAL_RANK}")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3000)
    LOGGER = logging.getLogger(f"GPU {GLOBAL_RANK}")
    DEVICE = torch.device(f'cuda:{LOCAL_RANK}')
    torch.cuda.set_device(DEVICE)
    MIXED_PRECISION_ENABLED = torch.amp.autocast_mode.is_autocast_available(DEVICE.type)

LOGGER.setLevel(logging.INFO)
stream_handler = PartialLineHandler(sys.stdout)
stream_handler.setFormatter(PartialFormatter(
    fmt="\033[95m%(asctime)s\033[0m - \033[94m%(levelname)s\033[0m - \033[96m%(name)s\033[0m - \033[93m%(message)s\033[0m"
))
LOGGER.addHandler(stream_handler)

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

ENV = {
    "python": sys.platform,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    "git_commit": get_git_commit(),
}


class Config:
    def to_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"
    
    def update(self, skip: list[str] = [], **kwargs):
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key) \
                and key not in skip:
                setattr(self, key, value)

class ModelConfig(Config):
    def __init__(self, **kwargs):
        self.embed_dim: int = kwargs.get("embed_dim", 512)
        self.n_blocks: int = kwargs.get("n_blocks", 6)
        self.vocab_size: int = kwargs.get("vocab_size", 25000)
        self.ff_dim: int = kwargs.get("ff_dim", 2048)
        self.heads: int = kwargs.get("heads", 8)
        self.dropout: float = kwargs.get("dropout", 0.1)
        self.seq_len: int = kwargs.get("seq_len", 50)
        self.metric_dim: int = kwargs.get("metric_dim", 128)
        self.epsilon: float = kwargs.get("metric_epsilon", 1e-02)


class ModelWithLoRAConfig(ModelConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lora_rank: int = kwargs.get("lora_rank", 16)
        self.lora_alpha: int = kwargs.get("lora_alpha", 32)
        self.lora_targets: dict = kwargs.get("lora_targets", {})
        self.lora_dropout: float = kwargs.get("lora_dropout", 0.05)


class TrainingConfig(Config):
    def __init__(self, **kwargs):
        kwargs = { k: v for k, v in kwargs.items() if v is not None}
        self.steps_per_epoch = None
        self.batches_per_epoch = None
        self.dl_workers: int = kwargs.get("dl_workers", 0)
        self.epochs: int = kwargs.get("epochs", 10)
        self.batch_size: int = kwargs.get("batch_size", 64)
        self.grad_accum_steps: int = kwargs.get("grad_accum_steps", 1)
        self.init_lr: float = kwargs.get("init_lr", 2e-04)
        self.min_lr: float = kwargs.get("min_lr", 8e-05)
        self.lr_scheduler: str = kwargs.get("lr_scheduler", "warmup_linear")
        self.weight_decay: float = kwargs.get("weight_decay", 0.01)
        self.beta1: float = kwargs.get("beta1", 0.9)
        self.beta2: float = kwargs.get("beta2", 0.999)
        self.epsilon: float = kwargs.get("epsilon", 1e-08)
        self.max_norm: float = kwargs.get("max_norm", 1.0)
        self.ema_alpha: float = kwargs.get("ema_alpha", 0.9)
        self.label_smoothing: float = kwargs.get("label_smoothing", 0)
        self.es_min_delta: float = kwargs.get("es_min_delta", 0.01)
        self.es_patience: float = kwargs.get("es_patience", 10000)
        self.tb_log_dir: str = kwargs.get("tb_log_dir", "logs")
        self.checkpoint: str = kwargs.get("checkpoint", "amharic-gpt")
        self.max_checkpoints_to_keep: int = kwargs.get("max_checkpoints_to_keep", 5)
        self.warmup_steps: int = kwargs.get("warmup_steps", 1000)
        self.save_every: int = kwargs.get("save_every", 1000)
        self.validate_every: int = kwargs.get("validate_every", 100)
        self.training_data: str = kwargs.get("training_data", None)
        self.validation_data: str = kwargs.get("validation_data", None)
        self.finetuning: bool = kwargs.get("finetuning", False)
        self.sampler_alpha: float = kwargs.get("sampler_alpha", 0.5)
        self.grad_scaler_init: float = kwargs.get("grad_scaler_init", 2 ** 14)
        self.vt_ratio: float = kwargs.get("vt_ratio", 1.0)
        
        if self.training_data and not os.path.isfile(self.training_data):
            raise FileNotFoundError(f"File '{self.training_data}' does not exist")
        
        if self.validation_data and not os.path.isfile(self.validation_data):
            raise FileNotFoundError(f"File '{self.validation_data}' does not exist")
        
class TrainingState:
    def __init__(self, epoch: int, global_step: int, training_loss: float, validation_loss: float, best_val_loss: float, optimizer_state: dict, lr_scheduler_state: dict):
        self.epoch = epoch
        self.global_step = global_step
        self.training_loss = training_loss
        self.validation_loss = validation_loss
        self.best_val_loss = best_val_loss
        self.optimizer_state = optimizer_state
        self.lr_scheduler_state = lr_scheduler_state


class InferenceConfig(Config):
    def __init__(self, **kwargs):
        self.top_k: int = kwargs.get("top_k", 0)
        self.top_p: float = kwargs.get("top_p", 1.0)
        self.temperature: float = kwargs.get("temperature", 1.0)
        self.max_temp: float = kwargs.get("max_temp", 2.0)
        self.repetition_penalty: float = kwargs.get("repetition_penalty", 1.15)
        self.presence_penalty: float = kwargs.get("presence_penalty", 0.0)
        self.freq_penalty: float = kwargs.get("freq_penalty", 0.3)
        self.no_repeat_ngram_size: int = kwargs.get("no_repeat_ngram_size", 3)
        self.rep_window: int = kwargs.get("rep_window", 200)
        self.kv_cache_size: int = kwargs.get("kv_cache_size", 0)
        
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()

THREAD_POOL = ThreadPoolExecutor(max_workers=os.cpu_count())
atexit.register(THREAD_POOL.shutdown, wait=True, cancel_futures=True)