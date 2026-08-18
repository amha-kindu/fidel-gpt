import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from tensorboard_logger import TensorboardLogger


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

@_non_blocking()
def log_confidence_metrics(tb_logger: TensorboardLogger, logits: torch.Tensor, global_step: int):
    with torch.no_grad():
        # Cast to fp32: under fp16 autocast, 1e-9 underflows to 0.0 making clamp a no-op.
        logits_f = logits.float()
        probs = torch.softmax(logits_f, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-9)), dim=-1).mean().item()
        max_prob = probs.max(dim=-1).values.mean().item()
        logit_std = logits_f.std(dim=-1).mean().item()
        tb_logger.log_scalar("Confidence/Entropy", entropy, global_step)
        tb_logger.log_scalar("Confidence/MaxProb", max_prob, global_step)
        tb_logger.log_scalar("Confidence/LogitStd", logit_std, global_step)

@_non_blocking()
def log_gradients(tb_logger: TensorboardLogger, grads: dict[str, torch.Tensor], global_step: int):
    with torch.no_grad():
        global_sq = 0.0
        component_sq: dict[str, float] = {}
        for name, grad in grads.items():
            if grad is None:
                continue
            norm_sq = torch.linalg.vector_norm(grad.float().view(-1)).item() ** 2
            global_sq += norm_sq
            if name.startswith("embedding"):
                key = "Embedding"
            elif name.startswith("decoders."):
                key = f"Decoder{name.split('.')[1]}"
            elif name.startswith("projection"):
                key = "Projection"
            else:
                key = "NormF"
            component_sq[key] = component_sq.get(key, 0.0) + norm_sq
        
        tb_logger.log_scalar("Gradients/Global", global_sq ** 0.5, global_step)
        for key, sq in component_sq.items():
            tb_logger.log_scalar(f"Gradients/{key}", sq ** 0.5, global_step)


def _cosine_gram(vecs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norm = vecs.norm(dim=-1)
    vecs_norm = vecs / norm.clamp_min(1e-12).unsqueeze(-1)  # eps: divide-by-exact-zero guard only
    gram = torch.matmul(vecs_norm, vecs_norm.transpose(-2, -1)).pow(2)
    return gram, norm


def _mode_gram(module) -> tuple[torch.Tensor, torch.Tensor]:
    B = torch.matmul(module.weight_U, module.weight_V.transpose(-2, -1)) * module.B_scale
    B = torch.where(module.lower_mask, B, 0.0)
    B_flat = B.reshape(module.heads, module.modes, -1)
    return _cosine_gram(B_flat)


class RiemannianMetricProbe:
    def __init__(self, model):
        self.stats = []
        self.handles = [
            dec.masked_multihead_attention.riemannian_metric.register_forward_hook(self._make_hook(i))
            for i, dec in enumerate(model.decoders)
            if dec.masked_multihead_attention.riemannian_metric is not None
        ]

    def _make_hook(self, layer_idx: int):
        def hook(module, inp, out):
            with torch.no_grad(), torch.autocast(device_type=inp[0].device.type, enabled=False):
                x_in = inp[0].detach().float()

                raw_gate = torch.einsum("nhsd,hdm->nhsm", x_in, module.weight_W.detach().float())
                gate = F.silu(raw_gate)  # (N,H,S,modes)

                U = module.weight_U.detach().float()
                V = module.weight_V.detach().float()
                B = torch.einsum("hmir,hmjr->hmij", U, V) * module.B_scale  # (H,modes,head_dim,head_dim)
                B = torch.where(module.lower_mask, B, torch.zeros_like(B))
                L = torch.asinh(torch.einsum("nhsm,hmij->nhsij", gate, B) * module.S_scale)  # (N,H,S,head_dim,head_dim)

                # G = I + L@L^T is the metric actually APPLIED to x (out = x @ G)
                trace_G = module.head_dim + L.pow(2).sum(dim=(-2, -1))   # (N,H,S) == trace(G) exactly

                eye = torch.eye(module.head_dim, device=x_in.device, dtype=L.dtype)
                G = eye + torch.matmul(L, L.transpose(-2, -1))  # (N,H,S,head_dim,head_dim), symmetric PD
                C = torch.linalg.cholesky(G)   # (N,H,S,head_dim,head_dim), lower-triangular, G = C @ C^T
                diag_C = torch.diagonal(C, dim1=-2, dim2=-1)
                # clamp_min is a pure fp-roundoff guard (G's eigenvalues are >=1 by construction, so
                # C's diagonal should stay well away from 0 too) -- not a masked-singularity floor,
                # since G can't be singular in the first place.
                log_det_G = 2.0 * diag_C.clamp_min(1e-12).log().sum(-1)   # (N,H,S) == log det(G) exactly

                trace_per_head = (trace_G / module.head_dim).mean(dim=(0, 2))       # (H,) arithmetic mean eigenvalue -- 1 at L=0 (G=I)
                log_det_per_head = (log_det_G / module.head_dim).mean(dim=(0, 2))   # (H,) log of geometric mean eigenvalue -- 0 at L=0 (G=I)

                # Isotropy = geometric_mean(eigenvalues) / arithmetic_mean(eigenvalues) in (0, 1],
                # by AM-GM -- computed here purely from the trace/log-det AGGREGATES above, without
                # ever needing G's individual eigenvalues. It equals 1 iff the metric is a scalar
                # multiple of the identity (perfectly isotropic), and approaches 0 as the metric
                # becomes ill-conditioned (some direction's eigenvalue dominates the others) -- NOT
                # "singular" anymore, since G's eigenvalues are bounded away from 0 by construction
                # (>=1). Unlike trace, this detects anisotropy rather than just overall scale.
                isotropy_per_head = log_det_per_head.exp() / trace_per_head.clamp_min(1e-12)   # (H,)

                mode_diagnostics = {}
                if module.modes >= 2:
                    off_diag_mask = ~torch.eye(module.modes, dtype=torch.bool, device=x_in.device)

                    # Structural/weight-space redundancy: do two modes' B_hm point the same
                    # direction in (head_dim*head_dim)-space? See _mode_gram's own comment.
                    weight_gram, weight_norm = _mode_gram(module)
                    weight_redundancy = weight_gram[:, off_diag_mask].mean().item()

                    # Behavioral/usage redundancy+balance: same squared-cosine-Gram idea, but
                    # over each mode's actual per-token gate activation THIS BATCH, not its
                    # weights -- catches two modes firing on the same tokens even if their
                    # B_hm are orthogonal, and separately (via the min/mean norm ratio below)
                    # catches a mode that's just gone dead (near-zero for everyone), which
                    # weight-space orthogonality has no way to see at all.
                    gate_flat = gate.permute(1, 3, 0, 2).reshape(module.heads, module.modes, -1)  # (H,modes,N*S)
                    gate_gram, gate_norm = _cosine_gram(gate_flat)
                    gate_redundancy = gate_gram[:, off_diag_mask].mean().item()
                    # 1.0 = every mode in every head fires with the same average magnitude;
                    # ->0 = at least one mode has gone (near-)dead in at least one head.
                    gate_balance = (gate_norm.min(dim=-1).values / gate_norm.mean(dim=-1).clamp_min(1e-12)).mean().item()

                    mode_diagnostics = {
                        "weight_redundancy": weight_redundancy,
                        "weight_norm": weight_norm.mean().item(),
                        "gate_redundancy": gate_redundancy,
                        "gate_balance": gate_balance,
                    }

                self.stats.append({
                    "layer": layer_idx,
                    "module": module,
                    "trace_per_head": trace_per_head.tolist(),
                    "log_det_per_head": log_det_per_head.tolist(),
                    "isotropy_per_head": isotropy_per_head.tolist(),
                    "mode_diagnostics": mode_diagnostics,
                })
        return hook

    def capture_grads(self):
        for s in self.stats:
            module = s.pop("module")
            
            grad_norms = {}
            for name, key in (("weight_W", "Gate"), ("weight_U", "U"), ("weight_V", "V")):
                param = getattr(module, name, None)
                if param is not None and param.grad is not None:
                    grad_norms[key] = torch.linalg.vector_norm(param.grad.detach().float()).item()
                else:
                    grad_norms[key] = 0.0
            s["grad_norms"] = grad_norms

    def close(self):
        for h in self.handles:
            h.remove()


@_non_blocking()
def log_riemannian_metrics(tb_logger: TensorboardLogger, stats: list[dict], global_step: int):
    if not stats:
        return
    with torch.no_grad():
        for s in stats:
            tag = f"Decoder{s['layer']}"
            tb_logger.log_scalars(f"Metrics/{tag}/Trace", {f"Head{h}": v for h, v in enumerate(s["trace_per_head"])}, global_step)
            tb_logger.log_scalars(f"Metrics/{tag}/LogDet", {f"Head{h}": v for h, v in enumerate(s["log_det_per_head"])}, global_step)
            tb_logger.log_scalars(f"Metrics/{tag}/Isotropy", {f"Head{h}": v for h, v in enumerate(s["isotropy_per_head"])}, global_step)
            tb_logger.log_scalars(f"Metrics/{tag}/Gradient", s["grad_norms"], global_step)

            md = s.get("mode_diagnostics")
            if md:
                tb_logger.log_scalars(f"Metrics/{tag}/ModeCollapse", {
                    "UVRedundancy": md["weight_redundancy"],
                    "UVNorm": md["weight_norm"],
                    "GateRedundancy": md["gate_redundancy"],
                    "GateBalance": md["gate_balance"],
                }, global_step)


@_non_blocking()
def log_weight_norms(tb_logger: TensorboardLogger, weights: dict[str, torch.Tensor], global_step: int):
    with torch.no_grad():
        component_sq: dict[str, float] = {}
        for name, param in weights.items():
            norm_sq = torch.linalg.vector_norm(param.float().view(-1)).item() ** 2
            if name.startswith("embedding"):
                key = "Embedding"
            elif name.startswith("decoders."):
                key = f"Decoder{name.split('.')[1]}"
            elif name.startswith("projection"):
                key = "Projection"
            else:
                key = "NormF"
            component_sq[key] = component_sq.get(key, 0.0) + norm_sq
        for key, sq in component_sq.items():
            tb_logger.log_scalar(f"WeightNorm/{key}", sq ** 0.5, global_step)


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
