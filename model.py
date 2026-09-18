import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from config import *
from lora import LoRAdapter
from cache import SlidingKVCache
from probes import register


class EmbeddingModule(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

    # Input shape: x -> (N_BATCHES, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(x))


class RoPeModule(nn.Module):
    def __init__(self, dim: torch.Tensor):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires an even head dimension"

        # (DIM // 2,)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, offset: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=self.inv_freq.dtype)

        # (SEQ_LEN, DIM // 2)
        phase_angles = torch.outer(positions, self.inv_freq)
        return phase_angles


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, config: ModelConfig):
        assert config.attn_dim % config.heads == 0, "ATTN_DIM is not divisible by heads"

        super().__init__()
        self.heads = config.heads
        self.attn_dim: int = config.attn_dim
        self.d_head: int = config.attn_dim // config.heads

        self.dropout_p: float = config.dropout
        self.Wqkv: nn.Linear = nn.Linear(config.embed_dim, 3*config.attn_dim, bias=False)
        self.Wo: nn.Linear = nn.Linear(config.attn_dim, config.embed_dim, bias=False)
    
    # Input shape: x(y) -> (N_BATCHES, HEADS, SEQ_LEN, HEAD_DIM); cos/sin -> (SEQ_LEN, HEAD_DIM // 2)
    # Output shape: (N_BATCHES, HEADS, SEQ_LEN, HEAD_DIM)
    def _apply_rotary(self, x: torch.Tensor, y: torch.Tensor, cos_phase: torch.Tensor, sin_phase: torch.Tensor):        
        # (1, 1, SEQ_LEN, HEAD_DIM // 2)
        cos = cos_phase.to(x.dtype).view(1, 1, x.shape[2], -1)
        sin = sin_phase.to(x.dtype).view(1, 1, x.shape[2], -1)

        # (N_BATCHES, HEADS, SEQ_LEN, HEADS, HEAD_DIM) -> 2x tuple[(N_BATCHES, HEADS, SEQ_LEN, HEAD_DIM // 2)]
        x1, x2 = x.chunk(2, dim=-1)
        y1, y2 = y.chunk(2, dim=-1)

        return torch.cat(
            [
                x1 * cos - x2 * sin,
                x2 * cos + x1 * sin,
                y1 * cos - y2 * sin,
                y2 * cos + y1 * sin,
            ],
            dim=-1,
        ).chunk(2, dim=-1)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM), attn_mask -> (SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
        use_cache: bool = False,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        cos_sin_phases: tuple[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # (N_BATCHES, SEQ_LEN, EMBED_DIM) @ (EMBED_DIM, 3 * ATTN_DIM) --> (N_BATCHES, SEQ_LEN, 3 * ATTN_DIM)
        qkv: torch.Tensor = self.Wqkv(x)
        
        # (N_BATCHES, SEQ_LEN, ATTN_DIM)
        query: torch.Tensor = qkv[..., :self.attn_dim]
        key: torch.Tensor = qkv[..., self.attn_dim: 2*self.attn_dim]
        value: torch.Tensor = qkv[..., 2*self.attn_dim:]

        # (N_BATCHES, SEQ_LEN, ATTN_DIM) --> (N_BATCHES, SEQ_LEN, HEADS, d_head) --> (N_BATCHES, HEADS, SEQ_LEN, d_head)
        query = query.view(query.shape[0], query.shape[1], self.heads, -1).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, -1).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, -1).transpose(1, 2)
        
        if cos_sin_phases is not None:
            query, key = self._apply_rotary(query, key, cos_sin_phases[0], cos_sin_phases[1])        

        # Cache accumulates past tokens; the model only returns the new KV pairs.
        # Concatenation of past+new is the cache's responsibility. Keys are cached
        # already rotated (rotation only depends on each token's own absolute
        # position), so SlidingKVCache's non-chronological ring-buffer order
        # doesn't affect correctness.
        new_kv = (key, value) if use_cache else None
        if use_cache and kv_cache is not None:
            key_past, value_past = kv_cache
            key = torch.cat([key_past, key], dim=2)
            value = torch.cat([value_past, value], dim=2)

        # attn_mask/is_causal are resolved once per forward pass by GPTmodel._decode
        # (identical for every block), instead of rebuilding a float bias here on
        # every one of the n_decoders calls. SDPA accepts a boolean attn_mask directly.
        output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        # (N_BATCHES, HEADS, SEQ_LEN, d_head) -> (N_BATCHES, SEQ_LEN, HEADS, d_head)
        output = output.transpose(1, 2)

        # (N_BATCHES, SEQ_LEN, HEADS, d_head) -> (N_BATCHES, SEQ_LEN, ATTN_DIM)
        output = output.contiguous().view(*x.shape[:-1], -1)

        # (N_BATCHES, SEQ_LEN, ATTN_DIM) @ (ATTN_DIM, EMBED_DIM) -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
        return self.Wo(output), new_kv
    

class GatedFeedForwardModule(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.Wug = nn.Linear(config.embed_dim, 2 * config.ff_dim)
        self.Wd = nn.Linear(config.ff_dim, config.embed_dim)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up, gate = self.Wug(x).chunk(2, dim=-1)
        
        return self.Wd(
            self.dropout(
                F.silu(gate) * up
            )
        )

class DecoderModule(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.strategy = config.norm_strategy
        self.dropout = nn.Dropout(config.dropout)

        norm_cls = nn.RMSNorm if "-rms" in self.strategy else nn.LayerNorm
        self.norm1 = norm_cls(config.embed_dim)
        self.norm2 = norm_cls(config.embed_dim)

        if self.strategy == "deepnorm":
            self.alpha = (2 * config.n_decoders) ** 0.25
        elif "rootdepth" in self.strategy:
            self.alpha = config.gain * (2 * config.n_decoders) ** -0.5
        else:
            self.alpha = 1.0

        self.feed_forward = GatedFeedForwardModule(config)
        self.attention = MultiHeadAttentionModule(config)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM), attn_mask -> (SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
        use_cache: bool = False,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        cos_sin_phases: tuple[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, SlidingKVCache | None]:
        if "post" in self.strategy:
            x_update, new_kv = self.attention(x, attn_mask, is_causal, use_cache, kv_cache, cos_sin_phases)
            x = self.norm1(x + self.dropout(x_update))
            x = self.norm2(x + self.dropout(self.feed_forward(x)))
        elif self.strategy == "deepnorm":
            x_update, new_kv = self.attention(x, attn_mask, is_causal, use_cache, kv_cache, cos_sin_phases)
            x = self.norm1(self.alpha * x + self.dropout(x_update))
            x = self.norm2(self.alpha * x + self.dropout(self.feed_forward(x)))
        elif "rootdepth" in self.strategy:
            x_update, new_kv = self.attention(self.norm1(x), attn_mask, is_causal, use_cache, kv_cache, cos_sin_phases)
            x = x + self.alpha * self.dropout(x_update)
            x = x + self.alpha * self.dropout(self.feed_forward(self.norm2(x)))
        else:
            x_update, new_kv = self.attention(self.norm1(x), attn_mask, is_causal, use_cache, kv_cache, cos_sin_phases)
            x = x + self.dropout(x_update)
            x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x, new_kv


class ProjectionModule(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
    # Output shape: (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class GPTmodel(nn.Module):
    def __init__(self, config: ModelConfig):
        assert config.norm_strategy in ModelConfig.NORM_STRATEGIES, \
            f"norm_strategy must be one of {ModelConfig.NORM_STRATEGIES}"

        super().__init__()
        self.config: ModelConfig = config
        
        self.embedding = EmbeddingModule(config)
        self.projection = ProjectionModule(config)
        self.rope = RoPeModule(config.attn_dim // config.heads)
        self.decoders = nn.ModuleList([DecoderModule(config) for _ in range(config.n_decoders)])
        self.norm_f = (nn.RMSNorm if "-rms" in config.norm_strategy else nn.LayerNorm)(config.embed_dim)
        self.activation_ckpt = False

    # Input shape: x -> (N_BATCHES, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
    # Output shape: (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

    # Input/Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def _final_norm(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.config.norm_strategy in ("post-ln", "post-rms", "deepnorm") else self.norm_f(x)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM), mask -> (SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def _decode(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        use_cache: bool = False,
        kv_caches: list[SlidingKVCache] | None = None,
        phase_angles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # During decode (single new token against full KV cache), Q is shorter than K.
        # The cache already enforces causal ordering, so no mask is needed.
        in_decode_phase = use_cache and kv_caches is not None and kv_caches[0].get() is not None
        attn_mask = mask if (mask is not None and not in_decode_phase) else None
        is_causal = (mask is None) and not in_decode_phase
        
        cos_sin_phases = phase_angles.cos().to(x.dtype), phase_angles.sin().to(x.dtype)

        for i, decoder in enumerate(self.decoders):
            kv_cache = None if not use_cache else kv_caches[i].get()
            if self.training and self.activation_ckpt and not use_cache:
                x, new_kv = torch.utils.checkpoint.checkpoint(
                    decoder, x, attn_mask, is_causal, use_cache, kv_cache, cos_sin_phases,
                    use_reentrant=False,
                )
            else:
                x, new_kv = decoder(x, attn_mask, is_causal, use_cache, kv_cache, cos_sin_phases)
            if use_cache:
                kv_caches[i].append(new_kv[0], new_kv[1])
        return self._final_norm(x)

    # Input shape: x -> (N_BATCHES, SEQ_LEN), mask -> (SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        use_cache: bool = False,
        kv_caches: list[SlidingKVCache] | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        x = self._embed(x)
        phase_angles = self.rope(x.shape[1], position_offset, x.device, x.dtype)
        x = self._decode(x, mask, use_cache, kv_caches, phase_angles)
        return self._project(x)
    

    @classmethod
    def build(
        cls,
        config: ModelConfig | ModelWithLoRAConfig,
        weights: dict | None = None,
    ):
        model = cls(config)
        weights = weights or {}

        lora_weights = {k: v for k, v in weights.items() if isinstance(config, ModelWithLoRAConfig) and k in LoRAdapter.get_lora_param_names(config.lora_targets)}
        base_weights = {k: v for k, v in weights.items() if k not in lora_weights}

        if weights:
            model.load_state_dict(base_weights, strict=True)
        else:
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
                    if m.weight is not None:
                        nn.init.ones_(m.weight)
                    if getattr(m, "bias", None) is not None:
                        nn.init.zeros_(m.bias)

            model.apply(init_weights)

            if config.norm_strategy == "deepnorm":
                # DeepNorm's other half: shrink the residual branch at init so the alpha
                # up-weighting of the identity path in DecoderModule.forward isn't just
                # post-norm with extra steps. apply() dispatches on module type and can't
                # tell a branch projection from any other Linear, so this needs a second,
                # name-aware pass. Q/K are deliberately left unscaled.
                beta = (8 * config.n_decoders) ** -0.25
                for block in model.decoders:
                    nn.init.xavier_uniform_(block.feed_forward.Wug.weight, gain=beta)
                    nn.init.xavier_uniform_(block.feed_forward.Wd.weight, gain=beta)
                    if hasattr(block.attention, "Wo"):
                        nn.init.xavier_uniform_(block.attention.Wo.weight, gain=beta)
                    with torch.no_grad():
                        # Wqkv is fused (embed_dim -> 3 * attn_dim); scale the V third only.
                        qkv_weight = block.attention.Wqkv.weight
                        qkv_weight[qkv_weight.shape[0] // 3 * 2:].mul_(beta)

        if isinstance(config, ModelWithLoRAConfig):
            LoRAdapter.apply_lora(model, config.lora_targets, config.lora_rank, config.lora_alpha, config.lora_dropout)
            
            if lora_weights:
                model.load_state_dict(lora_weights, strict=False)

        return model


# --------------------------------------------------------------------------- #
# Diagnostic Probes
# --------------------------------------------------------------------------- #


def _sublayer_metrics(module: nn.Module, inputs, output, ctx) -> dict[str, torch.Tensor]:
    """The four sublayer-health numbers, for whichever residual branch this is.

    Registered TWICE, once as `attn` and once as `ffn`, deliberately sharing one
    body: the pair only answers "which branch is responsible" if both sides are
    measured against identical definitions, and two copies would eventually drift
    apart exactly where comparing them is the point.

      input_norm        mean ||x|| entering the sublayer -- residual-stream drift,
                        and the scale its other magnitudes are relative to.
      update_ratio      ||update|| / ||x||, the sublayer's gain into the residual
                        stream. Far above 1 is a block shouting over the stream;
                        far below 1 is a sublayer that has switched off.
      update_cos        mean cos(update_i, x_i). ~0 is a sublayer writing genuinely
                        new content; -> 1 means it is mostly rescaling what each
                        token already held.
      update_isotropy   how evenly the update spreads over the directions
                        available to it. -> 0 is a sublayer writing everything into
                        a handful of directions however wide embed_dim is, which
                        cosine collapse cannot see.

    The two families share definitions but NOT resting values -- at init attention
    averages over positions with near-uniform weights, close to rank one, while
    the feed-forward's update comes through a random down-projection. Compare each
    against its own trajectory; comparing attn/x to ffn/x at a point in time reads
    a structural difference as a finding.
    """
    x = inputs[0].float()
    update = (output[0] if isinstance(output, tuple) else output).float()

    x_norm = x.norm(dim=-1, keepdim=True)
    update_norm = update.norm(dim=-1, keepdim=True)
    mean_x_norm = ctx.mean(x_norm)

    return {
        "input_norm": mean_x_norm,
        "update_ratio": ctx.mean(update_norm) / mean_x_norm.clamp_min(ctx.floor),
        "update_cos": ctx.mean((update * x).sum(dim=-1, keepdim=True)
                               / (update_norm * x_norm).clamp_min(ctx.floor)),
        "update_isotropy": ctx.isotropy(update),
    }


def _collapse_metric(metric: str, of_output: bool):
    """Mean pairwise token cosine, filed under one of collapse's three stages.

    One body, three registrations, so the three stages cannot come to mean
    slightly different things. `input` and `mid` read what a sublayer was HANDED
    (so under a pre-norm block they are post-norm1 and post-norm2), while
    `output` reads what left the block. mid - input is what attention added to
    the collapse and output - mid is what the feed-forward added, which is what
    attributes a block's collapse to the branch that caused it.
    """
    def measure(_module, inputs, output, ctx) -> dict[str, torch.Tensor]:
        hidden = (output[0] if isinstance(output, tuple) else output) if of_output else inputs[0]
        return {metric: ctx.collapse(hidden)}
    return measure

def _head_metrics(_module, inputs, output, ctx) -> dict[str, torch.Tensor]:
    """What the output head was handed, and how sharply it answers.

    A forward hook sees no labels, so nothing target-relative -- accuracy,
    calibration, per-token loss -- can live here. These two are what the head's
    own input and output say on their own.

      logit_std       per-token spread of the logits: the temperature the head
                      has taught itself. Rising while loss/val is flat is a head
                      sharpening rather than learning. It is also the tag that
                      reads RLMField on the projection, where the radial gain
                      multiplies logits directly and so acts as a per-token
                      temperature rather than the feature gate it is everywhere
                      else -- pair it with rlm/projection/g_dev.
      input_isotropy  how many directions the final representation actually uses,
                      as a fraction of those available. This is the head's raw
                      material: a rank-starved input caps what ANY head can
                      discriminate, however well trained. Read it against
                      ffn/update_isotropy at the last block to see whether the
                      narrowing happened in the stack or at norm_f.

    Cosine collapse is deliberately NOT here. Under -rms and -qrms, norm_f is a
    per-token rescale, which leaves pairwise cosine exactly unchanged, so it
    would duplicate collapse/output at every strategy but -ln.
    """
    logits = output
    # Reduced with dtype=float32 rather than by casting the whole tensor first:
    # logits are (B, S, VOCAB), the widest activation in the model, and .float()
    # on it would double an already large transient for a per-token scalar.
    # Population std, matching ProbeContext.std and QuantizedRMSNorm's running
    # statistics. torch.std's default applies Bessel's correction and so reads
    # sqrt(V/(V-1)) higher -- 0.002% at a 25k vocabulary, and not worth an
    # inconsistency with every other spread in the tree.
    mean = logits.mean(dim=-1, keepdim=True, dtype=torch.float32)
    mean_square = logits.square().mean(dim=-1, keepdim=True, dtype=torch.float32)

    return {
        "logit_std": ctx.mean((mean_square - mean.square()).clamp_min(0).sqrt()),
        "input_isotropy": ctx.isotropy(inputs[0]),
    }

register("attn", lambda m: isinstance(m, MultiHeadAttentionModule), per_site=False)(_sublayer_metrics)
register("ffn", lambda m: isinstance(m, GatedFeedForwardModule), per_site=False)(_sublayer_metrics)

register("collapse", lambda m: isinstance(m, MultiHeadAttentionModule), per_site=False)(
    _collapse_metric("input", of_output=False))
register("collapse", lambda m: isinstance(m, GatedFeedForwardModule), per_site=False)(
    _collapse_metric("mid", of_output=False))
register("collapse", lambda m: isinstance(m, DecoderModule), per_site=False)(
    _collapse_metric("output", of_output=True))

register("head", lambda m: isinstance(m, ProjectionModule), per_site=False)(_head_metrics)