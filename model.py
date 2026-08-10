import math
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from config import *
from lora import LoRAdapter
from cache import SlidingKVCache

try:
    from kernels.riemannian_metric import RiemannianMetricKernel
    _RIEMANNIAN_KERNEL_AVAILABLE = True
except ImportError:
    _RIEMANNIAN_KERNEL_AVAILABLE = False


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
    def __init__(self, config: ModelConfig):
        super().__init__()
        d_head = config.embed_dim // config.heads
        assert d_head % 2 == 0, "RoPE requires an even head dimension"

        # (HEAD_DIM // 2,)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, dtype=torch.float) / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, offset: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=self.inv_freq.dtype)

        # (SEQ_LEN, HEAD_DIM // 2)
        phase_angles = torch.outer(positions, self.inv_freq)
        return phase_angles


class RiemannianMetric(nn.Module):
    def __init__(self, heads: int, head_dim: int, rank: int, fused: bool = False):
        super().__init__()
        self.fused = fused
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** 0.5
        self.rank = min(rank, head_dim)
        self.diag_baseline = math.log1p(math.log(2.0))

        self.weight_diag = nn.Parameter(torch.zeros(heads, head_dim))
        self.weight_W = nn.Parameter(torch.zeros(heads, head_dim, self.rank))
        self.weight_U = nn.Parameter(torch.empty(heads, head_dim, self.rank))
        self.weight_V = nn.Parameter(torch.empty(heads, head_dim, self.rank))
        nn.init.normal_(self.weight_U, mean=0.0, std=head_dim ** -0.5)
        nn.init.normal_(self.weight_V, mean=0.0, std=head_dim ** -0.5)

        row = torch.arange(head_dim).view(head_dim, 1)
        col = torch.arange(head_dim).view(1, head_dim)
        self.register_buffer("lower_mask", row > col, persistent=False)

    # Input shape: x -> (N_BATCHES, HEADS, SEQ_LEN, HEAD_DIM)
    # Output shape: (N_BATCHES, HEADS, SEQ_LEN, HEAD_DIM)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return RiemannianMetricKernel.apply(
                x, self.weight_diag, self.weight_W, self.weight_U, self.weight_V, self.scale,
            )

        # (N_BATCHES, HEADS, SEQ_LEN, HEAD_DIM)
        raw_diag = x * self.weight_diag.view(1, self.heads, 1, self.head_dim)
        L_diag = 1.0 + torch.log1p(F.softplus(raw_diag)) - self.diag_baseline

        # (N_BATCHES, HEADS, SEQ_LEN, HEAD_DIM) @ (HEADS, HEAD_DIM, RANK) -> (N_BATCHES, HEADS, SEQ_LEN, RANK)
        gate = torch.asinh(
            torch.einsum("nhsd,hdr->nhsr", x, self.weight_W) / self.scale
        )

        # (N_BATCHES, HEADS, SEQ_LEN, HEAD_DIM, HEAD_DIM)
        L_offdiag = torch.einsum("nhsr,hir,hjr->nhsij", gate, self.weight_U, self.weight_V) / (self.rank ** 0.5)
        L_offdiag = torch.where(self.lower_mask, L_offdiag, torch.zeros_like(L_offdiag))
        L = L_offdiag + torch.diag_embed(L_diag)

        # U_j = ∑_i x_i . L_ij  ->  U = x.L  ---  (N_BATCHES, HEADS, SEQ_LEN, HEAD_DIM)
        u = torch.einsum("nhsd,nhsdj->nhsj", x, L)

        # O_i = ∑_j U_j . L_ij  -> O = U.L_T ---  (N_BATCHES, HEADS, SEQ_LEN, HEAD_DIM)
        return torch.einsum("nhsi,nhsji->nhsj", u, L)


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, config: ModelConfig):
        assert config.embed_dim % config.heads == 0, "EMBED_DIM is not divisible by heads"

        super().__init__()
        self.heads = config.heads
        self.d_head: int = config.embed_dim // config.heads

        self.dropout_p: float = config.dropout
        self.Wqkv: nn.Linear = nn.Linear(config.embed_dim, 3*config.embed_dim, bias=False)
        self.Wo: nn.Linear = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        
        self.riemannian_metric = None
        if config.riemannian:
            fused = _RIEMANNIAN_KERNEL_AVAILABLE and torch.cuda.is_available()
            rank = getattr(config, "metric_rank", None) or self.d_head
            self.riemannian_metric = RiemannianMetric(self.heads, self.d_head, rank, fused=fused)
        
    
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
        # (N_BATCHES, SEQ_LEN, EMBED_DIM) @ (EMBED_DIM, 3 * EMBED_DIM) --> (N_BATCHES, SEQ_LEN, 3 * EMBED_DIM)
        qkv: torch.Tensor = self.Wqkv(x)
        
        # (N_BATCHES, SEQ_LEN, EMBED_DIM)
        query: torch.Tensor = qkv[..., :x.shape[-1]]
        key: torch.Tensor = qkv[..., x.shape[-1]: 2*x.shape[-1]]
        value: torch.Tensor = qkv[..., 2*x.shape[-1]:]
        
        # (N_BATCHES, SEQ_LEN, EMBED_DIM) --> (N_BATCHES, SEQ_LEN, HEADS, d_head) --> (N_BATCHES, HEADS, SEQ_LEN, d_head)
        query = query.view(query.shape[0], query.shape[1], self.heads, -1).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, -1).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, -1).transpose(1, 2)

        if self.riemannian_metric is not None:
            key = self.riemannian_metric(key)

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

        # (N_BATCHES, SEQ_LEN, HEADS, d_head) -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
        output = output.contiguous().view(*x.shape[:-1], -1)
        
        return self.Wo(output), new_kv
    

class FeedForwardModule(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.linear1 = nn.Linear(config.embed_dim, config.ff_dim)
        self.linear2 = nn.Linear(config.ff_dim, config.embed_dim)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(
            self.dropout(self.gelu(self.linear1(x)))
        )


class DecoderModule(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.post_norm = config.post_norm
        self.dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)

        self.feed_forward = FeedForwardModule(config)
        self.masked_multihead_attention = MultiHeadAttentionModule(config)

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
        if self.post_norm:
            x_update, new_kv = self.masked_multihead_attention(x, attn_mask, is_causal, use_cache, kv_cache, cos_sin_phases)
            x = self.norm1(x + self.dropout(x_update))
            x = self.norm2(x + self.dropout(self.feed_forward(x)))
        else:
            x_update, new_kv = self.masked_multihead_attention(self.norm1(x), attn_mask, is_causal, use_cache, kv_cache, cos_sin_phases)
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
        super().__init__()
        self.config: ModelConfig = config
        
        self.embedding = EmbeddingModule(config)
        self.projection = ProjectionModule(config)
        self.rope = RoPeModule(config)
        self.decoders = nn.ModuleList([DecoderModule(config) for _ in range(config.n_decoders)])
        self.norm_f = nn.LayerNorm(config.embed_dim)
        self.activation_ckpt = False
        if config.tie_weights:
            # Tie input embedding and output projection weights (standard for decoder-only LMs).
            # Both are (vocab_size, embed_dim), sharing one tensor halves that parameter block.
            self.projection.linear.weight = self.embedding.embedding.weight

    # Input shape: x -> (N_BATCHES, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
    # Output shape: (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

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
        return self.norm_f(x) if not self.config.post_norm else x

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
    

    @staticmethod
    def build(
        config: ModelConfig | ModelWithLoRAConfig,
        weights: dict | None = None,
    ):
        model = GPTmodel(config)
        weights = weights or {}

        lora_weights = {k: v for k, v in weights.items() if isinstance(config, ModelWithLoRAConfig) and k in LoRAdapter.get_lora_param_names(config.lora_targets)}
        base_weights = {k: v for k, v in weights.items() if k not in lora_weights}

        if weights:
            if config.tie_weights and "embedding.embedding.weight" in base_weights:
                # Finetuned checkpoints saved via named_parameters() deduplicate tied
                # params and drop projection.linear.weight, so a merged weights dict can
                # hold a stale projection value. load_state_dict copies projection after
                # embedding into the one shared tensor, letting the stale value clobber
                # the finetuned embedding — keep the two keys in sync before loading.
                base_weights["projection.linear.weight"] = base_weights["embedding.embedding.weight"]
            model.load_state_dict(base_weights, strict=True)
        else:
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            
            model.apply(init_weights)
            if config.tie_weights:
                # apply() is children-first: EmbeddingModule gets normal(0, 0.02) then ProjectionModule
                # overwrites the shared tensor with xavier. Restore normal init.
                nn.init.normal_(model.embedding.embedding.weight, mean=0.0, std=0.02)

        if isinstance(config, ModelWithLoRAConfig):
            LoRAdapter.apply_lora(model, config.lora_targets, config.lora_rank, config.lora_alpha, config.lora_dropout)
            
            if lora_weights:
                model.load_state_dict(lora_weights, strict=False)

        return model