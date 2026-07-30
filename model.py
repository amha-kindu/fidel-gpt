import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from lora import LoRAdapter
from cache import SlidingKVCache


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

        # (d_head//2,)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, dtype=torch.float) / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # Returns cos/sin of shape (SEQ_LEN, d_head), covering absolute positions
    # [offset, offset + seq_len).
    def forward(self, seq_len: int, offset: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=self.inv_freq.dtype)

        # (SEQ_LEN, d_head//2)
        phase_angles = torch.outer(positions, self.inv_freq)

        # (SEQ_LEN, d_head)
        phase_angles = torch.cat((phase_angles, phase_angles), dim=-1)
        return phase_angles.cos().to(dtype), phase_angles.sin().to(dtype)


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, config: ModelConfig):
        assert config.embed_dim % config.heads == 0, "EMBED_DIM is not divisible by heads"

        super().__init__()
        self.heads = config.heads
        self.d_head: int = config.embed_dim // config.heads

        self.dropout_p: float = config.dropout
        self.Wq: nn.Linear = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.Wk: nn.Linear = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.Wv: nn.Linear = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.Wo: nn.Linear = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

    # x: (N_BATCHES, SEQ_LEN, EMBED_DIM); cos/sin: (SEQ_LEN, d_head)
    # RoPE rotates within each head's d_head slice, so the head boundary must be
    # exposed, but only via a free reshape (heads are already contiguous d_head
    # blocks within EMBED_DIM) -- no transpose needed for the rotation itself.
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n, s, e = x.shape
        x = x.view(n, s, self.heads, -1)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos + x_rotated * sin).view(n, s, e)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM), attn_mask -> (SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
        use_cache: bool = False,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # (N_BATCHES, SEQ_LEN, EMBED_DIM) @ (EMBED_DIM, EMBED_DIM) --> (N_BATCHES, SEQ_LEN, EMBED_DIM)
        key: torch.Tensor = self.Wk(x)
        query: torch.Tensor = self.Wq(x)
        value: torch.Tensor = self.Wv(x)

        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            query = self._apply_rotary(query, cos, sin)
            key = self._apply_rotary(key, cos, sin)

        # Cache accumulates past tokens; the model only returns the new KV pairs.
        # Concatenation of past+new is the cache's responsibility. Keys are cached
        # already rotated (rotation only depends on each token's own absolute
        # position), so SlidingKVCache's non-chronological ring-buffer order
        # doesn't affect correctness.
        new_kv = (key, value) if use_cache else None
        if use_cache and kv_cache is not None:
            key_past, value_past = kv_cache
            key = torch.cat([key_past, key], dim=1)
            value = torch.cat([value_past, value], dim=1)

        # (N_BATCHES, SEQ_LEN, EMBED_DIM) --> (N_BATCHES, SEQ_LEN, HEADS, d_head) --> (N_BATCHES, HEADS, SEQ_LEN, d_head)
        query = query.view(query.shape[0], query.shape[1], self.heads, -1).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, -1).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, -1).transpose(1, 2)

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
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        hist_residual: torch.Tensor | float = 0.0,
    ) -> tuple[torch.Tensor, SlidingKVCache | None]:
        if self.post_norm:
            x_update, new_kv = self.masked_multihead_attention(x, attn_mask, is_causal, use_cache, kv_cache, rope_cos_sin)
            x = self.norm1(x + self.dropout(x_update) + hist_residual)
            x = self.norm2(x + self.dropout(self.feed_forward(x)) + hist_residual)
        else:
            x_update, new_kv = self.masked_multihead_attention(self.norm1(x), attn_mask, is_causal, use_cache, kv_cache, rope_cos_sin)
            x = x + self.dropout(x_update) + hist_residual
            x = x + self.dropout(self.feed_forward(self.norm2(x))) + hist_residual
        return x, new_kv

class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        group_size = config.n_decoders // config.n_blocks
        self.decoders = nn.ModuleList([DecoderModule(config) for _ in range(group_size)])

        self.depth_query: nn.Parameter | None = None
        self.depth_norm: nn.RMSNorm | None = None
        if getattr(config, "depth_attention", False):
            self.depth_query = nn.Parameter(torch.empty(config.embed_dim))
            nn.init.normal_(self.depth_query, std=0.02)
            self.depth_norm = nn.RMSNorm(config.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
        use_cache: bool,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None],
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None,
        k_history: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None], torch.Tensor | None]:
        hist_residual = (
            combine_depth_signal(self.depth_query, k_history)
            if self.depth_query is not None else 0.0
        )

        new_kvs: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for decoder, kv_cache in zip(self.decoders, kv_caches):
            x, new_kv = decoder(x, attn_mask, is_causal, use_cache, kv_cache, rope_cos_sin, hist_residual)
            new_kvs.append(new_kv)

        k_block = self.depth_norm(x) if self.depth_norm is not None else None
        return x, new_kvs, k_block


class ProjectionModule(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.depth_query: nn.Parameter | None = None
        if getattr(config, "depth_attention", False):
            self.depth_query = nn.Parameter(torch.empty(config.embed_dim))
            nn.init.normal_(self.depth_query, std=0.02)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
    # Output shape: (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
    def forward(
        self,
        x: torch.Tensor,
        k_history: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.depth_query is not None:
            x = x + combine_depth_signal(self.depth_query, k_history or [])
        return self.linear(x)


class GPTmodel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config: ModelConfig = config
        
        assert config.n_decoders % config.n_blocks == 0, \
            f"n_decoders ({config.n_decoders}) must be divisible by n_blocks ({config.n_blocks})"

        self.embedding = EmbeddingModule(config)
        self.projection = ProjectionModule(config)
        self.rope = RoPeModule(config)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_blocks)])
        self.norm_f = nn.LayerNorm(config.embed_dim)
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
    def _project(
        self,
        x: torch.Tensor,
        k_history: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.projection(x, k_history)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM), mask -> (SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM), plus the depth-attention history
    def _decode(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        use_cache: bool = False,
        kv_caches: list[SlidingKVCache] | None = None,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # During decode (single new token against full KV cache), Q is shorter than K.
        # The cache already enforces causal ordering, so no mask is needed.
        in_decode_phase = use_cache and kv_caches is not None and kv_caches[0].get() is not None
        attn_mask = mask if (mask is not None and not in_decode_phase) else None
        is_causal = (mask is None) and not in_decode_phase

        depth_attention = getattr(self.config, "depth_attention", False)
        k_history: list[torch.Tensor] = []

        n_blocks = len(self.decoder_blocks)
        for g, block in enumerate(self.decoder_blocks):
            group_size = len(block.decoders)
            if use_cache:
                block_kv_caches = [kv_caches[g * group_size + j].get() for j in range(group_size)]
            else:
                block_kv_caches = [None] * group_size

            x, new_kvs, k_block = block(
                x, attn_mask, is_causal, use_cache, block_kv_caches, rope_cos_sin, k_history,
            )

            if use_cache:
                for j, new_kv in enumerate(new_kvs):
                    kv_caches[g * group_size + j].append(new_kv[0], new_kv[1])

            # The last block's output is fed directly into ProjectionModule as x, so
            # it never needs to be cached for anyone to consume.
            if depth_attention and g < n_blocks - 1:
                k_history.append(k_block)

        x = self.norm_f(x) if not self.config.post_norm else x
        return x, k_history

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
        rope_cos_sin = self.rope(x.shape[1], position_offset, x.device, x.dtype)
        x, k_history = self._decode(x, mask, use_cache, kv_caches, rope_cos_sin)
        return self._project(x, k_history)


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


# Depth-wise attention residual: combines a query against cached, RMSNorm'd block
# outputs from earlier blocks. Stateless (holds no parameters of its own) -- the
# caller (DecoderBlock / ProjectionModule) owns and passes in the learnable query.
# The same normalized tensor is used for both scoring and the mixed content
def combine_depth_signal(
    query: torch.Tensor,
    k_history: list[torch.Tensor],
) -> torch.Tensor | float:
    if not k_history:
        return 0.0

    # (L, N_BATCHES, SEQ_LEN, EMBED_DIM)
    all_k = torch.stack(k_history, dim=0)

    # (L, N_BATCHES, SEQ_LEN)
    score = torch.einsum("d,lbsd->lbs", query, all_k)
    alpha = F.softmax(score, dim=0)

    # (N_BATCHES, SEQ_LEN, EMBED_DIM)
    return torch.einsum("lbs,lbsd->bsd", alpha, all_k)
