import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from lora import LoRAdapter
from cache import SlidingKVCache


class Embedding(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

    # Input shape: x -> (N_BATCHES, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(x))


class PositionEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # (SEQ_LEN, 1)
        positions = torch.arange(0, config.seq_len, dtype=torch.float).unsqueeze(1)

        # (EMBED_DIM//2,)
        div_term = torch.exp(torch.arange(0, config.embed_dim, 2, dtype=torch.float) * -math.log(10000.0) / config.embed_dim)

        # (SEQ_LEN, EMBED_DIM)
        position_encodings: torch.Tensor = torch.zeros(config.seq_len, config.embed_dim)

        # PE(positions, 2i) = sin(positions / (10000 ^ (2i/embed_dim)))
        # PE(positions, 2i) = sin(positions * exp(-2i * log(10000) / embed_dim))
        position_encodings[:, ::2] = torch.sin(positions * div_term)

        # PE(positions, 2i+1) = cos(positions / (10000 ^ (2i/embed_dim)))
        # PE(positions, 2i+1) = cos(positions * exp(-2i * log(10000) / embed_dim))
        position_encodings[:, 1::2] = torch.cos(positions * div_term)

        # (SEQ_LEN, EMBED_DIM) --> (1, SEQ_LEN, EMBED_DIM)
        position_encodings = position_encodings.unsqueeze(0)

        self.register_buffer("position_encodings", position_encodings)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return x + self.position_encodings[:, offset:offset + x.shape[1], :]


class MultiHeadAttentionBlock(nn.Module):
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
        
    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM), attn_mask -> (SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
        use_cache: bool = False,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # (N_BATCHES, SEQ_LEN, EMBED_DIM) @ (EMBED_DIM, EMBED_DIM) --> (N_BATCHES, SEQ_LEN, EMBED_DIM)
        key: torch.Tensor = self.Wk(x)
        query: torch.Tensor = self.Wq(x)
        value: torch.Tensor = self.Wv(x)

        # Cache accumulates past tokens; the model only returns the new KV pairs.
        # Concatenation of past+new is the cache's responsibility.
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
        # every one of the n_blocks calls. SDPA accepts a boolean attn_mask directly.
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
    

class FeedForwardBlock(nn.Module):
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


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.post_norm = config.post_norm
        self.dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)

        self.feed_forward = FeedForwardBlock(config)
        self.masked_multihead_attention = MultiHeadAttentionBlock(config)

    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM), attn_mask -> (SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
        use_cache: bool = False,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, SlidingKVCache | None]:
        if self.post_norm:
            x_update, new_kv = self.masked_multihead_attention(x, attn_mask, is_causal, use_cache, kv_cache)
            x = self.norm1(x + self.dropout(x_update))
            x = self.norm2(x + self.dropout(self.feed_forward(x)))
        else:
            x_update, new_kv = self.masked_multihead_attention(self.norm1(x), attn_mask, is_causal, use_cache, kv_cache)
            x = x + self.dropout(x_update)
            x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x, new_kv


class Projection(nn.Module):
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
        
        self.embedding = Embedding(config)
        self.projection = Projection(config)
        self.position_encoder = PositionEncoder(config)
        self.decoders = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_blocks)])
        self.norm_f = nn.LayerNorm(config.embed_dim)
        if config.tie_weights:
            # Tie input embedding and output projection weights (standard for decoder-only LMs).
            # Both are (vocab_size, embed_dim), sharing one tensor halves that parameter block.
            self.projection.linear.weight = self.embedding.embedding.weight

    # Input shape: x -> (N_BATCHES, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def _embed_and_encode_position(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        x = self.embedding(x)
        return self.position_encoder(x, position_offset)

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
    ) -> torch.Tensor:
        # During decode (single new token against full KV cache), Q is shorter than K.
        # The cache already enforces causal ordering, so no mask is needed.
        in_decode_phase = use_cache and kv_caches is not None and kv_caches[0].get() is not None
        attn_mask = mask if (mask is not None and not in_decode_phase) else None
        is_causal = (mask is None) and not in_decode_phase

        for i, decoder in enumerate(self.decoders):
            kv_cache = None if not use_cache else kv_caches[i].get()
            x, new_kv = decoder(x, attn_mask, is_causal, use_cache, kv_cache)
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
        x = self._embed_and_encode_position(x, position_offset)
        x = self._decode(x, mask, use_cache, kv_caches)
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
                # apply() is children-first: Embedding gets normal(0, 0.02) then Projection
                # overwrites the shared tensor with xavier. Restore normal init.
                nn.init.normal_(model.embedding.embedding.weight, mean=0.0, std=0.02)

        if isinstance(config, ModelWithLoRAConfig):
            LoRAdapter.apply_lora(model, config.lora_targets, config.lora_rank, config.lora_alpha, config.lora_dropout)
            
            if lora_weights:
                model.load_state_dict(lora_weights, strict=False)

        return model