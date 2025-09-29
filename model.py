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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_encodings[:, :x.shape[1], :].requires_grad_(False)

    
class CholeskyFactor(nn.Module):
    """
        Given an input x ∈ R^{...xnxn}, it learns to fit a lower-triangular matrix L with positive diagonals that can be used to construct a symmetric positive-definite matrix
        M = L @ L.T
    """
    
    def __init__(self, metric_dim: int, input_dim: int, epsilon: float, dropout: float):
        super().__init__()
        self.epsilon = epsilon
        self.input_dim = input_dim
        self.metric_dim = metric_dim
        self.dropout = nn.Dropout(dropout)
        
        # Number of params for a lower-triangular matrix (including diagonal)
        n_params = input_dim * (input_dim + 1) // 2
        
        self.gelu = nn.GELU()
        self.linear1 = nn.Linear(input_dim, metric_dim, bias=False)
        self.linear2 = nn.Linear(metric_dim, n_params, bias=False)
        
        tri_rows, tri_cols = torch.tril_indices(self.input_dim, self.input_dim)
        self.register_buffer("I", torch.eye(self.input_dim), persistent=False)
        self.register_buffer("tri_rows", tri_rows, persistent=False)
        self.register_buffer("tri_cols", tri_cols, persistent=False)
        self.register_buffer("diag_idx", torch.arange(self.input_dim), persistent=False)
    
    # Input shape: x -> (..., input_dim)
    # Output shape: (..., input_dim, input_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., input_dim) @ (input_dim, metric_dim) --> (..., metric_dim)
        # (..., metric_dim) @ (metric_dim, n_params) --> (..., n_params)
        result: torch.Tensor = self.linear2(
            self.dropout(self.gelu(self.linear1(x)))
        )
        
        # (..., input_dim, input_dim)
        L = result.new_zeros(*result.shape[:-1],self.input_dim, self.input_dim)
        
        # Fill lower-triangular entries
        L[..., self.tri_rows, self.tri_cols] = result
        
        # Make diagonal strictly positive
        diagonals = L[..., self.diag_idx, self.diag_idx]
        
        # Apply safe softplus to avoid numerical instabilities
        with torch.autocast(device_type=DEVICE.type, enabled=False):
            diag_pos = F.softplus(diagonals.float()) # FP32 math
        L[..., self.diag_idx, self.diag_idx] = diag_pos.to(x.dtype)
        
        # Apply epsilon for numerical stability
        L = L + self.epsilon * self.I
        
        return L


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        assert config.embed_dim % config.heads == 0, "EMBED_DIM is not divisible by heads"

        super().__init__()
        self.heads = config.heads
        self.d_head: int = config.embed_dim // config.heads

        self.dropout = nn.Dropout(config.dropout)
        self.Wq: nn.Linear = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.Wk: nn.Linear = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.Wv: nn.Linear = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.Wo: nn.Linear = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        
        self.epsilon = config.epsilon
        self.cholesky_factor = CholeskyFactor(config)
    
    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM), mask -> (SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        use_cache: bool = False,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # (N_BATCHES, SEQ_LEN, EMBED_DIM) @ (EMBED_DIM, EMBED_DIM) --> (N_BATCHES, SEQ_LEN, EMBED_DIM)
        key: torch.Tensor = self.Wk(x)
        query: torch.Tensor = self.Wq(x)
        value: torch.Tensor = self.Wv(x)
                
        if use_cache:
            if kv_cache is not None:
                # (N_BATCHES, CACHE_SIZE, EMBED_DIM)
                key_past, value_past = kv_cache
                
                # (N_BATCHES, CACHE_SIZE + SEQ_LEN, EMBED_DIM)
                key = torch.cat([key_past, key], dim=1)
                value = torch.cat([value_past, value], dim=1)
            kv_cache = key, value

        # (N_BATCHES, SEQ_LEN, EMBED_DIM) --> (N_BATCHES, SEQ_LEN, HEADS, d_head) --> (N_BATCHES, HEADS, SEQ_LEN, d_head)
        query = query.view(query.shape[0], query.shape[1], self.heads, -1).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, -1).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, -1).transpose(1, 2)
                
        # (N_BATCHES, HEADS, SEQ_LEN, d_head) --> (N_BATCHES, HEADS, SEQ_LEN, d_head, d_head)
        L = self.cholesky_factor(query)
        
        # [d_head]∑(N_BATCHES, HEADS, SEQ_LEN, d_head) . (N_BATCHES, HEADS, SEQ_LEN, d_head, d_head) --> (N_BATCHES, HEADS, SEQ_LEN, d_head)
        key_tilde = torch.einsum("bhsd,bhsdd->bhsd", key, L)
        query_tilde = torch.einsum("bhsd,bhsdd->bhsd", query, L)
        
        attn_bias = None
        if mask is not None:
            attn_bias = torch.zeros_like(mask, dtype=query.dtype)
            attn_bias.masked_fill_(mask.logical_not(), -1e4)
        
        output = F.scaled_dot_product_attention(
            query_tilde, key_tilde, value,
            attn_mask=attn_bias,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=(mask is None),
        )

        # (N_BATCHES, HEADS, SEQ_LEN, d_head) -> (N_BATCHES, SEQ_LEN, HEADS, d_head)
        output = output.transpose(1, 2)

        # (N_BATCHES, SEQ_LEN, HEADS, d_head) -> (N_BATCHES, SEQ_LEN, EMBED_DIM)
        output = output.contiguous().view(*x.shape[:-1], -1)
        
        return self.Wo(output), kv_cache
    

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
        self.dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        
        self.feed_forward = FeedForwardBlock(config)
        self.masked_multihead_attention = MultiHeadAttentionBlock(config)
    
    # Input shape: x -> (N_BATCHES, SEQ_LEN, EMBED_DIM), mask -> (SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        use_cache: bool = False,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, SlidingKVCache | None]:
        # Post-Norm
        # x_update, kv_cache = self.masked_multihead_attention(x, mask, use_cache, kv_cache)
        # x = x + self.dropout(self.norm1(x_update))
        # x = x + self.dropout(self.norm2(self.feed_forward(x)))
        
        # Pre-Norm
        x_update, kv_cache = self.masked_multihead_attention(self.norm1(x), mask, use_cache, kv_cache)
        x = x + self.dropout(x_update)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x, kv_cache


class Projection(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear = nn.Linear(config.embed_dim, config.vocab_size)

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

    # Input shape: x -> (N_BATCHES, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, EMBED_DIM)
    def _embed_and_encode_position(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        return self.position_encoder(x)

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
        kv_caches: list[SlidingKVCache] = []
    ) -> torch.Tensor:
        for i, decoder in enumerate(self.decoders):
            kv_cache = None if not use_cache else kv_caches[i].get()
            x, new_kv_cache = decoder(x, mask, use_cache, kv_cache)
            if use_cache:
                kv_caches[i].append(new_kv_cache[0], new_kv_cache[1])
        return x

    # Input shape: x -> (N_BATCHES, SEQ_LEN), mask -> (SEQ_LEN, SEQ_LEN)
    # Output shape: (N_BATCHES, SEQ_LEN, VOCAB_SIZE)
    def forward(
        self,
        x: torch.Tensor, 
        mask: torch.Tensor,
        use_cache: bool = False,
        kv_caches: list[SlidingKVCache] = []
    ) -> torch.Tensor:
        x = self._embed_and_encode_position(x)
        x = self._decode(x, mask, use_cache, kv_caches)
        return self._project(x)
    

    @staticmethod
    def build(
        config: ModelConfig | ModelWithLoRAConfig,
        weights: dict = {}
    ):
        model = GPTmodel(config)
                
        lora_weights = {k: v for k, v in weights.items() if k in LoRAdapter.get_lora_param_names(config.lora_targets)}
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
            
        if isinstance(config, ModelWithLoRAConfig):
            LoRAdapter.apply_lora(model, config.lora_targets, config.lora_rank, config.lora_alpha, config.lora_dropout)
            
            if lora_weights:
                model.load_state_dict(lora_weights, strict=False)

        return model