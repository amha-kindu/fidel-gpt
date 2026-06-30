import torch


class SlidingKVCache:
    def __init__(self, size: int):
        self.size = size
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    @torch.no_grad()
    def append(self, new_keys: torch.Tensor, new_values: torch.Tensor) -> None:
        if self.keys is None:
            self.keys, self.values = new_keys, new_values
        else:
            self.keys = torch.cat([self.keys, new_keys], dim=1)
            self.values = torch.cat([self.values, new_values], dim=1)
        extra = self.keys.size(1) - self.size
        if extra > 0:
            self.keys = self.keys[..., extra:, :]
            self.values = self.values[..., extra:, :]

    def get(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.keys is None:
            return None
        return self.keys, self.values
