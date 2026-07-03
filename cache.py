import torch


class SlidingKVCache:
    """Fixed-capacity ring buffer for K/V. Allocated once (lazily, on the first
    append, once the batch/embed_dim/dtype/device are known) and written into
    in place from then on, instead of growing via torch.cat on every step.

    Once full, new tokens overwrite the oldest ones circularly, so get() may
    return the buffer in wrapped (non-chronological) order. That's fine: the
    cached keys/values are only ever attended to with no causal mask (the new
    query is always causally after everything cached), and softmax attention
    over the key/value axis is invariant to that axis's order.
    """

    def __init__(self, size: int):
        self.size = size
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.length = 0
        self.cursor = 0

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self.length = 0
        self.cursor = 0

    @torch.no_grad()
    def append(self, new_keys: torch.Tensor, new_values: torch.Tensor) -> None:
        batch, new_len, embed_dim = new_keys.shape

        if self.keys is None:
            self.keys = new_keys.new_zeros(batch, self.size, embed_dim)
            self.values = new_values.new_zeros(batch, self.size, embed_dim)

        if new_len >= self.size:
            # This chunk alone fills (or overflows) the window; keep just the tail.
            self.keys.copy_(new_keys[:, -self.size:, :])
            self.values.copy_(new_values[:, -self.size:, :])
            self.length = self.size
            self.cursor = 0
            return

        end = self.cursor + new_len
        if end <= self.size:
            self.keys[:, self.cursor:end, :] = new_keys
            self.values[:, self.cursor:end, :] = new_values
        else:
            head = self.size - self.cursor
            self.keys[:, self.cursor:, :] = new_keys[:, :head, :]
            self.values[:, self.cursor:, :] = new_values[:, :head, :]
            self.keys[:, :end - self.size, :] = new_keys[:, head:, :]
            self.values[:, :end - self.size, :] = new_values[:, head:, :]

        self.cursor = end % self.size
        self.length = min(self.length + new_len, self.size)

    def get(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.length == 0:
            return None
        if self.length < self.size:
            return self.keys[:, :self.length, :], self.values[:, :self.length, :]
        return self.keys, self.values
