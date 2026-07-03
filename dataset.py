import os
import json
import ijson
import math
import torch
import struct
import random
import threading
import numpy as np
import sentencepiece as spm
from bisect import bisect_left
from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset

_tls = threading.local()  # per-thread file handle cache: _tls.handles = {path: file}

from config import *
from preprocessor import AmharicPreprocessor
from utils import Conversation, get_causal_mask


class NLPDataset(Dataset):
    ignore_index = -100
    
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int, workers: int = 0) -> None:
        self.tokens = 0
        self.workers = workers
        self.max_len = max_len
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.preprocessor = AmharicPreprocessor()

        self.pad_token = self.tokenizer.pad_id()
        self._causal_mask = get_causal_mask(max_len)

    def get_loader(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=(sampler is None),      # Sampler itself will handle shuffling if provided
            sampler=sampler,
            num_workers=self.workers,       # Number of subprocesses to use for data loading
            pin_memory=torch.cuda.is_available(),  # pre-allocate batches in page-locked memory so that GPU transfers are faster and can be asynchronous
            drop_last=True,                 # drop the last incomplete(does not have the size 'batch_size') batch
        )
    
    def get_io_tensors(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        # Text:               A B C D E
        # Input Structure:    A B C D $ $ $ $
        # Output Structure:   B C D E $ $ $ $

        text = self.preprocessor.execute(text)
        token_ids = self.tokenizer.Encode(text, out_type=int)[:self.max_len]        
        padding = self.max_len - len(token_ids) + 1
        
        # (SEQ_LEN,)
        input: torch.Tensor = torch.concat([
            # (len(token_ids) - 1,)
            torch.tensor(token_ids[:-1], dtype=torch.int64),
            
            # (padding,)
            torch.tensor([self.pad_token] * padding, dtype=torch.int64)
        ])[:self.max_len]

        # (SEQ_LEN,)
        output = torch.concat([
            # (len(token_ids) - 1,)
            torch.tensor(token_ids[1:], dtype=torch.int64),
            
            # (padding,)
            torch.tensor([self.ignore_index] * padding, dtype=torch.int64)
        ])[:self.max_len]

        return input, output


class TextDataset(NLPDataset):
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int, workers: int = 0) -> None:
        super().__init__(file_path, tokenizer, max_len, workers)

        file_name = os.path.basename(file_path)
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        assert '.jsonl' in file_name, f"Only JSONL files are supported for raw text datasets!"
        with open(file_path, 'r', encoding='utf-8') as f:
            LOGGER.info(f"\033[93mLoading data from {file_name}...\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
            for line in f:
                text = json.loads(line.strip())
                if text and self.preprocessor.execute(text):
                    self.samples.append(self.get_io_tensors(text))
                    if self.samples and len(self.samples) % 100000 == 0:
                        LOGGER.info(f"\033[93mLoaded {len(self.samples)} samples from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[92mDone! Loaded {len(self.samples)} samples from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> dict:
        input, output = self.samples[index]
        
        # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (SEQ_LEN,) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN)
        mask = (input != self.pad_token) & self._causal_mask
        
        return input, output, mask
    

class TextStreamDataset(NLPDataset):
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int, workers: int = 0):
        super().__init__(file_path, tokenizer, max_len, workers)
        
        meta_data = {}
        index_path = file_path + '.index'
        meta_path = file_path + '.meta.json'
        file_name = os.path.basename(file_path)
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
            self.tokens = meta_data['tokens']
        
        if GLOBAL_RANK == COORDINATOR_RANK:
            if meta_data.get('mtime', None) != os.path.getmtime(file_path) or not os.path.exists(index_path):
                assert '.jsonl' in file_name, f"Only JSONL files are supported for raw text datasets!"
                with open(file_path, 'rb', buffering=1024 * 1024) as fin, open(index_path, 'wb') as fout:
                    LOGGER.info(f"\033[93mRegistering offsets from {file_name}...\033[0m")
                    count = 0
                    while True:
                        pos = fin.tell()
                        line = fin.readline()
                        if not line:    break

                        preprocessed_line = self.preprocessor.execute(json.loads(line.strip()))
                        self.tokens += len(self.tokenizer.Encode(preprocessed_line, out_type=int))

                        fout.write(struct.pack('<Q', pos))
                        count += 1
                        if count % 1_000_000 == 0:
                            LOGGER.info(f"\033[93mRegistered {count} offsets from {file_name}\033[0m")

                    with open(meta_path, 'w') as f:
                        json.dump({ "mtime": os.path.getmtime(file_path), "tokens": self.tokens, "samples": count }, f, indent=2)

                LOGGER.info(f"\033[92mDone! Registered {count} offsets from {file_name}\033[0m")
            else:
                LOGGER.info(f"\033[93mUsing existing index for {file_name}\033[0m")

        # Coordinator must finish writing the index before any rank mmaps it.
        if WORLD_SIZE > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()

        # Read-only memmap of uint64 offsets of each sample
        self.offsets = np.memmap(index_path, dtype=np.uint64, mode='r')
        if GLOBAL_RANK == COORDINATOR_RANK:
            index_file = os.path.basename(index_path)
            mib = self.offsets.nbytes / (1024 ** 2)
            LOGGER.info(f"\033[92mEstablished read-only memory mapping with {index_file} ({mib:.2f} MiB) for {len(self.offsets)} offsets\033[0m")
    
    def __len__(self) -> int:
        return len(self.offsets)
        
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = int(self.offsets[index])
        # One open handle per (thread, path) so interleaving train/val datasets doesn't thrash.
        if not hasattr(_tls, 'handles'):
            _tls.handles = {}
        f = _tls.handles.get(self.file_path)
        if f is None:
            f = open(self.file_path, 'rb')
            _tls.handles[self.file_path] = f
        f.seek(offset)
        line = f.readline()

        input, output = self.get_io_tensors(
            text=json.loads(line.strip())
        )
        
        # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (SEQ_LEN,) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN)
        mask = (input != self.pad_token) & self._causal_mask
        
        return input, output, mask


class PackedTextStreamDataset(TextStreamDataset, IterableDataset):

    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int, workers: int = 0) -> None:
        TextStreamDataset.__init__(self, file_path, tokenizer, max_len, workers)
        self._epoch: int = 0

    def __len__(self) -> int:
        return self.tokens // self.max_len

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        # All DDP ranks and DataLoader workers derive their index permutation
        # from the same epoch seed so the global shuffle is consistent, then
        # each party takes a non-overlapping shard.
        rng = random.Random(self._epoch)
        indices = list(range(len(self.offsets)))
        rng.shuffle(indices)

        # DDP shard: each rank gets 1/WORLD_SIZE of the shuffled indices.
        if WORLD_SIZE > 1:
            per_rank = math.ceil(len(indices) / WORLD_SIZE)
            indices = indices[GLOBAL_RANK * per_rank : (GLOBAL_RANK + 1) * per_rank]

        # DataLoader worker shard: split the rank's shard across workers.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = math.ceil(len(indices) / worker_info.num_workers)
            indices = indices[worker_info.id * per_worker : (worker_info.id + 1) * per_worker]

        inp_buf: list[int] = []
        out_buf: list[int] = []
        doc_ends: list[int] = []  # absolute end position of each document in the buffer

        for i in indices:
            inp, out, _ = self[i]
            actual_len = int((inp != self.pad_token).sum())
            if actual_len == 0:
                continue

            inp_buf.extend(inp[:actual_len].tolist())
            out_buf.extend(out[:actual_len].tolist())
            doc_ends.append(len(inp_buf))

            while len(inp_buf) >= self.max_len:
                chunk_inp = inp_buf[:self.max_len]
                chunk_out = out_buf[:self.max_len]

                # Build block-diagonal causal mask: token i attends to token j
                # iff i >= j AND both belong to the same document.
                mask = torch.zeros(1, self.max_len, self.max_len, dtype=torch.bool)
                prev = 0
                for end in doc_ends:
                    end = min(end, self.max_len)
                    n = end - prev
                    if n > 0:
                        mask[0, prev:end, prev:end] = torch.ones(n, n, dtype=torch.bool).tril()
                    prev = end
                    if end == self.max_len:
                        break

                yield (
                    torch.tensor(chunk_inp, dtype=torch.int64),
                    torch.tensor(chunk_out, dtype=torch.int64),
                    mask,
                )

                inp_buf = inp_buf[self.max_len:]
                out_buf = out_buf[self.max_len:]
                doc_ends = [e - self.max_len for e in doc_ends if e > self.max_len]

    def get_loader(self, batch_size: int, sampler: Sampler = None) -> DataLoader:
        # IterableDataset does not support external samplers; sharding is
        # handled inside __iter__ via set_epoch / worker_info / GLOBAL_RANK.
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=False,
            sampler=None,
            num_workers=self.workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )


class FineTuningDataset(NLPDataset):
    def __init__(self, intent: str, file_path: str, samples: list[dict], tokenizer: spm.SentencePieceProcessor, max_len: int) -> None:
        super().__init__(file_path, tokenizer, max_len)
        self.bot_token = self.tokenizer.PieceToId("[BOT]")
        self.user_token = self.tokenizer.PieceToId("[USER]")
        self.stop_token = self.tokenizer.PieceToId("[STOP]")
        self.system_token = self.tokenizer.PieceToId("[SYSTEM]")

        skips = 0
        self.samples = []
        file_name = os.path.basename(file_path)
        LOGGER.info(f"\033[93mTokenizing {len(samples)} samples from {file_name}{'/' + intent if intent else ''}...\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        for sample in samples:
            system_prompt = sample.get("system", "")
            if system_prompt:
                system_prompt = self.preprocessor.execute(system_prompt)
            conv = Conversation(sample['type'], system_prompt)
            for exchange in sample["exchanges"]:
                try:
                    conv.add_exchange(
                        self.preprocessor.execute(exchange["input"]),
                        self.preprocessor.execute(exchange["output"])
                    )
                except Exception as e:
                    LOGGER.error('File must be in JSON format [{"system": ..., "exchanges": [{"input": ..., "output": ...}, ...}]]')
                    exit(1)
            try:
                io_tensors = self.get_io_tensors(conv)
            except ValueError:
                skips += 1
                continue
            self.samples.append(io_tensors)
            self.tokens += int((io_tensors[0] != self.pad_token).sum())

            if self.samples and len(self.samples) % 30000 == 0:
                LOGGER.info(f"\033[93mLoaded {len(self.samples)} samples from {file_name}{'/' + intent if intent else ''}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[92mDone! Loaded {len(self.samples)} samples from {file_name}{'/' + intent if intent else ''}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[93mSkipped {skips} samples from {file_name}{'/' + intent if intent else ''}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[93mUsing {len(self.samples)} samples from {file_name}{'/' + intent if intent else ''}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None

    @classmethod
    def load_many(cls, intents: list[str], file_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int) -> dict[str, "FineTuningDataset"]:
        """Single streaming pass over file_path, bucketing raw samples by `type`
        before tokenizing - instead of every intent re-parsing the whole file
        just to filter out the ~1/len(intents) share it cares about."""
        buckets: dict[str, list[dict]] = {intent: [] for intent in intents}
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            LOGGER.info(f"\033[93mScanning {file_name} for {len(intents)} intents...\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
            for sample in ijson.items(f, 'item'):
                bucket = buckets.get(sample['type'])
                if bucket is not None:
                    bucket.append(sample)

        return {
            intent: cls(intent, file_path, buckets[intent], tokenizer, max_len)
            for intent in intents
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        input, output = self.samples[index]
        
        # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (SEQ_LEN,) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN)
        mask = (input != self.pad_token) & self._causal_mask
        
        return input, output, mask
    
    def get_io_tensors(self, conv: Conversation) -> tuple[torch.Tensor, torch.Tensor]:
        # System:              K L M N O
        # Conversation:        User: A B C D E 
        #                      Bot:  F G H I J
        #                      ...
        #                      User: Q R S T U
        #                      Bot:  V W X Y Z
        # Input Structure:     [SYSTEM] K L M N O [USER] A B C D E [BOT] F G H I   J    ... [USER] Q R S T U [BOT] V W X Y   Z    $ $ $
        # Output Structure:        $    $ $ $ $ $    $   $ $ $ $ $   F   G H I J [STOP] ...    $   $ $ $ $ $   V   W X Y Z [STOP] $ $ $
        
        input_ids: list[int] = []
        output_ids: list[int] = []
        if conv.system_text:
            input_ids.extend([
                self.system_token,
                *self.tokenizer.Encode(conv.system_text, out_type=int)
            ])
            output_ids.extend([self.ignore_index] * len(input_ids))
        
        exchanges_ipt, exchanges_opt = [], []
        for exchange in reversed(conv.exchanges):
            input_token_ids = self.tokenizer.Encode(exchange["input"], out_type=int)
            output_token_ids = self.tokenizer.Encode(exchange["output"], out_type=int)
            
            if len(input_ids) + len(exchanges_ipt) + len(input_token_ids) + len(output_token_ids) + 2 > self.max_len:
                break
            
            # [USER] A B C ... H I J [BOT] K L M ... X Y   Z   
            #   $    $ $ $ ... $ $ $   K   L M O ... Y Z [STOP]
            if input_token_ids and output_token_ids:
                exchanges_ipt = [
                    self.user_token,
                    *input_token_ids,
                    self.bot_token,
                    *output_token_ids
                ] + exchanges_ipt
                exchanges_opt = [
                    *[self.ignore_index] * (len(input_token_ids) + 1),
                    *output_token_ids,
                    self.stop_token
                ] + exchanges_opt
                
        if not exchanges_ipt:
            raise ValueError("Input text too long(or no exchanges)!")
        
        input_ids.extend(exchanges_ipt)
        output_ids.extend(exchanges_opt)
        
        padding = self.max_len - len(input_ids)
        
        # (SEQ_LEN,)
        input: torch.Tensor = torch.concat([
            # (len(input_ids),)
            torch.tensor(input_ids, dtype=torch.int64),
            
            # (padding,)
            torch.tensor([self.pad_token] * padding, dtype=torch.int64)
        ])[:self.max_len]
        
        # (SEQ_LEN,)
        output: torch.Tensor = torch.concat([
            # (len(output_ids),)
            torch.tensor(output_ids, dtype=torch.int64),
            
            # (padding,)
            torch.tensor([self.ignore_index] * padding, dtype=torch.int64),
        ])[:self.max_len]
        
        return input, output


class MultiTaskDataset(Dataset):
    ignore_index = NLPDataset.ignore_index
    
    def __init__(self, datasets: dict[str, NLPDataset], workers: int = 0) -> None:
        self.workers = workers
        self.task_names = list(datasets.keys())
        self.datasets = [datasets[k] for k in self.task_names]
        self.lengths = [len(ds) for ds in self.datasets]
        self.offsets = []
        total = 0
        for L in self.lengths:
            self.offsets.append(total)
            total += L
        self.total_len = total
        self.tokens = sum(getattr(ds, 'tokens', 0) for ds in self.datasets)

    def __len__(self):
        return self.total_len

    def task_probs(self, alpha: float) -> list[float]:
        weights = [L ** alpha for L in self.lengths]
        total = sum(weights)
        return [w / total for w in weights]

    def get_loader(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=(sampler is None),      # Sampler itself will handle shuffling if provided
            sampler=sampler,
            num_workers=self.workers,       # Number of subprocesses to use for data loading
            pin_memory=torch.cuda.is_available(),  # pre-allocate batches in page-locked memory so that GPU transfers are faster and can be asynchronous
            drop_last=True,                 # drop the last incomplete(does not have the size 'batch_size') batch
        )
    
    def __getitem__(self, global_index: int):
        task_id = bisect_left(self.offsets, global_index)
        if task_id == len(self.offsets) or global_index != self.offsets[task_id]:
            task_id -= 1
        return self.datasets[task_id][global_index - self.offsets[task_id]]


class TemperatureSampler(Sampler[int]):
    def __init__(self, mt_dataset: MultiTaskDataset, iter_size: int, alpha: float = 0.5):
        self.alpha = alpha
        self.mt = mt_dataset
        self.iter_size = iter_size
        self.lengths = torch.tensor(self.mt.lengths, dtype=torch.float)
        self.offsets = torch.tensor(self.mt.offsets, dtype=torch.long)
        self.task_probs = mt_dataset.task_probs(alpha)  # probs ∝ n^alpha

    def __len__(self):
        return self.iter_size

    def __iter__(self):
        probs = torch.tensor(self.task_probs, dtype=torch.float)
        for _ in range(self.iter_size):
            task = torch.multinomial(probs, 1).item()
            local_len = int(self.lengths[task].item())
            j = random.randrange(local_len)
            global_index = self.offsets[task].item() + j
            yield global_index


class PackedFineTuningDataset(IterableDataset):
    ignore_index = NLPDataset.ignore_index

    def __init__(self, mt_dataset: MultiTaskDataset, max_len: int, alpha: float = 0.5, samples_per_epoch: int = 0, workers: int = 0) -> None:
        self.mt = mt_dataset
        self.max_len = max_len
        self.workers = workers
        self.pad_token = mt_dataset.datasets[0].pad_token
        self.samples_per_epoch = samples_per_epoch
        self.task_probs = mt_dataset.task_probs(alpha)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Independent RNG stream per (epoch, rank, worker): DataLoader workers are
        # forked from the parent's already-seeded global RNG, so sampling with the
        # module-level `random` here would give every worker identical draws.
        rng = random.Random(f"{self._epoch}-{GLOBAL_RANK}-{worker_id}")
        per_worker_len = math.ceil(self.samples_per_epoch / num_workers)
        task_ids = list(range(len(self.mt.datasets)))

        inp_buf: list[int] = []
        out_buf: list[int] = []
        doc_ends: list[int] = []  # absolute end position of each conversation in the buffer
        yielded = 0

        while yielded < per_worker_len:
            task = rng.choices(task_ids, weights=self.task_probs, k=1)[0]
            j = rng.randrange(self.mt.lengths[task])
            inp, out, _ = self.mt.datasets[task][j]

            actual_len = int((inp != self.pad_token).sum())
            if actual_len == 0:
                continue

            inp_buf.extend(inp[:actual_len].tolist())
            out_buf.extend(out[:actual_len].tolist())
            doc_ends.append(len(inp_buf))

            while len(inp_buf) >= self.max_len:
                chunk_inp = inp_buf[:self.max_len]
                chunk_out = out_buf[:self.max_len]

                # Block-diagonal causal mask: token i attends to token j iff
                # i >= j AND both belong to the same packed conversation.
                mask = torch.zeros(1, self.max_len, self.max_len, dtype=torch.bool)
                prev = 0
                for end in doc_ends:
                    end = min(end, self.max_len)
                    n = end - prev
                    if n > 0:
                        mask[0, prev:end, prev:end] = torch.ones(n, n, dtype=torch.bool).tril()
                    prev = end
                    if end == self.max_len:
                        break

                yield (
                    torch.tensor(chunk_inp, dtype=torch.int64),
                    torch.tensor(chunk_out, dtype=torch.int64),
                    mask,
                )
                yielded += 1
                if yielded >= per_worker_len:
                    break

                inp_buf = inp_buf[self.max_len:]
                out_buf = out_buf[self.max_len:]
                doc_ends = [e - self.max_len for e in doc_ends if e > self.max_len]

    def get_loader(self, batch_size: int, sampler: Sampler = None) -> DataLoader:
        # IterableDataset does not support external samplers; sharding across
        # ranks happens via GLOBAL_RANK in the RNG seed, and across DataLoader
        # workers via worker_info inside __iter__.
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=False,
            sampler=None,
            num_workers=self.workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
