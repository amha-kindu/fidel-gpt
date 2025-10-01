import os
import json
import ijson
import torch
import struct
import random
import numpy as np
import sentencepiece as spm
from bisect import bisect_left
from torch.utils.data import Dataset, DataLoader, Sampler

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

    def get_loader(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=(sampler is None),      # Sampler itself will handle shuffling if provided
            sampler=sampler,
            num_workers=self.workers,       # Number of subprocesses to use for data loading
            pin_memory=True,                # pre-allocate batches in page-locked memory so that GPU transfers are faster and can be asynchronous
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
                preprocessed_text = self.preprocessor.execute(text)
                if preprocessed_text:
                    self.samples.append(self.get_io_tensors(preprocessed_text))
                    if self.samples and len(self.samples) % 100000 == 0:
                        LOGGER.info(f"\033[93mLoaded {len(self.samples)} samples from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[92mDone! Loaded {len(self.samples)} samples from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> dict:
        input, output = self.samples[index]
        
        # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (SEQ_LEN,) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN)
        mask = (input == self.pad_token) | get_causal_mask(self.max_len)
        
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
        
        if meta_data.get('mtime', None) != os.path.getmtime(file_path):
            assert '.jsonl' in file_name, f"Only JSONL files are supported for raw text datasets!"
            with open(file_path, 'rb', buffering=1024 * 1024) as fin, open(index_path, 'wb') as fout:
                LOGGER.info(f"\033[93mRegistering offsets from {file_name}...\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
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
                        LOGGER.info(f"\033[93mRegistered {count} offsets from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
                    
                with open(meta_path, 'w') as f:
                    json.dump({ "mtime": os.path.getmtime(file_path), "tokens": self.tokens, "samples": count }, f, indent=2)
            
            LOGGER.info(f"\033[92mDone! Registered {count} offsets from {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        else:
            LOGGER.info(f"\033[93mUsing existing index for {file_name}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        
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
        with open(self.file_path, 'rb', buffering=1024 * 1024) as f:
            f.seek(offset)
            line = f.readline()       
        
        input, output = self.get_io_tensors(
            text=json.loads(line.strip())
        )
        
        # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (SEQ_LEN,) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN)
        mask = (input == self.pad_token) | get_causal_mask(self.max_len)
        
        return input, output, mask


class FineTuningDataset(NLPDataset):
    def __init__(self, intent: str, file_path: str, tokenizer: spm.SentencePieceProcessor, max_len: int) -> None:
        super().__init__(file_path, tokenizer, max_len)
        self.bot_token = self.tokenizer.PieceToId("[BOT]")
        self.user_token = self.tokenizer.PieceToId("[USER]")
        self.stop_token = self.tokenizer.PieceToId("[STOP]")
        self.system_token = self.tokenizer.PieceToId("[SYSTEM]")

        skips = 0
        self.samples = []
        self.file = open(file_path, 'r', encoding='utf-8')
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            LOGGER.info(f"\033[93mLoading data from {file_name}{'/' + intent if intent else ''}...\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
            for sample in ijson.items(f, 'item'):
                if intent and sample['type'] != intent:
                    continue
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
                    self.samples.append(self.get_io_tensors(conv))
                except ValueError:
                    skips += 1
                    continue

                if self.samples and len(self.samples) % 30000 == 0:
                    LOGGER.info(f"\033[93mLoaded {len(self.samples)} samples from {file_name}{'/' + intent if intent else ''}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None    
        LOGGER.info(f"\033[92mDone! Loaded {len(self.samples)} samples from {file_name}{'/' + intent if intent else ''}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[93mSkipped {skips} samples from {file_name}{'/' + intent if intent else ''}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
        LOGGER.info(f"\033[93mUsing {len(self.samples)} samples from {file_name}{'/' + intent if intent else ''}\033[0m") if GLOBAL_RANK == COORDINATOR_RANK else None
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        input, output = self.samples[index]
        
        # (SEQ_LEN,) != (1,) --> (SEQ_LEN,) --> (SEQ_LEN,) & (1, SEQ_LEN, SEQ_LEN) --> (1, SEQ_LEN, SEQ_LEN)
        mask = (input == self.pad_token) | get_causal_mask(self.max_len)
        
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

    def __len__(self):
        return self.total_len    
    
    def get_loader(self, batch_size: int, sampler: Sampler=None) -> DataLoader:
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=(sampler is None),      # Sampler itself will handle shuffling if provided
            sampler=sampler,
            num_workers=self.workers,       # Number of subprocesses to use for data loading
            pin_memory=True,                # pre-allocate batches in page-locked memory so that GPU transfers are faster and can be asynchronous
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
        
        weights = (self.lengths.clamp(min=1.0)) ** alpha # probs ∝ n^alpha
        self.task_probs = (weights / weights.sum()).tolist()

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
