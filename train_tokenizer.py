import os
import json
import argparse
from config import *
import sentencepiece as spm
from typing import Generator, Iterator


class SentenceIterator(Iterator):
    def __init__(self, file_paths: list[str]):
        self.current_file = 0
        self.file_paths = file_paths
        self.generator = self.__gen__()

    def __iter__(self) -> Iterator:
        return self
    
    def __gen__(self) -> Generator:
        for file_path in self.file_paths:
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            # Parse each line as a separate JSON object
                            yield json.loads(line.strip())
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON line: {e}")
                            continue

    def __next__(self):
        item = next(self.generator)
        if item is not None:
            return item
        raise StopIteration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tokenizer model on a text corpus")
    parser.add_argument("--data", required=True, type=str, help="Comma separated paths to the datasets to train the tokenizer on")
    parser.add_argument("--max-sentence-length", type=int, default=1000, help="Maximum number of tokens in a sentence. Any sentences longer than this will be skipped.")
    parser.add_argument("--model-type", type=str, default="unigram", choices=["bpe", "unigram", "char", "word"], help="Model type to use, [bpe, unigram, char, word](default: unigram)")
    parser.add_argument("--model-prefix", type=str, default="tokenizer", help="Path to store the model files")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_MODEL_CONFIG.vocab_size, help="Vocabulary size to use")

    args = parser.parse_args()
    config = TrainingConfig(**args.__dict__)
    
    datasets = args.data.split(',')
    for dataset in datasets:
        if not os.path.exists(dataset):
            raise ValueError(f"Dataset {dataset} does not exist")

    iterator = SentenceIterator(datasets)    
    
    spm.SentencePieceTrainer.Train(
        sentence_iterator=iterator,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        model_type=args.model_type,
        byte_fallback=args.model_type == 'bpe',
        unk_id=0, pad_id=1, bos_id=2, eos_id=3,
        user_defined_symbols='[USER],[BOT],[SYSTEM],[CONTEXT],[STOP],[BREAK]',
        unk_piece='[UNK]', pad_piece='[PAD]', bos_piece='[BOS]', eos_piece='[EOS]',
        allow_whitespace_only_pieces= True,
        train_extremely_large_corpus=True,
        max_sentence_length=args.max_sentence_length
    )
