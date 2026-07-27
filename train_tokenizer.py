import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from datasets import load_dataset

# Hardcoded datasets for training code tokenizer & backbone
DATASETS_CONFIG = [
    {"name": "HuggingFaceTB/SmolLM-Corpus", "config": "python", "column": "content", "max_samples": 10000},
    {"name": "codeparrot/codeparrot-clean", "config": None, "column": "content", "max_samples": 10000},
    {"name": "bigcode/the-stack", "config": "data/python", "column": "content", "max_samples": 10000},
]


def train_bpe_tokenizer_from_iterator(
    iterator,
    output_path: str = "tokenizer.json",
    vocab_size: int = 16384,
    min_frequency: int = 2,
):
    """Train a custom Byte-Level BPE Tokenizer for code from a text iterator."""
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    special_tokens = [
        "<unk>",
        "<pad>",
        "<bos>",
        "<eos>",
        "<mask>",
        "```",
        "def ",
        "class ",
        "return ",
        "import ",
        "    ",
        "  ",
    ]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.train_from_iterator(iterator, trainer)
    tokenizer.save(output_path)
    print(f" Saved tokenizer to {output_path} (Actual Vocab Size: {tokenizer.get_vocab_size()})")


def train_bpe_tokenizer(output_path: str = "tokenizer.json", vocab_size: int = 16384):
    """Train a custom Byte-Level BPE Tokenizer hardcoded on the 3 code datasets."""
    print(f"Training Byte-Level BPE tokenizer (vocab size: {vocab_size:,}) across datasets...")

    def multi_dataset_stream_iterator():
        for target in DATASETS_CONFIG:
            ds_name = target["name"]
            ds_cfg = target["config"]
            col = target["column"]
            max_s = target["max_samples"]

            print(f"Streaming up to {max_s:,} samples from '{ds_name}' (config: {ds_cfg})...")
            kwargs = {}
            if ds_cfg:
                kwargs["name"] = ds_cfg

            try:
                ds = load_dataset(ds_name, split="train", streaming=True, **kwargs)
                count = 0
                for sample in ds:
                    text = sample.get(col, "")
                    if text and len(text.strip()) > 0:
                        yield text
                        count += 1
                        if count >= max_s:
                            break
            except Exception as e:
                print(f"Warning: Could not stream dataset '{ds_name}': {e}")

    train_bpe_tokenizer_from_iterator(multi_dataset_stream_iterator(), output_path=output_path, vocab_size=vocab_size)



if __name__ == "__main__":
    train_bpe_tokenizer()
