
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer


DATA_DIR = Path("data")

tokenizer = Tokenizer.from_file(
    str(DATA_DIR / "tokenizer.json")
)

eos_id = tokenizer.token_to_id("<|endoftext|>")

assert eos_id is not None

def prepare_split(split, limit, filename):
    print(f"\nPreparing {split} data...")

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split=split,
        streaming=True
    )

    output_path = DATA_DIR / filename

    buffer = []
    total_tokens = 0
    total_stories = 0

    with output_path.open("wb") as file:
        for row in dataset.take(limit):
            text = row["text"]

            token_ids = tokenizer.encode(text).ids
            token_ids.append(eos_id)

            buffer.extend(token_ids)
            total_stories += 1

            if len(buffer) >= 100000:
                array = np.array(
                    buffer, dtype=np.uint16
                )
                array.tofile(file)
                total_tokens += len(array)
                buffer = []

            if total_stories % 10000 == 0:
                print(
                    f"Processed {total_stories:,} stories"
                )

        if buffer:
            array = np.array(
                buffer, dtype=np.uint16
            )
            array.tofile(file)
            total_tokens += len(array)

    print(f"Saved: {output_path}")
    print(f"Stories: {total_stories:,}")
    print(f"Tokens: {total_tokens:,}")


prepare_split(
    "train",
    100000,
    "train.bin"
)

prepare_split(
    "validation",
    5000,
    "val.bin"
)

print("\nDataset preparation complete!")
