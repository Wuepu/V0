
from pathlib import Path

from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
    trainers
)

Path("data").mkdir(exist_ok=True)

print("Loading TinyStories dataset...")

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True
)

dataset = dataset.take(20000)

print("Creating tokenizer...")

tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = (
    pre_tokenizers.ByteLevel(add_prefix_space=False)
)

tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["<|endoftext|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

print("Training tokenizer from scratch...")

tokenizer.train_from_iterator(
    (row["text"] for row in dataset),
    trainer=trainer,
    length=20000
)

tokenizer.save("data/tokenizer.json")

print("Tokenizer saved!")
print("Vocabulary size:", tokenizer.get_vocab_size())

example = "The knight attacked the dragon."

encoded = tokenizer.encode(example)

print("Original:", example)
print("Token IDs:", encoded.ids)
print("Decoded:", tokenizer.decode(encoded.ids))
