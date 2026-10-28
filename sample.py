
import argparse

import torch
from tokenizers import Tokenizer

from model import GPT, Config


parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    default="Once upon a time, there was a little dragon"
)

parser.add_argument("--tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--repetition-penalty", type=float, default=1.12)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

if args.temperature <= 0:
    raise ValueError("Temperature must be positive.")

if args.repetition_penalty < 1:
    raise ValueError("Repetition penalty must be at least 1.")

torch.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_file("data/tokenizer.json")

checkpoint = torch.load(
    "checkpoints/best.pt",
    map_location="cpu",
    weights_only=True
)

config = Config(**checkpoint["config"])
model = GPT(config).to(device)

model.load_state_dict(checkpoint["model"])
model.eval()

print("V0 loaded!")
print("Training step:", checkpoint["step"])
print("Temperature:", args.temperature)
print("Repetition penalty:", args.repetition_penalty)

tokens = torch.tensor(
    [tokenizer.encode(args.prompt).ids],
    dtype=torch.long,
    device=device
)

prompt_length = tokens.shape[1]

eos_id = tokenizer.token_to_id("<|endoftext|>")

with torch.inference_mode():
    for _ in range(args.tokens):

        context = tokens[:, -config.seq_len:]

        logits, _ = model(context)
        logits = logits[:, -1, :].float()

        # Penalize tokens repeated in recent generated text
        recent_tokens = (
            tokens[0, prompt_length:][-64:].unique()
        )

        if recent_tokens.numel() > 0:
            previous_scores = logits[:, recent_tokens]

            penalty = args.repetition_penalty

            logits[:, recent_tokens] = torch.where(
                previous_scores > 0,
                previous_scores / penalty,
                previous_scores * penalty
            )

        logits /= args.temperature

        k = min(50, logits.size(-1))
        values, _ = torch.topk(logits, k=k)
        cutoff = values[:, -1:]

        logits[logits < cutoff] = float("-inf")

        probabilities = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1
        )

        if next_token.item() == eos_id:
            break

result = tokenizer.decode(
    tokens[0].tolist(),
    skip_special_tokens=True
)

print("\nV0 GENERATED STORY:\n")
print(result)
