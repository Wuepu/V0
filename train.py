
import argparse
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model import Config, GPT


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--micro-batch", type=int, default=4)
parser.add_argument("--accum", type=int, default=4)
parser.add_argument("--eval-every", type=int, default=20)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cuda":
    raise RuntimeError("CUDA is required for this training run.")

torch.backends.cuda.matmul.allow_tf32 = True

amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

scaler = torch.amp.GradScaler(
    "cuda", enabled=(amp_dtype == torch.float16)
)

Path("checkpoints").mkdir(exist_ok=True)

tokenizer = Tokenizer.from_file("data/tokenizer.json")
cfg = Config(vocab_size=tokenizer.get_vocab_size())

model = GPT(cfg).to(device)

print("GPU:", torch.cuda.get_device_name(0))
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Precision:", amp_dtype)

train_data = np.memmap(
    "data/train.bin", dtype=np.uint16, mode="r"
)

val_data = np.memmap(
    "data/val.bin", dtype=np.uint16, mode="r"
)

print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))

train_rng = np.random.default_rng(42)


def get_batch(data, rng):
    starts = rng.integers(
        0,
        len(data) - cfg.seq_len,
        size=args.micro_batch
    )

    chunks = np.stack([
        data[int(i):int(i) + cfg.seq_len + 1]
        for i in starts
    ]).astype(np.int64)

    tokens = torch.from_numpy(chunks).to(device)

    return tokens[:, :-1], tokens[:, 1:]


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1
)


def get_lr(step):
    warmup = min(100, max(2, args.steps // 10))

    if step <= warmup:
        return 5e-4 * step / warmup

    progress = (step - warmup) / max(
        1, args.steps - warmup
    )

    return 5e-4 * (
        0.1 + 0.45 * (1 + math.cos(math.pi * progress))
    )


@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    val_rng = np.random.default_rng(1234)

    for _ in range(10):
        x, y = get_batch(val_data, val_rng)

        with torch.autocast("cuda", dtype=amp_dtype):
            _, loss = model(x, y)

        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


best_val = float("inf")
start_step = 1

if args.resume:
    checkpoint = torch.load(
        "checkpoints/last.pt",
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["total_steps"] != args.steps:
        raise ValueError(
            "Resume with the original --steps value."
        )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    train_rng.bit_generator.state = checkpoint["numpy_rng"]
    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    start_step = checkpoint["step"] + 1
    best_val = checkpoint["best_val"]

    print("Resuming from step", start_step)

else:
    print("Initial validation loss:", evaluate())


model.train()

for step in range(start_step, args.steps + 1):
    lr = get_lr(step)

    for group in optimizer.param_groups:
        group["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0

    for _ in range(args.accum):
        x, y = get_batch(train_data, train_rng)

        with torch.autocast("cuda", dtype=amp_dtype):
            _, loss = model(x, y)

        running_loss += loss.item()

        scaler.scale(loss / args.accum).backward()

    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0
    )

    scaler.step(optimizer)
    scaler.update()

    if step % 5 == 0:
        print(
            f"Step {step}/{args.steps} | "
            f"Loss: {running_loss / args.accum:.4f}"
        )

    if step % args.eval_every == 0 or step == args.steps:
        val_loss = evaluate()

        print(
            f"STEP {step} | "
            f"Validation loss: {val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss

            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "step": step
                },
                "checkpoints/best.pt"
            )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": asdict(cfg),
                "step": step,
                "best_val": best_val,
                "total_steps": args.steps,
                "numpy_rng": train_rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            "checkpoints/last.pt"
        )

        print("Checkpoint saved!")

print("V0 training completed!")
