
import torch
from model import GPT, Config

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

model = GPT(Config()).to(device)

parameters = sum(p.numel() for p in model.parameters())

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")
print("Parameters:", f"{parameters:,}")

tokens = torch.randint(
    0, 8000, (2, 32), device=device
)

inputs = tokens[:, :-1]
targets = tokens[:, 1:]

logits, loss = model(inputs, targets)

loss.backward()

print("Output shape:", logits.shape)
print("Loss:", loss.item())
print("Backward pass successful!")
