import torch
from src.utils import encode, decode


def generate(prompt, model, block_size, max_new_tokens, device):
    X = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    X = X[:block_size].unsqueeze(0)
    results = decode(model.generate(X, max_new_tokens=max_new_tokens)[0].tolist())
    return results
