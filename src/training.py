import torch
from torch import nn

from src.utils import get_batch


@torch.no_grad()
def estimate_loss(model: nn.Module, eval_iters, block_size, batch_size, device):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size, batch_size)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(
    model,
    optimizer,
    max_iters,
    eval_interval,
    eval_iters,
    block_size,
    batch_size,
    device,
):
    val_loss = None
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, eval_iters, block_size, batch_size, device)
            print(
                f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if val_loss is not None:
                if losses["val"] < val_loss:
                    torch.save(model, "checkpoints/model.pth")
            else:
                val_loss = losses["val"]

        xb, yb = get_batch("train", block_size, batch_size)
        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
