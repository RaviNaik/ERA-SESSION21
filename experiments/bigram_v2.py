import torch
from torch import nn
import torch.nn.functional as F

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda:1" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embeds = 384
n_heads = 6
n_layers = 6
dropout = 0.2

torch.manual_seed(1123)

with open("input.txt") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return "".join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model: nn.Module):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, n_embed, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)  # (B,T,16) @ (B,16,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_embeds, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embeds, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embeds, n_embeds)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, n_embeds):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeds, 4 * n_embeds),
            nn.ReLU(),
            nn.Linear(4 * n_embeds, n_embeds),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embeds, n_heads):
        super().__init__()
        head_size = n_embeds // n_heads
        self.sa_heads = MultiHeadAttention(n_heads, n_embeds, head_size)
        self.ffwd = FeedForward(n_embeds)
        self.ln1 = nn.LayerNorm(n_embeds)
        self.ln2 = nn.LayerNorm(n_embeds)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embeds, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeds)
        self.position_embedding_table = nn.Embedding(block_size, n_embeds)
        self.blocks = nn.Sequential(
            *[Block(n_embeds, n_heads) for _ in range(n_layers)]
        )
        self.lnf = nn.LayerNorm(n_embeds)
        self.lm_head = nn.Linear(n_embeds, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_embeds = self.token_embedding_table(idx)  # BxTxNemb
        pos_embeds = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # TXNemb

        x = tok_embeds + pos_embeds  # BxTxNemb
        x = self.blocks(x)
        x = self.lnf(x)
        logits = self.lm_head(x)  # BxTxVocabSize

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)  # BxTxC
            logits = logits[:, -1, :]  # BxC
            probs = F.softmax(logits, dim=-1)  # BxC
            idx_next = torch.multinomial(probs, num_samples=1)  # Bx1
            idx = torch.cat((idx, idx_next), dim=1)  # BxT+1

        return idx


model = BigramLanguageModel(vocab_size, n_embeds, block_size)

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(
            f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
results = decode(model.generate(context, max_new_tokens=100)[0].tolist())
print(results)
