import torch
from torch import nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, n_embeds, head_size, block_size, dropout) -> None:
        super().__init__()
        self.key = nn.Linear(n_embeds, head_size, bias=False)
        self.query = nn.Linear(n_embeds, head_size, bias=False)
        self.value = nn.Linear(n_embeds, head_size, bias=False)
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
    def __init__(self, n_heads, n_embeds, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embeds, head_size, block_size, dropout) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_embeds, n_embeds)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, n_embeds, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeds, 4 * n_embeds),
            nn.ReLU(),
            nn.Linear(4 * n_embeds, n_embeds),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, n_embeds, n_heads, block_size, dropout):
        super().__init__()
        head_size = n_embeds // n_heads
        self.sa_heads = MultiHeadAttention(
            n_heads, n_embeds, head_size, block_size, dropout
        )
        self.ffwd = FeedForward(n_embeds, dropout)
        self.ln1 = nn.LayerNorm(n_embeds)
        self.ln2 = nn.LayerNorm(n_embeds)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(
        self, vocab_size, n_embeds, block_size, n_heads, n_layers, dropout, device
    ):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeds)
        self.position_embedding_table = nn.Embedding(block_size, n_embeds)
        self.blocks = nn.Sequential(
            *[Decoder(n_embeds, n_heads, block_size, dropout) for _ in range(n_layers)]
        )
        self.lnf = nn.LayerNorm(n_embeds)
        self.lm_head = nn.Linear(n_embeds, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_embeds = self.token_embedding_table(idx)  # BxTxNemb
        pos_embeds = self.position_embedding_table(
            torch.arange(T, device=self.device)
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
            idx_cond = idx[:, -self.block_size :]
            logits, loss = self(idx_cond)  # BxTxC
            logits = logits[:, -1, :]  # BxC
            probs = F.softmax(logits, dim=-1)  # BxC
            idx_next = torch.multinomial(probs, num_samples=1)  # Bx1
            idx = torch.cat((idx, idx_next), dim=1)  # BxT+1

        return idx
