import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# hyperparameters
batch_size = 16
block_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_heads = 4
n_layer = 3
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def precompute_freqs(dim, max_seq_len, base=10000):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cat([freqs, freqs], dim=-1)

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, freqs):
    # x: (B, T, C)
    # freqs: (T, C)
    cos_vals = torch.cos(freqs)
    sin_vals = torch.sin(freqs)
    return (x * cos_vals) + (rotate_half(x) * sin_vals)

class AttentionHead(nn.Module):
   
  def __init__(self, head_size):
      super().__init__()
      self.key = nn.Linear(n_embd, head_size, bias=False)
      self.query = nn.Linear(n_embd, head_size, bias=False)
      self.value = nn.Linear(n_embd, head_size, bias=False)
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
      self.register_buffer('freqs', precompute_freqs(head_size, block_size))

      self.dropout = nn.Dropout(dropout)

  def forward(self, x):
      B, T, C = x.shape
      k = self.key(x)
      q = self.query(x)
      
      # Apply RoPE to query and key
      q = apply_rope(q, self.freqs[:T])
      k = apply_rope(k, self.freqs[:T])

      wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
      wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
      wei = F.softmax(wei, dim=-1)
      wei = self.dropout(wei)

      v  = self.value(x)
      out = wei @ v

      return out

class MHA(nn.Module):
   
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out

class FFN(nn.Module):

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
       nn.Linear(n_embd, 4 * n_embd),
       nn.ReLU(),
       nn.Linear(4 * n_embd, n_embd),
       nn.Dropout(dropout)
    )
  
  def forward(self, x):
     return self.net(x)

class Block(nn.Module):
  
  def __init__(self, n_embd, n_heads):
    super().__init__()
    head_size = n_embd // n_heads
    self.sa = MHA(n_heads, head_size)
    self.ffwd = FFN(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLM(nn.Module):

  def __init__(self):
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    # No position embedding table needed for RoPE
    # self.sa_heads = MHA(4, n_embd//4)
    # self.ffwd = FFN(n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets = None):
    B, T = idx.shape

    token_emb = self.token_embedding_table(idx)
    # With RoPE, we don't add positional embeddings here
    # The positional information is applied in the attention mechanism
    x = token_emb
    # x = self.sa_heads(x)
    # x = self.ffwd(x)
    x = self.blocks(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, _loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

model = BigramLM()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
