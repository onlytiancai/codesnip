# tiny_transformer_mps.py
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 设备选择：优先 mps（macOS GPU），否则 CPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

# --- 小的 Synthetic Dataset ---
class RandomSeqDataset(Dataset):
    def __init__(self, n_samples=2000, seq_len=32, vocab_size=100, n_classes=2):
        self.n = n_samples
        self.seq_len = seq_len
        self.vocab = vocab_size
        self.n_classes = n_classes
        # Generate random integer token sequences and a binary label
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len), dtype=torch.long)
        # A toy rule: label = 1 if sum(tokens) % 2 == 0 else 0 (just synthetic)
        self.labels = (self.data.sum(dim=1) % n_classes == 0).long()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# --- Tiny Transformer Encoder-based Classifier ---
class TinyTransformerClassifier(nn.Module):
    def __init__(self, vocab_size=100, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, seq_len=32, n_classes=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # average across seq dimension
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, n_classes)
        )

    def forward(self, x):
        # x: [B, seq_len] (token ids)
        x = self.embedding(x)              # [B, seq_len, d_model]
        x = self.pos_enc(x)               # add positional encoding
        x = x.transpose(0, 1)             # Transformer expects [seq_len, B, d_model]
        x = self.transformer(x)           # [seq_len, B, d_model]
        x = x.transpose(0, 1)             # [B, seq_len, d_model]
        x = x.mean(dim=1)                 # simple mean pooling -> [B, d_model]
        return self.head(x)

# --- Training loop ---
def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        total_acc += (preds == yb).sum().item()
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            total_acc += (preds == yb).sum().item()
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)

# --- Hyperparams and run ---
def main():
    # hyperparams (tiny)
    vocab_size = 100
    seq_len = 32
    d_model = 64
    batch_size = 64
    epochs = 8
    lr = 1e-3

    train_ds = RandomSeqDataset(n_samples=1600, seq_len=seq_len, vocab_size=vocab_size)
    val_ds = RandomSeqDataset(n_samples=400, seq_len=seq_len, vocab_size=vocab_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TinyTransformerClassifier(vocab_size=vocab_size, d_model=d_model, seq_len=seq_len, n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        t1 = time.time()
        print(f"Epoch {ep:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}  time={t1-t0:.2f}s")

if __name__ == "__main__":
    main()
