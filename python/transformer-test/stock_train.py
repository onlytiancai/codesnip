import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm

# ==============================================================================
# Part 1: Data Generation and Saving
# ==============================================================================
def generate_and_save_data(filename="mock_data.csv", num_days=252):
    """生成模拟数据并保存到CSV文件。"""
    if os.path.exists(filename):
        print(f"数据文件 '{filename}' 已存在，跳过生成。")
        return pd.read_csv(filename, index_col='timestamp', parse_dates=True)

    print(f"正在生成 {num_days} 天的模拟数据并保存到 '{filename}'...")
    num_minutes = num_days * 4 * 60
    base_price = 4000
    data = []
    current_time = pd.to_datetime('2023-01-01')

    for i in range(num_minutes):
        day_minute = i % (4 * 60)
        if day_minute == 0:
            current_time = (current_time.date() + pd.Timedelta(days=1))
            while current_time.weekday() >= 5: # Skip weekends
                current_time += pd.Timedelta(days=1)
            base_price *= (1 + random.uniform(-0.02, 0.02))
        
        timestamp = pd.to_datetime(str(current_time.date()) + ' 09:00:00') + pd.Timedelta(minutes=day_minute)
        
        open_price = base_price + random.uniform(-5, 5)
        close_price = open_price + random.uniform(-3, 3)
        volume = random.randint(10000, 50000)
        open_interest = random.randint(100000, 200000)
        spread = random.uniform(0.1, 0.5)
        
        data.append({
            "timestamp": timestamp, "open": open_price, "close": close_price,
            "volume": volume, "open_interest": open_interest,
            "ask_price": close_price + spread, "bid_price": close_price - spread,
            "ask_depth": random.randint(50, 200), "bid_depth": random.randint(50, 200)
        })
        base_price = close_price

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print("数据生成并保存完毕。")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

# ==============================================================================
# Part 2: Feature Engineering and PyTorch Dataset
# ==============================================================================
def create_features(df):
    """根据原始数据计算模型所需的特征"""
    print("创建特征工程...")
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['oi_change'] = df['open_interest'].pct_change()
    df['spread'] = df['ask_price'] - df['bid_price']
    df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'])
    df['target_return'] = df['close'].pct_change(5).shift(-5)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

class FuturesDataset(Dataset):
    """为期货时间序列数据创建PyTorch数据集"""
    def __init__(self, data, features, target_col, seq_len):
        self.features = data[features].values.astype(np.float32)
        self.target = data[target_col].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        feature_seq = self.features[idx : idx + self.seq_len]
        target_val = self.target[idx + self.seq_len - 1]
        return torch.tensor(feature_seq), torch.tensor(target_val)

# ==============================================================================
# Part 3: Model Design (Model Design) - 统一版本
# ==============================================================================
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """自定义编码器层以总是返回注意力权重"""
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        try:
             attn_output, attn_weights = self.self_attn(src, src, src, 
                                                       attn_mask=src_mask,
                                                       key_padding_mask=src_key_padding_mask,
                                                       need_weights=True)
        except TypeError:
             attn_output, attn_weights = self.self_attn(src, src, src, 
                                                       attn_mask=src_mask,
                                                       key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class CausalTransformerEncoder(nn.Module):
    """统一的因果Transformer模型定义"""
    def __init__(self, num_features, model_dim, num_heads, num_layers, seq_len, dropout=0.1):
        super(CausalTransformerEncoder, self).__init__()
        self.model_dim = model_dim
        self.seq_len = seq_len
        self.input_projection = nn.Linear(num_features, model_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(model_dim, 1)
        self.register_buffer('causal_mask', self.generate_causal_mask(seq_len))

    def generate_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask
        
    def forward(self, src, return_attention=False):
        src = self.input_projection(src) * np.sqrt(self.model_dim)
        src += self.pos_encoder
        
        last_attention_weights = None
        for i, layer in enumerate(self.encoder_layers):
            src, attn = layer(src, src_mask=self.causal_mask)
            if i == len(self.encoder_layers) - 1:
                last_attention_weights = attn
        
        pooled_output = src[:, -1, :] 
        prediction = self.output_layer(pooled_output)
        
        if return_attention:
            return prediction.squeeze(-1), last_attention_weights
        else:
            return prediction.squeeze(-1)

# ==============================================================================
# Part 4: Training and Validation Loop
# ==============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_path):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for features, targets in train_pbar:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features, return_attention=False) # 训练时不需要attention
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features, return_attention=False)
                val_loss += criterion(outputs, targets).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Validation loss decreased. Saving model to {model_path}")

# ==============================================================================
# Part 5: Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    DATA_FILE, MODEL_SAVE_PATH = "mock_data.csv", "transformer_hft_model.pth"
    INPUT_FEATURES = ['price_change', 'volume_change', 'oi_change', 'spread', 'depth_imbalance']
    NUM_FEATURES, MODEL_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN = len(INPUT_FEATURES), 32, 4, 4, 60
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, TRAIN_TEST_SPLIT = 10, 64, 0.001, 0.8
    
    raw_df = generate_and_save_data(filename=DATA_FILE)
    featured_data = create_features(raw_df)
    
    split_idx = int(len(featured_data) * TRAIN_TEST_SPLIT)
    train_df, val_df = featured_data.iloc[:split_idx], featured_data.iloc[split_idx:]
    print(f"\n数据划分: 训练集 {len(train_df)} 条, 验证集 {len(val_df)} 条。")
    
    train_dataset = FuturesDataset(train_df, INPUT_FEATURES, 'target_return', SEQ_LEN)
    val_dataset = FuturesDataset(val_df, INPUT_FEATURES, 'target_return', SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = CausalTransformerEncoder(
        num_features=NUM_FEATURES, model_dim=MODEL_DIM, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, seq_len=SEQ_LEN
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n开始模型训练...")
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device, MODEL_SAVE_PATH)
    print(f"\n训练完成！最优模型已保存至 '{MODEL_SAVE_PATH}'")
