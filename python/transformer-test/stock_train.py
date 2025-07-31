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
    """
    生成模拟数据并保存到CSV文件。
    Args:
        filename (str): 保存数据的文件名。
        num_days (int): 模拟的交易天数。
    """
    if os.path.exists(filename):
        print(f"数据文件 '{filename}' 已存在，跳过生成。")
        return pd.read_csv(filename, index_col='timestamp', parse_dates=True)

    print(f"正在生成 {num_days} 天的模拟数据并保存到 '{filename}'...")
    # 每个交易日4小时，每小时60分钟
    num_minutes = num_days * 4 * 60
    base_price = 4000
    data = []
    start_time = pd.to_datetime('2023-01-01 09:00:00')
    
    current_time = start_time
    for i in range(num_minutes):
        # 模拟交易日之间的跳空
        if i > 0 and i % (4 * 60) == 0:
            current_time = current_time.date() + pd.Timedelta(days=1)
            current_time = pd.to_datetime(str(current_time) + ' 09:00:00')
            base_price *= (1 + random.uniform(-0.02, 0.02)) # 夜盘跳空
        else:
            current_time += pd.Timedelta(minutes=1)

        open_price = base_price + random.uniform(-5, 5)
        close_price = open_price + random.uniform(-3, 3)
        high = max(open_price, close_price) + random.uniform(0, 2)
        low = min(open_price, close_price) - random.uniform(0, 2)
        volume = random.randint(10000, 50000)
        open_interest = random.randint(100000, 200000)
        
        spread = random.uniform(0.1, 0.5)
        ask_price = close_price + spread
        bid_price = close_price - spread
        ask_depth = random.randint(50, 200)
        bid_depth = random.randint(50, 200)

        data.append({
            "timestamp": current_time,
            "open": open_price,
            "close": close_price,
            "high": high,
            "low": low,
            "volume": volume,
            "open_interest": open_interest,
            "ask_price": ask_price,
            "bid_price": bid_price,
            "ask_depth": ask_depth,
            "bid_depth": bid_depth
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
    
    # 目标变量：未来5分钟的收益率
    df['target_return'] = df['close'].pct_change(5).shift(-5)
    
    # 清理因计算产生NaN值的行
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

class FuturesDataset(Dataset):
    """为期货时间序列数据创建PyTorch数据集"""
    def __init__(self, data, features, target_col, seq_len):
        self.data = data
        self.features = data[features].values.astype(np.float32)
        self.target = data[target_col].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # 特征序列
        feature_seq = self.features[idx : idx + self.seq_len]
        # 目标值对应序列的最后一个时间点
        target_val = self.target[idx + self.seq_len - 1]
        
        return torch.tensor(feature_seq, dtype=torch.float32), torch.tensor(target_val, dtype=torch.float32)

# ==============================================================================
# Part 3: Model Design (Same as before, but included for completeness)
# ==============================================================================
class CausalTransformerEncoder(nn.Module):
    def __init__(self, num_features, model_dim, num_heads, num_layers, seq_len, dropout=0.1):
        super(CausalTransformerEncoder, self).__init__()
        self.model_dim = model_dim
        self.seq_len = seq_len
        self.input_projection = nn.Linear(num_features, model_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, 1)
        self.register_buffer('causal_mask', self.generate_causal_mask(seq_len))

    def generate_causal_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src):
        src = self.input_projection(src) * np.sqrt(self.model_dim)
        src += self.pos_encoder
        output = self.transformer_encoder(src, mask=self.causal_mask)
        pooled_output = output[:, -1, :] 
        prediction = self.output_layer(pooled_output)
        return prediction.squeeze(-1)

# ==============================================================================
# Part 4: Training and Validation Loop
# ==============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_path):
    """
    执行完整的模型训练和验证流程。
    """
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        
        for features, targets in train_pbar:
            features, targets = features.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = running_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for features, targets in val_pbar:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")
        
        # Save the model if validation loss has decreased
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Validation loss decreased. Saving model to {model_path}")

# ==============================================================================
# Part 5: Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    # --- Hyperparameters and Configuration ---
    # Data params
    DATA_FILE = "mock_data.csv"
    SEQ_LEN = 60
    TRAIN_TEST_SPLIT = 0.8
    
    # Model params
    INPUT_FEATURES = ['price_change', 'volume_change', 'oi_change', 'spread', 'depth_imbalance']
    NUM_FEATURES = len(INPUT_FEATURES)
    MODEL_DIM = 32
    NUM_HEADS = 4
    NUM_LAYERS = 4
    
    # Training params
    NUM_EPOCHS = 1 # 为快速演示设为10，实际可设为50-100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    MODEL_SAVE_PATH = "transformer_hft_model.pth"
    
    # --- Step 1: Generate or Load Data ---
    raw_df = generate_and_save_data(filename=DATA_FILE)
    
    # --- Step 2: Feature Engineering ---
    featured_data = create_features(raw_df)
    
    # --- Step 3: Split Data and Create Datasets ---
    split_idx = int(len(featured_data) * TRAIN_TEST_SPLIT)
    train_df = featured_data.iloc[:split_idx]
    val_df = featured_data.iloc[split_idx:]
    
    print(f"\n数据划分: 训练集 {len(train_df)} 条, 验证集 {len(val_df)} 条。")
    
    train_dataset = FuturesDataset(train_df, INPUT_FEATURES, 'target_return', SEQ_LEN)
    val_dataset = FuturesDataset(val_df, INPUT_FEATURES, 'target_return', SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Step 4: Initialize Model, Loss, and Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = CausalTransformerEncoder(
        num_features=NUM_FEATURES,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN
    ).to(device)
    
    criterion = nn.MSELoss() # 均方误差损失，适用于回归任务
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Step 5: Run Training ---
    print("\n开始模型训练...")
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device, MODEL_SAVE_PATH)
    print("\n训练完成！最优模型已保存至 'transformer_hft_model.pth'")

