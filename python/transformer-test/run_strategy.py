import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
import argparse

# ==============================================================================
# Part 1: Data Generation and Feature Engineering
# ==============================================================================
def generate_and_save_data(filename="mock_data.csv", num_days=252):
    """生成模拟数据并保存到CSV文件。"""
    if os.path.exists(filename):
        print(f"Data file '{filename}' already exists, skipping generation.")
        return pd.read_csv(filename, index_col='timestamp', parse_dates=True)

    print(f"Generating {num_days} days of mock data and saving to '{filename}'...")
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
    print("Data generation and saving complete.")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def create_features(df):
    """根据原始数据计算模型所需的特征"""
    print("Creating features...")
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['oi_change'] = df['open_interest'].pct_change()
    df['spread'] = df['ask_price'] - df['bid_price']
    df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'])
    df['is_uptime'] = (df['close'] > df['open']).astype(int)
    df['target_return'] = df['close'].pct_change(5).shift(-5)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# ==============================================================================
# Part 2: PyTorch Dataset and Model Design (Unified Version)
# ==============================================================================
class FuturesDataset(Dataset):
    """PyTorch Dataset for futures time series data."""
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

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Custom encoder layer to always return attention weights."""
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
    """Unified Causal Transformer model definition."""
    def __init__(self, num_features, model_dim, num_heads, num_layers, seq_len, dropout=0.1):
        super(CausalTransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(num_features, model_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(model_dim, 1)
        self.register_buffer('causal_mask', self.generate_causal_mask(seq_len))

    def generate_causal_mask(self, size):
        return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        
    def forward(self, src, return_attention=False):
        src = self.input_projection(src) * np.sqrt(self.input_projection.out_features)
        src += self.pos_encoder
        
        last_attention_weights = None
        for i, layer in enumerate(self.encoder_layers):
            src, attn = layer(src, src_mask=self.causal_mask)
            if i == len(self.encoder_layers) - 1:
                last_attention_weights = attn
        
        prediction = self.output_layer(src[:, -1, :])
        
        if return_attention:
            return prediction.squeeze(-1), last_attention_weights
        return prediction.squeeze(-1)

# ==============================================================================
# Part 3: Training and Backtesting Logic
# ==============================================================================
def run_training(args):
    """Handles the entire training process."""
    print("\n--- Mode: Training ---")
    raw_df = generate_and_save_data(filename=args.data_file)
    featured_data = create_features(raw_df)
    
    split_idx = int(len(featured_data) * args.train_split)
    train_df, val_df = featured_data.iloc[:split_idx], featured_data.iloc[split_idx:]
    print(f"\nData split: {len(train_df)} training samples, {len(val_df)} validation samples.")
    
    train_dataset = FuturesDataset(train_df, args.features, 'target_return', args.seq_len)
    val_dataset = FuturesDataset(val_df, args.features, 'target_return', args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CausalTransformerEncoder(
        num_features=len(args.features), model_dim=args.model_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, seq_len=args.seq_len
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
        for features, targets in train_pbar:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
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
                outputs = model(features)
                val_loss += criterion(outputs, targets).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.model_file)
            print(f"Validation loss decreased. Saving model to {args.model_file}")
    print(f"\nTraining complete! Best model saved to '{args.model_file}'")

class Backtester:
    """Handles the backtesting process."""
    def __init__(self, model, data, features, seq_len=60, hold_period=15):
        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()
        self.data = data.reset_index() # Ensure we have integer-based index
        self.features = features
        self.seq_len = seq_len
        self.hold_period = hold_period
        self.trades, self.position, self.entry_price, self.entry_time_step = [], 'NONE', 0, -1

    def run(self):
        print("\n--- Mode: Backtesting ---")
        for i in tqdm(range(self.seq_len, len(self.data)), desc="Backtesting Progress"):
            if self.position != 'NONE' and (i - self.entry_time_step) >= self.hold_period:
                self.close_position(i)

            if self.position == 'NONE':
                input_df = self.data.iloc[i - self.seq_len : i]
                model_input = torch.tensor(input_df[self.features].values.astype(np.float32)).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    pred_return, attn_weights = self.model(model_input, return_attention=True)
                
                pred_return = pred_return.item()
                last_step_attn = attn_weights[0, -1, :].cpu().numpy()
                bullish_score = np.sum(last_step_attn * input_df['is_uptime'].values)
                
                if pred_return > 0.001 and bullish_score > 0.6: self.open_position('LONG', i)
                elif pred_return < -0.001 and (1 - bullish_score) > 0.6: self.open_position('SHORT', i)

        if self.position != 'NONE': self.close_position(len(self.data) - 1)
        self.calculate_performance()

    def open_position(self, side, time_step):
        self.position, self.entry_time_step = side, time_step
        self.entry_price = self.data.at[time_step, 'close']
        print(f"\n{self.data.at[time_step, 'timestamp']}: Open {side} @ {self.entry_price:.2f}")

    def close_position(self, time_step):
        close_price = self.data.at[time_step, 'close']
        pnl = (close_price - self.entry_price) if self.position == 'LONG' else (self.entry_price - close_price)
        self.trades.append({"pnl_pct": (pnl / self.entry_price) * 100})
        print(f"{self.data.at[time_step, 'timestamp']}: Close {self.position} @ {close_price:.2f}, PnL: {self.trades[-1]['pnl_pct']:.4f}%")
        self.position = 'NONE'

    def calculate_performance(self):
        print("\n--- Backtest Performance ---")
        if not self.trades:
            print("No trades were executed.")
            return
        trade_df = pd.DataFrame(self.trades)
        win_rate = (trade_df['pnl_pct'] > 0).mean()
        sharpe = (trade_df['pnl_pct'].mean() / trade_df['pnl_pct'].std()) * np.sqrt(252*4*60) if trade_df['pnl_pct'].std() > 0 else 0
        print(f"Total Trades: {len(trade_df)}, Win Rate: {win_rate:.2%}, Sharpe Ratio (ann, approx.): {sharpe:.2f}")

def run_backtesting(args):
    """Handles the entire backtesting process."""
    if not os.path.exists(args.model_file):
        print(f"Error: Trained model '{args.model_file}' not found. Please run training first.")
        return
    
    raw_df = pd.read_csv(args.data_file, index_col='timestamp', parse_dates=True)
    featured_data = create_features(raw_df)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CausalTransformerEncoder(
        num_features=len(args.features), model_dim=args.model_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, seq_len=args.seq_len
    ).to(device)
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    
    backtester = Backtester(model, featured_data, args.features, args.seq_len)
    backtester.run()

# ==============================================================================
# Part 4: Main Execution Block with Argument Parsing
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or Backtest a Transformer HFT Strategy.")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'backtest'], help="Execution mode: 'train' or 'backtest'")
    
    # Shared arguments
    parser.add_argument('--data_file', type=str, default='mock_data.csv', help="Path to the data file.")
    parser.add_argument('--model_file', type=str, default='transformer_hft_model.pth', help="Path to save/load the model.")
    parser.add_argument('--seq_len', type=int, default=60, help="Input sequence length for the model.")
    parser.add_argument('--model_dim', type=int, default=32, help="Model dimension (d_model).")
    parser.add_argument('--num_heads', type=int, default=4, help="Number of attention heads.")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of transformer layers.")

    # Training-specific arguments
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--train_split', type=float, default=0.8, help="Train/validation split ratio.")

    args = parser.parse_args()
    
    # Define features here to be accessible by args
    args.features = ['price_change', 'volume_change', 'oi_change', 'spread', 'depth_imbalance']

    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'backtest':
        run_backtesting(args)

