import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import random
import os

# ==============================================================================
# 1. 数据模拟与准备 (Data Simulation & Preparation)
# ==============================================================================
def generate_mock_data(num_minutes=5000):
    """生成模拟的1分钟K线和订单簿数据"""
    print(f"生成 {num_minutes} 分钟的模拟数据...")
    base_price = 4000
    data = []
    for i in range(num_minutes):
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
            "timestamp": pd.to_datetime('2023-01-01 09:00:00') + pd.Timedelta(minutes=i),
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
    df.set_index('timestamp', inplace=True)
    return df

def create_features(df):
    """根据原始数据计算模型所需的特征"""
    print("创建特征工程...")
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['oi_change'] = df['open_interest'].pct_change()
    df['spread'] = df['ask_price'] - df['bid_price']
    df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'])
    
    df['is_uptime'] = (df['close'] > df['open']).astype(int)
    
    df['target_return'] = df['close'].pct_change(5).shift(-5)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df

# ==============================================================================
# 2. 模型设计 (Model Design) - 统一版本
# ==============================================================================
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """自定义编码器层以总是返回注意力权重"""
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False): # 兼容新版PyTorch
        # Pytorch 2.0+ has a new argument 'is_causal'
        # We handle both old and new versions
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
# 3. 策略与回测 (Strategy & Backtesting)
# ==============================================================================
class Backtester:
    def __init__(self, model, data, features, seq_len=60, hold_period=15):
        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()
        self.data = data
        self.features = features
        self.seq_len = seq_len
        self.hold_period = hold_period
        
        self.trades = []
        self.position = 'NONE'
        self.entry_price = 0
        self.entry_time_step = -1

    def run(self):
        print("\n开始执行回测...")
        num_steps = len(self.data)
        
        for i in range(self.seq_len, num_steps):
            if self.position != 'NONE' and (i - self.entry_time_step) >= self.hold_period:
                self.close_position(i)

            if self.position == 'NONE':
                input_sequence_df = self.data.iloc[i - self.seq_len : i]
                model_input_np = input_sequence_df[self.features].values.astype(np.float32)
                model_input = torch.tensor(model_input_np).unsqueeze(0).to(self.device)
                is_uptime_sequence = input_sequence_df['is_uptime'].values
                
                with torch.no_grad():
                    predicted_return, attention_weights = self.model(model_input, return_attention=True)
                
                predicted_return = predicted_return.item()
                last_step_attention = attention_weights[0, -1, :].cpu().numpy()
                bullish_attention_score = np.sum(last_step_attention * is_uptime_sequence)
                
                if predicted_return > 0.001 and bullish_attention_score > 0.6:
                    self.open_position('LONG', i)
                elif predicted_return < -0.001 and (1 - bullish_attention_score) > 0.6:
                    self.open_position('SHORT', i)

        if self.position != 'NONE':
            self.close_position(num_steps - 1)
        
        self.calculate_performance()
        
    def open_position(self, side, time_step):
        self.position = side
        self.entry_time_step = time_step
        self.entry_price = self.data.at[time_step, 'close']
        print(f"{self.data.at[time_step, 'timestamp']}: 开仓 {side} @ {self.entry_price:.2f}")

    def close_position(self, time_step):
        close_price = self.data.at[time_step, 'close']
        pnl = (close_price - self.entry_price) if self.position == 'LONG' else (self.entry_price - close_price)
        pnl_pct = (pnl / self.entry_price) * 100
        
        self.trades.append({
            "entry_time": self.data.at[self.entry_time_step, 'timestamp'],
            "exit_time": self.data.at[time_step, 'timestamp'],
            "side": self.position, "entry_price": self.entry_price, "exit_price": close_price,
            "pnl_pct": pnl_pct, "holding_period": time_step - self.entry_time_step
        })
        print(f"{self.data.at[time_step, 'timestamp']}: 平仓 {self.position} @ {close_price:.2f}, PnL: {pnl_pct:.4f}%")
        self.position = 'NONE'

    def calculate_performance(self):
        print("\n--- 回测性能评估 ---")
        if not self.trades:
            print("没有发生任何交易。")
            return
            
        trade_df = pd.DataFrame(self.trades)
        win_rate = (trade_df['pnl_pct'] > 0).mean()
        avg_return = trade_df['pnl_pct'].mean()
        std_return = trade_df['pnl_pct'].std()
        sharpe_ratio = (avg_return / std_return) * np.sqrt(252 * 4 * 60) if std_return > 0 else 0
        max_drawdown = trade_df['pnl_pct'].min()
        
        print(f"总交易次数: {len(trade_df)}")
        print(f"胜率: {win_rate:.2%}")
        print(f"平均每笔收益: {avg_return:.4f}%")
        print(f"夏普比率 (年化, 粗算): {sharpe_ratio:.2f}")
        print(f"最大单笔亏损 (作为最大回撤代理): {max_drawdown:.4f}%")
        print("\n部分交易记录:\n", trade_df.head().to_string())

# ==============================================================================
# 4. 主程序 (Main Execution)
# ==============================================================================
if __name__ == '__main__':
    DATA_FILE = "mock_data.csv"
    MODEL_SAVE_PATH = "transformer_hft_model.pth"
    
    INPUT_FEATURES = ['price_change', 'volume_change', 'oi_change', 'spread', 'depth_imbalance']
    NUM_FEATURES = len(INPUT_FEATURES)
    MODEL_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN = 32, 4, 4, 60
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"错误: 未找到已训练的模型 '{MODEL_SAVE_PATH}'。请先运行训练脚本。")
    else:
        # 使用与训练时相同的数据进行回测
        if os.path.exists(DATA_FILE):
            raw_df = pd.read_csv(DATA_FILE, index_col='timestamp', parse_dates=True)
        else:
            raw_df = generate_mock_data() # 如果没有csv，则生成临时数据
        
        featured_data = create_features(raw_df)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        print(f"\n从 '{MODEL_SAVE_PATH}' 加载已训练的模型...")
        model = CausalTransformerEncoder(
            num_features=NUM_FEATURES, model_dim=MODEL_DIM, num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS, seq_len=SEQ_LEN
        ).to(device)
        
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        
        backtester = Backtester(
            model=model, data=featured_data, features=INPUT_FEATURES,
            seq_len=SEQ_LEN, hold_period=15
        )
        backtester.run()
