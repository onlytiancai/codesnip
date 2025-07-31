# ==============================================================================
#                 高频趋势跟踪策略: Transformer 模型训练与回测
#
# 介绍:
#   本脚本实现了一个基于Transformer模型的期货高频趋势跟踪策略。它被设计为
#   一个一体化的工具，可以通过命令行参数控制其运行模式：训练或回测。
#
# 功能:
#   1. 数据模拟: 如果本地没有数据，脚本会自动生成模拟的1分钟K线数据。
#   2. 特征工程: 基于原始数据计算模型所需的多种技术特征。
#   3. 模型训练 (--mode train):
#      - 使用历史数据训练一个4层因果Transformer模型。
#      - 模型的目标是预测未来5分钟的收益率。
#      - 训练过程中会自动保存验证集上表现最好的模型。
#   4. 策略回测 (--mode backtest):
#      - 加载已训练好的模型。
#      - 在整个数据集上模拟交易，应用策略规则。
#      - 输出详细的性能报告并绘制包含权益曲线和交易点位的图表。
#
# 依赖库:
#   - torch
#   - pandas
#   - numpy
#   - matplotlib
#   - tqdm
#   请使用 'pip install torch pandas numpy matplotlib tqdm' 命令安装。
#
# 使用方式:
#   1. 训练模型:
#      python this_script_name.py --mode train --epochs 20 --lr 0.001
#
#   2. 执行回测 (在训练完成后):
#      python this_script_name.py --mode backtest
#
# ==============================================================================

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
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==============================================================================
# Part 0: 日志配置 (Logging Configuration)
# ==============================================================================
# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# ==============================================================================
# Part 1: 数据生成与特征工程 (Data Generation and Feature Engineering)
# ==============================================================================
def generate_and_save_data(filename="mock_data.csv", num_days=252):
    """
    生成模拟的1分钟K线和订单簿数据，并保存到CSV文件。
    这个函数只在数据文件不存在时运行。
    """
    if os.path.exists(filename):
        logging.info(f"数据文件 '{filename}' 已存在，跳过生成。")
        return pd.read_csv(filename, index_col='timestamp', parse_dates=True)

    logging.info(f"正在生成 {num_days} 天的模拟数据并保存到 '{filename}'...")
    num_minutes = num_days * 4 * 60  # 假设每个交易日4小时
    base_price = 4000
    data = []
    current_date = pd.to_datetime('2023-01-01').normalize()

    for i in range(num_minutes):
        day_minute = i % (4 * 60)
        if day_minute == 0:
            # 进入下一个交易日
            current_date += pd.Timedelta(days=1)
            # 跳过周末
            while current_date.weekday() >= 5:
                current_date += pd.Timedelta(days=1)
            # 模拟隔夜跳空
            base_price *= (1 + random.uniform(-0.02, 0.02))
        
        timestamp = current_date + pd.Timedelta(hours=9, minutes=day_minute)
        
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
    logging.info("数据生成并保存完毕。")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def create_features(df):
    """根据原始数据计算模型所需的特征"""
    logging.info("正在创建特征工程...")
    # 价格、成交量、持仓量变化率
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['oi_change'] = df['open_interest'].pct_change()
    # 买卖价差
    df['spread'] = df['ask_price'] - df['bid_price']
    # 深度不平衡
    df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'])
    # 是否为上涨K线（收盘价>开盘价），用于计算注意力权重
    df['is_uptime'] = (df['close'] > df['open']).astype(int)
    # 目标变量：未来5分钟的收益率，用于模型训练
    df['target_return'] = df['close'].pct_change(5).shift(-5)
    # 处理计算过程中可能产生的无穷大值和空值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# ==============================================================================
# Part 2: PyTorch数据集和模型设计 (PyTorch Dataset and Model Design)
# ==============================================================================
class FuturesDataset(Dataset):
    """为期货时间序列数据创建PyTorch数据集"""
    def __init__(self, data, features, target_col, seq_len):
        self.features = data[features].values.astype(np.float32)
        self.target = data[target_col].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        # 特征序列是长度为seq_len的窗口
        feature_seq = self.features[idx : idx + self.seq_len]
        # 目标值对应序列的最后一个时间点
        target_val = self.target[idx + self.seq_len - 1]
        return torch.tensor(feature_seq), torch.tensor(target_val)

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """自定义编码器层，以方便地获取注意力权重"""
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        try:
             # 新版PyTorch需要need_weights参数
             attn_output, attn_weights = self.self_attn(src, src, src, 
                                                       attn_mask=src_mask,
                                                       key_padding_mask=src_key_padding_mask,
                                                       need_weights=True)
        except TypeError:
             # 兼容旧版PyTorch
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
        # 输入层：将N维特征映射到模型内部的D维空间
        self.input_projection = nn.Linear(num_features, model_dim)
        # 位置编码：让模型学习到序列中的位置信息
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        # Transformer编码器层列表
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        # 输出层：从D维空间映射回1维的预测值
        self.output_layer = nn.Linear(model_dim, 1)
        # 注册因果掩码为模型的缓冲区，它不是模型参数，但应随模型移动（如.to(device)）
        self.register_buffer('causal_mask', self.generate_causal_mask(seq_len))

    def generate_causal_mask(self, size):
        """生成因果关系掩码，防止模型在预测时看到未来的信息"""
        return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        
    def forward(self, src, return_attention=False):
        # 1. 输入投影和位置编码
        src = self.input_projection(src) * np.sqrt(self.input_projection.out_features)
        src += self.pos_encoder
        
        # 2. 依次通过Transformer编码器层
        last_attention_weights = None
        for i, layer in enumerate(self.encoder_layers):
            src, attn = layer(src, src_mask=self.causal_mask)
            # 只保留最后一层的注意力权重用于分析
            if i == len(self.encoder_layers) - 1:
                last_attention_weights = attn
        
        # 3. 使用序列最后一个时间步的输出来进行预测
        prediction = self.output_layer(src[:, -1, :])
        
        # 根据需要返回预测值和注意力权重
        if return_attention:
            return prediction.squeeze(-1), last_attention_weights
        return prediction.squeeze(-1)

# ==============================================================================
# Part 3: 训练与回测逻辑 (Training and Backtesting Logic)
# ==============================================================================
def run_training(args):
    """处理完整的模型训练流程"""
    logging.info("--- 模式: 训练 ---")
    raw_df = generate_and_save_data(filename=args.data_file)
    featured_data = create_features(raw_df)
    
    split_idx = int(len(featured_data) * args.train_split)
    train_df, val_df = featured_data.iloc[:split_idx], featured_data.iloc[split_idx:]
    logging.info(f"数据划分: {len(train_df)} 条训练样本, {len(val_df)} 条验证样本。")
    
    train_dataset = FuturesDataset(train_df, args.features, 'target_return', args.seq_len)
    val_dataset = FuturesDataset(val_df, args.features, 'target_return', args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 自动设备选择逻辑，优先MPS (Apple Silicon), 其次CUDA, 最后CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"使用设备: {device}")

    model = CausalTransformerEncoder(
        num_features=len(args.features), model_dim=args.model_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, seq_len=args.seq_len
    ).to(device)
    
    criterion = nn.MSELoss()  # 均方误差损失，适用于回归任务
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train() # 设置为训练模式
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [训练中]")
        for features, targets in train_pbar:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        model.eval() # 设置为评估模式
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, targets).item()
        
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1}, 训练损失: {running_loss/len(train_loader):.6f}, 验证损失: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.model_file)
            logging.info(f"验证损失降低。保存模型到 {args.model_file}")
    logging.info(f"\n训练完成！最优模型已保存至 '{args.model_file}'")

class Backtester:
    """处理回测流程的类"""
    def __init__(self, model, data, features, seq_len=60, hold_period=15):
        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()
        self.data = data.reset_index() # 确保有从0开始的整数索引
        self.features = features
        self.seq_len = seq_len
        self.hold_period = hold_period
        self.trades = []
        self.position = 'NONE' # 当前持仓状态: 'NONE', 'LONG', 'SHORT'
        self.entry_price = 0
        self.entry_time_step = -1

    def run(self):
        """执行回测的主循环"""
        logging.info("--- 模式: 回测 ---")
        for i in tqdm(range(self.seq_len, len(self.data)), desc="回测进度"):
            # 规则1：检查是否达到最大持仓周期，如果达到则平仓
            if self.position != 'NONE' and (i - self.entry_time_step) >= self.hold_period:
                self.close_position(i, reason="Max holding period reached")

            # 规则2：如果没有持仓，则寻找新的交易机会
            if self.position == 'NONE':
                input_df = self.data.iloc[i - self.seq_len : i]
                model_input = torch.tensor(input_df[self.features].values.astype(np.float32)).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    pred_return, attn_weights = self.model(model_input, return_attention=True)
                
                pred_return = pred_return.item()
                last_step_attn = attn_weights[0, -1, :].cpu().numpy()
                # 计算多头注意力权重：模型对过去上涨时段的关注度总和
                bullish_score = np.sum(last_step_attn * input_df['is_uptime'].values)
                
                # 信号判断
                if pred_return > 0.001 and bullish_score > 0.6: self.open_position('LONG', i)
                elif pred_return < -0.001 and (1 - bullish_score) > 0.6: self.open_position('SHORT', i)

        # 回测结束时，强制平掉所有剩余仓位
        if self.position != 'NONE': self.close_position(len(self.data) - 1, reason="End of backtest")
        self.calculate_and_show_performance()

    def open_position(self, side, time_step):
        """开仓"""
        self.position, self.entry_time_step = side, time_step
        self.entry_price = self.data.at[time_step, 'close']
        logging.info(f"Open {side} @ {self.entry_price:.2f} | Time: {self.data.at[time_step, 'timestamp']}")

    def close_position(self, time_step, reason=""):
        """平仓并记录交易"""
        close_price = self.data.at[time_step, 'close']
        pnl = (close_price - self.entry_price) if self.position == 'LONG' else (self.entry_price - close_price)
        pnl_pct = (pnl / self.entry_price)
        
        self.trades.append({
            "entry_time": self.data.at[self.entry_time_step, 'timestamp'],
            "exit_time": self.data.at[time_step, 'timestamp'],
            "entry_price": self.entry_price,
            "exit_price": close_price,
            "side": self.position,
            "pnl_pct": pnl_pct
        })
        logging.info(f"Close {self.position} @ {close_price:.2f} | PnL: {pnl_pct:.4%} | Reason: {reason}")
        self.position = 'NONE'

    def calculate_and_show_performance(self):
        """计算并展示详细的回测性能报告和图表"""
        logging.info("\n" + "="*25 + " Backtest Performance Report " + "="*25)
        if not self.trades:
            logging.info("No trades were executed during the backtest.")
            return
        
        trade_df = pd.DataFrame(self.trades)
        
        # --- 核心指标计算 ---
        total_trades = len(trade_df)
        wins = trade_df[trade_df['pnl_pct'] > 0]
        losses = trade_df[trade_df['pnl_pct'] <= 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        gross_profit = wins['pnl_pct'].sum()
        gross_loss = abs(losses['pnl_pct'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl_pct'].mean()) if len(losses) > 0 else 0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # --- 打印报告 ---
        print("\n--- General Performance ---")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {len(wins)}")
        print(f"Losing Trades: {len(losses)}")
        print(f"Win Rate: {win_rate:.2%}")
        
        print("\n--- Profit and Loss Analysis ---")
        print(f"Gross Profit (%): {gross_profit:.2%}")
        print(f"Gross Loss (%): {gross_loss:.2%}")
        print(f"Profit Factor (Gross Profit / Gross Loss): {profit_factor:.2f}")
        print(f"Average Win (%): {avg_win:.4%}")
        print(f"Average Loss (%): {avg_loss:.4%}")
        print(f"Average Win/Loss Ratio: {avg_win_loss_ratio:.2f}")
        
        self.plot_performance(trade_df)

    def plot_performance(self, trade_df):
        """绘制权益曲线和交易点位图"""
        logging.info("Generating backtest performance chart...")
        
        # 1. 计算权益曲线
        trade_df = trade_df.sort_values(by='exit_time').reset_index(drop=True)
        trade_df['equity_curve'] = (1 + trade_df['pnl_pct']).cumprod() - 1

        # 2. 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
        fig.suptitle('HFT Trend Following Strategy - Backtest Performance', fontsize=16)

        # 3. 绘制权益曲线
        ax1.plot(trade_df['exit_time'], trade_df['equity_curve'] * 100, label='Equity Curve (%)', color='blue')
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.set_title('Equity Curve')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()

        # 4. 绘制价格和交易点位
        ax2.plot(self.data['timestamp'], self.data['close'], label='Close Price', color='black', alpha=0.7)
        
        long_trades = trade_df[trade_df['side'] == 'LONG']
        ax2.scatter(long_trades['entry_time'], long_trades['entry_price'], label='Long Entry', marker='^', color='red', s=100, zorder=5)
        ax2.scatter(long_trades['exit_time'], long_trades['exit_price'], label='Long Exit', marker='o', color='red', s=50, zorder=5)

        short_trades = trade_df[trade_df['side'] == 'SHORT']
        ax2.scatter(short_trades['entry_time'], short_trades['entry_price'], label='Short Entry', marker='v', color='green', s=100, zorder=5)
        ax2.scatter(short_trades['exit_time'], short_trades['exit_price'], label='Short Exit', marker='o', color='green', s=50, zorder=5)

        ax2.set_ylabel('Price')
        ax2.set_title('Price Chart with Trade Entries and Exits')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # 格式化X轴日期
        fig.autofmt_xdate()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def run_backtesting(args):
    """处理完整的回测流程"""
    if not os.path.exists(args.model_file):
        logging.error(f"错误: 未找到已训练的模型 '{args.model_file}'。请先运行训练脚本。")
        return
    
    raw_df = pd.read_csv(args.data_file, index_col='timestamp', parse_dates=True)
    featured_data = create_features(raw_df)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"使用设备: {device}")
    
    model = CausalTransformerEncoder(
        num_features=len(args.features), model_dim=args.model_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, seq_len=args.seq_len
    ).to(device)
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    
    backtester = Backtester(model, featured_data, args.features, args.seq_len)
    backtester.run()

# ==============================================================================
# Part 4: 主程序入口 (Main Execution Block)
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练或回测一个基于Transformer的高频交易策略。")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'backtest'], help="执行模式: 'train' (训练) 或 'backtest' (回测)")
    
    # --- 通用参数 ---
    parser.add_argument('--data_file', type=str, default='mock_data.csv', help="数据文件的路径")
    parser.add_argument('--model_file', type=str, default='transformer_hft_model.pth', help="保存或加载模型的路径")
    parser.add_argument('--seq_len', type=int, default=60, help="模型的输入序列长度")
    parser.add_argument('--model_dim', type=int, default=32, help="模型内部维度 (d_model)")
    parser.add_argument('--num_heads', type=int, default=4, help="多头注意力机制的头数")
    parser.add_argument('--num_layers', type=int, default=4, help="Transformer编码器的层数")

    # --- 训练专用参数 ---
    parser.add_argument('--epochs', type=int, default=10, help="训练周期数")
    parser.add_argument('--batch_size', type=int, default=64, help="训练批次大小")
    parser.add_argument('--lr', type=float, default=0.001, help="优化器的学习率")
    parser.add_argument('--train_split', type=float, default=0.8, help="训练集/验证集划分比例")

    args = parser.parse_args()
    
    # 将特征列表定义为全局可访问
    args.features = ['price_change', 'volume_change', 'oi_change', 'spread', 'depth_imbalance']

    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'backtest':
        run_backtesting(args)

