import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import random

# ==============================================================================
# 1. 数据模拟与准备 (Data Simulation & Preparation)
# ==============================================================================
# 在实际应用中，您需要从文件或数据库加载您的真实数据。
# 这里我们生成一个模拟的DataFrame作为示例。
def generate_mock_data(num_minutes):
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
        
        # 模拟订单簿数据
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
    # 价格变化率
    df['price_change'] = df['close'].pct_change()
    # 成交量变化
    df['volume_change'] = df['volume'].pct_change()
    # 持仓量变化
    df['oi_change'] = df['open_interest'].pct_change()
    # 买卖价差
    df['spread'] = df['ask_price'] - df['bid_price']
    # 深度不平衡
    df['depth_imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'])
    
    # 标记是否为上涨时段 (用于注意力权重计算)
    df['is_uptime'] = (df['close'] > df['open']).astype(int)
    
    # 定义目标变量：未来5分钟的收益率 (用于模型训练)
    df['target_return'] = df['close'].pct_change(5).shift(-5)
    
    # 清理因计算产生NaN值的行
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df

# ==============================================================================
# 2. 模型设计 (Model Design)
# ==============================================================================
# 我们需要一个自定义的Transformer编码器层来方便地获取注意力权重
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 重写forward方法以同时返回输出和注意力权重
        # 注意：PyTorch 1.9+版本中，self_attn的forward签名可能需要is_causal参数
        attn_output, attn_weights = self.self_attn(src, src, src, 
                                                   attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask,
                                                   need_weights=True)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class CausalTransformerEncoder(nn.Module):
    def __init__(self, num_features, model_dim, num_heads, num_layers, seq_len, dropout=0.1):
        """
        Args:
            num_features (int): 输入特征的数量 (e.g., price_change, volume_change, etc.)
            model_dim (int): Transformer模型的内部维度 (d_model)
            num_heads (int): 多头注意力机制的头数
            num_layers (int): Transformer编码器的层数
            seq_len (int): 输入序列的长度
            dropout (float): Dropout比率
        """
        super(CausalTransformerEncoder, self).__init__()
        self.model_dim = model_dim
        self.seq_len = seq_len
        
        # 输入线性层：将输入特征映射到模型维度
        self.input_projection = nn.Linear(num_features, model_dim)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        
        # 自定义的Transformer编码器层列表
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=model_dim, 
                nhead=num_heads, 
                dropout=dropout,
                batch_first=True # 重要：输入数据格式为 (batch, seq, feature)
            ) for _ in range(num_layers)
        ])
        
        # 输出层：从模型维度映射到单个预测值 (5分钟收益率)
        self.output_layer = nn.Linear(model_dim, 1)
        
        # 因果关系掩码：确保模型在预测时不能看到未来的信息
        self.causal_mask = self.generate_causal_mask(seq_len).to('cpu') # 默认在cpu

    def generate_causal_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src):
        """
        前向传播，并返回预测和最后一层的注意力权重。
        """
        # 确保掩码在正确的设备上
        self.causal_mask = self.causal_mask.to(src.device)
        
        # 1. 投影和位置编码
        src = self.input_projection(src) * np.sqrt(self.model_dim)
        src += self.pos_encoder
        
        # 2. 通过Transformer编码器
        attention_weights = None
        for i, layer in enumerate(self.encoder_layers):
            src, attn = layer(src, src_mask=self.causal_mask)
            # 我们只关心最后一层的注意力权重
            if i == len(self.encoder_layers) - 1:
                attention_weights = attn
        
        # 3. 平均池化和输出
        # 取序列最后一个时间步的输出来做预测
        pooled_output = src[:, -1, :] 
        prediction = self.output_layer(pooled_output)
        
        return prediction.squeeze(-1), attention_weights

# ==============================================================================
# 3. 策略与回测 (Strategy & Backtesting)
# ==============================================================================
class Backtester:
    def __init__(self, model, data, features, seq_len=60, hold_period=15):
        self.model = model
        self.model.eval() # 设置为评估模式
        self.data = data
        self.features = features
        self.seq_len = seq_len
        self.hold_period = hold_period
        
        self.trades = []
        self.position = 'NONE' # 'NONE', 'LONG', 'SHORT'
        self.entry_price = 0
        self.entry_time_step = -1

    def run(self):
        """执行回测循环"""
        print("\n开始执行回测...")
        num_steps = len(self.data)
        
        for i in range(self.seq_len, num_steps):
            
            # --- 检查持仓是否超时 ---
            if self.position != 'NONE' and (i - self.entry_time_step) >= self.hold_period:
                self.close_position(i)

            # --- 生成交易信号 ---
            if self.position == 'NONE': # 只有在没有持仓时才寻找新机会
                
                # 准备模型输入
                input_sequence_df = self.data.iloc[i - self.seq_len : i]
                
                # 提取特征并转换为tensor
                model_input_np = input_sequence_df[self.features].values.astype(np.float32)
                model_input = torch.tensor(model_input_np).unsqueeze(0) # 增加batch维度
                
                # 提取上涨时段标记
                is_uptime_sequence = input_sequence_df['is_uptime'].values
                
                # 获取模型预测和注意力权重
                with torch.no_grad():
                    predicted_return, attention_weights = self.model(model_input)
                
                # attention_weights shape is (batch, seq_len, seq_len)
                # 我们取最后一个时间步对所有历史的注意力
                last_step_attention = attention_weights[0, -1, :].numpy()
                
                # 计算多头注意力权重
                # 权重总和为1，我们计算关注在"上涨时段"的权重比例
                bullish_attention_score = np.sum(last_step_attention * is_uptime_sequence)
                
                # --- 信号规则判断 ---
                # 做多信号
                if predicted_return > 0.001 and bullish_attention_score > 0.6:
                    self.open_position('LONG', i)
                # 做空信号 (反向规则)
                elif predicted_return < -0.001 and (1 - bullish_attention_score) > 0.6: # 假设空头关注度是 1 - 多头关注度
                    self.open_position('SHORT', i)

        # 回测结束时关闭所有剩余仓位
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
        
        if self.position == 'LONG':
            pnl = close_price - self.entry_price
        elif self.position == 'SHORT':
            pnl = self.entry_price - close_price
        else:
            pnl = 0

        pnl_pct = (pnl / self.entry_price) * 100
        
        trade_log = {
            "entry_time": self.data.at[self.entry_time_step, 'timestamp'],
            "exit_time": self.data.at[time_step, 'timestamp'],
            "side": self.position,
            "entry_price": self.entry_price,
            "exit_price": close_price,
            "pnl_pct": pnl_pct,
            "holding_period": time_step - self.entry_time_step
        }
        self.trades.append(trade_log)
        
        print(f"{self.data.at[time_step, 'timestamp']}: 平仓 {self.position} @ {close_price:.2f}, PnL: {pnl_pct:.4f}%")
        
        # 重置仓位状态
        self.position = 'NONE'
        self.entry_price = 0
        self.entry_time_step = -1

    def calculate_performance(self):
        print("\n--- 回测性能评估 ---")
        if not self.trades:
            print("没有发生任何交易。")
            return
            
        trade_df = pd.DataFrame(self.trades)
        
        # 胜率
        wins = trade_df[trade_df['pnl_pct'] > 0]
        win_rate = len(wins) / len(trade_df) if len(trade_df) > 0 else 0
        
        # 总收益和年化收益 (简化计算)
        total_return = trade_df['pnl_pct'].sum()
        
        # 最大回撤 (简化版：只看单笔最大亏损)
        max_drawdown = trade_df['pnl_pct'].min()
        
        # 夏普比率 (简化版：假设无风险利率为0)
        if len(trade_df) > 1:
            avg_return = trade_df['pnl_pct'].mean()
            std_return = trade_df['pnl_pct'].std()
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252 * 4 * 60) if std_return > 0 else 0 # 简单年化
        else:
            sharpe_ratio = 0
        
        print(f"总交易次数: {len(trade_df)}")
        print(f"胜率: {win_rate:.2%}")
        print(f"平均每笔收益: {trade_df['pnl_pct'].mean():.4f}%")
        print(f"夏普比率 (年化, 粗算): {sharpe_ratio:.2f}")
        print(f"最大单笔亏损 (作为最大回撤代理): {max_drawdown:.4f}%")
        
        # 打印部分交易记录
        print("\n部分交易记录:")
        print(trade_df.head().to_string())


# ==============================================================================
# 4. 主程序 (Main Execution)
# ==============================================================================
if __name__ == '__main__':
    # --- 参数设置 ---
    INPUT_FEATURES = ['price_change', 'volume_change', 'oi_change', 'spread', 'depth_imbalance']
    NUM_FEATURES = len(INPUT_FEATURES)
    MODEL_DIM = 32  # 模型维度
    NUM_HEADS = 4   # 注意力头数
    NUM_LAYERS = 4  # Transformer层数 (与您描述的一致)
    SEQ_LEN = 60    # 输入序列长度 (60分钟)
    
    # 1. 准备数据
    # 使用1整年的分钟数据进行模拟 (约 240 * 252 = 60480 分钟)
    # 为了快速演示，我们只用几千分钟
    raw_data = generate_mock_data(num_minutes=5000)
    featured_data = create_features(raw_data)

    # 2. 初始化模型
    # 重要：这里模型是随机初始化的。在真实场景中，你需要加载已经训练好的模型权重。
    print("\n初始化未训练的Transformer模型...")
    model = CausalTransformerEncoder(
        num_features=NUM_FEATURES,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN
    )
    
    # 3. 运行回测
    backtester = Backtester(
        model=model,
        data=featured_data,
        features=INPUT_FEATURES,
        seq_len=SEQ_LEN,
        hold_period=15 # 持仓不超过15分钟
    )
    backtester.run()
