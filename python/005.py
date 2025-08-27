import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

# 1. 获取历史数据
# 使用yfinance下载S&P 500的历史数据
data = yf.download('^GSPC', start='2018-01-01', end='2023-10-01')
data = data['Adj Close']

# 2. 计算对数收益率
returns = np.log(data / data.shift(1)).dropna()

# 3. 构建GARCH模型
# 使用对数收益率的序列来拟合GARCH模型
model = arch_model(returns, vol='Garch', p=1, q=1)
model_fit = model.fit(disp='off')
print(model_fit.summary())

# 4. 预测未来5天的波动率（假设是未来一周的波动率）
forecast_horizon = 5
volatility_forecast = model_fit.forecast(horizon=forecast_horizon)

# 提取预测的波动率均值（年化）
predicted_volatility = volatility_forecast.variance.values[-1, :]
predicted_volatility_annualized = np.sqrt(predicted_volatility * 252)

# 5. 输出下周的波动率预测
print("未来5天（1周）每日预测波动率（年化）：")
for i, vol in enumerate(predicted_volatility_annualized):
    print(f"第{i+1}天：{vol:.2%}")

# 6. 可视化
plt.figure(figsize=(10, 5))
plt.plot(data.index, returns, label='Daily Returns', color='blue')
plt.plot(data.index[-forecast_horizon:], np.sqrt(predicted_volatility) * np.sqrt(252), label='Predicted Volatility', color='red')
plt.title('S&P 500 Returns and Predicted Volatility (Annualized)')
plt.legend()
plt.show()
