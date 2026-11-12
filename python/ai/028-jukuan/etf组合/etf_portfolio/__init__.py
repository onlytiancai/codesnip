"""ETF 组合优化工具包（聚宽研究环境）。

模块清单：
    universe          候选池预过滤（5 年上市 + 5000 万成交 + 0.92 相关去重）
    data_loader       取价、复权、收益转换、parquet 缓存
    covariance        Ledoit-Wolf 协方差收缩 + EWM 均值估计
    optimizers        SLSQP 求解器（Sharpe / Calmar / MinVar）
    risk_parity       风险平价对照组（Newton 法）
    metrics           empyrical 指标封装 + 衰减率计算
    wfa               Walk-Forward 滚动样本外主框架
    robust_baselines  3 个硬编码稳健基线组合
    reporting         净值曲线、衰减率表、稳健度评分生成

设计原则：
    1. 所有依赖限定为聚宽研究环境预装库（numpy/pandas/scipy/sklearn/empyrical）。
    2. 每个模块都可在 notebook 中独立 import，不依赖 jupyter magic。
    3. 卡玛比率手写（MDD 是路径依赖指标，不能用 ep.calmar_ratio）。
"""

__version__ = "0.1.0"
