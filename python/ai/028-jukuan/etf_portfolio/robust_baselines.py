"""3 个硬编码稳健基线组合。

无需任何优化，可直接实盘或对照参考。所有 ETF 都满足上市 ≥ 5 年。

说明：
    - "经典60_40"：股票 + 债券，最朴素的资产配置范式
    - "全天候简化"：桥水全天候的中国简化版，股 + 债 + 海外 + 黄金 + 商品
    - "红利+海外+黄金"：A 股红利 + 海外股票 + 黄金 + 少量债券，进攻型稳健组合
"""

from typing import Dict, List, TypedDict


class BaselineHolding(TypedDict):
    code: str
    name: str
    weight: float


class Baseline(TypedDict):
    description: str
    holdings: List[BaselineHolding]


BASELINES: Dict[str, Baseline] = {
    "经典60_40": {
        "description": "60% 沪深300 + 40% 上证5年国债，朴素资产配置范式",
        "holdings": [
            {"code": "510300.XSHG", "name": "沪深300",     "weight": 0.60},
            {"code": "511010.XSHG", "name": "上证5年国债", "weight": 0.40},
        ],
    },
    "全天候简化": {
        "description": "桥水全天候的中国简化版：股（A股+海外+港股）+ 红利 + 长期债 + 黄金 + 商品",
        "holdings": [
            {"code": "510300.XSHG", "name": "沪深300",     "weight": 0.30},
            {"code": "513500.XSHG", "name": "标普500",     "weight": 0.10},
            {"code": "159920.XSHE", "name": "恒生ETF",     "weight": 0.10},
            {"code": "510880.XSHG", "name": "红利ETF",     "weight": 0.15},
            {"code": "511010.XSHG", "name": "上证5年国债", "weight": 0.20},
            {"code": "518880.XSHG", "name": "黄金ETF",     "weight": 0.10},
            {"code": "162411.XSHE", "name": "油气LOF",     "weight": 0.05},
        ],
    },
    "红利+海外+黄金": {
        "description": "进攻型稳健：A 股红利（防御）+ 海外科技（弹性）+ 港股 + 黄金（对冲）+ 少量债券",
        "holdings": [
            {"code": "510880.XSHG", "name": "红利ETF",          "weight": 0.40},
            {"code": "513100.XSHG", "name": "纳斯达克100ETF",   "weight": 0.20},
            {"code": "159920.XSHE", "name": "恒生ETF",          "weight": 0.10},
            {"code": "518880.XSHG", "name": "黄金ETF",          "weight": 0.20},
            {"code": "511010.XSHG", "name": "上证5年国债",      "weight": 0.10},
        ],
    },
}


def get_baseline_weights(baseline_name: str) -> Dict[str, float]:
    """获取某基线的 {code: weight} 字典。"""
    bl = BASELINES[baseline_name]
    return {h["code"]: h["weight"] for h in bl["holdings"]}


def list_baselines() -> List[str]:
    """列出所有基线名。"""
    return list(BASELINES.keys())
