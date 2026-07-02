# scripts/002 — 002-transformer-attention.md 配图脚本

为 `002-transformer-attention.md` 生成 10 张 matplotlib 配图，输出到 `images/002/`。

## 文件清单

| 脚本 | 输出 PNG | 文档中位置 |
|---|---|---|
| `gen_01_vector.py` | `01_vector_geometry.png` | §1.1 标量与向量 |
| `gen_02_matvec.py` | `02_matvec.png` | §1.2 矩阵乘向量 |
| `gen_03_dot_product.py` | `03_dot_product_angle.png` | §1.3 点积几何意义 |
| `gen_04_softmax.py` | `04_softmax.png` | §1.4 softmax |
| `gen_05_embedding.py` | `05_embedding_heatmap.png` | §2 Embedding |
| `gen_06_qk_scores.py` | `06_qk_scores.png` | §4.1 Q·K 分数 |
| `gen_07_attn_weights.py` | `07_attn_weights.png` | §4.2 softmax 权重 |
| `gen_08_output.py` | `08_output_decomposition.png` | §5 / §6 output 分解 |
| `gen_09_pipeline.py` | `09_qkv_pipeline.png` | §4.4 整体流程 |
| `gen_10_multihead.py` | `10_multihead.png` | §7.3 Multi-Head |

公共样式（颜色、字号、字体、输出目录）见 `_style.py`。

## 运行

### 一键生成全部
```bash
cd scripts/002
python gen_all.py
```

### 单独生成某张图
```bash
cd scripts/002
python gen_05_embedding.py
```

## 依赖

```
pip install matplotlib
```

- Python 3.8+
- macOS 自带 PingFang SC 字体，无需额外安装
- 若在 Linux 上运行需把 `_style.py` 里 `font.sans-serif` 改成系统可用中文字体（如 `Noto Sans CJK SC`）

## 数字一致性

所有图里的数字（7 token 的 Q·K 分数、softmax 权重、output 分解）**与 `code/attention_demo.py` 当前版本的输出一一对应**：

- 7 个 Q·K 分数：`[0, 1, 0, 0, 0, 0, 0]`
- 7 个 softmax 权重：`[0.1147, 0.3118, 0.1147, 0.1147, 0.1147, 0.1147, 0.1147]`
- `output(it) = [0.43, 0, 0, 0]`（dim0 = 0.3118 + 0.1147 = 0.4265）

如要重新校准，先跑 `python code/attention_demo.py` 拿到新数字，再改 `gen_06/07/08` 里的 `scores` 列表即可。
