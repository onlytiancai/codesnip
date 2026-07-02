"""
一键生成 002-transformer-attention.md 全部配图。

用法：
    cd scripts/002
    python gen_all.py

依赖：
    pip install matplotlib
    Python 3.8+
"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GENERATORS = [
    ("gen_01_vector",        "01_vector_geometry.png"),
    ("gen_02_matvec",        "02_matvec.png"),
    ("gen_03_dot_product",   "03_dot_product_angle.png"),
    ("gen_04_softmax",       "04_softmax.png"),
    ("gen_05_embedding",     "05_embedding_heatmap.png"),
    ("gen_06_qk_scores",     "06_qk_scores.png"),
    ("gen_07_attn_weights",  "07_attn_weights.png"),
    ("gen_08_output",        "08_output_decomposition.png"),
    ("gen_09_pipeline",      "09_qkv_pipeline.png"),
    ("gen_10_multihead",     "10_multihead.png"),
]


def main():
    print("=" * 60)
    print(" 生成 002-transformer-attention.md 配图")
    print(" 输出目录: images/002/")
    print("=" * 60)
    for mod_name, png in GENERATORS:
        print(f"\n[{mod_name}]")
        mod = importlib.import_module(mod_name)
        mod.main()
    print("\n" + "=" * 60)
    print(" ✓ 全部完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
