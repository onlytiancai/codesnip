"""
020.py — 大矩阵乘法的「增量算子」优化与对比
================================================

场景
----
1. 先把 C = A @ B 算出来，并把它缓存在 GPU 上。
2. 之后 A、B 在行/列方向被「扩展」（追加 ΔA 行 / ΔB 列），需要算
    C' = A' @ B'。
3. 利用分块矩阵恒等式

        A' @ B' = [  A  ] @ [B  ΔB]   =  [  A @ B ,  A @ ΔB ]
                    [ ΔA  ]   [         ]      [ ΔA @ B, ΔA @ ΔB]

       上式左上的 A @ B 就是已经缓存好的 C —— 不必再算一次。
       这就是本题所说的「增量算子」在分块矩阵中的体现；
       PyTorch 内置的 `torch.addmm(input, A, B)`  = input + A @ B
       也属于「增量」算子族。

对比维度
    ● naive           :  A_ext @ B_ext 一把整算（不缓存）
    ● incremental     :  4 块分别相乘/拷贝 —— 左上直接复用缓存的 C

输出
    ● 各方法的耗时中位数（每种跑 10 趟 + 3 趟 warm-up）
    ● 节省的 FLOPs
    ● 正确性（与 naive 对比的绝对误差）
    ● matplotlib 柱状图（附带：PingFang SC + mathtext CM 字体）
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  imports
# ─────────────────────────────────────────────────────────────────────────────
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# matplotlib 中文字体 — 按优先级尝试，找到第一个存在的就行
_CJK_CANDIDATES = [
    "PingFang SC",        # 用户偏好 (若系统存在)
    "Hiragino Sans GB",   # macOS 自带的中文黑体
    "Heiti TC",           # macOS 自带
    "STHeiti",
    "Microsoft YaHei",    # Windows
    "Noto Sans CJK SC",   # Linux
    "SimHei",
]
from matplotlib import font_manager as _fm
_available = {f.name for f in _fm.fontManager.ttflist}
for _c in _CJK_CANDIDATES:
    if _c in _available:
        rcParams["font.family"] = _c
        rcParams["font.sans-serif"] = [_c] + rcParams.get("font.sans-serif", [])
        break
else:
    rcParams["font.family"] = "sans-serif"
rcParams["axes.unicode_minus"] = False
rcParams["mathtext.fontset"] = "cm"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  device 选择 — 在 Mac 上优先走 MPS，NVIDIA 上走 CUDA
# ─────────────────────────────────────────────────────────────────────────────
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """对应 backend 的同步原语，保证计时区间被 GPU 真正执行完。"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def bench(device: torch.device, fn, iters: int = 10) -> float:
    """返回 fn() 运行 iters 次的中位数耗时 (毫秒)。"""
    for _ in range(3):           # warm-up
        fn()
    sync(device)
    samples = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  通用算子 — naive vs incremental
# ─────────────────────────────────────────────────────────────────────────────
def matmul_naive(A_ext, B_ext):
    """整块重算，没有任何复用。"""
    return A_ext @ B_ext


def matmul_incremental(A, B, dA, dB, C_cached, m, n):
    """
    分块矩阵公式 (增量算子):
        out = [  C_cached  ,  A  @ dB ]
              [  dA  @ B   ,  dA @ dB ]

    左上的 m×n 块直接复用 Step 1 缓存下来的 C，剩下三块各算一次小乘法。
    """
    dm_, dn_ = dA.shape[0], dB.shape[1]
    out = A.new_empty((m + dm_, n + dn_))
    out[:m, :n] = C_cached                      # ← 复用缓存
    # 下面的 addmm 用 beta=0，等价于把 mat1@mat2 写到指定切片
    torch.addmm(A.new_zeros((m,  dn_)),  A,  dB, beta=0, alpha=1, out=out[:m, n:])
    torch.addmm(A.new_zeros((dm_, n )),  dA, B,    beta=0, alpha=1, out=out[m:, :n])
    torch.addmm(A.new_zeros((dm_, dn_)), dA, dB,   beta=0, alpha=1, out=out[m:, n:])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3.  benchmark 主流程
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = pick_device()
    dev_name = (torch.cuda.get_device_name(0)
                if device.type == "cuda" else device.type)
    torch.manual_seed(0)

    print(f"\ndevice = {device}  ({dev_name})\n")

    # ── 主测试规模 ──────────────────────────────────────────────────────────
    base_cfg = dict(m=4096, k=4096, n=4096, dm=512, dn=512)
    m, k, n, dm, dn = base_cfg.values()

    A  = torch.randn(m,  k,  device=device)
    B  = torch.randn(k,  n,  device=device)
    dA = torch.randn(dm, k,  device=device)
    dB = torch.randn(k,  dn,  device=device)

    print(f"shape: A={tuple(A.shape)}  B={tuple(B.shape)}  "
          f"dA={tuple(dA.shape)}  dB={tuple(dB.shape)}\n")

    # ── Step 1：缓存 ────────────────────────────────────────────────────────
    t_cache_ms = bench(device, lambda: A @ B, iters=10)
    C_cached   = A @ B                              # 真正留在显存里的
    sync(device)
    print(f"[1] 初始 C = A @ B  ({m}×{k} · {k}×{n})")
    print(f"    缓存耗时 (median of 10) : {t_cache_ms:7.2f} ms")

    # ── Step 2：扩展 ────────────────────────────────────────────────────────
    A_ext = torch.cat([A, dA], dim=0)               # (m+dm, k)
    B_ext = torch.cat([B, dB], dim=1)               # (k, n+dn)
    print(f"\n[2] 扩展后：A_ext={tuple(A_ext.shape)}  "
          f"B_ext={tuple(B_ext.shape)}")

    # ── Step 3：naive & incremental ─────────────────────────────────────────
    # 提前各跑一遍以便做正确性校验
    C_naive = matmul_naive(A_ext, B_ext)
    C_inc   = matmul_incremental(A, B, dA, dB, C_cached, m, n)
    sync(device)

    t_naive_ms = bench(device, lambda: matmul_naive(A_ext, B_ext))
    t_inc_ms   = bench(device, lambda: matmul_incremental(
                            A, B, dA, dB, C_cached, m, n))
    sync(device)

    max_err = (C_naive - C_inc).abs().max().item()

    # FLOPs
    flops_full   = 2 * (m + dm) * (n + dn) * k
    flops_cache  = 2 * m * k * n                          # 这部分被复用掉
    flops_inc_new = (2 * m * k * dn
                     + 2 * dm * k * n
                     + 2 * dm * k * dn)                   # 增量新增

    # ── 报告 ────────────────────────────────────────────────────────────────
    print("\n[3] 正确性")
    print(f"    max|naive - incremental| = {max_err:.3e}")

    print("\n[4] FLOPs 对比")
    print(f"    naive        : {flops_full        / 1e9:7.2f} GFLOPs")
    print(f"    incremental  : {flops_inc_new     / 1e9:7.2f} GFLOPs   "
          f"+ 复用 {flops_cache/1e9:.2f} GFLOPs (跳过)")
    print(f"    总节省比例    : "
          f"{(1 - flops_inc_new / flops_full) * 100:5.1f} %")

    print("\n[5] 耗时 (median of 10)")
    print(f"    naive        : {t_naive_ms:7.2f} ms")
    print(f"    incremental  : {t_inc_ms:7.2f} ms")
    print(f"    加速比        : {t_naive_ms / t_inc_ms:5.2f} x   "
          f"(节省 {(1 - t_inc_ms / t_naive_ms) * 100:5.1f} %)")

    # ── Step 4：跨多种扩展规模的扫描 ───────────────────────────────────────
    sweep_ext = [128, 256, 512, 1024, 2048]
    naive_ms, inc_ms, flops_ratio = [], [], []
    for ext in sweep_ext:
        Da = torch.randn(ext, k, device=device)
        Db = torch.randn(k,   ext, device=device)
        A_e = torch.cat([A, Da], dim=0)
        B_e = torch.cat([B, Db], dim=1)

        nv = bench(device, lambda: matmul_naive(A_e, B_e))
        ic = bench(device, lambda: matmul_incremental(
                        A, B, Da, Db, C_cached, m, n))
        naive_ms.append(nv)
        inc_ms.append(ic)

        full = 2 * (m + ext) * (n + ext) * k
        new  = 2 * (m * k * ext + ext * k * n + ext * k * ext)
        flops_ratio.append((1 - new / full) * 100)
        print(f"   dm=dn={ext:>4}  naive={nv:7.2f} ms  "
              f"incremental={ic:7.2f} ms  加速={nv/ic:5.2f}x  "
              f"FLOPs 节省={flops_ratio[-1]:5.1f}%")

    plot_chart(sweep_ext, naive_ms, inc_ms, flops_ratio, device)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  matplotlib 可视化
# ─────────────────────────────────────────────────────────────────────────────
def plot_chart(extensions, naive_ms, inc_ms, flops_ratio, device):
    fig, ax1 = plt.subplots(figsize=(9, 5.2))

    x = range(len(extensions))
    w = 0.38

    bar1 = ax1.bar([i - w/2 for i in x], naive_ms, width=w,
                   label="naive  (A' @ B' 重算)",
                   color="#4F8BD9")
    bar2 = ax1.bar([i + w/2 for i in x], inc_ms, width=w,
                   label="incremental  (分块复用缓存)",
                   color="#E67E22")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"{e}" for e in extensions])
    ax1.set_xlabel(r"$dm = dn$  (额外追加的行列数)")
    ax1.set_ylabel("耗时 (ms)  —  median of 10")
    ax1.set_title(f"矩阵乘法增量算子对比  ·  base $m=n=k=4096$  ·  {device}")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper left")

    # 在柱顶标注数值
    for bars in (bar1, bar2):
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h,
                     f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    # 右轴：FLOPs 节省百分比
    ax2 = ax1.twinx()
    ax2.plot(x, flops_ratio, color="#27AE60", marker="o",
             linewidth=2, label="FLOPs 节省 %")
    ax2.set_ylabel(r"FLOPs 节省 %  (右轴)", color="#27AE60")
    ax2.tick_params(axis="y", colors="#27AE60")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    out = "/Users/huhao/src/codesnip/python/ai/020.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n[图] 已保存  {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
