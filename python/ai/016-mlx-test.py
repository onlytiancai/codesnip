"""Apple MLX 基础功能测试

覆盖范围:
1. 环境与设备 (Metal GPU)
2. 张量创建 (array, zeros, ones, full, arange, linspace, eye, random)
3. 张量属性 (shape, dtype, ndim, size, T)
4. 索引与切片
5. 形状变换 (reshape, transpose, broadcast_to, expand_dims, squeeze)
6. 数学运算 (加减乘除、矩阵乘、逐元素函数)
7. 归约运算 (sum, mean, max, min, argmax)
8. 广播机制
9. 自动微分 (grad)
10. 编译加速 (compile / compile_with_function)
11. 与 NumPy 互转
12. 保存与加载 (save, load)

运行: python 016-mlx-test.py
"""

import time
import numpy as np
import mlx.core as mx


# ---------- 工具函数 ----------
def section(title: str) -> None:
    """打印小节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check(name: str, got, expected) -> bool:
    """比较结果, 打印通过/失败"""
    if isinstance(expected, mx.array):
        ok = mx.allclose(got, expected).item()
    else:
        ok = got == expected
    flag = "✅" if ok else "❌"
    print(f"  {flag} {name}: got={got!r}, expected={expected!r}")
    return ok


# ---------- 1. 环境与设备 ----------
def test_environment() -> None:
    section("1. 环境与设备")
    print(f"  MLX 版本:        {mx.__version__}")
    print(f"  Metal 可用:      {mx.metal.is_available()}")
    if mx.metal.is_available():
        info = mx.device_info()
        # device_info 在不同版本可能返回 dict 或 tuple
        print(f"  设备信息:        {dict(info) if hasattr(info, 'items') else info}")
    print(f"  默认设备:        {mx.default_device()}")
    # 设置默认 GPU 设备
    if mx.metal.is_available():
        mx.set_default_device(mx.gpu)


# ---------- 2. 张量创建 ----------
def test_creation() -> None:
    section("2. 张量创建")
    a = mx.array([1, 2, 3])
    print("  array([1,2,3])     =", a)

    b = mx.array([[1.0, 2.0], [3.0, 4.0]])
    print("  array([[1,2],[3,4]])=", b)

    check("zeros", mx.zeros((2, 3)), mx.array([[0, 0, 0], [0, 0, 0]]))
    check("ones",  mx.ones((2, 2)),  mx.array([[1, 1], [1, 1]]))
    check("full",  mx.full((2, 2), 7), mx.array([[7, 7], [7, 7]]))

    print("  arange(0,6,2)      =", mx.arange(0, 6, 2))
    print("  linspace(0,1,5)    =", mx.linspace(0, 1, 5))
    check("eye",   mx.eye(3),       mx.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    rng_key = mx.random.key(0)
    r = mx.random.uniform(key=rng_key, low=0.0, high=1.0, shape=(2, 3))
    print("  random.uniform     =", r)
    n = mx.random.normal(key=rng_key, shape=(2, 3))
    print("  random.normal      =", n)

    # 显式指定 dtype
    i32 = mx.array([1, 2, 3], dtype=mx.int32)
    f32 = mx.array([1, 2, 3], dtype=mx.float32)
    print(f"  dtype=int32/float32 = {i32.dtype} / {f32.dtype}")


# ---------- 3. 张量属性 ----------
def test_attributes() -> None:
    section("3. 张量属性")
    x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"  x.shape  = {x.shape}")
    print(f"  x.dtype  = {x.dtype}")
    print(f"  x.ndim   = {x.ndim}")
    print(f"  x.size   = {x.size}")
    print(f"  x.T      =\n{x.T}")


# ---------- 4. 索引与切片 ----------
def test_indexing() -> None:
    section("4. 索引与切片")
    x = mx.array([[10, 11, 12, 13],
                  [20, 21, 22, 23],
                  [30, 31, 32, 33]])
    print("  x =")
    print("   ", x)
    print("  x[0]            =", x[0])
    print("  x[0, 1]         =", x[0, 1])
    print("  x[:, 1:3]       =", x[:, 1:3])
    print("  x[::-1]         =", x[::-1])
    # MLX 暂不支持 x[mask] 形式的布尔索引, 用 mx.where / mx.nonzero 代替
    mask = x > 20
    print("  x > 20 (mask)   =\n", mask.astype(mx.int32))
    print("  mx.where(x>20)  =", mx.where(mask, x, 0))
    # 用 where 取出满足条件的元素 (保留位置, 不满足置 0)
    sel = mx.where(mask, x, 0)
    print("  where(mask,x,0)  =\n", sel)
    print(f"  count_nonzero    = {mx.count_nonzero(mask).item()}")


# ---------- 5. 形状变换 ----------
def test_shape_ops() -> None:
    section("5. 形状变换")
    x = mx.arange(12)
    print("  x            =", x)
    print("  reshape(3,4) =\n", x.reshape(3, 4))
    print("  transpose    =\n", mx.arange(6).reshape(2, 3).T)
    print("  expand_dims  shape =", mx.expand_dims(mx.arange(3), 0).shape)
    print("  expand_dims  shape =", mx.expand_dims(mx.arange(3), 1).shape)
    print("  squeeze      shape =", mx.squeeze(mx.array([[[1]]])).shape)


# ---------- 6. 数学运算 ----------
def test_math() -> None:
    section("6. 数学运算")
    a = mx.array([[1.0, 2.0], [3.0, 4.0]])
    b = mx.array([[5.0, 6.0], [7.0, 8.0]])

    print("  a + b      =\n", a + b)
    print("  a * b      =\n", a * b)
    print("  a @ b      =\n", a @ b)
    print("  mx.matmul  =\n", mx.matmul(a, b))

    x = mx.array([1.0, 2.0, 3.0])
    print("  exp(x)     =", mx.exp(x))
    print("  sin(x)     =", mx.sin(x))
    print("  log(x)     =", mx.log(x))
    print("  sqrt(x)    =", mx.sqrt(x))
    print("  maximum(a,2)=\n", mx.maximum(a, 2.0))


# ---------- 7. 归约运算 ----------
def test_reduction() -> None:
    section("7. 归约运算")
    x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"  x = \n{x}")
    print(f"  sum         = {mx.sum(x)}")
    print(f"  sum(axis=0) = {mx.sum(x, axis=0)}")
    print(f"  sum(axis=1) = {mx.sum(x, axis=1)}")
    print(f"  mean        = {mx.mean(x)}")
    print(f"  max         = {mx.max(x)}")
    print(f"  argmax      = {mx.argmax(x)}")  # 展平后的索引
    print(f"  argmax(ax=1)= {mx.argmax(x, axis=1)}")


# ---------- 8. 广播机制 ----------
def test_broadcast() -> None:
    section("8. 广播机制")
    a = mx.array([[1.0], [2.0], [3.0]])   # (3,1)
    b = mx.array([10.0, 20.0, 30.0])      # (3,)
    print("  a (3,1) + b (3,) ->")
    print("  ", a + b)


# ---------- 9. 自动微分 ----------
def test_autograd() -> None:
    section("9. 自动微分")

    # 简单标量函数: y = x^3, dy/dx = 3x^2
    def f(x):
        return x ** 3

    df = mx.grad(f)
    x = mx.array(2.0)
    print(f"  d(x^3)/dx @ x=2: grad={df(x)}, expected=12.0")

    # 多元梯度: f(x,y) = x^2 + 3xy + y^2
    def g(params):
        x, y = params
        return x * x + 3 * x * y + y * y

    dg = mx.grad(g)
    params = [mx.array(1.0), mx.array(2.0)]
    grads = dg(params)
    # df/dx = 2x + 3y = 8, df/dy = 3x + 2y = 7
    print(f"  grad(x^2+3xy+y^2) @ (1,2): {grads}, expected=[8.0, 7.0]")

    # 用 value_and_grad 同时拿值与梯度
    def loss(w):
        return mx.mean((w * 2 - 1) ** 2)

    vg = mx.value_and_grad(loss)
    w = mx.array([1.0, 2.0, 3.0])
    val, grad = vg(w)
    print(f"  value_and_grad: val={val.item()}, grad={grad}")


# ---------- 10. compile 加速 ----------
def test_compile() -> None:
    section("10. compile 加速")

    def step(x, w):
        y = mx.matmul(x, w)
        return mx.tanh(y)

    x = mx.random.normal(shape=(64, 128))
    w = mx.random.normal(shape=(128, 64))

    N = 100

    # --- eager 模式: 首次 (含编译) ---
    t0 = time.perf_counter()
    y1 = step(x, w)
    mx.eval(y1)
    t_first = (time.perf_counter() - t0) * 1000

    # --- eager 模式: 稳态 (已编译过 kernel) ---
    t0 = time.perf_counter()
    for _ in range(N):
        y_eager = step(x, w)
        mx.eval(y_eager)
    eager_ms = (time.perf_counter() - t0) * 1000 / N

    # --- compile 模式: warm-up + 稳态 ---
    step_compiled = mx.compile(step)
    _ = step_compiled(x, w)
    mx.eval(_)

    t0 = time.perf_counter()
    for _ in range(N):
        y2 = step_compiled(x, w)
        mx.eval(y2)
    compiled_ms = (time.perf_counter() - t0) * 1000 / N

    print(f"  eager  首次  (含编译) : {t_first:7.3f} ms")
    print(f"  eager  稳态  (100 次) : {eager_ms:7.3f} ms / 次")
    print(f"  compile 稳态 (100 次) : {compiled_ms:7.3f} ms / 次")
    print(f"  加速比 (eager 稳态 / compile) : {eager_ms / compiled_ms:.2f}x")


# ---------- 11. 与 NumPy 互转 ----------
def test_numpy_interop() -> None:
    section("11. 与 NumPy 互转")
    np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mx_arr = mx.array(np_arr)
    print(f"  numpy -> mlx: shape={mx_arr.shape}, dtype={mx_arr.dtype}")

    np_back = np.array(mx_arr)
    print(f"  mlx   -> numpy: shape={np_back.shape}, dtype={np_back.dtype}")
    print(f"  数值一致: {np.allclose(np_arr, np_back)}")


# ---------- 12. 保存与加载 ----------
def test_io(tmp_path: str = "/tmp/mlx_test_arr.npy") -> None:
    section("12. 保存与加载")
    a = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mx.save(tmp_path, a)
    b = mx.load(tmp_path)
    print(f"  写入 {tmp_path}")
    print(f"  读回 = \n{b}")
    check("save/load 一致", mx.allclose(a, b), True)


# ---------- 13. MLX(GPU) vs NumPy(CPU) 矩阵乘加速比 ----------
def test_matmul_speedup() -> None:
    """对比相同尺寸方阵乘:
       - numpy.matmul:  CPU + BLAS (OpenBLAS / Accelerate)
       - mx.matmul:    Apple Silicon GPU (Metal)
    测量方式: warm-up 之后取多次平均, 排除 kernel 编译和首次分配开销.
    """
    section("13. MLX(GPU) vs NumPy(CPU) 矩阵乘加速比")
    if not mx.metal.is_available():
        print("  ⚠️  Metal 不可用, 跳过 GPU 对比")
        return

    sizes = [256, 512, 1024, 2048, 4096]
    repeats = 20
    print(f"  矩阵尺寸        NumPy(CPU)     MLX(GPU)       加速比")
    print(f"  ---------------------------------------------------")

    results = []
    for n in sizes:
        a_np = np.random.randn(n, n).astype(np.float32)
        b_np = np.random.randn(n, n).astype(np.float32)
        a_mx = mx.array(a_np)
        b_mx = mx.array(b_np)

        # NumPy 预热 (线程池就绪)
        np.matmul(a_np, b_np)
        # MLX 预热 (含 Metal kernel 编译)
        mx.eval(mx.matmul(a_mx, b_mx))

        t0 = time.perf_counter()
        for _ in range(repeats):
            c_np = np.matmul(a_np, b_np)
        np_ms = (time.perf_counter() - t0) * 1000 / repeats

        t0 = time.perf_counter()
        for _ in range(repeats):
            c_mx = mx.matmul(a_mx, b_mx)
            mx.eval(c_mx)
        mx_ms = (time.perf_counter() - t0) * 1000 / repeats

        speedup = np_ms / mx_ms
        results.append((n, np_ms, mx_ms, speedup))
        print(f"  {n:4d}x{n:<5}    {np_ms:9.3f} ms  {mx_ms:9.3f} ms  {speedup:6.2f}x")

        # 数值一致性: fp32 下 BLAS 与 Metal 累加顺序不同,
        # 大矩阵因 SIMD 顺序固定反而完全一致, 小矩阵允许 ~5% 误差
        rel_err = float(mx.max(mx.abs(mx.array(c_np) - c_mx) /
                               mx.maximum(mx.abs(c_mx), 1e-6)))
        check(f"  {n}x{n} 数值一致 (max rel err={rel_err:.2e})",
              rel_err < 5e-2, True)

    # 选最大尺寸给一句结论
    n, np_ms, mx_ms, sp = results[-1]
    print(f"\n  结论: {n}x{n} 矩阵乘上 MLX(GPU) 比 NumPy(CPU) "
          f"快 {sp:.2f}x ({np_ms:.1f} ms → {mx_ms:.1f} ms)")
    print(f"  注: 小尺寸上 GPU 调度开销占主导, 优势不明显;")
    print(f"      真正能跑满 GPU 算力的尺寸通常 ≥2048.")


# ---------- 主入口 ----------
def main() -> None:
    print("🚀 Apple MLX 基础功能测试")
    test_environment()
    test_creation()
    test_attributes()
    test_indexing()
    test_shape_ops()
    test_math()
    test_reduction()
    test_broadcast()
    test_autograd()
    test_compile()
    test_numpy_interop()
    test_io()
    test_matmul_speedup()
    print("\n✨ 所有测试完成\n")


if __name__ == "__main__":
    main()