"""蒙特卡洛交叉验证脚本。

对 Black-Scholes 闭式解执行蒙特卡洛交叉验证，输出一份可读报告，包含：

1. 目标指标：绝对误差、相对误差、标准误、95% 置信区间覆盖判断；
2. 收敛性检验：样本量 N 递增时，误差按 1/√N 下降（log-log 斜率 ≈ -0.5）。

用法::

    python scripts/validate_mc.py

随机性来源：numpy `default_rng(seed=42)`（PCG64）+ 对偶变量。
收敛判据：|MC - 闭式| ≤ 1.96 × SE（落在 95% 置信区间内），
          且 log-log 收敛斜率接近 -0.5。
"""

import sys
from pathlib import Path

import numpy as np

# 允许脚本在任意工作目录直接运行（如 `python scripts/validate_mc.py`）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from black_scholes import price, mc_price


def main():
    # 基础参数（ATM 平价期权）
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    print("=" * 72)
    print("Black-Scholes 蒙特卡洛交叉验证报告")
    print(f"参数: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
    print("随机性: numpy default_rng(seed=42) + 对偶变量")
    print("=" * 72)

    # ---- 1. 单点验证（call 与 put） ----
    print("\n[1] 定价一致性（n_paths = 1,000,000）")
    print(f"{'类型':<6}{'闭式解':>12}{'MC 估计':>12}{'标准误':>12}"
          f"{'绝对误差':>12}{'相对误差':>10}{'95%覆盖':>8}")
    for opt in ("call", "put"):
        closed = price(S, K, T, r, sigma, opt)
        est, se = mc_price(S, K, T, r, sigma, opt,
                           n_paths=1_000_000, seed=42)
        abs_err = est - closed
        rel_err = abs_err / closed if closed else float("nan")
        covered = abs(abs_err) <= 1.96 * se
        print(f"{opt:<6}{closed:>12.5f}{est:>12.5f}{se:>12.5f}"
              f"{abs_err:>12.5f}{rel_err:>10.2%}{str(covered):>8}")

    # ---- 2. 收敛性检验 ----
    print("\n[2] 收敛性检验（误差应随 N 以 1/√N 下降）")
    n_list = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    closed = price(S, K, T, r, sigma, "call")
    print(f"{'N(路径数)':>14}{'误差':>12}{'标准误':>12}")
    log_n, log_err = [], []
    for n in n_list:
        est, se = mc_price(S, K, T, r, sigma, "call",
                           n_paths=n, seed=42)
        err = abs(est - closed)
        print(f"{n:>14,}{err:>12.6f}{se:>12.6f}")
        log_n.append(np.log10(n))
        log_err.append(np.log10(err))

    # 线性回归求 log-log 斜率
    slope = np.polyfit(log_n, log_err, 1)[0]
    print(f"\nlog-log 回归斜率 = {slope:.3f}  (理论值 -0.5)")

    # ---- 结论 ----
    print("\n" + "=" * 72)
    if abs(slope + 0.5) < 0.15:
        print("结论: 通过 —— MC 与闭式解一致，且收敛阶符合 1/√N 理论预期。")
    else:
        print("结论: 异常 —— 收敛阶偏离 -0.5，请检查 MC 实现。")
    print("=" * 72)


if __name__ == "__main__":
    main()
