"""最小使用示例：演示定价、Greeks 与蒙特卡洛验证。"""

import sys
from pathlib import Path

# 允许脚本在任意工作目录直接运行（如 `python examples/quickstart.py`）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from black_scholes import price, greeks, mc_price


def main():
    # 参数：平价 ATM 期权
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    # 1. 闭式定价
    call = price(S, K, T, r, sigma, "call")
    put = price(S, K, T, r, sigma, "put")
    print(f"看涨期权价格: {call:.4f}")
    print(f"看跌期权价格: {put:.4f}")

    # 2. Greeks
    g = greeks(S, K, T, r, sigma, "call")
    print("\nGreeks (call):")
    for name, val in g.items():
        print(f"  {name:<6}: {val:.6f}")

    # 3. 蒙特卡洛验证
    est, se = mc_price(S, K, T, r, sigma, "call",
                       n_paths=500_000, seed=42)
    print(f"\n蒙特卡洛估计: {est:.4f} ± {se:.5f}  (闭式 {call:.4f})")


if __name__ == "__main__":
    main()
