"""蒙特卡洛欧式期权定价（用于与闭式解交叉验证）。

原理：在风险中性测度下，期权价格 = 折现后的到期期望 payoff。
通过模拟标的在几何布朗运动下的到期价格 S_T，用样本均值近似期望值。

关键点：
- 漂移项必须用无风险利率 r（而非真实漂移 μ）；
- 欧式期权 payoff 仅依赖 S_T，可用单步精确抽样（无需离散时间步）：
      S_T = S·exp((r - σ²/2)·T + σ·√T·Z),  Z ~ N(0,1)
- 支持对偶变量（antithetic variates）以降低方差；
- 固定随机种子保证可复现；
- 返回标准误以量化估计精度。
"""

import numpy as np

__all__ = ["mc_price"]


def mc_price(S, K, T, r, sigma, option_type="call", n_paths=1_000_000,
             seed=42, use_antithetic=True):
    """蒙特卡洛定价，返回 (估计价格, 标准误)。

    参数
    ----
    n_paths : 模拟路径数（越大估计越精确，误差按 1/√n 下降）
    seed : 随机种子，固定后结果可复现
    use_antithetic : 是否使用对偶变量（Z 与 -Z 成对，可降低方差）
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type 必须为 'call' 或 'put'")

    rng = np.random.default_rng(seed)

    if use_antithetic:
        n = n_paths // 2 * 2          # 确保成对，总路径数接近 n_paths
        half = rng.standard_normal(n // 2)
        Z = np.concatenate([half, -half])
    else:
        n = n_paths
        Z = rng.standard_normal(n)

    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    discounted = np.exp(-r * T) * payoff
    estimate = float(discounted.mean())
    std_err = float(discounted.std(ddof=1) / np.sqrt(n))
    return estimate, std_err
