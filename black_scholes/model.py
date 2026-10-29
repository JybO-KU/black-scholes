"""Black-Scholes 欧式期权闭式定价与 Greeks。

模型假设（在使用前务必理解其边界）：
- 标的服从几何布朗运动，价格服从对数正态分布；
- 波动率 σ 恒定、无风险利率 r 恒定、连续复利；
- 标的在期权存续期内不分红；
- 欧式行权（仅到期可行权）。

本模块为核心纯函数：给定输入参数，返回确定性的定价 / 敏感度结果，
无任何 I/O 或随机性，便于单元测试与交叉验证。
"""

import numpy as np

from .normal import cdf, pdf

__all__ = ["price", "greeks", "d1_d2"]


def d1_d2(S, K, T, r, sigma):
    """计算 BS 中间变量 d1、d2。

    d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
    d2 = d1 - σ·√T
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def price(S, K, T, r, sigma, option_type="call"):
    """欧式期权 BS 定价。

    参数
    ----
    S : 标的现价
    K : 执行价
    T : 到期期限（年）
    r : 无风险利率（连续复利，如 0.05 表示 5%）
    sigma : 年化波动率（如 0.20 表示 20%）
    option_type : "call" 或 "put"

    返回
    ----
    float : 期权理论价格
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type 必须为 'call' 或 'put'")

    d1, d2 = d1_d2(S, K, T, r, sigma)
    disc = np.exp(-r * T)

    if option_type == "call":
        return S * cdf(d1) - K * disc * cdf(d2)
    return K * disc * cdf(-d2) - S * cdf(-d1)


def greeks(S, K, T, r, sigma, option_type="call"):
    """计算 BS 一阶/二阶 Greeks。

    返回 dict，键与单位口径如下：
    - delta : 每单位标的价格变化（call ∈ [0,1]，put ∈ [-1,0]）
    - gamma : delta 对 S 的二阶导数（call 与 put 相同）
    - vega  : 每 1 个波动率单位（dσ=1.0）的价格变化（call 与 put 相同）；
              常见"每 1%"市场口径为 vega / 100
    - theta : 每 1 年（除以 365 得每日）的时间衰减（通常为负）
    - rho   : 每 1 个利率单位（dr=1.0）的价格变化
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type 必须为 'call' 或 'put'")

    d1, d2 = d1_d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    disc = np.exp(-r * T)
    pdf_d1 = pdf(d1)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T

    if option_type == "call":
        delta = cdf(d1)
        theta = -S * pdf_d1 * sigma / (2 * sqrt_T) - r * K * disc * cdf(d2)
        rho = K * T * disc * cdf(d2)
    else:
        delta = cdf(d1) - 1.0
        theta = -S * pdf_d1 * sigma / (2 * sqrt_T) + r * K * disc * cdf(-d2)
        rho = -K * T * disc * cdf(-d2)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
