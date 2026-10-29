"""Black-Scholes 模型单元测试。

覆盖：
1. Hull 教科书基准值
2. 看涨看跌平价（自洽性）
3. Greeks 有限差分对比
4. 蒙特卡洛与闭式解一致性（置信区间覆盖）
5. 蒙特卡洛随机种子复现性
6. 非法参数异常处理
"""

import math

import pytest

from black_scholes import price, greeks, mc_price


# ---------------------------------------------------------------------------
# 1. Hull 教科书基准值
# ---------------------------------------------------------------------------
def test_hull_call_benchmark():
    c = price(42, 40, 0.5, 0.10, 0.20, "call")
    assert c == pytest.approx(4.76, abs=0.02)


def test_hull_put_benchmark():
    p = price(42, 40, 0.5, 0.10, 0.20, "put")
    assert p == pytest.approx(0.81, abs=0.02)


# ---------------------------------------------------------------------------
# 2. 看涨看跌平价
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "S,K,T,r,sigma",
    [
        (100, 100, 1.0, 0.05, 0.20),
        (42, 40, 0.5, 0.10, 0.20),
        (50, 60, 0.25, 0.03, 0.40),   # OTM
        (120, 100, 2.0, 0.05, 0.15),  # ITM
    ],
)
def test_put_call_parity(S, K, T, r, sigma):
    c = price(S, K, T, r, sigma, "call")
    p = price(S, K, T, r, sigma, "put")
    # C + K·e^(-rT) = P + S
    residual = (c + K * math.exp(-r * T)) - (p + S)
    assert residual == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# 3. Greeks 有限差分对比
# ---------------------------------------------------------------------------
def test_delta_finite_difference():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    h = 0.01
    c_up = price(S + h, K, T, r, sigma, "call")
    c_dn = price(S - h, K, T, r, sigma, "call")
    delta_fd = (c_up - c_dn) / (2 * h)
    assert greeks(S, K, T, r, sigma, "call")["delta"] == pytest.approx(delta_fd, abs=1e-4)


def test_gamma_finite_difference():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    h = 0.1
    c0 = price(S, K, T, r, sigma, "call")
    c_up = price(S + h, K, T, r, sigma, "call")
    c_dn = price(S - h, K, T, r, sigma, "call")
    gamma_fd = (c_up - 2 * c0 + c_dn) / (h ** 2)
    assert greeks(S, K, T, r, sigma, "call")["gamma"] == pytest.approx(gamma_fd, abs=1e-4)


def test_vega_finite_difference():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    h = 1e-4
    c_up = price(S, K, T, r, sigma + h, "call")
    c_dn = price(S, K, T, r, sigma - h, "call")
    vega_fd = (c_up - c_dn) / (2 * h)
    assert greeks(S, K, T, r, sigma, "call")["vega"] == pytest.approx(vega_fd, rel=1e-3)


def test_theta_finite_difference():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    h = 1e-4
    c_now = price(S, K, T, r, sigma, "call")
    c_later = price(S, K, T + h, r, sigma, "call")
    theta_fd = -(c_later - c_now) / h   # theta = -dC/dT
    assert greeks(S, K, T, r, sigma, "call")["theta"] == pytest.approx(theta_fd, rel=1e-3)


# ---------------------------------------------------------------------------
# 4. 蒙特卡洛与闭式解一致性
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_within_confidence_interval(option_type):
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    closed = price(S, K, T, r, sigma, option_type)
    est, se = mc_price(S, K, T, r, sigma, option_type, n_paths=200_000, seed=42)
    # 以 3σ 为容差，覆盖 ~99.7% 的抽样波动，稳健且能暴露系统性偏差
    assert abs(est - closed) < 3 * se + 1e-6


def test_mc_antithetic_reduces_variance():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    _, se_plain = mc_price(S, K, T, r, sigma, "call", n_paths=200_000, seed=1,
                           use_antithetic=False)
    _, se_anti = mc_price(S, K, T, r, sigma, "call", n_paths=200_000, seed=1,
                          use_antithetic=True)
    assert se_anti < se_plain


# ---------------------------------------------------------------------------
# 5. 随机种子复现性
# ---------------------------------------------------------------------------
def test_mc_reproducibility():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    e1, s1 = mc_price(S, K, T, r, sigma, "call", n_paths=50_000, seed=42)
    e2, s2 = mc_price(S, K, T, r, sigma, "call", n_paths=50_000, seed=42)
    assert e1 == e2 and s1 == s2


# ---------------------------------------------------------------------------
# 6. 非法参数
# ---------------------------------------------------------------------------
def test_invalid_option_type():
    with pytest.raises(ValueError):
        price(100, 100, 1.0, 0.05, 0.20, "american")
    with pytest.raises(ValueError):
        mc_price(100, 100, 1.0, 0.05, 0.20, "american")
