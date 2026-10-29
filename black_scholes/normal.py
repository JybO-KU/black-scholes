"""标准正态分布封装。

集中封装 scipy.stats.norm 的 CDF 与 PDF，避免在核心模块中反复构造
分布对象，同时便于未来替换实现（如改用 math.erf 消除 scipy 依赖）。
"""

from scipy.stats import norm as _norm

__all__ = ["cdf", "pdf"]


def cdf(x):
    """标准正态分布累积分布函数 N(x)。"""
    return _norm.cdf(x)


def pdf(x):
    """标准正态分布概率密度函数 φ(x) = exp(-x^2/2) / sqrt(2π)。"""
    return _norm.pdf(x)
