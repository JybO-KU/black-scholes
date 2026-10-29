"""Black-Scholes 期权定价模型（含蒙特卡洛交叉验证）。

基础用法::

    from black_scholes import price, greeks, mc_price

    call = price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
"""

from .model import price, greeks
from .monte_carlo import mc_price

__all__ = ["price", "greeks", "mc_price"]
__version__ = "0.1.0"
