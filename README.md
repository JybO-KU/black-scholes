# Black-Scholes 期权定价模型

一个用于**教学与研究**的 Black-Scholes（BS）欧式期权定价模型，提供解析闭式解、Greeks 风险敏感度，以及独立的**蒙特卡洛（Monte Carlo）交叉验证**。

> **免责声明**：本项目仅用于学习与学术研究，不构成任何投资建议。BS 模型基于强假设，实际市场存在波动率微笑、肥尾等偏离，请勿直接用于实盘定价或交易决策。

---

## 特性

- **闭式定价**：欧式看涨 / 看跌期权解析解（无分红、恒定波动率）
- **Greeks**：Delta、Gamma、Vega、Theta、Rho
- **蒙特卡洛验证**：独立数值实现，与闭式解交叉验证，含标准误与 95% 置信区间
- **收敛性检验**：验证误差按 `1/√N` 理论速率下降
- **完整单元测试**：Hull 基准值、看涨看跌平价、Greeks 有限差分、MC 一致性

## 模型假设与局限

- 标的价格服从几何布朗运动（对数正态分布）
- 波动率 σ 与无风险利率 r 恒定（连续复利）
- 存续期内不分红
- **欧式行权**（仅到期可行权）——不适用于美式期权

BS 公式与蒙特卡洛共享同一套假设，两者一致仅证明「模型内部自洽 + 实现正确」，**不能证明**模型符合真实市场。

## 安装

```bash
git clone <你的仓库地址>
cd black-scholes
python -m venv .venv
# Windows:  .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt        # 运行时依赖（numpy、scipy）
pip install -e .                       # 可选：以可编辑模式安装本项目
pip install pytest                     # 可选：运行测试
```

## 快速开始

```python
from black_scholes import price, greeks, mc_price

# 平价期权：S=100, K=100, T=1年, r=5%, sigma=20%
call = price(100, 100, 1.0, 0.05, 0.20, "call")   # 10.4506
g = greeks(100, 100, 1.0, 0.05, 0.20, "call")     # delta/gamma/vega/theta/rho

# 蒙特卡洛验证
est, se = mc_price(100, 100, 1.0, 0.05, 0.20, "call", n_paths=1_000_000, seed=42)
print(f"{est:.4f} ± {se:.5f}")
```

完整示例见 `examples/quickstart.py`。

## 参数说明

| 参数 | 符号 | 说明 | 典型取值 |
|---|---|---|---|
| 标的现价 | `S` | 股票 / ETF 当前价格 | — |
| 执行价 | `K` | 行权价 | — |
| 到期期限 | `T` | 单位：年 | 0 ~ 2 |
| 无风险利率 | `r` | 连续复利 | 0.03 ~ 0.05 |
| 波动率 | `sigma` | 年化波动率 | 0.20 ~ 0.40 |

## 核心公式

```
d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
d2 = d1 - σ·√T

call = S·N(d1) - K·e^(-rT)·N(d2)
put  = K·e^(-rT)·N(-d2) - S·N(-d1)
```

其中 `N(·)` 为标准正态分布累积分布函数。Greeks 为一阶 / 二阶偏导闭式解，见 `black_scholes/model.py`。

## 蒙特卡洛交叉验证

蒙特卡洛在风险中性测度下模拟标的路径，用样本均值近似折现期望 payoff：

```
S_T = S·exp((r - σ²/2)·T + σ·√T·Z),  Z ~ N(0,1)
price ≈ e^(-rT) · mean(max(S_T - K, 0))
```

验证设计要点：

| 维度 | 实现 |
|---|---|
| 目标指标 | 绝对误差、相对误差、标准误、95% 置信区间覆盖 |
| 样本量 | 单点验证 1e6；收敛性检验 1e3 → 1e7 |
| 随机性来源 | numpy `default_rng(seed=42)`（PCG64）+ 对偶变量 |
| 收敛判据 | `\|MC - 闭式\| ≤ 1.96·SE` 且 log-log 斜率 ≈ -0.5 |

运行验证：

```bash
python scripts/validate_mc.py
```

## 测试

```bash
pytest tests/ -v
```

覆盖 Hull 教科书基准值、看涨看跌平价、Greeks 有限差分、MC 置信区间覆盖与复现性。

## 项目结构

```
black-scholes/
├── black_scholes/
│   ├── __init__.py        # 公共 API
│   ├── normal.py          # 标准正态 CDF/PDF
│   ├── model.py           # 闭式定价 + Greeks
│   └── monte_carlo.py     # 蒙特卡洛定价（对偶变量）
├── tests/                 # 单元测试
├── scripts/validate_mc.py # MC 交叉验证脚本
├── examples/quickstart.py # 最小示例
├── README.md / requirements.txt / pyproject.toml / LICENSE
```

## 扩展方向

- 隐含波动率反解（Newton-Raphson）
- 美式期权（二叉树 / 有限差分）
- 分红、利率期限结构
- 波动率微笑修正（SABR / 局部波动率）

## License

[MIT](LICENSE)
