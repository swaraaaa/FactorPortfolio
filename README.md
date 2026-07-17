# Factor-Based Long-Short Portfolio Allocation under Beta Constraints

**Fama-French Three-Factor Model · Portfolio Optimization · Long-Short Strategy · Risk Management · Python**

> FE-630 Final Project — MS Financial Engineering, Stevens Institute of Technology (Dec 2025)

> Author: Swara Dave

> Advisor: Prof. Papa Momar Ndiaye

---

## 📌 Overview

This project constructs and compares two factor-based long-short portfolio allocation strategies under explicit beta constraints using the Fama-French three-factor model. Portfolios are rebalanced weekly over **March 2007 – October 2025** across a 12-ETF global universe covering equities, commodities, currencies, and fixed income.

**Two strategies compared:**
1. **Strategy I — Robust Utility Optimization:** Low-beta, volatility-penalizing allocation targeting market de-correlation
2. **Strategy II — Information Ratio Optimization:** Benchmark-relative active strategy maximizing risk-adjusted excess returns over SPY

---

## 📊 Key Results

### Full-Sample Performance (S=90 days, λ=0.5)

| Metric | Strategy I | Strategy II | SPY |
|---|---|---|---|
| Cumulative Return | 17.96% | **828.80%** | 535.62% |
| Annualized Volatility | 17.51% | 18.98% | 19.91% |
| Sharpe Ratio | 0.06 | **0.66** | 0.54 |
| Max 10-Day Drawdown | -31.50% | -20.73% | -24.95% |
| CVaR (daily, 5%) | 2.49% | 2.94% | 3.08% |

**Strategy II outperforms SPY** on cumulative return (828.80% vs 535.62%) and Sharpe ratio (0.66 vs 0.54) while maintaining lower volatility (18.98% vs 19.91%).

### Regime-Based Performance (S=90 days, λ=0.5)

| Period | Strategy I Sharpe | Strategy II Sharpe | SPY Sharpe |
|---|---|---|---|
| Pre-Crisis | -1.31 | -0.90 | 0.49 |
| GFC 2008 | **0.30** | -0.57 | -0.84 |
| Post-Crisis | 0.26 | **1.04** | 1.03 |
| COVID-19 | -0.81 | **0.86** | 0.85 |
| Post-COVID | -0.08 | **0.64** | 0.57 |

Strategy I excels during crisis periods (GFC Sharpe: 0.30 vs SPY's -0.84). Strategy II dominates during recovery and expansion phases.

---

## 🗂️ Investment Universe

12 global ETFs covering major asset classes (Mar 2007 – Oct 2025):

| ETF | Description |
|---|---|
| SPY | SPDR S&P 500 ETF (benchmark) |
| QQQ | Invesco NASDAQ-100 ETF |
| GLD | SPDR Gold Trust |
| USO | United States Oil Fund |
| DBA | Invesco DB Agriculture Fund |
| SHV | iShares Short Treasury Bond ETF |
| EWJ | iShares MSCI Japan ETF |
| FXE | CurrencyShares Euro Trust |
| XBI | SPDR S&P Biotech ETF |
| ILF | iShares Latin America 40 ETF |
| EPP | iShares MSCI Pacific ex-Japan ETF |
| FEZ | SPDR EURO STOXX 50 ETF |

---

## ⚙️ Methodology

### Factor Model
- **Fama-French 3-Factor Model:** Market (MKT), Size (SMB), Value (HML)
- Factor loadings estimated via rolling time-series regressions
- Factor-based covariance: Σ = B·Ωf·Bᵀ + D (avoids noisy sample covariance)

### Strategy I — Robust Utility Optimization
```
maximize: ρᵀω − λ·√(ωᵀΣω)
subject to: Σωi = 1, −2 ≤ ωi ≤ 2, −0.5 ≤ β_portfolio ≤ 0.5
```
Solved using **CVXPY** (convex optimization)

### Strategy II — Information Ratio Optimization
```
maximize: (ρᵀω − rSPY) / TEV(ω) − λ·√(ωᵀΣω)
subject to: Σωi = 1, −2 ≤ ωi ≤ 2, −2 ≤ β_portfolio ≤ 2
```
Solved using **SLSQP** (SciPy nonlinear optimizer)

### Sensitivity Analysis
- **Risk aversion:** λ ∈ {0.1, 0.5, 1.0}
- **Estimation horizons:** Short (40d), Medium (90d), Long (180d) for both returns and covariance
- **Market regimes:** Pre-Crisis, GFC, Post-Crisis, COVID-19, Post-COVID

---

## 📁 Repository Structure

```
FactorPortfolio/
├── factor_portfolio_optimization.py    # Full implementation — factor model, optimization, backtesting
└── Plots/
    ├── CumulativePnL.jpeg              # Growth of $100: Strategy I vs II vs SPY
    ├── StrategyI.jpeg                  # Strategy I return distribution
    ├── StrategyII.jpeg                 # Strategy II return distribution
    ├── PerformanceComparison.jpeg      # Side-by-side performance metrics
    ├── TermStructureSensitivity.jpeg   # Sensitivity to estimation horizons
    └── SensitivityAcrossLambda.jpeg    # Sensitivity to risk-aversion parameter
```

---

## 🚀 How to Run

1. Clone the repo
2. Install dependencies:
```bash
pip install numpy pandas matplotlib scipy cvxpy statsmodels yfinance
```
3. Run the script:
```bash
python factor_portfolio_optimization.py
```

> **Note:** Stock data is fetched automatically via `yfinance`. Fama-French factor data is downloaded from [Ken French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). No manual data download required.

---

## 👤 Author

**Swara Dave** — MS Financial Engineering, Stevens Institute of Technology
[![LinkedIn](https://img.shields.io/badge/LinkedIn-swara--dave-blue?style=flat&logo=linkedin)](https://linkedin.com/in/swara-dave) [![GitHub](https://img.shields.io/badge/GitHub-swaraaaa-black?style=flat&logo=github)](https://github.com/swaraaaa)
