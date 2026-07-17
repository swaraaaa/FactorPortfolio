from __future__ import annotations

import io
import os
import zipfile
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf

import statsmodels.api as sm
import cvxpy as cp
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis, norm

# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    # Investment universe (must include SPY and SHV per assignment setup)
    tickers: List[str]

    benchmark: str = "SPY"
    cash_proxy: str = "SHV"

    # March 1, 2007 to end of Oct 2025 (inclusive)
    start: str = "2007-03-01"
    end: str = "2025-11-01"  # exclusive bound to include Oct 2025

    # weekly re-optimization: every 5 trading days
    rebalance_step: int = 5

    # Term-structure choices
    cov_windows: Tuple[int, ...] = (40, 90, 180)
    ret_windows: Tuple[int, ...] = (40, 90, 180)
    lambdas: Tuple[float, ...] = (0.1, 0.5, 1.0)

    # Weight constraints
    weight_bounds: Tuple[float, float] = (-2.0, 2.0)

    # Beta constraints
    beta_bounds_strat1: Tuple[float, float] = (-0.5, 0.5)
    beta_bounds_strat2: Tuple[float, float] = (-2.0, 2.0)

    # Metrics
    alpha_var: float = 0.05
    annualization: int = 250

    # Regimes (assignment: choose approximative dates)
    # Feel free to adjust to your report’s chosen dates.
    regimes: Tuple[Tuple[str, str, str], ...] = (
        ("PreCrisis", "2007-03-01", "2007-08-31"),
        ("Crisis08",  "2007-09-01", "2009-03-31"),
        ("Normal",    "2009-04-01", "2019-12-31"),
        ("COVID",     "2020-01-01", "2021-06-30"),
        ("PostCOVID", "2021-07-01", "2025-10-31"),
    )

    # For *separate* per-regime backtests, we need history to estimate inputs at the start.
    warmup_buffer_days: int = 10

    # Output
    out_dir: str = "fe630_outputs"


# -----------------------------
# Data
# -----------------------------
def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    px = yf.download(
        tickers, start=start, end=end, progress=False, auto_adjust=True
    )["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(how="all").ffill()
    return px


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def download_ff3_daily() -> pd.DataFrame:
    """
    Ken French daily data factors: Mkt-RF, SMB, HML, RF (returned as decimals, not %).
    """
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    with urllib.request.urlopen(url) as resp:
        zbytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        raw = zf.read(name).decode("utf-8", errors="ignore")

    lines = raw.splitlines()
    data_lines = [ln for ln in lines if len(ln) >= 8 and ln[:8].isdigit()]
    csv_text = "Date,Mkt-RF,SMB,HML,RF\n" + "\n".join(data_lines)

    ff = pd.read_csv(io.StringIO(csv_text))
    ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m%d")
    ff = ff.set_index("Date").sort_index()
    ff = ff.astype(float) / 100.0
    return ff


# -----------------------------
# Factor covariance model: Sigma = B Ω_f B' + D
# -----------------------------
def estimate_factor_model(
    returns: pd.DataFrame,
    ff: pd.DataFrame,
) -> Tuple[np.ndarray, pd.Index]:
    """
    Given a window of DAILY returns for assets and daily FF factors aligned to the same dates:
      - regress each asset's excess return on (Mkt-RF, SMB, HML)
      - compute B (Nx3), D (NxN diagonal), Ω_f (3x3) covariance of factors
      - return Sigma (NxN) and the asset ordering (Index)
    """
    # Align
    common = returns.index.intersection(ff.index)
    R = returns.loc[common].dropna(how="any")  # require all assets present
    F = ff.loc[R.index, ["Mkt-RF", "SMB", "HML", "RF"]].dropna()

    common2 = R.index.intersection(F.index)
    R = R.loc[common2]
    F = F.loc[common2]

    if len(R) < 30:
        raise ValueError("Not enough observations in window to estimate factor model.")

    rf = F["RF"]
    R_ex = R.sub(rf, axis=0)

    X = sm.add_constant(F[["Mkt-RF", "SMB", "HML"]])

    B_rows = []
    resid_vars = []

    for col in R_ex.columns:
        y = R_ex[col].values
        fit = sm.OLS(y, X.values).fit()
        beta = fit.params[1:]  # exclude intercept
        B_rows.append(beta)
        resid_vars.append(np.var(fit.resid, ddof=1))

    B = np.vstack(B_rows)  # N x 3
    D = np.diag(np.clip(np.array(resid_vars, dtype=float), 1e-12, None))  # N x N

    Omega_f = np.cov(F[["Mkt-RF", "SMB", "HML"]].values.T, ddof=1)  # 3x3

    Sigma = B @ Omega_f @ B.T + D
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-10 * np.eye(Sigma.shape[0])  # stabilize

    return Sigma, R.columns


def capm_betas_vs_spy(returns: pd.DataFrame, spy: str = "SPY") -> pd.Series:
    """
    CAPM beta_i = cov(r_i, r_SPY) / var(r_SPY) from a window of returns.
    """
    rM = returns[spy].dropna()
    varM = float(np.var(rM.values, ddof=1)) if len(rM) > 2 else 1e-12
    varM = max(varM, 1e-12)

    betas = {}
    for a in returns.columns:
        ra = returns[a].dropna()
        idx = ra.index.intersection(rM.index)
        if len(idx) < 30:
            betas[a] = 0.0
        else:
            cov = float(np.cov(ra.loc[idx].values, rM.loc[idx].values, ddof=1)[0, 1])
            betas[a] = cov / varM
    return pd.Series(betas).reindex(returns.columns)


def chol_psd(Sigma: np.ndarray) -> np.ndarray:
    """
    Robust-ish Cholesky for PSD matrices by adding jitter; fallback to eigen sqrt.
    """
    jitter = 1e-12
    for _ in range(12):
        try:
            return np.linalg.cholesky(Sigma + jitter * np.eye(Sigma.shape[0]))
        except np.linalg.LinAlgError:
            jitter *= 10
    w, V = np.linalg.eigh(Sigma)
    w = np.clip(w, 1e-12, None)
    return V @ np.diag(np.sqrt(w))


# -----------------------------
# Strategy I (convex, CVXPY)
# max rho^T w - lambda * sqrt(w' Σ w)
# s.t. sum w = 1, -2 <= w_i <= 2, -0.5 <= beta^T w <= 0.5
# -----------------------------
def solve_strategy_1(
    rho: np.ndarray,
    Sigma: np.ndarray,
    beta_m: np.ndarray,
    lam: float,
    beta_bounds: Tuple[float, float],
    w_bounds: Tuple[float, float],
) -> np.ndarray:
    n = len(rho)
    w = cp.Variable(n)

    L = chol_psd(Sigma)
    vol = cp.norm(L @ w, 2)

    constraints = [
        cp.sum(w) == 1.0,
        w >= w_bounds[0],
        w <= w_bounds[1],
        beta_m @ w >= beta_bounds[0],
        beta_m @ w <= beta_bounds[1],
    ]
    prob = cp.Problem(cp.Maximize(rho @ w - lam * vol), constraints)

    try:
        prob.solve(solver="ECOS", verbose=False)
    except Exception:
        prob.solve(solver="SCS", verbose=False)

    if w.value is None:
        raise RuntimeError("Strategy I optimization failed.")
    return np.asarray(w.value).reshape(-1)


# -----------------------------
# Strategy II (nonlinear, SLSQP)
# max (rho^T w - r_SPY)/TEV(w)  -  lambda * sqrt(w' Σ w)
# s.t. sum w = 1, -2 <= w_i <= 2, -2 <= beta^T w <= 2
#
# TEV(w) = std(rP - rSPY) with closed-form:
#   TEV(w)^2 = w' Σ w - 2 w' cov(:,SPY) + var(SPY)
# -----------------------------
def solve_strategy_2(
    rho: np.ndarray,
    Sigma: np.ndarray,
    beta_m: np.ndarray,
    lam: float,
    beta_bounds: Tuple[float, float],
    w_bounds: Tuple[float, float],
    spy_idx: int,
) -> np.ndarray:
    n = len(rho)
    Sigma = 0.5 * (Sigma + Sigma.T)

    cov_vec = Sigma[:, spy_idx].copy()
    var_spy = float(Sigma[spy_idx, spy_idx])
    r_spy = float(rho[spy_idx])

    eps = 1e-6

    def port_vol(w: np.ndarray) -> float:
        return float(np.sqrt(max(w @ Sigma @ w, eps)))

    def tev(w: np.ndarray) -> float:
        v = float(w @ Sigma @ w - 2.0 * w @ cov_vec + var_spy)
        return float(np.sqrt(max(v, eps)))

    def neg_obj(w: np.ndarray) -> float:
        w = np.asarray(w)
        val = (float(rho @ w) - r_spy) / tev(w) - lam * port_vol(w)
        return -val  # maximize -> minimize negative

    cons = [
        {"type": "eq",   "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: (beta_m @ w) - beta_bounds[0]},
        {"type": "ineq", "fun": lambda w: beta_bounds[1] - (beta_m @ w)},
    ]
    bounds = [w_bounds] * n
    x0 = np.ones(n) / n

    res = minimize(
        neg_obj,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2500, "ftol": 1e-10, "disp": False},
    )

    if not res.success:
        # Retry from a few random starts (helps with nonconvexity)
        best = None
        rng = np.random.default_rng(123)
        for _ in range(10):
            x0 = rng.normal(0.0, 0.25, size=n)
            x0 = np.clip(x0, w_bounds[0], w_bounds[1])
            x0 = x0 - x0.mean() + 1.0 / n  # roughly enforce sum=1
            x0 = np.clip(x0, w_bounds[0], w_bounds[1])
            r2 = minimize(
                neg_obj,
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 3000, "ftol": 1e-10, "disp": False},
            )
            if r2.success and (best is None or r2.fun < best.fun):
                best = r2
        if best is not None:
            res = best

    if not res.success:
        raise RuntimeError(f"Strategy II optimization failed: {res.message}")

    return np.asarray(res.x).reshape(-1)


# -----------------------------
# Backtest engine (weekly, non-anticipative)
# -----------------------------
def backtest_weekly(
    rets: pd.DataFrame,
    ff: pd.DataFrame,
    cov_window: int,
    ret_window: int,
    lam: float,
    cfg: Config,
) -> Dict[str, pd.Series]:
    tickers = list(rets.columns)
    if cfg.benchmark not in tickers:
        raise ValueError("Benchmark SPY must be present in tickers.")
    if cfg.cash_proxy not in tickers:
        raise ValueError("Cash proxy SHV must be present in tickers.")

    spy_idx = tickers.index(cfg.benchmark)

    max_lb = max(cov_window, ret_window)
    nT = len(rets)

    # first rebalance date index (needs enough history *before* i)
    start_i = max_lb
    rebalance_points = list(range(start_i, nT, cfg.rebalance_step))

    r1 = pd.Series(index=rets.index, dtype=float)
    r2 = pd.Series(index=rets.index, dtype=float)

    for k, i in enumerate(rebalance_points):
        # history windows end at i-1 (non-anticipative)
        cov_slice = rets.iloc[i - cov_window:i]
        ret_slice = rets.iloc[i - ret_window:i]

        # Trend-following expected return estimator (rho): rolling mean of returns
        rho = ret_slice.mean().values  # N-vector

        # Factor model sigma estimated from cov_slice dates
        ff_hist = ff.reindex(cov_slice.index).dropna()
        # Must align factors to the return window used for covariance estimation
        try:
            Sigma, cols = estimate_factor_model(cov_slice, ff_hist)
        except Exception:
            continue

        # CAPM betas vs SPY (estimated from same cov_slice)
        beta_m = capm_betas_vs_spy(cov_slice, cfg.benchmark).values

        w1 = solve_strategy_1(rho, Sigma, beta_m, lam, cfg.beta_bounds_strat1, cfg.weight_bounds)
        w2 = solve_strategy_2(rho, Sigma, beta_m, lam, cfg.beta_bounds_strat2, cfg.weight_bounds, spy_idx)

        # Apply weights until next rebalance point
        i_next = rebalance_points[k + 1] if k + 1 < len(rebalance_points) else nT
        hold = rets.iloc[i:i_next]  # realized returns after weights are chosen

        r1.iloc[i:i_next] = hold.values @ w1
        r2.iloc[i:i_next] = hold.values @ w2

    spy = rets[cfg.benchmark].copy()
    return {"StrategyI": r1, "StrategyII": r2, "SPY": spy}


def backtest_for_period(
    rets_full: pd.DataFrame,
    ff_full: pd.DataFrame,
    cov_window: int,
    ret_window: int,
    lam: float,
    cfg: Config,
    period_start: str,
    period_end: str,
) -> Dict[str, pd.Series]:
    """
    "Separate backtest per sub-period" with warmup history.
    - We include a warmup lookback BEFORE period_start so we can estimate inputs.
    - We return PnL only inside [period_start, period_end].
    """
    start = pd.to_datetime(period_start)
    end = pd.to_datetime(period_end)

    idx_period = rets_full.loc[start:end].index
    if len(idx_period) == 0:
        return {"StrategyI": pd.Series(dtype=float), "StrategyII": pd.Series(dtype=float), "SPY": pd.Series(dtype=float)}

    max_lb = max(cov_window, ret_window) + cfg.warmup_buffer_days

    pos0 = rets_full.index.get_indexer([idx_period[0]])[0]
    warm_pos0 = max(0, pos0 - max_lb)
    warm_start = rets_full.index[warm_pos0]

    rets_slice = rets_full.loc[warm_start:end]
    ff_slice = ff_full.reindex(rets_slice.index).dropna()

    rd = backtest_weekly(rets_slice, ff_slice, cov_window, ret_window, lam, cfg)
    return {k: v.loc[idx_period] for k, v in rd.items()}


# -----------------------------
# Metrics (assignment table)
# -----------------------------
def wealth(r: pd.Series, start: float = 100.0) -> pd.Series:
    r = r.dropna()
    if len(r) == 0:
        return pd.Series(dtype=float)
    return start * (1.0 + r).cumprod()


def max_drawdown(w: pd.Series) -> float:
    if len(w) == 0:
        return np.nan
    peak = w.cummax()
    dd = (w / peak) - 1.0
    return float(dd.min())


def max_10day_drawdown(w: pd.Series, window: int = 10) -> float:
    if len(w) < window:
        return np.nan
    vals = w.values
    worst = 0.0
    for i in range(len(vals) - window + 1):
        seg = pd.Series(vals[i:i + window])
        worst = min(worst, max_drawdown(seg))
    return float(worst)


def modified_var_cf(r: pd.Series, alpha: float = 0.05) -> float:
    """
    Modified VaR via Cornish-Fisher expansion (reported as positive loss number).
    """
    x = r.dropna().values
    if len(x) < 30:
        return np.nan
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    if s <= 0:
        return np.nan
    sk = float(skew(x, bias=False))
    kt = float(kurtosis(x, fisher=False, bias=False))  # normal=3
    z = norm.ppf(alpha)
    z_cf = (
        z
        + (sk / 6.0) * (z**2 - 1.0)
        + ((kt - 3.0) / 24.0) * (z**3 - 3.0 * z)
        - (sk**2 / 36.0) * (2.0 * z**3 - 5.0 * z)
    )
    var = float(-(m + z_cf * s))
    return max(var, 0.0) if np.isfinite(var) else var


def cvar_hist(r: pd.Series, alpha: float = 0.05) -> float:
    """
    Historical CVaR (expected shortfall) reported as positive loss number.
    """
    x = r.dropna().values
    if len(x) < 30:
        return np.nan
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    return float(-np.mean(tail)) if len(tail) else np.nan


def summarize(
    r: pd.Series,
    rf_daily: Optional[pd.Series],
    cfg: Config,
) -> pd.Series:
    x = r.dropna()
    if len(x) < 30:
        return pd.Series(dtype=float)

    w = wealth(x, 100.0)
    ann = cfg.annualization

    arith = float(x.mean())
    geom = float((1.0 + x).prod() ** (1.0 / len(x)) - 1.0)
    vol_ann = float(x.std(ddof=1) * np.sqrt(ann))

    sharpe = np.nan
    if rf_daily is not None:
        idx = x.index.intersection(rf_daily.index)
        ex = (x.loc[idx] - rf_daily.loc[idx]).dropna()
        s = float(ex.std(ddof=1))
        if s > 0:
            sharpe = float(ex.mean() / s * np.sqrt(ann))

    return pd.Series({
        "Cumulated PnL/Return": float((1.0 + x).prod() - 1.0),
        "Daily Mean (Arithmetic)": arith,
        "Daily Mean (Geometric)": geom,
        "Daily Min Return": float(x.min()),
        "Max 10-Day Drawdown": max_10day_drawdown(w, 10),
        "Volatility (Ann.)": vol_ann,
        "Sharpe Ratio (Ann.)": sharpe,
        "Skewness": float(skew(x.values, bias=False)) if len(x) > 3 else np.nan,
        "Kurtosis": float(kurtosis(x.values, fisher=False, bias=False)) if len(x) > 3 else np.nan,
        "Modified VaR (daily, 5%)": modified_var_cf(x, cfg.alpha_var),
        "CVaR (daily, 5%)": cvar_hist(x, cfg.alpha_var),
    })


# -----------------------------
# Plots
# -----------------------------
def plot_cumulative_pnl(rd: Dict[str, pd.Series], title: str, outpath: Optional[str] = None):
    plt.figure()
    for name, r in rd.items():
        w = wealth(r, 100.0)
        if len(w):
            plt.plot(w.index, w.values, label=name)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Growth of $100")
    plt.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=160)
        plt.close()
    else:
        plt.show()


def plot_return_hist(r: pd.Series, title: str, logy: bool = False, outpath: Optional[str] = None):
    plt.figure()
    x = r.dropna().values
    if len(x):
        plt.hist(x, bins=80)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=160)
        plt.close()
    else:
        plt.show()


# -----------------------------
# Run everything
# -----------------------------
def run_all(cfg: Config) -> pd.DataFrame:
    os.makedirs(cfg.out_dir, exist_ok=True)

    prices = download_prices(cfg.tickers, cfg.start, cfg.end)
    rets_full = compute_returns(prices).dropna()

    ff_full = download_ff3_daily()
    ff_full = ff_full.loc[rets_full.index.min():rets_full.index.max()]
    rf_daily_full = ff_full["RF"].reindex(rets_full.index).fillna(0.0)

    periods = [("Whole", cfg.start, "2025-10-31")] + list(cfg.regimes)

    rows = []

    for period_name, a, b in periods:
        print(f"\n=== Period: {period_name} ({a} to {b}) ===")
        idx_period = rets_full.loc[pd.to_datetime(a):pd.to_datetime(b)].index

        for cov_w in cfg.cov_windows:
            for ret_w in cfg.ret_windows:
                for lam in cfg.lambdas:
                    print(f"  Backtest: cov={cov_w}, ret={ret_w}, lambda={lam}")

                    if period_name == "Whole":
                        rd = backtest_weekly(
                            rets_full,
                            ff_full.reindex(rets_full.index).dropna(),
                            cov_w,
                            ret_w,
                            lam,
                            cfg,
                        )
                        start_plot = rd["StrategyI"].first_valid_index()
                        idx_use = rets_full.loc[start_plot:].index if start_plot is not None else rets_full.index
                    else:
                        rd = backtest_for_period(
                            rets_full,
                            ff_full,
                            cov_w,
                            ret_w,
                            lam,
                            cfg,
                            a,
                            b,
                        )
                        idx_use = idx_period

                    for series_name in ["StrategyI", "StrategyII", "SPY"]:
                        r = rd[series_name].reindex(idx_use).dropna()
                        if len(r) < 30:
                            continue
                        s = summarize(r, rf_daily_full.loc[idx_use], cfg)
                        if s.empty:
                            continue
                        s["Period"] = period_name
                        s["cov_window"] = cov_w
                        s["ret_window"] = ret_w
                        s["lambda"] = lam
                        s["Series"] = series_name
                        s["S_notation"] = f"S{ret_w}_{cov_w}"
                        rows.append(s)

    summary = pd.DataFrame(rows).set_index(["Period", "S_notation", "lambda", "Series"]).sort_index()

    # Save summary table
    summary_path = os.path.join(cfg.out_dir, "fe630_summary.csv")
    summary.to_csv(summary_path)
    print(f"\nSaved summary table: {summary_path}")

    # Example plots for one combo (you can change this)
    example_key = (90, 90, 0.5)
    rd_ex = backtest_weekly(
        rets_full,
        ff_full.reindex(rets_full.index).dropna(),
        example_key[0],
        example_key[1],
        example_key[2],
        cfg,
    )
    start_plot = rd_ex["StrategyI"].first_valid_index()
    if start_plot is not None:
        rd_plot = {k: v.loc[start_plot:].dropna() for k, v in rd_ex.items()}
    else:
        rd_plot = {k: v.dropna() for k, v in rd_ex.items()}

    plot_cumulative_pnl(
        rd_plot,
        f"Cumulative PnL (aligned start) — cov={example_key[0]}, ret={example_key[1]}, λ={example_key[2]}",
        outpath=os.path.join(cfg.out_dir, "cum_pnl_example.png"),
    )
    plot_return_hist(
        rd_ex["StrategyI"],
        "Strategy I — Daily Return Distribution",
        outpath=os.path.join(cfg.out_dir, "hist_strat1.png"),
    )
    plot_return_hist(
        rd_ex["StrategyI"],
        "Strategy I — Daily Return Distribution (log frequency)",
        logy=True,
        outpath=os.path.join(cfg.out_dir, "hist_strat1_log.png"),
    )
    plot_return_hist(
        rd_ex["StrategyII"],
        "Strategy II — Daily Return Distribution",
        outpath=os.path.join(cfg.out_dir, "hist_strat2.png"),
    )

    print(f"Saved example plots in: {cfg.out_dir}/")
    return summary


def main():
    tickers = [
        "FXE", "EWJ", "GLD", "QQQ", "SPY", "SHV",
        "DBA", "USO", "XBI", "ILF", "EPP", "FEZ",
    ]
    cfg = Config(tickers=tickers)
    run_all(cfg)


if __name__ == "__main__":
    main()
