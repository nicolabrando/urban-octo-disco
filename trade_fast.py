import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
from scipy.optimize import minimize
from joblib import Parallel, delayed
from numba import njit, float64, int64

# =========================
# Config
# =========================
DATA_PATH = "out_yfinance_sp500/sp500_long_robust_2005_2025.csv"
PRICE_COL = "Adj Close"
DATE_COL = "Date"
TICKER_COL = "Ticker"

V0 = 100.0
X0 = 0.0
TRADING_DAYS = 252

# Parameter grids
L_grid = [1.0, 2.0, 3.0]
k_grid = [1.0, 5.0, 10.0, 20.0, 50.0]
tau_grid = [0.0, 0.0005, 0.0010]

# RSI
RSI_WINDOW = 14
RSI_LO = 30.0
RSI_HI = 70.0

# GARCH
GARCH_WINDOW = 252
GARCH_REFIT_EVERY = 21
GARCH_EPS = 1e-9

# Bootstrap
BOOT_N = 500
BOOT_BLOCK = 10
RNG_SEED = 42

# =========================
# Numba Optimized Helpers
# =========================

@njit(fastmath=True)
def clip(x, lo, hi):
    return min(max(x, lo), hi)

@njit(fastmath=True)
def rsi_wilder_numba(prices: np.ndarray, window: int) -> np.ndarray:
    n = len(prices)
    rsi = np.full(n, np.nan, dtype=np.float64)
    if n < window + 1:
        return rsi

    # Diff
    delta = np.diff(prices)
    
    # Init first window
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(window):
        val = delta[i]
        if val > 0:
            avg_gain += val
        else:
            avg_loss += -val
            
    avg_gain /= window
    avg_loss /= window
    
    # Calculate first RSI
    if avg_loss == 0:
        rsi[window] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[window] = 100.0 - (100.0 / (1.0 + rs))
        
    # Smoothing
    for i in range(window + 1, n):
        change = delta[i - 1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        
        avg_gain = (avg_gain * (window - 1) + gain) / window
        avg_loss = (avg_loss * (window - 1) + loss) / window
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            
    return rsi

@njit(fastmath=True)
def garch11_negloglik_numba(params, r):
    omega, alpha, beta = params
    # Bounds check inside JIT to avoid python overhead
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
        return 1e12

    T = len(r)
    h = np.empty(T, dtype=np.float64)
    
    # Init with variance
    # Numba doesn't like np.var on array slice sometimes, manual calc safe
    mean_r = 0.0
    for val in r: mean_r += val
    mean_r /= T
    var_r = 0.0
    for val in r: var_r += (val - mean_r)**2
    var_r /= T
    
    h[0] = var_r + 1e-9
    
    log_lik = 0.0
    
    for t in range(1, T):
        h[t] = omega + alpha * (r[t-1] ** 2) + beta * h[t-1]
        if h[t] <= 0:
            return 1e12
        log_lik += -0.5 * (np.log(2*np.pi) + np.log(h[t]) + (r[t]**2)/h[t])
        
    return -log_lik

# Wrapper per scipy minimize che chiama la funzione JIT
def fit_garch11_fast(r: np.ndarray) -> Tuple[float, float, float]:
    r = r - np.mean(r)
    v = np.var(r) + GARCH_EPS
    x0 = np.array([0.1 * v, 0.05, 0.9])
    bounds = [(1e-12, None), (0.0, 1.0), (0.0, 1.0)]
    
    # Passiamo la funzione compilata a minimize
    res = minimize(
        garch11_negloglik_numba, x0, args=(r,),
        method="L-BFGS-B", bounds=bounds,
        tol=1e-4, # Rilassiamo leggermente la tolleranza per velocità
        options={"maxiter": 100}
    )
    
    if not res.success:
        return (0.05 * v, 0.05, 0.9)
        
    omega, alpha, beta = res.x
    if alpha + beta >= 0.999:
        beta = min(beta, 0.94)
        alpha = min(alpha, 0.05)
    return float(omega), float(alpha), float(beta)

def garch_filtered_sigma_optimized(logret: np.ndarray, window: int, refit_every: int) -> np.ndarray:
    n = len(logret)
    sigma = np.full(n, np.nan, dtype=float)
    if n < window + 2: return sigma

    omega, alpha, beta = 0.0, 0.0, 0.0
    h_prev = 0.0

    # Main loop in python because fit_garch uses scipy (cannot JIT), 
    # but the inner likelihood is JIT-ed.
    for t in range(window, n):
        if (t == window) or ((t - window) % refit_every == 0):
            r_win = logret[t-window:t]
            omega, alpha, beta = fit_garch11_fast(r_win)
            h_prev = np.var(r_win) + GARCH_EPS

        if t == 0:
            h_t = np.var(logret[:window]) + GARCH_EPS
        else:
            h_t = omega + alpha * (logret[t-1] ** 2) + beta * h_prev
            h_t = max(h_t, GARCH_EPS)

        sigma[t] = np.sqrt(h_t)
        h_prev = h_t
    return sigma

# =========================
# Backtest Logic (Numba)
# =========================

@njit(float64[:](float64[:], int64, int64, int64))
def block_bootstrap_mean_numba(x, n_boot, block, seed):
    np.random.seed(seed)
    # Filtra NaNs
    valid_mask = np.isfinite(x)
    clean_x = x[valid_mask]
    n = len(clean_x)
    out = np.full(n_boot, np.nan)
    
    if n < 5:
        return out

    n_blocks = int(np.ceil(n / block))
    
    for b in range(n_boot):
        sum_val = 0.0
        count = 0
        for _ in range(n_blocks):
            start = np.random.randint(0, max(1, n - block + 1))
            # manual loop for slicing speed
            end = min(n, start + block)
            for i in range(start, end):
                sum_val += clean_x[i]
                count += 1
        if count > 0:
            out[b] = sum_val / count
    return out

@njit(float64(float64[:]))
def calc_max_drawdown_numba(equity):
    peak = -1e99
    max_dd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = (peak - x) / peak if peak != 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd

# Return tuple: (cum_return, vol_ann, sharpe, mdd, equity, logret)
@njit
def backtest_core(prices, k, L, tau, use_rsi, rsi, use_vol, sigma, is_sls):
    n = len(prices)
    V = np.zeros(n)
    x = np.zeros(n)
    costs = np.zeros(n)
    
    V[0] = V0
    x[0] = X0
    
    for t in range(1, n):
        Rt = (prices[t] - prices[t-1]) / prices[t-1]
        
        # Determine Target x[t]
        if t == 1:
            x[t] = X0 if is_sls else V[t-1] # SLS starts 0, BH starts invested
        else:
            if not is_sls:
                # BH Logic
                x[t] = x[t-1] # Hold
            else:
                # SLS Logic
                delta_tm1 = (prices[t-1] - prices[t-2]) / prices[t-2]
                V_tm2 = V[t-2]
                
                k_eff = k
                if use_vol:
                    sig = sigma[t-1]
                    if not np.isnan(sig) and sig > 0:
                        k_eff = k / sig
                
                m = 1.0
                if use_rsi:
                    val = rsi[t-1]
                    # RSI check
                    if not np.isnan(val) and (val < RSI_LO or val > RSI_HI):
                        m = 0.0
                
                x_tilde = x[t-1] + (m * k_eff) * delta_tm1 * V_tm2
                bound = L * V[t-1]
                
                # Clip
                if x_tilde > bound: x_tilde = bound
                if x_tilde < -bound: x_tilde = -bound
                x[t] = x_tilde
                
        # Cost
        costs[t] = tau * abs(x[t] - x[t-1])
        
        # Update Wealth
        V[t] = V[t-1] + x[t-1] * Rt - costs[t]
        
        if V[t] <= 1e-9:
            V[t] = 1e-9
            
    # Metrics calculation inside JIT to save memory/passes
    logret = np.empty(n-1)
    for i in range(n-1):
        logret[i] = np.log(V[i+1] / V[i])
        
    cum_ret = (V[n-1] - V[0]) / V[0]
    
    # Std & Mean
    lr_sum = 0.0
    lr_sq_sum = 0.0
    cnt = 0
    for val in logret:
        if np.isfinite(val):
            lr_sum += val
            lr_sq_sum += val*val
            cnt += 1
            
    if cnt > 1:
        lr_mean = lr_sum / cnt
        lr_var = (lr_sq_sum / cnt) - (lr_mean * lr_mean)
        lr_std = np.sqrt(lr_var) if lr_var > 0 else 0.0
    else:
        lr_mean = 0.0
        lr_std = 0.0
        
    vol_ann = lr_std * np.sqrt(252.0)
    sharpe = (lr_mean / (lr_std + 1e-12)) * np.sqrt(252.0)
    
    peak = -1e99
    mdd = 0.0
    for val in V:
        if val > peak: peak = val
        dd = (peak - val) / peak if peak != 0 else 0.0
        if dd > mdd: mdd = dd
        
    return cum_ret, vol_ann, sharpe, mdd, logret

# =========================
# Worker Function
# =========================
def process_ticker(tic, df_tic):
    P = df_tic[PRICE_COL].astype(float).to_numpy()
    if len(P) < 400:
        return []

    logret_raw = np.log(P[1:] / P[:-1])
    rsi = rsi_wilder_numba(P, RSI_WINDOW)
    
    # GARCH fitting is the slowest part remaining, but optimized
    sigma = garch_filtered_sigma_optimized(logret_raw, GARCH_WINDOW, GARCH_REFIT_EVERY)
    sigma_pad = np.full(len(P), np.nan)
    sigma_pad[1:] = sigma

    local_rows = []

    for tau in tau_grid:
        # BH
        # Backtest core signature: prices, k, L, tau, use_rsi, rsi, use_vol, sigma, is_sls
        bh_metrics = backtest_core(P, 0.0, 0.0, tau, False, rsi, False, sigma_pad, False)
        bh_logret = bh_metrics[4] # 5th element
        
        # Bootstrap BH
        # Numba bootstrap returns array of means
        # bh_boot = block_bootstrap_mean_numba(bh_logret, BOOT_N, BOOT_BLOCK, RNG_SEED)

        for L in L_grid:
            for k in k_grid:
                # SLS Pure
                sls_m = backtest_core(P, float(k), float(L), tau, False, rsi, False, sigma_pad, True)
                
                # SLS RSI
                sls_rsi_m = backtest_core(P, float(k), float(L), tau, True, rsi, False, sigma_pad, True)
                
                # SLS Vol
                sls_vol_m = backtest_core(P, float(k), float(L), tau, False, rsi, True, sigma_pad, True)

                # Bootstrap Excess Returns (SLS Pure vs BH)
                ex_lr = sls_m[4] - bh_logret
                ex_boot = block_bootstrap_mean_numba(ex_lr, BOOT_N, BOOT_BLOCK, RNG_SEED)
                ex_mu = np.nanmean(ex_lr)
                ci_lo, ci_hi = np.nanquantile(ex_boot, [0.025, 0.975])

                local_rows.append({
                    "ticker": tic, "tau": tau, "L": L, "k": k,
                    
                    "BH_cum_return": bh_metrics[0], "BH_vol_ann": bh_metrics[1], 
                    "BH_mdd": bh_metrics[3], "BH_sharpe": bh_metrics[2],

                    "SLS_cum_return": sls_m[0], "SLS_vol_ann": sls_m[1], 
                    "SLS_mdd": sls_m[3], "SLS_sharpe": sls_m[2],
                    "SLS_alpha": sls_m[0] - bh_metrics[0],

                    "SLS_RSI_cum_return": sls_rsi_m[0], "SLS_RSI_vol_ann": sls_rsi_m[1], 
                    "SLS_RSI_mdd": sls_rsi_m[3], "SLS_RSI_sharpe": sls_rsi_m[2],
                    "SLS_RSI_alpha": sls_rsi_m[0] - bh_metrics[0],

                    "SLS_VOL_cum_return": sls_vol_m[0], "SLS_VOL_vol_ann": sls_vol_m[1], 
                    "SLS_VOL_mdd": sls_vol_m[3], "SLS_VOL_sharpe": sls_vol_m[2],
                    "SLS_VOL_alpha": sls_vol_m[0] - bh_metrics[0],

                    "excess_mu_logret": ex_mu, "excess_ci_lo": ci_lo, "excess_ci_hi": ci_hi,
                })
    return local_rows

# =========================
# Main
# =========================
def run():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([TICKER_COL, DATE_COL])
    tickers = df[TICKER_COL].unique().tolist()
    
    print(f"Starting parallel processing on {len(tickers)} tickers...")
    
    # Prepare data chunks for parallelism
    tasks = []
    for tic in tickers:
        tasks.append((tic, df[df[TICKER_COL] == tic].copy()))

    # Run Parallel
    # n_jobs=-1 uses all cores.
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_ticker)(tic, data) for tic, data in tasks
    )

    # Flatten results
    rows = [item for sublist in results for item in sublist]
    
    print("Processing results...")
    out = pd.DataFrame(rows)

    # Aggregation
    grp = out.groupby(["tau", "L", "k"], as_index=False).mean(numeric_only=True)

    out_path_asset = "results_by_asset_fast.csv"
    out_path_agg = "results_aggregated_fast.csv"
    out.to_csv(out_path_asset, index=False)
    grp.to_csv(out_path_agg, index=False)

    print(f"\nDone! Saved:\n- {out_path_asset}\n- {out_path_agg}")

if __name__ == "__main__":
    run()