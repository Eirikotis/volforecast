"""
Volatility Forecasting Engine
Refactored from original Colab notebook to production-ready module.
"""

import os
import json
import time
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ccxt
from arch import arch_model

# ==== Constants ====
TF_MS = {"15m": 15*60*1000, "1h": 60*60*1000, "4h": 4*60*60*1000, "1d": 24*60*60*1000}
TF_FREQ = {"15m": "15T", "1h": "1H", "4h": "4H", "1d": "1D"}
PERIODS_PER_DAY = {"15m": 96, "1h": 24, "4h": 6, "1d": 1}

# VPS-safe caps
MAX_BARS_BY_TF = {"15m": 1344, "1h": 2160, "4h": 2190, "1d": 1095}
MAX_SIMS = 2500
MAX_H = 100
MAX_PQ_SUM = 3
MIN_SLEEP_MS = 200


# ==== Helper Functions ====
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def now_utc():
    return datetime.now(timezone.utc)


def ms_since_epoch(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fresh_enough(last_ts_ms: int, tf_ms: int) -> bool:
    """Cache considered fresh if last bar is within one TF of current time."""
    return (ms_since_epoch(now_utc()) - last_ts_ms) <= tf_ms


def backoff_sleep(attempt: int, base_ms: int = 300, max_ms: int = 5000):
    """Exponential backoff with jitter."""
    wait = min(max_ms, base_ms * (2 ** attempt))
    wait = int(wait * (0.75 + 0.5 * random.random()))
    time.sleep(wait / 1000)


# ==== Cache Management ====
def get_cache_dirs(base_dir: str = None):
    """Initialize cache directories."""
    if base_dir is None:
        base_dir = os.getcwd()
    cache_dir = os.path.join(base_dir, "cache")
    ohlcv_dir = os.path.join(cache_dir, "ohlcv")
    fit_dir = os.path.join(cache_dir, "fit")
    os.makedirs(ohlcv_dir, exist_ok=True)
    os.makedirs(fit_dir, exist_ok=True)
    return cache_dir, ohlcv_dir, fit_dir


def ohlcv_cache_path(symbol: str, tf: str, cache_dir: str) -> str:
    safe = symbol.replace("/", "_")
    return os.path.join(cache_dir, "ohlcv", f"{safe}_{tf}.parquet")


def save_ohlcv_parquet(df: pd.DataFrame, path: str):
    df.to_parquet(path, index=False)


def load_ohlcv_parquet(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


# ==== Symbol Listing ====
def list_spot_symbols(exchange_id: str,
                      quote: str | None = "USDT",
                      cache_hours: int = 24,
                      cache_dir: str = None,
                      search_text: str | None = None,
                      strict_quote: bool = True) -> list[tuple[str, str]]:
    """Fetch and cache spot symbols with flexible filtering.
    
    Returns list of (symbol, exchange_id) tuples.
    - spot == True, active != False (None is OK)
    - Quote filter: if strict_quote, only provided quote; else ["USDT","USDC","USD","BTC","ETH"] for Auto
    - Base filter: only if search_text provided
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_key_quote = (quote or "AUTO").upper()
    cache_file = os.path.join(cache_dir, f"symbols_{exchange_id}_spot_{cache_key_quote}.json")

    # try cache (with recovery if < 20 symbols)
    cached_syms = None
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            ts = data.get("_cached_at", 0)
            cached_list = data.get("symbols", [])
            # convert old format (list[str]) to new format (list[tuple])
            if cached_list and isinstance(cached_list[0], str):
                cached_list = [(s, exchange_id) for s in cached_list]
            if (time.time() - ts) <= cache_hours * 3600 and len(cached_list) >= 20:
                cached_syms = cached_list
        except Exception:
            pass

    # fetch fresh
    try:
        ex_cls = getattr(ccxt, exchange_id)
        ex = ex_cls({"enableRateLimit": True})
        markets = ex.load_markets()
    except Exception:
        # fallback to cache if fetch fails
        if cached_syms:
            return cached_syms
        return []

    # determine allowed quotes
    if strict_quote and quote:
        allowed_quotes = [quote.upper()]
    else:
        allowed_quotes = ["USDT","USDC","USD","BTC","ETH"]
    
    base_search = (search_text or "").upper().strip()

    syms = []
    tao_found = None
    tao_alt_quote = None
    
    for m in markets.values():
        if not m.get("spot", False):
            continue
        if m.get("active") is False:  # exclude only explicitly False
            continue
        q = (m.get("quote") or "").upper()
        b = (m.get("base") or "").upper()
        
        if q not in allowed_quotes:
            # track TAO for later inclusion
            if b == "TAO":
                if not tao_found:
                    tao_found = m.get("symbol")
                    tao_alt_quote = q
            continue
        
        if base_search and (base_search not in b):
            continue
        
        if not ex.has.get("fetchOHLCV", False):
            continue
        
        syms.append((m["symbol"], exchange_id))
        
        # mark TAO if found in allowed quote
        if b == "TAO" and not tao_found:
            tao_found = m["symbol"]
            tao_alt_quote = None

    # Force-include TAO if found under any allowed quote (even if active=None)
    if tao_found and "TAO" not in "|".join([s[0].split("/")[0] for s in syms]):
        # try to find TAO under allowed quote even if active=None
        for m in markets.values():
            if m.get("spot", False):
                b = (m.get("base") or "").upper()
                q = (m.get("quote") or "").upper()
                if b == "TAO" and q in allowed_quotes:
                    syms.insert(0, (m.get("symbol"), exchange_id))  # at top
                    break

    # Sort alphabetically by symbol (but keep TAO at top if it was inserted)
    if syms and syms[0][0].split("/")[0].upper() == "TAO":
        # TAO is already at top, sort the rest
        rest = syms[1:]
        rest.sort(key=lambda x: x[0].upper())
        syms = [syms[0]] + rest
    else:
        syms.sort(key=lambda x: x[0].upper())

    # save cache
    try:
        with open(cache_file, "w") as f:
            json.dump({"_cached_at": time.time(), "symbols": syms}, f)
    except Exception:
        pass

    return syms


def union_symbols_across_exchanges(exchange_ids: list[str],
                                   quote: str | None = None,
                                   strict_quote: bool = False,
                                   cache_dir: str = None) -> tuple[list[tuple[str, str]], set[str]]:
    """Union symbols across multiple exchanges, deduplicating by normalized symbol.
    
    Returns (list of (symbol, exchange_id) tuples, set of exchange_ids seen).
    First exchange that provides a symbol gets tagged.
    """
    seen_symbols = set()
    result = []
    sources = set()
    
    for ex_id in exchange_ids:
        try:
            syms = list_spot_symbols(ex_id, quote=quote, strict_quote=strict_quote, cache_dir=cache_dir)
            for sym, ex in syms:
                if sym not in seen_symbols:
                    seen_symbols.add(sym)
                    result.append((sym, ex))
                    sources.add(ex)
        except Exception:
            continue
    
    # Sort alphabetically by symbol
    result.sort(key=lambda x: x[0].upper())
    
    # Force TAO to top if present (after sorting)
    tao_idx = None
    for i, (sym, ex) in enumerate(result):
        if sym.split("/")[0].upper() == "TAO":
            tao_idx = i
            break
    if tao_idx is not None and tao_idx > 0:
        tao_item = result.pop(tao_idx)
        result.insert(0, tao_item)
    
    return result, sources


# ==== Data Loading ====
def load_ohlcv_ccxt(exchange_id: str, symbol: str, timeframe: str, lookback_bars: int,
                    use_cache: bool = True, cache_dir: str = None,
                    warnings: list = None, max_attempts: int = 3) -> tuple[pd.DataFrame, bool, bool]:
    """Load OHLCV from CCXT with caching. Returns (df_raw, stale_flag, used_cache_flag)."""
    if warnings is None:
        warnings = []
        
    if timeframe not in TF_MS:
        raise ValueError(f"timeframe must be one of {list(TF_MS.keys())}")
    
    tf_ms = TF_MS[timeframe]
    stale = False
    
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "cache")
    
    # separate cache per exchange
    cache_dir_ex = os.path.join(cache_dir, exchange_id)
    os.makedirs(cache_dir_ex, exist_ok=True)
    cache_path = ohlcv_cache_path(symbol, timeframe, cache_dir_ex)
    df_cache = load_ohlcv_parquet(cache_path) if use_cache else None

    now_ms = ms_since_epoch(now_utc())
    since_ms = now_ms - lookback_bars * tf_ms

    ex_cls = getattr(ccxt, exchange_id)
    ex = ex_cls({"enableRateLimit": True})
    rate_ms = int(getattr(ex, "rateLimit", 500))
    rate_ms = max(rate_ms, MIN_SLEEP_MS)

    rows = []
    fetch_since = since_ms
    limit = 100

    if df_cache is not None and len(df_cache) > 0:
        try:
            last_cached_ts = int(pd.to_datetime(df_cache["timestamp"]).astype("int64").max() // 1_000_000)
        except Exception:
            last_cached_ts = int(pd.to_datetime(df_cache["timestamp"]).view("int64").max() // 1_000_000)
        fetch_since = max(fetch_since, last_cached_ts + tf_ms)

    try:
        attempts = 0
        while True:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
            if not ohlcv:
                break
            rows.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            fetch_since = last_ts + tf_ms
            if fetch_since >= (now_ms - tf_ms):
                break
            time.sleep(rate_ms / 1000)
            
        if rows:
            df_new = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
            if df_cache is not None and len(df_cache) > 0:
                df_cache = df_cache[["timestamp","open","high","low","close","volume"]].copy()
                df_all = pd.concat([df_cache, df_new], ignore_index=True)
            else:
                df_all = df_new
        else:
            if df_cache is not None and len(df_cache) > 0:
                df_all = df_cache.copy()
            else:
                while attempts < max_attempts and not rows:
                    attempts += 1
                    backoff_sleep(attempts)
                    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
                    if ohlcv:
                        rows.extend(ohlcv)
                if not rows:
                    raise RuntimeError("OKX fetchOHLCV returned no data.")
                df_all = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])

        df_all = df_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df_all = df_all[df_all["timestamp"] >= since_ms]
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], unit="ms", utc=True)

        used_cache_flag = False
        if use_cache:
            try:
                save_ohlcv_parquet(df_all, cache_path)
            except Exception as e:
                warnings.append(f"Failed to write OHLCV cache: {e}")

        last_ts_ms = int(df_all["timestamp"].astype("int64").max() // 1_000_000)
        if not fresh_enough(last_ts_ms, tf_ms):
            stale = True

        return df_all.reset_index(drop=True), stale, False

    except Exception as e:
        if df_cache is not None and len(df_cache) > 0:
            stale = True
            df_cache = df_cache.drop_duplicates("timestamp").sort_values("timestamp")
            if not np.issubdtype(df_cache["timestamp"].dtype, np.datetime64):
                df_cache["timestamp"] = pd.to_datetime(df_cache["timestamp"], utc=True)
            df_cache = df_cache[df_cache["timestamp"].astype("int64") // 1_000_000 >= since_ms]
            warnings.append(f"Served cached OHLCV due to live fetch error: {e}")
            return df_cache.reset_index(drop=True), stale, True
        raise RuntimeError(f"Failed to fetch OHLCV and no cache available: {e}")


# ==== Data Preprocessing ====
def regularize_grid(df_raw: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df_raw.empty:
        raise RuntimeError("No OHLCV rows loaded.")
    df = df_raw.copy()
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]
    full = pd.date_range(start, end, freq=TF_FREQ[timeframe], tz="UTC")
    df = df.set_index("timestamp").reindex(full)

    df["close"] = df["close"].ffill()
    df["open"] = df["open"].fillna(df["close"].shift(1)).ffill()
    mx = df[["open","close"]].max(axis=1)
    mn = df[["open","close"]].min(axis=1)
    df["high"] = df["high"].fillna(mx)
    df["low"] = df["low"].fillna(mn)
    df["volume"] = df["volume"].fillna(0.0)

    df = df.reset_index().rename(columns={"index":"timestamp"})
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df


def compute_returns_and_rv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_ret"] = 100.0 * np.log(out["close"] / out["close"].shift(1))
    parkinson = (1.0/(4*np.log(2))) * (np.log(out["high"]/out["low"])**2)
    gk = 0.5*(np.log(out["high"]/out["low"])**2) - (2*np.log(2)-1)*(np.log(out["close"]/out["open"])**2)
    rs = (np.log(out["high"]/out["close"])*np.log(out["high"]/out["open"])
          + np.log(out["low"]/out["close"])*np.log(out["low"]/out["open"]))
    out["rv_parkinson"] = np.sqrt(parkinson).clip(lower=0) * 100.0
    out["rv_gk"] = np.sqrt(np.clip(gk, 0, None)) * 100.0
    out["rv_rs"] = np.sqrt(np.clip(rs, 0, None)) * 100.0
    out = out.dropna(subset=["log_ret"]).reset_index(drop=True)
    return out


def assert_sane(df: pd.DataFrame, timeframe: str):
    if df[["open","high","low","close"]].isna().any().any():
        raise RuntimeError("NaNs present after fill — check data.")
    n = len(df)
    if n < 300:
        raise RuntimeError(f"Too few bars after prep: {n} < 300.")
    std_lr = df["log_ret"].std()
    if not np.isfinite(std_lr) or std_lr > 15.0:
        raise RuntimeError(f"Unreasonable log-return std ({std_lr:.2f} %/bar) — reduce lookback or change TF.")


# ==== Model Fitting ====
def ewma_vol_percent(log_ret_pct: pd.Series, lam: float = 0.94) -> np.ndarray:
    r = (log_ret_pct.values / 100.0)
    var = np.empty_like(r)
    v = np.var(r[:min(60, len(r))]) if len(r) >= 60 else np.var(r)
    for i, x in enumerate(r):
        v = lam * v + (1 - lam) * (x * x)
        var[i] = v
    vol_pct = np.sqrt(var) * 100.0
    return vol_pct


def fit_arch_family(df: pd.DataFrame, model: str = "egarch", dist: str = "t",
                    p: int = 1, q: int = 1, mean_spec: str = "constant",
                    warnings: list = None):
    if warnings is None:
        warnings = []
        
    y = df["log_ret"].astype("float64")
    diagnostics = {}
    params = {}

    if model.lower() in ("egarch","garch"):
        vol_tag = "EGARCH" if model.lower() == "egarch" else "Garch"
        dist_tag = "t" if dist.lower() == "t" else "normal"
        mean_tag = {"constant":"Constant","zero":"Zero","ar1":"AR"}[mean_spec.lower()]
        try:
            am = arch_model(y, vol=vol_tag, p=p, q=q, mean=mean_tag, dist=dist_tag)
            res = am.fit(disp="off")
            cond_vol = np.asarray(res.conditional_volatility)
            params = {k: float(v) for k, v in res.params.to_dict().items()}
            diagnostics = {
                "aic": float(res.aic),
                "bic": float(res.bic),
                "loglik": float(res.loglikelihood),
                "converged": bool(getattr(res, "convergence_flag", 0) == 0),
                "iters": int(getattr(res, "iterations", 0)),
            }
            fit_name = f"{model.upper()}({p},{q})_{dist_tag}"
            return res, fit_name, params, cond_vol, diagnostics
        except Exception as e:
            warnings.append(f"{model.upper()} fit failed ({e}); trying fallback.")

    if model.lower() != "garch" or dist.lower() != "normal" or p != 1 or q != 1:
        try:
            am = arch_model(y, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
            res = am.fit(disp="off")
            cond_vol = np.asarray(res.conditional_volatility)
            params = {k: float(v) for k, v in res.params.to_dict().items()}
            diagnostics = {
                "aic": float(res.aic),
                "bic": float(res.bic),
                "loglik": float(res.loglikelihood),
                "converged": bool(getattr(res, "convergence_flag", 0) == 0),
                "iters": int(getattr(res, "iterations", 0)),
            }
            warnings.append("Fell back to GARCH(1,1) normal.")
            return res, "GARCH(1,1)_normal", params, cond_vol, diagnostics
        except Exception as e:
            warnings.append(f"GARCH fallback failed ({e}); using EWMA.")

    vol_pct = ewma_vol_percent(df["log_ret"])
    params = {"lambda": 0.94}
    diagnostics = {"aic": np.nan, "bic": np.nan, "loglik": np.nan, "converged": True, "iters": 0}
    return None, "EWMA(0.94)", params, vol_pct, diagnostics


# ==== Forecasting ====
def forecast_vol_steps(res, fit_name: str, hist_vol: np.ndarray, horizon: int,
                       dist: str = "t", seed: int = 42) -> np.ndarray:
    if res is None:
        last_vol = hist_vol[-1]
        return np.full(horizon, float(last_vol), dtype="float64")

    if fit_name.startswith("EGARCH") and dist.lower() == "t":
        fc = res.forecast(horizon=horizon, method="simulation", simulations=2000,
                          random_state=seed, reindex=False)
    else:
        fc = res.forecast(horizon=horizon, reindex=False)

    vol_fc = np.sqrt(fc.variance.values[-1])
    return np.asarray(vol_fc, dtype="float64")


# ==== Simulation ====
def var_es(arr, alpha=0.95):
    q = np.quantile(arr, 1 - alpha)
    es = arr[arr <= q].mean() if np.any(arr <= q) else q
    return float(q), float(es)


def run_pipeline(params: dict) -> dict:
    """
    Main pipeline function. Takes parameters and returns structured payload.
    
    params dict keys:
        symbol, timeframe, horizon, sims, model, dist, p, q, mean_spec,
        lookback_bars, seed, zcap, vol_scale, drift_mode, drift_user_pct,
        use_cache, base_dir (optional)
    """
    # Extract and validate params
    symbol = params.get("symbol", "BTC/USDT")
    timeframe = params.get("timeframe", "4h")
    horizon = int(params.get("horizon", 18))
    sims = int(params.get("sims", 1500))
    model = params.get("model", "egarch")
    dist = params.get("dist", "t")
    p_order = int(params.get("p", 1))
    q_order = int(params.get("q", 1))
    mean_spec = params.get("mean_spec", "constant")
    lookback_bars = int(params.get("lookback_bars", 2000))
    seed = int(params.get("seed", 42))
    zcap = float(params.get("zcap", 6.0))
    vol_scale = float(params.get("vol_scale", 1.0))
    drift_mode = params.get("drift_mode", "neutral")
    drift_user_pct = float(params.get("drift_user_pct", 0.0))
    use_cache = params.get("use_cache", True)
    base_dir = params.get("base_dir", os.getcwd())
    data_source_requested = (params.get("data_source", "auto") or "auto").lower()
    failover = bool(params.get("failover", True))
    quote_filter = params.get("quote", "USDT") or "USDT"
    
    warnings = []
    
    # Validate timeframe
    if timeframe not in TF_MS:
        raise ValueError(f"timeframe must be one of {list(TF_MS.keys())}")
    
    # Clamp parameters
    cap_bars = MAX_BARS_BY_TF[timeframe]
    if lookback_bars > cap_bars:
        warnings.append(f"LOOKBACK_BARS clamped from {lookback_bars} to {cap_bars} for {timeframe}.")
    lookback_bars = clamp(int(lookback_bars), 100, cap_bars)
    
    if horizon > MAX_H:
        warnings.append(f"H clamped from {horizon} to {MAX_H}.")
    horizon = clamp(int(horizon), 1, MAX_H)
    
    if sims > MAX_SIMS:
        warnings.append(f"SIMS clamped from {sims} to {MAX_SIMS}.")
    sims = clamp(int(sims), 100, MAX_SIMS)
    
    if (p_order + q_order) > MAX_PQ_SUM:
        warnings.append(f"(p+q) clamped from {p_order+q_order} to {MAX_PQ_SUM}. Using (1,1).")
        p_order, q_order = 1, 1
    
    # Initialize cache
    cache_dir, ohlcv_dir, fit_dir = get_cache_dirs(base_dir)
    
    # Load OHLCV with multi-source fallback
    exchanges_priority = ["okx","binance","bybit","kucoin","kraken"]
    if data_source_requested != "auto":
        exchanges_to_try = [data_source_requested]
        if failover:
            exchanges_to_try += [e for e in exchanges_priority if e != data_source_requested]
    else:
        exchanges_to_try = exchanges_priority

    df_raw = None
    stale_data = False
    used_cache_flag = False
    data_source_used = None
    used_fallback = False
    last_err = None
    for i, ex_id in enumerate(exchanges_to_try):
        try:
            df_raw, stale_data, used_cache_flag = load_ohlcv_ccxt(
                ex_id, symbol, timeframe, lookback_bars,
                use_cache=use_cache, cache_dir=cache_dir, warnings=warnings
            )
            data_source_used = ex_id
            used_fallback = (i > 0)
            break
        except Exception as e:
            last_err = e
            if not failover:
                break
            continue
    if df_raw is None:
        raise RuntimeError(f"Failed to load OHLCV from all sources. Last error: {last_err}")
    
    # Preprocess
    df = regularize_grid(df_raw, timeframe)
    df = compute_returns_and_rv(df)
    assert_sane(df, timeframe)
    
    # Fit model
    res, fit_name, fit_params, hist_vol, diag = fit_arch_family(
        df, model=model, dist=dist, p=p_order, q=q_order,
        mean_spec=mean_spec, warnings=warnings
    )
    
    t_hist = df["timestamp"].iloc[-len(hist_vol):]
    
    # Forecast
    vol_fc = forecast_vol_steps(res, fit_name, hist_vol, horizon, dist=dist, seed=seed)
    
    if vol_scale != 1.0:
        warnings.append(f"VOL_SCALE applied: ×{vol_scale:.2f}")
    vol_fc = np.clip(vol_fc * float(vol_scale), 0.0, None)
    
    per_day = PERIODS_PER_DAY[timeframe]
    vol_fc_daily = vol_fc * np.sqrt(per_day)
    vol_fc_annual = vol_fc_daily * np.sqrt(365)
    
    last_bar = df["timestamp"].iloc[-1]
    t_fc = pd.date_range(
        last_bar + pd.Timedelta(milliseconds=TF_MS[timeframe]),
        periods=horizon, freq=TF_FREQ[timeframe], tz="UTC"
    )
    
    # Simulation
    rng = np.random.default_rng(seed)
    P0 = float(df["close"].iloc[-1])
    
    if drift_mode == "from_mu" and fit_params and ("mu" in fit_params):
        mu_pct = float(fit_params["mu"])
    elif drift_mode == "user":
        mu_pct = float(drift_user_pct)
    else:
        mu_pct = 0.0
    
    nu = float(fit_params.get("nu", np.nan)) if fit_params else np.nan
    use_t = (not np.isnan(nu)) and (nu > 2.0) and ("EGARCH" in fit_name or "GARCH" in fit_name) and dist.lower()=="t"
    
    if use_t:
        z = rng.standard_t(df=nu, size=(sims, horizon)) / np.sqrt(nu/(nu-2.0))
    else:
        z = rng.standard_normal(size=(sims, horizon))
    z = np.clip(z, -abs(zcap), abs(zcap))
    
    mu_dec = mu_pct / 100.0
    sig_dec = (vol_fc / 100.0).astype("float64")
    
    drift = mu_dec - 0.5 * (sig_dec ** 2)
    steps = drift + sig_dec * z
    
    log_paths = np.empty((sims, horizon+1), dtype="float64")
    log_paths[:, 0] = np.log(P0)
    log_paths[:, 1:] = log_paths[:, [0]] + np.cumsum(steps, axis=1)
    price_paths = np.exp(log_paths)
    
    p10 = np.percentile(price_paths[:, 1:], 10, axis=0)
    p50 = np.percentile(price_paths[:, 1:], 50, axis=0)
    p90 = np.percentile(price_paths[:, 1:], 90, axis=0)
    p05 = np.percentile(price_paths[:, 1:], 5, axis=0)
    p95 = np.percentile(price_paths[:, 1:], 95, axis=0)
    
    r1 = (price_paths[:, 1] / price_paths[:, 0]) - 1.0
    rH = (price_paths[:, -1] / price_paths[:, 0]) - 1.0
    
    # Downside (existing)
    risk = {
        "VaR95_1": var_es(r1, 0.95)[0], "ES95_1": var_es(r1, 0.95)[1],
        "VaR99_1": var_es(r1, 0.99)[0], "ES99_1": var_es(r1, 0.99)[1],
        "VaR95_H": var_es(rH, 0.95)[0], "ES95_H": var_es(rH, 0.95)[1],
        "VaR99_H": var_es(rH, 0.99)[0], "ES99_H": var_es(rH, 0.99)[1],
    }
    
    risk["CVaR95_1"] = risk["ES95_1"]
    risk["CVaR99_1"] = risk["ES99_1"]
    risk["CVaR95_H"] = risk["ES95_H"]
    risk["CVaR99_H"] = risk["ES99_H"]
    
    # Upside (mirror on positive tail)
    def var_es_up(arr, alpha=0.95):
        q = np.quantile(arr, alpha)
        es = arr[arr >= q].mean() if np.any(arr >= q) else q
        return float(q), float(es)

    VaR95_1_up, ES95_1_up = var_es_up(r1, 0.95)
    VaR99_1_up, ES99_1_up = var_es_up(r1, 0.99)
    VaR95_H_up, ES95_H_up = var_es_up(rH, 0.95)
    VaR99_H_up, ES99_H_up = var_es_up(rH, 0.99)

    risk_pct = {
        "VaR95_1_pct": 100.0 * risk["VaR95_1"],
        "ES95_1_pct": 100.0 * risk["ES95_1"],
        "VaR99_1_pct": 100.0 * risk["VaR99_1"],
        "ES99_1_pct": 100.0 * risk["ES99_1"],
        "VaR95_H_pct": 100.0 * risk["VaR95_H"],
        "ES95_H_pct": 100.0 * risk["ES95_H"],
        "VaR99_H_pct": 100.0 * risk["VaR99_H"],
        "ES99_H_pct": 100.0 * risk["ES99_H"],
        "VaR95_1_up_pct": 100.0 * VaR95_1_up,
        "ES95_1_up_pct": 100.0 * ES95_1_up,
        "VaR99_1_up_pct": 100.0 * VaR99_1_up,
        "ES99_1_up_pct": 100.0 * ES99_1_up,
        "VaR95_H_up_pct": 100.0 * VaR95_H_up,
        "ES95_H_up_pct": 100.0 * ES95_H_up,
        "VaR99_H_up_pct": 100.0 * VaR99_H_up,
        "ES99_H_up_pct": 100.0 * ES99_H_up,
    }
    
    risk_price = {
        "VaR95_1_price": P0 * (1.0 + risk["VaR95_1"]),
        "ES95_1_price": P0 * (1.0 + risk["ES95_1"]),
        "VaR99_1_price": P0 * (1.0 + risk["VaR99_1"]),
        "ES99_1_price": P0 * (1.0 + risk["ES99_1"]),
        "VaR95_H_price": P0 * (1.0 + risk["VaR95_H"]),
        "ES95_H_price": P0 * (1.0 + risk["ES95_H"]),
        "VaR99_H_price": P0 * (1.0 + risk["VaR99_H"]),
        "ES99_H_price": P0 * (1.0 + risk["ES99_H"]),
        "VaR95_1_up_price": P0 * (1.0 + VaR95_1_up),
        "ES95_1_up_price": P0 * (1.0 + ES95_1_up),
        "VaR99_1_up_price": P0 * (1.0 + VaR99_1_up),
        "ES99_1_up_price": P0 * (1.0 + ES99_1_up),
        "VaR95_H_up_price": P0 * (1.0 + VaR95_H_up),
        "ES95_H_up_price": P0 * (1.0 + ES95_H_up),
        "VaR99_H_up_price": P0 * (1.0 + VaR99_H_up),
        "ES99_H_up_price": P0 * (1.0 + ES99_H_up),
    }
    
    # Prepare historical price series for context (last 300 bars)
    hist_window = int(min(300, len(df)))
    price_hist_times = df["timestamp"].iloc[-hist_window:]
    price_hist_values = df["close"].iloc[-hist_window:]

    # Sample a subset of simulation paths for plotting
    sample_n = int(min(50, sims))
    sample_idx = rng.choice(sims, size=sample_n, replace=False)
    paths_sample = price_paths[sample_idx, 1:]

    # Debug info
    n_obs = int(len(df))
    last_hist_time = df["timestamp"].iloc[-1].isoformat()
    first_fc_time = t_fc[0].isoformat() if len(t_fc) > 0 else None
    alignment_ok = (pd.to_datetime(first_fc_time) > pd.to_datetime(last_hist_time)) if first_fc_time else False
    returns_unit = "pct"
    # rolling std of last 288 bars (or min sensible)
    roll_window = int(min(288, len(df)))
    rolling_std_pct = float(np.std(100.0 * np.log(df["close"].iloc[-roll_window:] / df["close"].iloc[-roll_window:].shift(1)).dropna()))
    model_sigma_last_pct = float(hist_vol[-1]) if len(hist_vol) > 0 else float("nan")
    sigma_ratio = float(model_sigma_last_pct / rolling_std_pct) if rolling_std_pct > 0 else float("nan")
    converged = bool(diag.get("converged", True))
    iterations = int(diag.get("iters", 0))
    aic = float(diag.get("aic", np.nan))
    bic = float(diag.get("bic", np.nan))
    # persistence proxy: beta or close equivalent if present
    persistence_proxy = float(fit_params.get("beta[1]", fit_params.get("beta", np.nan))) if fit_params else float("nan")
    fc_sigma_first_pct = float(vol_fc[0]) if len(vol_fc) > 0 else float("nan")

    # Warnings per rules
    if not alignment_ok:
        warnings.append("Forecast not ahead of history (index alignment issue).")
    if np.isfinite(sigma_ratio) and (sigma_ratio < 0.5 or sigma_ratio > 2.0):
        warnings.append("Model σ out of line with rolling stdev (check units/variance/duplication).")
    if iterations == 0 and converged:
        warnings.append("Fit result not applied (check you’re reading result, not spec).")
    if used_fallback:
        warnings.append(f"Primary {data_source_requested} failed; using {data_source_used}.")
    if used_cache_flag:
        warnings.append("Served cached OHLCV due to fetch issues.")
    warnings.append("Two-sided VaR/CVaR enabled.")

    debug_payload = {
        "n_obs": n_obs,
        "last_hist_time": last_hist_time,
        "first_fc_time": first_fc_time,
        "alignment_ok": bool(alignment_ok),
        "returns_unit": returns_unit,
        "rolling_std_pct": rolling_std_pct,
        "model_sigma_last_pct": model_sigma_last_pct,
        "sigma_ratio": sigma_ratio,
        "converged": converged,
        "iterations": iterations,
        "aic": aic,
        "bic": bic,
        "persistence_proxy": persistence_proxy,
        "fc_sigma_first_pct": fc_sigma_first_pct,
        "data_source_requested": data_source_requested,
        "data_source_used": data_source_used,
        "used_fallback": bool(used_fallback),
        "used_cache": bool(used_cache_flag),
    }

    # Expected range (symmetric two-sided) from H-step returns
    # rH is simple return already from price paths
    def pct_band(arr, low_q, high_q):
        lo = float(np.quantile(arr, low_q))
        hi = float(np.quantile(arr, high_q))
        return lo, hi
    p90_lo, p90_hi = pct_band(rH, 0.05, 0.95)
    p95_lo, p95_hi = pct_band(rH, 0.025, 0.975)
    p99_lo, p99_hi = pct_band(rH, 0.005, 0.995)
    expected_range = {
        "timeframe": timeframe,
        "horizon_steps": int(horizon),
        "price": {
            "p90": {"low": float(P0 * (1.0 + p90_lo)), "high": float(P0 * (1.0 + p90_hi))},
            "p95": {"low": float(P0 * (1.0 + p95_lo)), "high": float(P0 * (1.0 + p95_hi))},
            "p99": {"low": float(P0 * (1.0 + p99_lo)), "high": float(P0 * (1.0 + p99_hi))},
        },
        "return_pct": {
            "p90": {"low": float(100.0 * p90_lo), "high": float(100.0 * p90_hi)},
            "p95": {"low": float(100.0 * p95_lo), "high": float(100.0 * p95_hi)},
            "p99": {"low": float(100.0 * p99_lo), "high": float(100.0 * p99_hi)},
        },
    }

    # Build payload
    payload = {
        "meta": {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars_used": int(len(df)),
            "last_close": P0,
            "stale": bool(stale_data),
            "horizon": int(horizon),
            "sims": int(sims),
            "seed": int(seed),
            "data_source_requested": data_source_requested,
            "data_source_used": data_source_used,
            "used_fallback": bool(used_fallback),
            "used_cache": bool(used_cache_flag),
            "quote_filter": quote_filter,
            "caps": {
                "max_sims": int(MAX_SIMS),
                "max_h": int(MAX_H),
                "max_bars_by_tf": MAX_BARS_BY_TF
            }
        },
        "fit": {
            "model": fit_name,
            "dist": dist,
            "params": fit_params,
            "diagnostics": diag
        },
        "vol": {
            "times_hist": pd.to_datetime(t_hist).astype(str).tolist(),
            "cond_hist": [float(x) for x in hist_vol],
            "forecast": [float(x) for x in vol_fc],
            "daily": [float(x) for x in vol_fc_daily],
            "annual": [float(x) for x in vol_fc_annual]
        },
        "sim": {
            "times": [ts.isoformat() for ts in t_fc],
            "p10": [float(x) for x in p10],
            "p50": [float(x) for x in p50],
            "p90": [float(x) for x in p90],
            "p05": [float(x) for x in p05],
            "p95": [float(x) for x in p95],
            "risk": {k: float(v) for k, v in risk.items()},
            "risk_pct": {k: float(v) for k, v in risk_pct.items()},
            "risk_price": {k: float(v) for k, v in risk_price.items()},
            # include a small sample of simulated paths for plotting
            "paths_sample": [[float(v) for v in row] for row in paths_sample],
            "price_hist_times": pd.to_datetime(price_hist_times).astype(str).tolist(),
            "price_hist": [float(x) for x in price_hist_values],
        },
        "warnings": warnings,
        "debug": debug_payload,
        "expected_range": expected_range,
    }
    
    return payload

