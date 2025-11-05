"""
FastAPI Volatility Forecasting App
"""

import os
import secrets
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .engine import run_pipeline, list_spot_symbols, union_symbols_across_exchanges

# Get password from environment or default to 'void'
# For VPS: Set VOL_PASS and SESSION_SECRET as environment variables
VOL_PASS = os.getenv("VOL_PASS", "void")
SESSION_SECRET = os.getenv("SESSION_SECRET", secrets.token_urlsafe(32))

# Detect if running in production (for HTTPS-only cookies)
# Set DEPLOY_ENV=production on VPS or use HTTPS detection
IS_PRODUCTION = os.getenv("DEPLOY_ENV") == "production" or os.getenv("HTTPS") == "true"

app = FastAPI(title="Volatility Forecasting Dashboard")

# Add session middleware with secure cookie settings
# For VPS: Set DEPLOY_ENV=production to enable HTTPS-only cookies
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    max_age=86400,  # 24 hours
    same_site="lax",
    https_only=IS_PRODUCTION  # True in production with HTTPS
)

# Static and templates
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
static_dir = os.path.join(project_root, "static")
templates_dir = os.path.join(project_root, "templates")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Add custom Jinja2 filter for number formatting
def format_float(value, decimals=2):
    """Format float to specified decimal places."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"

templates.env.filters["ff"] = format_float


# Authentication dependency
def require_auth(request: Request):
    """Check if user is authenticated."""
    if not request.session.get("authenticated", False):
        return None
    return True


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    """Login page."""
    # If already authenticated, redirect to home
    if request.session.get("authenticated", False):
        return RedirectResponse(url="/", status_code=303)
    
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": error
        }
    )


@app.post("/login")
async def login(request: Request, password: str = Form(...)):
    """Login handler."""
    if password == VOL_PASS:
        request.session["authenticated"] = True
        return RedirectResponse(url="/", status_code=303)
    else:
        return RedirectResponse(url="/login?error=invalid", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    """Logout handler."""
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


def get_mock_payload():
    """Return a mock payload for Step 1 testing."""
    return {
        "meta": {
            "symbol": "BTC/USDT",
            "timeframe": "4h",
            "bars_used": 2190,
            "last_close": 109317.0,
            "stale": False,
            "horizon": 18,
            "sims": 1500,
            "seed": 42,
            "caps": {
                "max_sims": 2500,
                "max_h": 100,
                "max_bars_by_tf": {"15m": 1344, "1h": 2160, "4h": 2190, "1d": 1095}
            }
        },
        "fit": {
            "model": "EGARCH(1,1)_t",
            "dist": "t",
            "params": {
                "mu": 0.05,
                "omega": 0.01,
                "alpha[1]": 0.1,
                "beta[1]": 0.85,
                "nu": 5.2
            },
            "diagnostics": {
                "converged": True,
                "aic": -5.2,
                "bic": -5.1,
                "loglik": 1200.5,
                "iters": 15
            }
        },
        "vol": {
            "times_hist": ["2024-01-01T00:00:00+00:00"] * 200,  # Simplified for demo
            "cond_hist": [2.5 + i * 0.01 for i in range(200)],
            "forecast": [3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5],
            "daily": [15.6, 15.2, 14.7, 14.2, 13.7, 13.2, 12.7, 12.2, 11.7, 11.2, 10.7, 10.2, 9.8, 9.4, 9.0, 8.6, 8.2, 7.8],
            "annual": [298.5, 290.7, 281.0, 271.6, 261.8, 252.4, 243.0, 233.3, 223.7, 214.1, 204.6, 195.1, 187.3, 179.8, 172.1, 164.4, 156.8, 149.2]
        },
        "sim": {
            "times": [f"2024-01-20T{i*4:02d}:00:00+00:00" for i in range(18)],
            "p10": [105000 + i * 500 for i in range(18)],
            "p50": [109317 - i * 100 for i in range(18)],
            "p90": [114000 + i * 300 for i in range(18)],
            "risk": {
                "VaR95_1": -0.0093,
                "ES95_1": -0.0115,
                "VaR99_1": -0.0132,
                "ES99_1": -0.0158,
                "VaR95_H": -0.098,
                "ES95_H": -0.115,
                "VaR99_H": -0.132,
                "ES99_H": -0.158,
                "CVaR95_1": -0.0115,
                "CVaR99_1": -0.0158,
                "CVaR95_H": -0.115,
                "CVaR99_H": -0.158
            },
            "risk_pct": {
                "VaR95_1_pct": -0.93,
                "ES95_1_pct": -1.15,
                "VaR99_1_pct": -1.32,
                "ES99_1_pct": -1.58,
                "VaR95_H_pct": -9.8,
                "ES95_H_pct": -11.5,
                "VaR99_H_pct": -13.2,
                "ES99_H_pct": -15.8
            },
            "risk_price": {
                "VaR95_1_price": 109317 * 0.9907,
                "ES95_1_price": 109317 * 0.9885,
                "VaR99_1_price": 109317 * 0.9868,
                "ES99_1_price": 109317 * 0.9842,
                "VaR95_H_price": 99900.0,
                "ES95_H_price": 98450.0,
                "VaR99_H_price": 96820.0,
                "ES99_H_price": 94890.0
            }
        },
        "warnings": []
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main form page - protected."""
    # Check authentication
    if not request.session.get("authenticated", False):
        return RedirectResponse(url="/login", status_code=303)
    # Get form values from query params if coming back from results/error
    form_values = {
        "symbol": request.query_params.get("symbol", ""),
        "timeframe": request.query_params.get("timeframe", ""),
        "horizon": request.query_params.get("horizon", ""),
        "sims": request.query_params.get("sims", ""),
        "model": request.query_params.get("model", ""),
        "dist": request.query_params.get("dist", ""),
        "p": request.query_params.get("p", ""),
        "q": request.query_params.get("q", ""),
        "mean_spec": request.query_params.get("mean_spec", ""),
        "seed": request.query_params.get("seed", ""),
        "vol_scale": request.query_params.get("vol_scale", ""),
        "zcap": request.query_params.get("zcap", ""),
        "lookback_bars": request.query_params.get("lookback_bars", ""),
        "drift_mode": request.query_params.get("drift_mode", ""),
        "drift_user_pct": request.query_params.get("drift_user_pct", ""),
        "data_source": request.query_params.get("data_source", "auto"),
        "failover": request.query_params.get("failover", "on"),
        "quote": request.query_params.get("quote", "USDT"),
        "show_debug": request.query_params.get("show_debug", "off"),
    }
    
    # Build symbol list with tags
    try:
        ds = (form_values.get("data_source") or "auto").lower()
        quote_raw = (form_values.get("quote") or "AUTO").upper()
        strict_quote = (quote_raw != "AUTO")
        quote = None if not strict_quote else quote_raw
        
        if ds == "auto":
            # Union across exchanges
            exchanges = ["okx", "binance", "bybit", "kucoin", "kraken"]
            symbol_tuples, symbol_sources = union_symbols_across_exchanges(
                exchanges, quote=quote, strict_quote=strict_quote
            )
            symbols = symbol_tuples
            symbol_count = len(symbols)
        else:
            # Single exchange
            symbol_tuples = list_spot_symbols(ds, quote=quote, strict_quote=strict_quote, cache_hours=24)
            symbols = symbol_tuples
            symbol_sources = {ds}
            symbol_count = len(symbols)
    except Exception as e:
        # Fallback
        symbols = [("BTC/USDT", "okx"), ("ETH/USDT", "okx"), ("SOL/USDT", "okx"), ("BNB/USDT", "okx"), ("ADA/USDT", "okx")]
        symbol_sources = {"okx"}
        symbol_count = 5
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "symbols": symbols,
            "symbol_count": symbol_count,
            "symbol_sources": sorted(symbol_sources),
            "timeframes": ["15m", "1h", "4h", "1d"],
            "models": ["EGARCH", "GARCH", "EWMA"],
            "dists": ["t", "normal"],
            "mean_specs": ["constant", "zero", "ar1"],
            "data_sources": ["Auto","OKX","Binance","Bybit","KuCoin","Kraken"],
            "quotes": ["USDT","USDC"],
            "form_values": form_values
        }
    )


@app.post("/forecast", response_class=HTMLResponse)
async def forecast(request: Request,
                   symbol: str = Form(...),
                   timeframe: str = Form(...),
                   horizon: int = Form(...),
                   sims: int = Form(...),
                   model: str = Form("egarch"),
                   dist: str = Form("t"),
                   p: int = Form(1),
                   q: int = Form(1),
                   mean_spec: str = Form("constant"),
                   seed: int = Form(42),
                   vol_scale: float = Form(1.0),
                   zcap: float = Form(6.0),
                   lookback_bars: int = Form(2000),
                   drift_mode: str = Form("neutral"),
                   drift_user_pct: float = Form(0.0),
                   data_source: str = Form("auto"),
                   failover: str = Form("on"),
                   quote: str = Form("USDT"),
                   show_debug: str = Form("off")):
    """Forecast endpoint - renders results with charts."""
    # Check authentication
    if not request.session.get("authenticated", False):
        return RedirectResponse(url="/login", status_code=303)
    
    # Prepare parameters for engine
    params = {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon": horizon,
        "sims": sims,
        "model": model.lower(),
        "dist": dist,
        "p": p,
        "q": q,
        "mean_spec": mean_spec,
        "lookback_bars": lookback_bars,
        "seed": seed,
        "zcap": zcap,
        "vol_scale": vol_scale,
        "drift_mode": drift_mode,
        "drift_user_pct": drift_user_pct,
        "use_cache": True,
        "base_dir": project_root,
        "data_source": data_source,
        "failover": (failover.lower() in ("on","true","1","yes")),
        "quote": quote,
        "show_debug": (show_debug.lower() in ("on","true","1","yes")),
    }
    
    # Try to run the engine
    try:
        payload = run_pipeline(params)
        error_message = None
    except Exception as e:
        # On error, show error page but keep form values
        error_message = str(e)
        payload = None
        
        # Try to get symbols for form
        try:
            symbols = list_spot_symbols("okx", quote="USDT", cache_hours=24, strict_quote=True)
        except Exception:
            symbols = [("BTC/USDT", "okx"), ("ETH/USDT", "okx"), ("SOL/USDT", "okx"), ("BNB/USDT", "okx"), ("ADA/USDT", "okx")]
        
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error_message": error_message,
                "form_values": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "horizon": horizon,
                    "sims": sims,
                    "model": model,
                    "dist": dist,
                    "p": p,
                    "q": q,
                    "mean_spec": mean_spec,
                    "seed": seed,
                    "vol_scale": vol_scale,
                    "zcap": zcap,
                    "lookback_bars": lookback_bars,
                    "drift_mode": drift_mode,
                    "drift_user_pct": drift_user_pct
                },
                "symbols": symbols,
                "timeframes": ["15m", "1h", "4h", "1d"],
                "models": ["EGARCH", "GARCH", "EWMA"],
                "dists": ["t", "normal"],
                "mean_specs": ["constant", "zero", "ar1"],
                "data_sources": ["Auto","OKX","Binance","Bybit","KuCoin","Kraken"],
                "quotes": ["USDT","USDC"]
            }
        )
    
    # Success - render results
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "payload": payload,
            "form_values": {
                "symbol": symbol,
                "timeframe": timeframe,
                "horizon": horizon,
                "sims": sims,
                "model": model,
                "dist": dist,
                "p": p,
                "q": q,
                "mean_spec": mean_spec,
                "seed": seed,
                "vol_scale": vol_scale,
                "zcap": zcap,
                "lookback_bars": lookback_bars,
                "drift_mode": drift_mode,
                "drift_user_pct": drift_user_pct,
                "data_source": data_source,
                "failover": failover,
                "quote": quote,
                "show_debug": show_debug,
            }
        }
    )

