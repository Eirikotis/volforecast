# Quick Start Guide

## 🚀 How to Run & Test

### Step 1: Install Dependencies
```bash
cd volforecast
pip install -r requirements.txt  # First time only
```

### Step 2: Configure Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env and set your password and session secret
# VOL_PASS=your_secure_password
# SESSION_SECRET=your_generated_secret
```

Generate a session secret:
```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

### Step 3: Start the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4: Login
1. Open browser and navigate to your server address (or `http://localhost:8000` for local dev)
2. You'll be redirected to the login page
3. Enter the password you set in `VOL_PASS` environment variable
4. Click "Login"

## 📍 Symbol Selection

**The symbol dropdown is the FIRST field on the form:**

1. Look at the top of the page - you'll see "**Symbol**" label
2. **Use the search box** to filter symbols (e.g., type "BTC" to find BTC/USDT)
3. Click on a symbol from the dropdown list
4. Symbols show exchange tags: `BTC/USDT • Okx`

### Symbol Features

- **1000+ symbols** from 5 exchanges (OKX, Binance, Bybit, KuCoin, Kraken)
- **Real-time search** - type to filter instantly
- **Alphabetically sorted** - easy to find what you need
- **Auto-fetched** - automatically loads from exchanges
- **Cached for 24 hours** - fast after first load
- **Universe mode** - combines symbols from all exchanges when "Auto" data source is selected

### Data Source Options

- **Auto**: Combines symbols from all 5 exchanges (recommended)
- **Specific Exchange**: Choose OKX, Binance, Bybit, KuCoin, or Kraken
- **Failover**: Automatically switches if primary exchange fails (enabled by default)
- **Quote Filter**: Auto (USDT/USDC/USD/BTC/ETH) or specific quote currency

## 📋 Form Layout

When you open the dashboard, you'll see:

```
┌─────────────────────────────────────────┐
│  Volatility Forecasting      [Logout]   │
│                                           │
│  Symbol: [Search box...]                 │
│          [BTC/USDT • Okx ▼]  ← SELECT   │
│  Timeframe: [4h ▼]                       │
│  Horizon: [18]                           │
│  Simulations: [1500]                     │
│  Model: [EGARCH ▼]                       │
│  Distribution: [t ▼]                     │
│  Data Source: [Auto ▼]                   │
│  Failover: [On ✓]                        │
│                                           │
│  [Generate Forecast]                     │
└─────────────────────────────────────────┘
```

## 🧪 Testing Steps

1. **Login** with your password (set in `VOL_PASS` environment variable)
2. **Search for a symbol** (e.g., type "BTC" in the search box)
3. **Select Symbol** from dropdown (e.g., BTC/USDT)
4. **Choose Timeframe** (15m, 1h, 4h, 1d)
5. **Set Horizon** (default: 18 steps = 3 days for 4h timeframe)
6. **Set Simulations** (default: 1500 - higher = more accurate but slower)
7. **Choose Model**:
   - **EGARCH** (recommended) - handles asymmetric volatility
   - **GARCH** - standard volatility clustering
   - **EWMA** - simple exponential smoothing
8. **Click "Generate Forecast"**
9. **Wait 30-60 seconds** (first run fetches data, subsequent runs are faster with cache)
10. **View results**:
    - Volatility history & forecast chart
    - Price projection with simulated paths and VaR bands
    - Risk metrics table (VaR/CVaR downside and upside)
    - Expected Range card (90%/95%/99% price ranges)
    - Debug Info panel (if enabled)

## 📊 Understanding Results

### Expected Range Card
Shows where the price is expected to trade:
- **90% range**: 90% of simulations fall within this price range
- **95% range**: 95% of simulations fall within this price range  
- **99% range**: 99% of simulations fall within this price range

### Risk Metrics Table
- **VaR95**: Value at Risk at 95% confidence (downside/upside)
- **CVaR95**: Expected loss beyond VaR95
- **1-step**: Next period forecast
- **H-step**: Cumulative forecast over entire horizon

### Charts
- **Volatility Chart**: Historical volatility with forecasted path
- **Price Chart**: Historical prices with:
  - Simulated future paths (Monte Carlo)
  - VaR95 confidence bands
  - Forecast horizon indicator

### Data Source Badge
- Shows which exchange was used
- **(fallback)**: Primary exchange failed, used backup
- **(cache)**: Using cached data (faster)

## 🔍 Symbol Fetching Details

**Code Location:**
- `app/main.py`: Symbol list building with multi-exchange support
- `app/engine.py`: `list_spot_symbols()` and `union_symbols_across_exchanges()`

**What it does:**
- Connects to exchange APIs (OKX, Binance, Bybit, KuCoin, Kraken)
- Gets all spot trading pairs with flexible quote filtering
- Combines and deduplicates symbols when "Auto" mode is selected
- Caches symbols in `cache/symbols_{exchange}_spot.json`
- Caches OHLCV data in `cache/ohlcv/` (Parquet format)

**If symbol list is empty:**
- Check internet connection
- Exchange API might be down (try failover mode)
- Check cache directory permissions
- Fallback symbols will show: BTC/USDT, ETH/USDT, etc.

## 💡 Tips

- **First run is slow** - data is being fetched and cached
- **Subsequent runs are faster** - using cached data
- **Higher simulations = more accurate** but slower (1500+ recommended)
- **Enable Debug Info** to troubleshoot convergence issues
- **Use Auto data source** for maximum symbol coverage
- **Check Debug Info** if model doesn't converge (try GARCH or reduce lookback_bars)

## 🚨 Troubleshooting

### "Model didn't converge" warning
- Try switching to **GARCH** model (more stable)
- Reduce **lookback_bars** (e.g., from 2000 to 1000)
- Use **normal** distribution instead of t-distribution

### "No data available" error
- Enable **Failover** mode
- Check internet connection
- Try a different exchange manually
- Check cache directory exists and is writable

### Slow performance
- Reduce **simulations** count (but lower accuracy)
- Reduce **lookback_bars**
- Use cached data (indicated by "(cache)" badge)
