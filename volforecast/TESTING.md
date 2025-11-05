# How to Test the Volatility Forecasting App

## Quick Start

1. **Navigate to the project directory:**
```bash
cd volforecast
```

2. **Install dependencies (if not already installed):**
```bash
pip install -r requirements.txt
```

3. **Run the FastAPI server:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

4. **Open your browser:**
   - Go to: **http://localhost:8080**
   - You should see the volatility forecasting form

## Where to Input Symbol

The symbol selection is the **first dropdown field** at the top of the form:

1. **Look for "Symbol" label** - it's the first field on the page
2. **Click the dropdown** - it shows all available OKX spot trading pairs
3. **Select a symbol** - e.g., "BTC/USDT", "ETH/USDT", etc.

### Symbol List Source

The app automatically fetches symbols from OKX exchange:
- Fetches all USDT-quoted spot pairs
- Caches the list for 24 hours
- Falls back to a small default list if API fails

### Example Symbols Available:
- BTC/USDT
- ETH/USDT
- SOL/USDT
- BNB/USDT
- ADA/USDT
- And many more...

## Testing Workflow

1. **Fill out the form:**
   - **Symbol**: Select from dropdown (e.g., BTC/USDT)
   - **Timeframe**: Choose 15m, 1h, 4h, or 1d
   - **Horizon**: Number of forecast steps (default: 18)
   - **Simulations**: Number of Monte Carlo paths (default: 1500)
   - **Model**: EGARCH, GARCH, or EWMA
   - **Distribution**: t or normal

2. **Click "Generate Forecast"**

3. **Wait for results** (may take 30-60 seconds for first run as it fetches data)

4. **View results:**
   - Volatility history + forecast chart
   - Price simulation bands (p10/p50/p90)
   - VaR/CVaR risk metrics table

## Troubleshooting

### If symbols dropdown is empty:
- Check internet connection (needs to fetch from OKX API)
- Check if OKX API is accessible
- Fallback symbols should still appear: BTC/USDT, ETH/USDT, etc.

### If forecast fails:
- Check error message on error page
- Try a different symbol
- Try reducing lookback_bars (default: 2000)
- Try a simpler model (EWMA instead of EGARCH)

### First run may be slow:
- Initial data fetch from OKX
- Model fitting takes time
- Subsequent runs use cached data (faster)

## Quick Test Command

To test with a specific symbol quickly, you can modify the default in `app/main.py` or just use the form dropdown.

