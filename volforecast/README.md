# Volatility Forecasting Dashboard

A comprehensive web application for cryptocurrency volatility forecasting with advanced risk metrics, multi-exchange data support, and professional-grade analytics.

## Features

### 🔐 Security
- **Password-protected access** with secure session management
- Environment variable-based authentication
- Secure cookie sessions (HTTPS-ready for production)

### 📊 Multi-Exchange Data Support
- **Automatic data source selection** with intelligent fallback
- Support for 5 major exchanges: **OKX**, **Binance**, **Bybit**, **KuCoin**, **Kraken**
- Automatic failover when primary exchange is unavailable
- Symbol caching for faster load times
- **Universe mode**: Combines symbols across all exchanges with deduplication
- Flexible quote filtering (USDT, USDC, USD, BTC, ETH, or custom)
- Special handling for TAO token (always prioritized)

### 📈 Volatility Forecasting Models
- **EGARCH** (Exponential GARCH) - Recommended for asymmetric volatility
- **GARCH** - Standard volatility clustering model
- **EWMA** - Exponentially Weighted Moving Average
- Multiple distribution assumptions: **t-distribution** or **normal**
- Configurable mean specifications: constant, zero, or AR(1)

### 🎯 Risk Metrics

#### Two-Sided VaR/CVaR
- **Downside Risk**:
  - VaR95/CVaR95 (1-step and H-step horizon)
  - VaR99/CVaR99 (1-step and H-step horizon)
- **Upside Risk**:
  - VaR95/CVaR95 upside (1-step and H-step horizon)
  - VaR99/CVaR99 upside (1-step and H-step horizon)
- All metrics expressed in both **percentage** and **price** terms

#### Expected Range
- Symmetric probability ranges (90%, 95%, 99%)
- Shows expected price range for forecast horizon
- Displays both absolute prices and percentage returns
- Includes warnings for low simulation counts

### 📉 Interactive Visualizations
- **Volatility History & Forecast Chart**: Historical volatility with projected path
- **Price Projection Chart**: Historical prices with:
  - Monte Carlo simulated paths
  - VaR95 confidence bands
  - Forecast horizon visualization
- Dark theme optimized for professional trading environments

### 🔍 Advanced Features
- **Symbol Search**: Real-time filtering of 1000+ cryptocurrency pairs
- **Debug Info Panel**: Comprehensive model diagnostics including:
  - Model convergence status
  - Iteration count
  - AIC/BIC metrics
  - Volatility ratios
  - Data source tracking
- **Data Source Badge**: Shows which exchange was used, fallback status, and cache usage
- **Universe Statistics**: Displays total symbol count and source exchanges

### 🎨 User Interface
- **Modern Dark Theme**: Professional trading terminal aesthetic
- **Responsive Design**: Works on desktop and mobile devices
- **Form Value Preservation**: Automatically restores form values after submission
- **Error Handling**: Graceful error messages with troubleshooting tips

## Project Structure

```
volforecast/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application with auth and routing
│   └── engine.py        # Core volatility forecasting engine
├── templates/
│   ├── index.html       # Main input form with symbol search
│   ├── result.html      # Results page with charts and metrics
│   ├── login.html       # Login page
│   └── error.html       # Error page
├── static/
│   ├── theme.css        # Dark theme stylesheet
│   └── styles.css       # Additional styles
├── cache/               # Auto-created data cache
│   ├── ohlcv/          # OHLCV data cache
│   └── fit/            # Model fit cache
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (required for security):
```bash
# Copy the example file
cp .env.example .env

# Edit .env and set your values:
# - VOL_PASS: Your secure password (required)
# - SESSION_SECRET: Random secret key (required - generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))')
```

**Important**: Never commit the `.env` file to git. It's already in `.gitignore`.

Alternatively, you can set environment variables directly:
```bash
# Windows PowerShell
$env:VOL_PASS = "your_secure_password"
$env:SESSION_SECRET = "your_random_secret"

# Linux/Mac
export VOL_PASS="your_secure_password"
export SESSION_SECRET="your_random_secret"
```

## Running the App

### Development

```bash
cd volforecast
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the app at `http://localhost:8000` (or your server's IP address). You'll be redirected to the login page.

### Production (VPS)

**Important**: If port 8000 is already in use, change the port number in the commands below (e.g., use `8001`).

1. **Set up environment variables** (create `.env` file):
```bash
cd /path/to/volforecast
cp .env.example .env
nano .env  # Edit and set your values
```

Or set environment variables directly:
```bash
export VOL_PASS="your_secure_password"
export SESSION_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export DEPLOY_ENV="production"  # Enables HTTPS-only cookies
```

2. **Run with uvicorn** (use a different port if 8000 is taken):
```bash
# If port 8000 is free:
uvicorn app.main:app --host 0.0.0.0 --port 8000

# If port 8000 is taken, use 8001:
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

3. **Use a process manager** (recommended for production):

**systemd service** (`/etc/systemd/system/volforecast.service`):
```ini
[Unit]
Description=Volatility Forecasting App
After=network.target

[Service]
User=your_user
WorkingDirectory=/path/to/volforecast
# Load environment from .env file (if using python-dotenv, add it to requirements)
# Or set environment variables directly:
Environment="VOL_PASS=your_secure_password"
Environment="SESSION_SECRET=your_generated_secret"
Environment="DEPLOY_ENV=production"
ExecStart=/path/to/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable volforecast
sudo systemctl start volforecast
sudo systemctl status volforecast  # Check status
```

4. **Behind a reverse proxy** (nginx example - adjust port if needed):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8001;  # Change port if app uses different port
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Deployment Checklist:**
- [ ] Set strong `VOL_PASS` in `.env` or environment variables
- [ ] Generate secure `SESSION_SECRET` (never reuse across deployments)
- [ ] Set `DEPLOY_ENV=production` for HTTPS-only cookies
- [ ] Choose available port (8000, 8001, etc.)
- [ ] Configure firewall to allow traffic on chosen port
- [ ] Set up reverse proxy (nginx) if using HTTPS/domain
- [ ] Test login functionality
- [ ] Verify `.env` file is not committed to git (it's in `.gitignore`)

## Usage Guide

### Basic Workflow

1. **Login**: Enter your password (set in `VOL_PASS` environment variable) to access the dashboard
2. **Select Symbol**: Use the search box to filter 1000+ cryptocurrency pairs
3. **Configure Parameters**:
   - **Timeframe**: 15m, 1h, 4h, or 1d
   - **Horizon**: Forecast steps ahead (e.g., 18 steps of 4h = 3 days)
   - **Simulations**: Number of Monte Carlo paths (recommended: 1000+)
   - **Model**: EGARCH (recommended), GARCH, or EWMA
   - **Distribution**: t-distribution (recommended) or normal
4. **Advanced Options**:
   - GARCH(p,q) parameters
   - Mean specification
   - Volatility scaling
   - Z-cap for outlier handling
   - Lookback period
   - Drift mode
5. **Data Source**: Choose Auto (with fallback) or specific exchange
6. **Generate Forecast**: Click "Generate Forecast" to run analysis

### Understanding Results

#### Model Diagnostics
- **Converged**: Model successfully optimized
- **Iterations**: Number of optimization iterations
- **AIC/BIC**: Model selection criteria (lower is better)

#### Risk Metrics Table
- **VaR95**: Value at Risk at 95% confidence (downside/upside)
- **CVaR95**: Conditional VaR (expected loss beyond VaR95)
- **1-step**: Next period forecast
- **H-step**: Cumulative forecast over horizon

#### Expected Range
- Shows the price range where the asset is expected to trade
- 90% range: 90% of simulations fall within this range
- 95% range: 95% of simulations fall within this range
- 99% range: 99% of simulations fall within this range

#### Charts
- **Volatility Chart**: Historical volatility with forecasted path
- **Price Chart**: Historical prices with simulated paths and VaR bands

#### Debug Info
- Click "Debug Info" to see detailed model diagnostics
- Useful for troubleshooting convergence issues
- Shows data source tracking (exchange, fallback, cache usage)

## Configuration

### Model Parameters

- **p, q**: GARCH lag parameters (typically 1,1 or 1,2)
- **vol_scale**: Volatility scaling factor (default: 1.0)
- **zcap**: Z-score cap for outlier handling (default: 6.0)
- **lookback_bars**: Historical data window (default: 2000)
- **drift_mode**: neutral, bullish, or bearish
- **drift_user_pct**: Custom drift percentage (if mode is "custom")

### Data Source Options

- **Auto**: Automatically selects from OKX → Binance → Bybit → KuCoin → Kraken
- **Failover**: Automatically switches if primary exchange fails
- **Quote Filter**: Auto (USDT/USDC/USD/BTC/ETH) or specific quote currency
- **Cache**: Symbols and OHLCV data cached for 24 hours

## Troubleshooting

### Model Convergence Issues
- Try reducing `lookback_bars`
- Switch to GARCH model (more stable than EGARCH)
- Use normal distribution instead of t-distribution
- Check Debug Info for convergence warnings

### Data Fetching Issues
- Enable "Failover" for automatic exchange switching
- Check internet connection
- Verify exchange API access (some exchanges require IP whitelisting)
- Symbols are cached for 24 hours - wait for cache refresh

### Performance
- Reduce `sims` count for faster results (but lower accuracy)
- Reduce `lookback_bars` for faster data loading
- Use cached data when available (indicated by "(cache)" badge)

## Security Notes

- **Never commit `.env` file** to git (already in `.gitignore`)
- **Set a strong password** via `VOL_PASS` environment variable
- **Generate a unique SESSION_SECRET** for each deployment (use `python -c 'import secrets; print(secrets.token_urlsafe(32))'`)
- **Use HTTPS** in production (set `DEPLOY_ENV=production` to enable HTTPS-only cookies)
- **Protect environment variables** - use `.env` file or system environment variables
- Sessions expire after 24 hours of inactivity
- The repository is public, but authentication keeps the app secure

## Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **Forecasting**: ARCH library (GARCH/EGARCH models)
- **Data**: CCXT (cryptocurrency exchange library)
- **Visualization**: Plotly.js
- **Frontend**: Tailwind CSS, custom dark theme
- **Caching**: File-based (Parquet format)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Support

For issues, questions, or feature requests, please open an issue on GitHub.
