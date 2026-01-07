import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from functools import wraps
from scipy import stats

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Stock Calculator Pro", layout="wide", initial_sidebar_state="expanded")

# =====================================================
# CONSTANTS
# =====================================================
VOL_FLOOR = 0.10
VOL_CAP = 1.5
KELLY_MAX = 0.15
KELLY_MIN = 0.01
MONTE_CARLO_SIMS = 5000
RISK_FREE_RATE = 0.045  # 4.5% annual risk-free rate

# =====================================================
# DECORATORS
# =====================================================
def retry_on_rate_limit(max_retries=3, delay=2):
    """Decorator to retry functions on rate limit errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    if "Rate limit" in error_msg or "Too Many Requests" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = delay * (2 ** attempt)  # Exponential backoff
                            time.sleep(wait_time)
                            continue
                    raise e
            return None
        return wrapper
    return decorator

# =====================================================
# FUNCTIONS
# =====================================================
@st.cache_data(ttl=3600)
@retry_on_rate_limit(max_retries=3, delay=2)
def load_stock(ticker):
    """Download stock data with improved error handling."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y", timeout=10)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Handle multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure Close column exists
        if "Close" not in df.columns:
            return pd.DataFrame()
        
        # Clean data
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
        
        if len(df) < 60:
            return pd.DataFrame()
            
        return df
    except Exception as e:
        st.error(f"Error loading {ticker}: {str(e)}")
        return pd.DataFrame()

def annual_volatility(df):
    """Calculate annualized volatility using EWMA."""
    if df.empty or len(df) < 10:
        return 0.30
    
    returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if len(returns) < 10:
        return 0.30
    
    vol = returns.ewm(span=20).std().iloc[-1] * np.sqrt(252)
    return float(np.clip(vol, VOL_FLOOR, VOL_CAP))

def get_historical_returns(df, days=252):
    """Get historical returns for bootstrap sampling."""
    if df.empty or len(df) < days:
        return np.array([])
    
    returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    return returns.tail(days).values

def expected_move(price, vol, days=30):
    """Expected price move over a given number of days."""
    return price * vol * np.sqrt(days / 252)

def prob_hit_mc_advanced(S, K, vol, days=30, sims=MONTE_CARLO_SIMS, method="student_t", df_hist=None):
    """
    Advanced Monte Carlo with fat-tail distributions.
    
    Parameters:
    - method: "normal", "student_t", or "bootstrap"
    - df_hist: Historical dataframe for bootstrap method
    """
    if vol <= 0 or S <= 0 or K <= 0:
        return 0.0
    
    dt = 1/252
    
    if method == "student_t":
        # Use Student's t-distribution with df=5 for fat tails
        dof = 5
        random_shocks = stats.t.rvs(df=dof, size=(sims, days)) / np.sqrt(dof / (dof - 2))
        
    elif method == "bootstrap" and df_hist is not None:
        # Historical bootstrap
        hist_returns = get_historical_returns(df_hist)
        if len(hist_returns) < 60:
            # Fallback to Student's t if insufficient data
            dof = 5
            random_shocks = stats.t.rvs(df=dof, size=(sims, days)) / np.sqrt(dof / (dof - 2))
        else:
            # Sample with replacement from historical returns
            random_indices = np.random.randint(0, len(hist_returns), size=(sims, days))
            random_shocks = hist_returns[random_indices]
            # Normalize to match current volatility
            current_vol = np.std(hist_returns) * np.sqrt(252)
            if current_vol > 0:
                random_shocks = random_shocks * (vol / current_vol)
    else:
        # Standard normal distribution (baseline)
        random_shocks = np.random.normal(size=(sims, days))
    
    # Generate price paths
    price_paths = S * np.exp(np.cumsum((-0.5*vol**2)*dt + vol*np.sqrt(dt)*random_shocks, axis=1))
    hits = (price_paths >= K).any(axis=1)
    
    return float(hits.mean())

def kelly_fraction(df):
    """Calculate Kelly fraction for position sizing."""
    if df.empty:
        return 0.02
    
    returns = df["Close"].pct_change().dropna()
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    
    if returns.empty or len(returns) < 30:
        return 0.02
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(wins) < 10 or len(losses) < 10:
        return 0.02
    
    win_rate = len(wins) / len(returns)
    loss_rate = 1 - win_rate
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    
    if avg_loss == 0:
        return 0.02
    
    R = avg_win / avg_loss
    fraction = win_rate - (loss_rate / R)
    
    # Conservative Kelly: use half-Kelly with caps
    return float(np.clip(fraction * 0.5, KELLY_MIN, KELLY_MAX))

def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown from cumulative returns."""
    if len(cumulative_returns) == 0:
        return 0.0
    
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / (running_max + 1e-10)
    max_dd = drawdown.min()
    
    return float(max_dd)

def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    return float(sharpe)

def backtest(df, days=30):
    """Enhanced backtest with risk metrics."""
    if df.empty or len(df) < days + 30:
        return pd.DataFrame({"PnL": [], "Return": []})
    
    pnl = []
    returns = []
    
    for i in range(len(df) - days):
        entry = df["Close"].iloc[i]
        window = df["Close"].iloc[i:i+days].copy()
        
        # Calculate stop-loss
        vol_at_entry = annual_volatility(df.iloc[:i+1]) if i > 20 else annual_volatility(df)
        stop_price = entry - expected_move(entry, vol_at_entry, days) * 0.5
        
        window = pd.to_numeric(window, errors="coerce").dropna()
        if window.empty:
            continue
        
        # Check for stop-loss hit
        hits = window[window <= stop_price]
        exit_price = float(hits.iloc[0]) if not hits.empty else float(window.iloc[-1])
        
        trade_pnl = exit_price - entry
        trade_return = (exit_price - entry) / entry
        
        pnl.append(trade_pnl)
        returns.append(trade_return)
    
    if not pnl:
        return pd.DataFrame({"PnL": [], "Return": []})
    
    bt = pd.DataFrame({
        "PnL": pnl,
        "Return": returns
    })
    
    bt["PnL"] = pd.to_numeric(bt["PnL"], errors="coerce").fillna(0)
    bt["Return"] = pd.to_numeric(bt["Return"], errors="coerce").fillna(0)
    bt["CumPnL"] = bt["PnL"].cumsum()
    bt["CumReturn"] = (1 + bt["Return"]).cumprod() - 1
    
    return bt

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "ticker" not in st.session_state:
    st.session_state.ticker = ""
if "calculations_cache" not in st.session_state:
    st.session_state.calculations_cache = {}

# =====================================================
# SIDEBAR CONFIGURATION
# =====================================================
with st.sidebar:
    st.markdown("### 📊 Stock Calculator Pro")
    st.markdown("---")
    
    # Stock input
    ticker_input = st.text_input("Stock Symbol", value="AAPL", key="ticker_input").upper()
    load_button = st.button("🔄 Load Stock", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Configuration")
    
    # Account size
    account_size = st.number_input(
        "Account Size ($)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000,
        key="account_size"
    )
    
    # Simulation days
    sim_days = st.slider(
        "Simulation Period (Days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1,
        key="sim_days"
    )
    
    # Monte Carlo method
    mc_method = st.selectbox(
        "Monte Carlo Method",
        options=["student_t", "bootstrap", "normal"],
        index=0,
        format_func=lambda x: {
            "student_t": "Student's t (Fat Tails)",
            "bootstrap": "Historical Bootstrap",
            "normal": "Standard Normal"
        }[x],
        key="mc_method"
    )
    
    st.markdown("---")
    st.caption("💡 **Tip:** Student's t distribution accounts for market crashes better than normal distribution.")
    st.caption("⚠️ Data cached for 1 hour to avoid rate limits.")

# =====================================================
# LOAD STOCK
# =====================================================
if load_button:
    with st.spinner(f"Loading {ticker_input}..."):
        df = load_stock(ticker_input)
        if not df.empty:
            st.session_state.df = df
            st.session_state.ticker = ticker_input
            st.session_state.calculations_cache = {}  # Clear cache on new load
            st.success(f"✅ Loaded {ticker_input} with {len(df)} days of data")
        else:
            st.error(f"❌ Could not load data for {ticker_input}. Try again later.")

if st.session_state.df.empty:
    st.info("👈 Enter a stock symbol and press **Load Stock**")
    st.stop()

df = st.session_state.df
ticker = st.session_state.ticker

# =====================================================
# CALCULATIONS (with caching)
# =====================================================
cache_key = f"{ticker}_{sim_days}_{mc_method}_{account_size}"

if cache_key not in st.session_state.calculations_cache:
    price = float(df["Close"].iloc[-1])
    vol = annual_volatility(df)
    
    move = expected_move(price, vol, sim_days)
    buy = price - move
    sell = price + move
    stop = buy - move * 0.5
    
    # Use advanced Monte Carlo
    buy_prob = 1 - prob_hit_mc_advanced(price, buy, vol, sim_days, MONTE_CARLO_SIMS, mc_method, df)
    sell_prob = prob_hit_mc_advanced(price, sell, vol, sim_days, MONTE_CARLO_SIMS, mc_method, df)
    
    RRR = (sell - buy) / max(buy - stop, 0.01)
    kelly = kelly_fraction(df)
    qty = int((account_size * kelly) / max(buy - stop, 0.01))
    
    # Cache results
    st.session_state.calculations_cache[cache_key] = {
        "price": price,
        "vol": vol,
        "move": move,
        "buy": buy,
        "sell": sell,
        "stop": stop,
        "buy_prob": buy_prob,
        "sell_prob": sell_prob,
        "RRR": RRR,
        "kelly": kelly,
        "qty": qty
    }

# Retrieve cached calculations
calc = st.session_state.calculations_cache[cache_key]
price = calc["price"]
vol = calc["vol"]
buy = calc["buy"]
sell = calc["sell"]
stop = calc["stop"]
buy_prob = calc["buy_prob"]
sell_prob = calc["sell_prob"]
RRR = calc["RRR"]
kelly = calc["kelly"]
qty = calc["qty"]

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["💰 Trade Setup", "📊 Price Chart", "📜 Backtest & Risk", "📁 Portfolio", "🚀 Market Scanner"]
)

# =====================================================
# TRADE TAB
# =====================================================
with tab1:
    st.title(f"{ticker} - Trade Analysis")
    
    # Price metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Price", f"${price:.2f}")
    col2.metric("Volatility (Annual)", f"{vol*100:.1f}%")
    col3.metric("Expected Move", f"${calc['move']:.2f}")
    col4.metric("Kelly %", f"{kelly*100:.1f}%")
    col5.metric("Shares", qty)
    
    st.markdown("---")
    
    # Trade levels
    st.subheader("📍 Trade Levels")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🟢 Buy Target", f"${buy:.2f}", f"{buy_prob*100:.1f}% prob")
        st.caption(f"Entry if price drops to ${buy:.2f}")
    
    with col2:
        st.metric("🔵 Sell Target", f"${sell:.2f}", f"{sell_prob*100:.1f}% prob")
        st.caption(f"Exit at profit target")
    
    with col3:
        st.metric("🔴 Stop-Loss", f"${stop:.2f}")
        st.caption(f"Max loss per share: ${buy - stop:.2f}")
    
    st.markdown("---")
    
    # Risk-Reward analysis
    st.subheader("⚖️ Risk-Reward Analysis")
    col1, col2, col3 = st.columns(3)
    
    expected_profit = (sell - buy) * qty
    max_loss = (buy - stop) * qty
    
    col1.metric("Risk-Reward Ratio", f"{RRR:.2f}x")
    col2.metric("Expected Profit", f"${expected_profit:,.2f}")
    col3.metric("Maximum Loss", f"${max_loss:,.2f}")
    
    # Color-coded assessment
    if RRR >= 2:
        st.success(f"✅ **Excellent** - Risk-reward ratio of {RRR:.2f} indicates strong trade setup")
    elif RRR >= 1.5:
        st.warning(f"⚠️ **Acceptable** - Risk-reward ratio of {RRR:.2f} is reasonable but not ideal")
    else:
        st.error(f"❌ **Poor** - Risk-reward ratio of {RRR:.2f} is too low for this trade")
    
    st.markdown("---")
    
    # Add to portfolio
    if st.button("➕ Add to Portfolio", type="primary", use_container_width=True):
        trade = {
            "Ticker": ticker,
            "Buy": round(buy, 2),
            "Sell": round(sell, 2),
            "Stop": round(stop, 2),
            "Quantity": qty,
            "Expected PnL": round(expected_profit, 2),
            "Max Loss": round(max_loss, 2),
            "RRR": round(RRR, 2),
            "Method": mc_method
        }
        st.session_state.portfolio.append(trade)
        st.success(f"✅ Added {ticker} to portfolio!")
        st.balloons()

# =====================================================
# CHART TAB
# =====================================================
with tab2:
    st.subheader(f"{ticker} - Price Chart with Trade Levels")
    
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(f'{ticker} Price', 'Volume'),
        shared_xaxes=True
    )
    
    # Price line
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color="blue", width=2)),
        row=1, col=1
    )
    
    # Trade levels
    colors = {"Buy": "green", "Sell": "red", "Stop": "orange"}
    for y_val, name in [(buy, "Buy"), (sell, "Sell"), (stop, "Stop")]:
        if not np.isnan(y_val):
            fig.add_hline(
                y=y_val, line_dash="dash", line_color=colors[name],
                annotation_text=f"{name}: ${y_val:.2f}",
                annotation_position="right",
                row=1, col=1
            )
    
    # Volume bars
    if "Volume" in df.columns:
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="lightblue"),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    fig.update_layout(height=700, showlegend=True, hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# BACKTEST TAB
# =====================================================
with tab3:
    st.subheader("📜 Backtest Results & Risk Metrics")
    st.info(f"📘 Strategy: {sim_days}-day buy & hold with 50% stop-loss")
    
    bt = backtest(df, sim_days)
    
    if not bt.empty and len(bt) > 0:
        # Calculate metrics
        win_rate = (bt["PnL"] > 0).mean() * 100
        total_trades = len(bt)
        winning_trades = (bt["PnL"] > 0).sum()
        losing_trades = (bt["PnL"] < 0).sum()
        
        profit_factor = bt[bt["PnL"]>0]["PnL"].sum() / max(abs(bt[bt["PnL"]<0]["PnL"].sum()), 0.01)
        total_pnl = bt["PnL"].sum()
        avg_win = bt[bt["PnL"]>0]["PnL"].mean() if winning_trades > 0 else 0
        avg_loss = bt[bt["PnL"]<0]["PnL"].mean() if losing_trades > 0 else 0
        
        # Advanced risk metrics
        max_dd = calculate_max_drawdown(bt["CumReturn"].values)
        sharpe = calculate_sharpe_ratio(bt["Return"].values)
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", total_trades)
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Profit Factor", f"{profit_factor:.2f}")
        col4.metric("Total P&L", f"${total_pnl:,.2f}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Win", f"${avg_win:.2f}")
        col2.metric("Avg Loss", f"${avg_loss:.2f}")
        col3.metric("Max Drawdown", f"{max_dd*100:.2f}%", delta_color="inverse")
        col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        st.markdown("---")
        
        # Performance chart
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.1,
            subplot_titles=('Cumulative P&L', 'Trade Distribution')
        )
        
        # Cumulative P&L
        fig.add_trace(
            go.Scatter(x=list(range(len(bt))), y=bt["CumPnL"], 
                      name="Cumulative P&L", fill='tozeroy',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Trade distribution histogram
        fig.add_trace(
            go.Histogram(x=bt["PnL"], name="Trade P&L Distribution",
                        marker_color='lightblue', nbinsx=30),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Trade Number", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
        fig.update_xaxes(title_text="P&L per Trade ($)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk interpretation
        st.markdown("### 📊 Risk Analysis")
        
        if sharpe > 1.5:
            st.success(f"✅ **Excellent Sharpe Ratio ({sharpe:.2f})** - Risk-adjusted returns are strong")
        elif sharpe > 1.0:
            st.info(f"ℹ️ **Good Sharpe Ratio ({sharpe:.2f})** - Reasonable risk-adjusted returns")
        elif sharpe > 0:
            st.warning(f"⚠️ **Low Sharpe Ratio ({sharpe:.2f})** - Returns may not justify the risk")
        else:
            st.error(f"❌ **Negative Sharpe Ratio ({sharpe:.2f})** - Strategy underperforming risk-free rate")
        
        if abs(max_dd) < 0.10:
            st.success(f"✅ **Low Drawdown ({max_dd*100:.1f}%)** - Capital preservation is good")
        elif abs(max_dd) < 0.20:
            st.warning(f"⚠️ **Moderate Drawdown ({max_dd*100:.1f}%)** - Some volatility present")
        else:
            st.error(f"❌ **High Drawdown ({max_dd*100:.1f}%)** - Significant capital at risk")
    else:
        st.warning("❌ Insufficient data for backtesting")

# =====================================================
# PORTFOLIO TAB
# =====================================================
with tab4:
    st.subheader("📁 Your Portfolio")
    
    if not st.session_state.portfolio:
        st.info("📭 No trades added yet. Go to the **Trade Setup** tab to add positions.")
    else:
        pf = pd.DataFrame(st.session_state.portfolio)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Positions", len(pf))
        col2.metric("Total Expected P&L", f"${pf['Expected PnL'].sum():,.2f}")
        col3.metric("Total Risk", f"${pf['Max Loss'].sum():,.2f}")
        col4.metric("Avg RRR", f"{pf['RRR'].mean():.2f}x")
        
        st.markdown("---")
        
        # Display portfolio table
        st.dataframe(
            pf.style.format({
                'Buy': '${:.2f}',
                'Sell': '${:.2f}',
                'Stop': '${:.2f}',
                'Expected PnL': '${:,.2f}',
                'Max Loss': '${:,.2f}',
                'RRR': '{:.2f}x'
            }),
            use_container_width=True,
            height=400
        )
        
        # Portfolio visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pf['Ticker'],
            y=pf['Expected PnL'],
            name='Expected P&L',
            marker_color='lightgreen'
        ))
        fig.add_trace(go.Bar(
            x=pf['Ticker'],
            y=pf['Max Loss'],
            name='Max Loss',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Portfolio Risk/Reward Breakdown",
            xaxis_title="Ticker",
            yaxis_title="Amount ($)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Clear portfolio button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Portfolio", use_container_width=True):
                st.session_state.portfolio = []
                st.rerun()

# =====================================================
# OPPORTUNITY TAB
# =====================================================
with tab5:
    st.subheader("🚀 Market Scanner - Best Opportunities")
    
    universe = ["AAPL","MSFT","NVDA","META","AMZN","GOOGL","TSLA","AMD","NFLX","AVGO",
                "INTC","ORCL","CSCO","ADBE","CRM","PYPL","UBER","ABNB","COIN","RBLX"]
    
    scan_button = st.button("🔍 Scan Market", type="primary", use_container_width=True)
    
    if scan_button:
        candidates = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, sym in enumerate(universe):
            status_text.text(f"Scanning {sym}... ({idx+1}/{len(universe)})")
            progress_bar.progress((idx + 1) / len(universe))
            
            df_try = load_stock(sym)
            if df_try.empty or len(df_try) < 60:
                continue
            
            try:
                price_t = df_try["Close"].iloc[-1]
                ma50 = df_try["Close"].rolling(50).mean().iloc[-1]
                
                # Only consider stocks above 50-day MA
                if price_t < ma50:
                    continue
                
                vol_t = annual_volatility(df_try)
                move_t = expected_move(price_t, vol_t, sim_days)
                buy_t = price_t - move_t
                sell_t = price_t + move_t
                stop_t = buy_t - move_t * 0.5
                
                prob_t = 1 - prob_hit_mc_advanced(price_t, buy_t, vol_t, sim_days, MONTE_CARLO_SIMS, mc_method, df_try)
                RRR_t = (sell_t - buy_t) / max(buy_t - stop_t, 0.01)
                EV = prob_t * (sell_t - buy_t)
                
                # Filter criteria
                if prob_t >= 0.55 and RRR_t >= 1.5:
                    candidates.append({
                        'EV': EV,
                        'Ticker': sym,
                        'Price': price_t,
                        'Buy': buy_t,
                        'Sell': sell_t,
                        'Stop': stop_t,
                        'RRR': RRR_t,
                        'Prob': prob_t,
                        'Vol': vol_t
                    })
            except Exception:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if not candidates:
            st.warning("❌ No qualified opportunities found in current market conditions.")
            st.info("💡 Try adjusting simulation days or wait for better market setups.")
        else:
            # Sort by expected value
            candidates_df = pd.DataFrame(candidates).sort_values('EV', ascending=False)
            
            st.success(f"✅ Found {len(candidates_df)} opportunities!")
            
            # Display top 3
            st.markdown("### 🏆 Top 3 Opportunities")
            
            for i, row in candidates_df.head(3).iterrows():
                with st.expander(f"#{candidates_df.index.get_loc(i)+1} - {row['Ticker']} (EV: ${row['EV']:.2f})", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Price", f"${row['Price']:.2f}")
                    col2.metric("Buy Target", f"${row['Buy']:.2f}")
                    col3.metric("Sell Target", f"${row['Sell']:.2f}")
                    col4.metric("Stop Loss", f"${row['Stop']:.2f}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Probability", f"{row['Prob']*100:.1f}%")
                    col2.metric("Risk-Reward", f"{row['RRR']:.2f}x")
                    col3.metric("Volatility", f"{row['Vol']*100:.1f}%")
            
            st.markdown("---")
            st.markdown("### 📋 All Opportunities")
            
            # Display full table
            st.dataframe(
                candidates_df[['Ticker', 'Price', 'Buy', 'Sell', 'RRR', 'Prob', 'EV']].style.format({
                    'Price': '${:.2f}',
                    'Buy': '${:.2f}',
                    'Sell': '${:.2f}',
                    'RRR': '{:.2f}x',
                    'Prob': '{:.1%}',
                    'EV': '${:.2f}'
                }),
                use_container_width=True
            )
