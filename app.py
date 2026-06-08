import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. PAGE & THEME CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Institutional Trading & IB Hub",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI typography
st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef; }
    .stHeading h1 { color: #1E3A8A; }
    .stHeading h2 { color: #0F766E; }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. DATABASE INITIALIZATION (PART B)
# ==========================================
def init_db():
    """Creates a local SQLite database and table if they do not exist."""
    conn = sqlite3.connect("deals.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mna_deals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target TEXT NOT NULL,
            sector TEXT,
            value_m REAL,
            ev_ebitda REAL,
            stage TEXT
        )
    """)
    # Insert mock seed data if the database table is completely brand new
    cursor.execute("SELECT COUNT(*) FROM mna_deals")
    if cursor.fetchone()[0] == 0:
        mock_data = [
            ("Acme Software", "Technology", 450.0, 14.2, "Due Diligence"),
            ("Nexus Pharma", "Healthcare", 1200.0, 19.5, "LOI Signed"),
            ("Apex Logistics", "Industrials", 850.0, 9.8, "Pitching")
        ]
        cursor.executemany("INSERT INTO mna_deals (target, sector, value_m, ev_ebitda, stage) VALUES (?, ?, ?, ?, ?)", mock_data)
        conn.commit()
    conn.close()

def load_deals():
    """Reads data records directly from the SQLite database."""
    conn = sqlite3.connect("deals.db")
    df = pd.read_sql_query("SELECT target, sector, value_m, ev_ebitda, stage FROM mna_deals", conn)
    conn.close()
    return df

# Trigger the database setup on app initialization
init_db()


# ==========================================
# 3. MACHINE LEARNING ENGINE (PART A)
# ==========================================
def predict_next_close(price_series):
    """Uses simple linear regression on rolling features to predict tomorrow's price."""
    df_ml = pd.DataFrame(price_series)
    df_ml.columns = ['Close']
    
    # Feature Engineering
    df_ml['MA_5'] = df_ml['Close'].rolling(window=5).mean()
    df_ml['MA_20'] = df_ml['Close'].rolling(window=20).mean()
    df_ml['Target'] = df_ml['Close'].shift(-1) # Shift target to represent tomorrow
    
    df_ml = df_ml.dropna()
    
    if len(df_ml) < 30:
        return None # Return None if historical window is too short to train
        
    X = df_ml[['MA_5', 'MA_20']]
    y = df_ml['Target']
    
    # Fit Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Run dynamic inference for the immediate upcoming session
    latest_features = np.array([[price_series.iloc[-5:].mean(), price_series.iloc[-20:].mean()]])
    prediction = model.predict(latest_features)[0]
    
    return prediction


# ==========================================
# 4. DATA PIPELINE (MARKET DATA)
# ==========================================
@st.cache_data(ttl=1800) # Auto-refresh market data from APIs every 30 minutes
def fetch_market_data(tickers, period="1y"):
    """Fetches historical market data and calculates daily returns."""
    data = yf.download(tickers, period=period)['Close']
    returns = data.pct_change().dropna()
    return data, returns

@st.cache_data
def calculate_portfolio_metrics(returns, weights):
    """Computes basic annualized metrics and Sharpe Ratio risk adjustment."""
    mean_returns = returns.mean() * 252
    port_return = np.sum(mean_returns * weights)
    
    cov_matrix = returns.cov() * 252
    port_variance = np.dot(weights, np.dot(cov_matrix, weights))
    port_volatility = np.sqrt(port_variance)
    
    risk_free_rate = 0.05 # 5% benchmark
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility
    
    return port_return, port_volatility, sharpe_ratio


# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🏛️ Execution Desk")
app_mode = st.sidebar.selectbox("Choose Workspace", ["Market Trading Analytics", "Investment Banking Deal Room"])


# ==========================================
# 6. MODULE 1: MARKET TRADING ANALYTICS
# ==========================================
if app_mode == "Market Trading Analytics":
    st.title("📈 Quantitative Trading & Portfolio Risk Desk")
    st.markdown("Real-time portfolio optimization, volatility tracking, and machine learning inferences.")
    
    st.sidebar.subheader("Portfolio Configuration")
    selected_tickers = st.sidebar.multiselect("Select Portfolio Assets", ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], default=["AAPL", "MSFT", "NVDA"])
    
    if len(selected_tickers) < 2:
        st.warning("Please select at least 2 assets to evaluate portfolio risk metrics.")
    else:
        prices, returns = fetch_market_data(selected_tickers)
        weights = np.array([1/len(selected_tickers)] * len(selected_tickers))
        p_return, p_vol, p_sharpe = calculate_portfolio_metrics(returns, weights)
        
        # Performance KPI Blocks
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Expected Annual Return", f"{p_return*100:.2f}%")
        m_col2.metric("Portfolio Volatility (σ)", f"{p_vol*100:.2f}%")
        m_col3.metric("Portfolio Sharpe Ratio", f"{p_sharpe:.2f}")
        
        st.markdown("---")
        
        # Machine Learning Forecasting UI Action
        st.subheader("🔮 Predictive Machine Learning Engine")
        if st.button("Run Predictive Model (Linear Regression)"):
            ml_cols = st.columns(len(selected_tickers))
            for idx, ticker in enumerate(selected_tickers):
                predicted_val = predict_next_close(prices[ticker])
                current_val = prices[ticker].iloc[-1]
                
                if predicted_val:
                    direction = "🔼 Upward Trend" if predicted_val > current_val else "🔽 Downward Trend"
                    ml_cols[idx].metric(
                        label=f"{ticker} Forecasted Close", 
                        value=f"${predicted_val:.2f}", 
                        delta=direction
                    )
        
        st.markdown("---")
        
        # Interactive Plotly Data Visualizations
        chart_col1, chart_col2 = st.columns([2, 1])
        with chart_col1:
            st.subheader("Normalized Asset Performance (Base 100)")
            normalized_prices = (prices / prices.iloc[0]) * 100
            fig_perf = px.line(normalized_prices, labels={"value": "Index Value", "Date": "Date"})
            fig_perf.update_layout(template="plotly_white")
            st.plotly_chart(fig_perf, use_container_width=True)
            
        with chart_col2:
            st.subheader("Asset Allocation Summary")
            fig_pie = go.Figure(data=[go.Pie(labels=selected_tickers, values=weights, hole=.4)])
            fig_pie.update_layout(template="plotly_white", margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)


# ==========================================
# 7. MODULE 2: INVESTMENT BANKING DEAL ROOM
# ==========================================
else:
    st.title("💼 Investment Banking M&A Advisory Desk")
    st.markdown("Automated corporate valuation, peer multi-comparison, and persistent SQL database pipelines.")
    
    ib_col1, ib_col2 = st.columns([1, 2])
    
    with ib_col1:
        st.subheader("➕ Log New Transaction")
        with st.form("new_deal_form", clear_on_submit=True):
            target_name = st.text_input("Target Company Name")
            sector = st.selectbox("Industry Sector", ["Technology", "Healthcare", "Industrials", "Consumer", "Energy"])
            deal_val = st.number_input("Enterprise Value ($ Millions)", min_value=1, value=100)
            multiple = st.number_input("Implied EV / EBITDA Multiple", min_value=1.0, max_value=50.0, value=12.5, step=0.1)
            stage = st.selectbox("Advisory Stage", ["Pitching", "Due Diligence", "LOI Signed", "Closed"])
            
            submit_deal = st.form_submit_button("Commit Deal to Database")
            
            # Persistent Write Action directly to SQLite
            if submit_deal and target_name:
                conn = sqlite3.connect("deals.db")
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO mna_deals (target, sector, value_m, ev_ebitda, stage) VALUES (?, ?, ?, ?, ?)",
                    (target_name, sector, deal_val, multiple, stage)
                )
                conn.commit()
                conn.close()
                st.success(f"Successfully committed {target_name} to database file!")
                st.rerun()

    with ib_col2:
        st.subheader("📂 Active Corporate Finance Pipeline (Fetched from SQL)")
        # Load the newly updated database context frame
        df_deals = load_deals()
        st.dataframe(df_deals, use_container_width=True, hide_index=True)
        
        st.subheader("📊 Sector Valuation Multiples Benchmarking (EV/EBITDA)")
        fig_bar = px.bar(
            df_deals, 
            x="target", 
            y="ev_ebitda", 
            color="sector", 
            text_auto=True,
            labels={"ev_ebitda": "EV / EBITDA Multiple", "target": "Target Asset"}
        )
        fig_bar.update_layout(template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)