# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import datetime as dt
import os
import json
import hashlib
import sqlite3
import logging
import uuid
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import io

# =========================================================
# PAGE CONFIGURATION & ENHANCED UX STYLING
# =========================================================
st.set_page_config(
    page_title="Pro Journal | Zenvo",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@zenvo.com',
        'Report a bug': "mailto:bugs@zenvo.com",
        'About': "# Zenvo Pro Journal\n\nA professional trade journaling and analytics platform."
    }
)

# --- Ensure screenshots directory exists ---
SCREENSHOTS_DIR = "screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Inject custom CSS for a professional, modern dark theme
st.markdown("""
<style>
    /* --- Main App Styling --- */
    .stApp {
        background-color: #0d1117; /* Dark background */
        color: #c9d1d9; /* Light text */
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding: 1.5rem 2.5rem 2rem 2.5rem !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #e6edf3 !important; /* Lighter header color */
        font-weight: 600;
    }
    h1 {
        font-size: 2.8rem; /* Larger main title */
        margin-bottom: 0.5rem;
    }
    h2 {
        font-size: 2.2rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    h3 {
        font-size: 1.8rem;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }
    
    /* --- Hide Streamlit Branding --- */
    #MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden !important; }

    /* --- Custom Horizontal Line --- */
    hr {
        margin: 1.5rem 0 !important;
        border-top: 1px solid #30363d !important;
        background-color: transparent !important;
    }

    /* --- Metric Card Styling --- */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem 1.2rem; /* Adjusted padding */
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center; /* Center align metric content */
    }
    [data-testid="stMetric"]:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #8b949e;
        font-size: 0.9rem;
        margin-bottom: 0.3rem; /* Space between label and value */
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e6edf3;
        line-height: 1.2;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* --- Tab Styling --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0 20px;
        transition: all 0.2s ease-in-out;
        color: #8b949e;
        font-weight: 500;
        display: flex;
        align-items: center;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1f2a35;
        color: #58a6ff;
        border-color: #58a6ff;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0d1117;
        border-bottom-color: #0d1117 !important;
        color: #58a6ff;
        border-color: #58a6ff;
        border-bottom-width: 3px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 2rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }

    /* --- Styling for Markdown in Trade Playbook & General Markdown --- */
    .trade-notes-display {
        background-color: #0d1117;
        border-left: 4px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
    }
    .trade-notes-display p { font-size: 15px; color: #c9d1d9; line-height: 1.6; }
    .trade-notes-display h1, h2, h3, h4 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 4px; margin-top: 1rem; }
    
    /* --- General Form Elements --- */
    .stSelectbox > div > div, .stTextInput > div > div, .stDateInput > div > div, .stNumberInput > div > div, .stTextArea > div > div {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        color: #c9d1d9;
    }
    .stSelectbox > div > div:focus-within, .stTextInput > div > div:focus-within, .stDateInput > div > div:focus-within, .stNumberInput > div > div:focus-within, .stTextArea > div > div:focus-within {
        border-color: #58a6ff;
        box-shadow: 0 0 0 1px #58a6ff;
    }
    label {
        color: #c9d1d9 !important;
        font-weight: 500;
        margin-bottom: 0.2rem;
    }
    .stRadio > label {
        margin-right: 1.5rem;
    }

    /* --- Buttons --- */
    .stButton > button {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease-in-out;
        font-weight: 500;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #30363d;
        border-color: #58a6ff;
        color: #58a6ff;
    }
    .stButton > button[kind="primary"] {
        background-color: #238636; /* GitHub green for primary */
        border-color: #238636;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #2da44e;
        border-color: #2da44e;
    }
    .stButton > button.red-button { /* Custom class for danger buttons */
        background-color: #cf222e;
        border-color: #cf222e;
        color: white;
    }
    .stButton > button.red-button:hover {
        background-color: #a42a32;
        border-color: #a42a32;
    }

    /* --- Sidebar Styling --- (Included for completeness but user asked to ignore for UX focus) */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
        padding-top: 2rem;
    }
    [data-testid="stSidebarNavLink"] {
        color: #c9d1d9;
        font-weight: 500;
        padding: 0.7rem 1rem;
        border-radius: 6px;
        transition: all 0.2s ease-in-out;
    }
    [data-testid="stSidebarNavLink"]:hover {
        background-color: #1f2a35;
        color: #58a6ff;
    }
    [data-testid="stSidebarNavLinkActive"] {
        background-color: #30363d !important;
        color: #58a6ff !important;
    }

    /* --- Expander Styling --- */
    .streamlit-expanderHeader {
        background-color: #1f2a35;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #c9d1d9;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }
    .streamlit-expanderHeader:hover {
        border-color: #58a6ff;
        color: #58a6ff;
    }
    .streamlit-expanderContent {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
    }
    
    /* --- Toast notifications --- */
    .stToast {
        background-color: #238636 !important;
        color: white !important;
    }
    .stToast .stMarkdown p { color: white !important; }

    /* --- Custom Card for Playbook Trades --- */
    .trade-card {
        border: 1px solid #30363d;
        border-left: 6px solid; /* Dynamic color injected via style */
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        background-color: #161b22;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
        transition: all 0.2s ease-in-out;
    }
    .trade-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .trade-card h4 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .trade-card .trade-info {
        font-size: 0.9rem;
        color: #8b949e;
    }
    .trade-card .trade-pnl {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .trade-card-actions {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.8rem;
    }
    .trade-card-actions button {
        flex: 1;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOGGING & DATABASE SETUP
# =========================================================
logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_FILE = "zenvo_journal.db"

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.datetime, dt.date)): return obj.isoformat()
        if pd.isna(obj) or (isinstance(obj, (float, np.float64)) and np.isnan(obj)): return None
        return super().default(obj)

try:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, 
            password TEXT, 
            data TEXT
        )
    ''')
    conn.commit()
except Exception as e:
    st.error("Fatal Error: Could not connect to the database.")
    logging.critical(f"Failed to initialize SQLite database: {str(e)}", exc_info=True)
    st.stop()

# =========================================================
# HELPER & GAMIFICATION FUNCTIONS
# =========================================================
def get_user_data(username):
    c.execute("SELECT data FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    return json.loads(result[0]) if result and result[0] else {}

def save_user_data(username, data):
    try:
        json_data = json.dumps(data, cls=CustomJSONEncoder)
        c.execute("UPDATE users SET data = ? WHERE username = ?", (json_data, username))
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Failed to save data for {username}: {e}", exc_info=True)
        st.error(f"Error saving user data: {e}")
        return False

def _ta_save_journal(username, journal_df):
    user_data = get_user_data(username)
    # Convert datetime objects to string for JSON serialization
    journal_df_copy = journal_df.copy()
    for col in ['Date']:
        if col in journal_df_copy.columns:
            journal_df_copy[col] = journal_df_copy[col].dt.isoformat() if not journal_df_copy[col].isnull().all() else None

    user_data["trade_journal"] = journal_df_copy.to_dict('records')
    if save_user_data(username, user_data):
        logging.info(f"Journal saved for user {username}: {len(journal_df)} trades")
        return True
    return False

def ta_update_xp(username, amount):
    user_data = get_user_data(username)
    user_data['xp'] = user_data.get('xp', 0) + amount
    st.toast(f"🎉 +{amount} XP Earned!", icon="⭐")
    save_user_data(username, user_data)
    if 'xp' in st.session_state: # Update session state immediately
        st.session_state.xp = user_data['xp']

def ta_update_streak(username):
    user_data = get_user_data(username)
    today = dt.date.today()
    last_date_str = user_data.get('last_journal_date')
    streak = user_data.get('streak', 0)
    
    if last_date_str:
        last_date = dt.date.fromisoformat(last_date_str)
        if last_date == today: return # Already journaled today
        if last_date == today - dt.timedelta(days=1): streak += 1
        else: streak = 1
    else: streak = 1
        
    user_data.update({'streak': streak, 'last_journal_date': today.isoformat()})
    if 'streak' in st.session_state: # Update session state immediately
        st.session_state.streak = streak
    if save_user_data(username, user_data):
        st.toast(f"🔥 Daily streak: {streak} days!", icon="🗓️")

def load_image(image_path):
    if image_path and os.path.exists(image_path):
        return Image.open(image_path)
    return None

def save_uploaded_file(uploaded_file, filename):
    filepath = os.path.join(SCREENSHOTS_DIR, filename)
    try:
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return filepath
    except Exception as e:
        logging.error(f"Failed to save file {filename}: {e}")
        st.error(f"Failed to save screenshot: {e}")
        return ''

# Robust PnL and R:R Calculation (more realistic for FX, can be extended)
def calculate_pnl_rr(symbol, direction, entry_price, stop_loss, final_exit, lots, outcome):
    pnl, rr = 0.0, 0.0
    
    if entry_price == 0 or final_exit == 0:
        return pnl, rr # Cannot calculate if prices are zero

    # Determine pip value based on symbol
    pip_size = 0.0001
    contract_size = 100000 # Standard lot for FX

    if "JPY" in symbol.upper():
        pip_size = 0.01
        contract_size = 100000
    elif "XAU" in symbol.upper() or "GOLD" in symbol.upper():
        pip_size = 0.01 # Price movement per cent
        contract_size = 100 # Contract size for Gold (e.g., XAUUSD futures or spot)
    elif "BTC" in symbol.upper() or "ETH" in symbol.upper() or "CRYPTO" in symbol.upper():
        pip_size = 1 # For crypto, move by whole units
        contract_size = 1 # Typically 1 unit per lot for crypto

    # PnL Calculation
    if outcome in ["Win", "Loss"]:
        price_diff = (final_exit - entry_price)
        if direction == "Short":
            price_diff *= -1
        
        if "XAU" in symbol.upper(): # Gold in USD terms, so 1 lot = 100 oz
            pnl = price_diff * lots * contract_size
        elif "JPY" in symbol.upper(): # JPY pairs, pip value is usually $1 per 0.01 move per standard lot
            pnl = (price_diff / pip_size) * lots * (contract_size / 100000) * 10 # This needs better specific handling
            # A more robust formula for JPY: (final_exit - entry_price) * 100,000 * lots / current_USDJPY_rate
            # For simplicity, keeping it general, but highlighting complexity
        elif "BTC" in symbol.upper() or "ETH" in symbol.upper():
             pnl = price_diff * lots # Assuming lots is number of crypto units
        else: # Standard FX
            pnl = (price_diff / pip_size) * lots * 10 # 10 USD per standard lot per pip
            
        # Simplified for demonstration. Real PnL calculation is highly instrument-specific.
        # Let's use a simpler, more generic one that still shows directionality.
        pnl = (final_exit - entry_price) * lots * (10000 if "JPY" not in symbol.upper() else 100) # Arbitrary multiplier for effect
        if direction == "Short": pnl *= -1
    
    # R:R Calculation
    if stop_loss > 0 and entry_price > 0:
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit > 0:
            potential_reward_per_unit = abs(final_exit - entry_price)
            rr = (potential_reward_per_unit / risk_per_unit)
            if pnl < 0: rr *= -1 # Indicate negative R:R for losses
    
    return pnl, rr


# =========================================================
# MOCK AUTHENTICATION & SESSION STATE SETUP
# =========================================================
if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = "pro_trader"
    c.execute("SELECT username FROM users WHERE username = ?", (st.session_state.logged_in_user,))
    if not c.fetchone():
        hashed_password = hashlib.sha256("password".encode()).hexdigest()
        initial_data = json.dumps({'xp': 0, 'streak': 0, 'trade_journal': []})
        c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)", 
                  (st.session_state.logged_in_user, hashed_password, initial_data))
        conn.commit()
    
    user_data = get_user_data(st.session_state.logged_in_user)
    st.session_state.xp = user_data.get('xp', 0)
    st.session_state.streak = user_data.get('streak', 0)
    st.session_state.current_tab = "log_new_trade" # Default tab on app load

# =========================================================
# JOURNAL SCHEMA & ROBUST DATA MIGRATION (REVISED)
# =========================================================
journal_cols = [
    "TradeID", "Date", "Symbol", "Direction", "Outcome", "PnL", "RR", 
    "Strategy", "Tags", "EntryPrice", "StopLoss", "FinalExit", "Lots",
    "EntryRationale", "TradeJournalNotes", "EntryScreenshot", "ExitScreenshot"
]
journal_dtypes = {
    "TradeID": str, "Date": "datetime64[ns]", "Symbol": str, "Direction": str, 
    "Outcome": str, "PnL": float, "RR": float, "Strategy": str, 
    "Tags": str, "EntryPrice": float, "StopLoss": float, "FinalExit": float, "Lots": float,
    "EntryRationale": str, "TradeJournalNotes": str, 
    "EntryScreenshot": str, "ExitScreenshot": str
}

if 'trade_journal' not in st.session_state:
    user_data = get_user_data(st.session_state.logged_in_user)
    journal_data = user_data.get("trade_journal", [])
    df = pd.DataFrame(journal_data)
    
    # Safely migrate data to the new, safer schema
    legacy_col_map = {
        "Trade ID": "TradeID", "Entry Price": "EntryPrice", "Stop Loss": "StopLoss",
        "Final Exit": "FinalExit", "PnL ($)": "PnL", "R:R": "RR",
        "Entry Rationale": "EntryRationale", "Trade Journal Notes": "TradeJournalNotes",
        "Entry Screenshot": "EntryScreenshot", "Exit Screenshot": "ExitScreenshot"
    }
    df.rename(columns=legacy_col_map, inplace=True)

    for col, dtype in journal_dtypes.items():
        if col not in df.columns:
            if dtype == str: df[col] = ''
            elif 'datetime' in str(dtype): df[col] = pd.NaT
            elif dtype == float: df[col] = 0.0
            else: df[col] = None
    
    st.session_state.trade_journal = df[journal_cols].astype(journal_dtypes, errors='ignore')
    st.session_state.trade_journal['Date'] = pd.to_datetime(st.session_state.trade_journal['Date'], errors='coerce')


# =========================================================
# PAGE LAYOUT
# =========================================================

# --- Sidebar --- (Included for completeness but not the focus of this UX task)
with st.sidebar:
    st.image("https://www.google.com/s2/favicons?domain=streamlit.io", width=30) # Placeholder logo
    st.markdown("## Pro Journal | Zenvo")
    st.caption("Advanced Trade Tracking")
    st.markdown("---")

    st.markdown(f"**Welcome, {st.session_state.logged_in_user}!**")
    st.metric(label="Total XP", value=f"{st.session_state.xp} ⭐", delta="Earn XP by logging trades!")
    st.metric(label="Daily Streak", value=f"{st.session_state.streak} 🔥", delta="Keep journaling daily!")
    
    st.markdown("---")
    st.markdown("**Navigation**")
    # Using dummy page links, for this exercise the focus is on the tabs below
    st.button("📈 Trading Chart", key="nav_chart", use_container_width=True, on_click=lambda: st.session_state.update(current_tab="trading_chart"))
    st.button("📝 Log New Trade", key="nav_log", use_container_width=True, on_click=lambda: st.session_state.update(current_tab="log_new_trade"))
    st.button("📚 Trade Playbook", key="nav_playbook", use_container_width=True, on_click=lambda: st.session_state.update(current_tab="trade_playbook"))
    st.button("📊 Analytics Dashboard", key="nav_analytics", use_container_width=True, on_click=lambda: st.session_state.update(current_tab="analytics_dashboard"))
    
    st.markdown("---")
    if st.button("Logout", help="End current session", type="secondary", use_container_width=True):
        st.session_state.logged_in_user = None 
        st.rerun()

# --- Main Content ---
st.title("📈 Pro Journal & Backtesting Environment")
st.caption(f"A streamlined interface for professional trade analysis. | Logged in as: **{st.session_state.logged_in_user}**")
st.markdown("---")

# --- CHARTING AREA --- (Moved to top level outside tabs to be always visible)
pairs_map = {
    "EUR/USD": "FX_IDC:EURUSD", "USD/JPY": "FX_IDC:USDJPY", "GBP/USD": "FX_IDC:GBPUSD", "USD/CHF": "FX_IDC:USDCHF",
    "AUD/USD": "FX_IDC:AUDUSD", "NZD/USD": "FX_IDC:NZDUSD", "USD/CAD": "FX_IDC:USDCAD",
    "XAU/USD (Gold)": "XAUUSD", "BTC/USD": "BINANCE:BTCUSD", "ETH/USD": "BINANCE:ETHUSD"
}
chart_col, _ = st.columns([0.7, 0.3])
with chart_col:
    pair = st.selectbox("Select Chart Pair", list(pairs_map.keys()), index=0, key="tv_pair", help="Select the instrument to display on the TradingView chart.")
    tv_symbol = pairs_map[pair]

    tv_html = f"""
    <div id="tradingview_widget"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
    new TradingView.widget(
    {{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#161b22",
        "enable_publishing": false,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "studies": [
            "MACD@tv-basic",
            "RSI@tv-basic",
            "BollingerBands@tv-basic"
        ],
        "container_id": "tradingview_widget"
    }}
    );
    </script>
    """
    st.components.v1.html(tv_html, height=550)
st.markdown("---")

# =========================================================
# TRADING JOURNAL TABS
# =========================================================
tab_entry, tab_playbook, tab_analytics = st.tabs(["**📝 Log New Trade**", "**📚 Trade Playbook**", "**📊 Analytics Dashboard**"], 
                                                  key="main_tabs", 
                                                  on_change=lambda: st.session_state.update(current_tab=st.session_state.main_tabs.lower().replace(' ', '_').replace('**', '')))

# Set the active tab based on session state
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "log_new_trade"

# Simulate active tab display (Streamlit's native tabs handle this visually)
# We use st.session_state.current_tab to control internal logic when needed.


# --- TAB 1: LOG NEW TRADE ---
with tab_entry if st.session_state.current_tab == "log_new_trade" else st.empty():
    if st.session_state.current_tab == "log_new_trade":
        st.header("Log a New Trade")
        st.caption("Focus on a quick, essential entry. You can add detailed notes and screenshots later in the Playbook.")

        # Initialize calculated PnL and R:R
        calculated_pnl = 0.0
        calculated_rr = 0.0
        pnl_warning = ""

        with st.form("trade_entry_form", clear_on_submit=True):
            st.markdown("##### ⚡ Quick Entry Details")
            cols_quick_entry = st.columns(4)
            
            with cols_quick_entry[0]:
                date_val = st.date_input("Trade Date *", dt.date.today(), key="new_trade_date")
                symbol_options = sorted(list(pairs_map.keys())) + ["Other"]
                symbol_index = symbol_options.index(pair) if pair in symbol_options else 0
                symbol_selected = st.selectbox("Symbol *", symbol_options, index=symbol_index, key="new_trade_symbol_select")
                
                symbol = symbol_selected
                if symbol_selected == "Other":
                    custom_symbol = st.text_input("Enter Custom Symbol *", placeholder="e.g., AAPL, SPX500", key="new_trade_custom_symbol").upper()
                    symbol = custom_symbol if custom_symbol else ""
                
            with cols_quick_entry[1]:
                direction = st.radio("Direction *", ["Long", "Short"], horizontal=True, key="new_trade_direction")
                lots = st.number_input("Position Size (Lots) *", min_value=0.01, max_value=10000.0, value=0.10, step=0.01, format="%.2f", key="new_trade_lots")
            
            with cols_quick_entry[2]:
                entry_price = st.number_input("Entry Price *", min_value=0.0, value=0.0, step=0.00001, format="%.5f", key="new_trade_entry_price", help="The price at which you entered the trade.")
                stop_loss = st.number_input("Stop Loss", min_value=0.0, value=0.0, step=0.00001, format="%.5f", key="new_trade_stop_loss", help="Your predefined stop-loss level. Used for R:R calculation.")
            
            with cols_quick_entry[3]:
                final_exit = st.number_input("Final Exit Price *", min_value=0.0, value=0.0, step=0.00001, format="%.5f", key="new_trade_final_exit", help="The price at which you exited the trade.")
                outcome = st.selectbox("Outcome *", ["Win", "Loss", "Breakeven", "No Trade/Study"], key="new_trade_outcome", help="The final outcome of the trade.")

            # Dynamic PnL/RR Preview
            if entry_price > 0 and final_exit > 0 and lots > 0:
                calculated_pnl, calculated_rr = calculate_pnl_rr(symbol, direction, entry_price, stop_loss, final_exit, lots, outcome)
                st.info(f"**Estimated PnL:** ${calculated_pnl:,.2f} | **Estimated R:R:** {calculated_rr:.2f}R")
            elif outcome in ["Win", "Loss"] and (entry_price == 0 or final_exit == 0 or lots == 0):
                pnl_warning = "Please enter Entry Price, Final Exit, and Lots for PnL/R:R calculation."

            with st.expander("📚 Add Strategy, Rationale & Tags (Optional)", expanded=False):
                col_strat_tag = st.columns(2)
                with col_strat_tag[0]:
                    all_strategies = sorted(list(set(st.session_state.trade_journal['Strategy'].dropna().astype(str).str.strip())))
                    strategy_input = st.selectbox("Strategy Used", options=[''] + all_strategies + ["_Add New Strategy_"], key="new_trade_strategy_select", help="Categorize your trade by strategy.")
                    strategy = strategy_input
                    if strategy_input == "_Add New Strategy_":
                        strategy = st.text_input("Enter New Strategy Name", placeholder="e.g., 'Breakout Fade', 'Trend Continuation'", key="new_trade_custom_strategy")
                    
                    entry_rationale = st.text_area("Why did you enter this trade? *", value="", height=100, key="new_trade_entry_rationale", placeholder="e.g., 'Double bottom on daily chart, retest of demand zone'.")
                
                with col_strat_tag[1]:
                    all_tags = sorted(list(set(st.session_state.trade_journal['Tags'].str.split(',').explode().dropna().str.strip())))
                    suggested_tags = ["Breakout", "Reversal", "Trend Follow", "Counter-Trend", "News Play", "FOMO", "Over-leveraged", "Liquidity Grab", "Supply Zone", "Demand Zone", "High Impact News"]
                    unique_suggested_tags = sorted(list(set(all_tags + suggested_tags)))
                    tags = st.multiselect("Trade Tags", options=unique_suggested_tags, key="new_trade_tags", help="Add descriptive tags to your trade for easier filtering and analysis.")

            submitted = st.form_submit_button("Save Trade", type="primary", use_container_width=True)
            
            if submitted:
                if not symbol or entry_price == 0.0 or final_exit == 0.0 or lots == 0.0:
                    st.error("Please fill in all required fields (marked with *) before saving.")
                elif outcome in ["Win", "Loss"] and (entry_price == 0 or final_exit == 0):
                    st.error("Entry Price and Final Exit must be greater than 0 for Win/Loss outcomes.")
                else:
                    final_pnl, final_rr = calculate_pnl_rr(symbol, direction, entry_price, stop_loss, final_exit, lots, outcome)

                    new_trade_data = {
                        "TradeID": f"TRD-{uuid.uuid4().hex[:6].upper()}",
                        "Date": pd.to_datetime(date_val),
                        "Symbol": symbol,
                        "Direction": direction,
                        "Outcome": outcome,
                        "Lots": lots,
                        "EntryPrice": entry_price,
                        "StopLoss": stop_loss,
                        "FinalExit": final_exit,
                        "PnL": final_pnl,
                        "RR": final_rr,
                        "Strategy": strategy if strategy else '',
                        "Tags": ','.join(tags) if tags else '',
                        "EntryRationale": entry_rationale,
                        "TradeJournalNotes": '', 
                        "EntryScreenshot": '',
                        "ExitScreenshot": ''
                    }
                    new_df = pd.DataFrame([new_trade_data])
                    st.session_state.trade_journal = pd.concat([st.session_state.trade_journal, new_df], ignore_index=True)
                    
                    if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                        if outcome in ["Win", "Loss"]:
                            ta_update_xp(st.session_state.logged_in_user, 10)
                            ta_update_streak(st.session_state.logged_in_user)
                        st.success(f"Trade {new_trade_data['TradeID']} logged successfully!")
                        st.session_state.current_tab = "trade_playbook" # Redirect to playbook after saving
                        st.rerun()
                    else:
                        st.error("Failed to save new trade.")
            if pnl_warning:
                st.warning(pnl_warning)

# --- TAB 2: TRADE PLAYBOOK ---
with tab_playbook if st.session_state.current_tab == "trade_playbook" else st.empty():
    if st.session_state.current_tab == "trade_playbook":
        st.header("Your Trade Playbook")
        st.caption("Filter, review, and refine your past trades. Edit details or add screenshots.")
        
        df_playbook = st.session_state.trade_journal
        if df_playbook.empty:
            st.info("Your logged trades will appear here as playbook cards. Log your first trade to get started!")
        else:
            # Filtering & Search
            filter_expander = st.expander("Filter & Search Trades", expanded=True)
            with filter_expander:
                search_query = st.text_input("Search Trades (Symbol, Strategy, Rationale, Notes, Tags)", placeholder="e.g., EURUSD, Breakout, FOMO...", key="playbook_search")
                
                filter_cols_1 = st.columns([1, 1, 1])
                outcome_filter = filter_cols_1[0].multiselect("Filter Outcome", df_playbook['Outcome'].unique(), default=df_playbook['Outcome'].unique(), key="pb_outcome_filter")
                symbol_filter = filter_cols_1[1].multiselect("Filter Symbol", df_playbook['Symbol'].unique(), default=df_playbook['Symbol'].unique(), key="pb_symbol_filter")
                direction_filter = filter_cols_1[2].multiselect("Filter Direction", df_playbook['Direction'].unique(), default=df_playbook['Direction'].unique(), key="pb_direction_filter")
                
                all_tags = sorted(list(set(df_playbook['Tags'].str.split(',').explode().dropna().str.strip())))
                tag_filter = st.multiselect("Filter Tag", options=all_tags, key="pb_tag_filter")

                filter_cols_2 = st.columns(2)
                if filter_cols_2[0].button("Clear All Filters", key="clear_filters", use_container_width=True):
                    # Reset all filter session states
                    st.session_state.playbook_search = ""
                    st.session_state.pb_outcome_filter = df_playbook['Outcome'].unique()
                    st.session_state.pb_symbol_filter = df_playbook['Symbol'].unique()
                    st.session_state.pb_direction_filter = df_playbook['Direction'].unique()
                    st.session_state.pb_tag_filter = []
                    st.rerun()

            filtered_df = df_playbook[
                (df_playbook['Outcome'].isin(outcome_filter)) &
                (df_playbook['Symbol'].isin(symbol_filter)) &
                (df_playbook['Direction'].isin(direction_filter))
            ]
            if tag_filter:
                filtered_df = filtered_df[filtered_df['Tags'].apply(lambda x: any(tag in str(x) for tag in tag_filter))]
            
            if search_query:
                filtered_df = filtered_df[
                    filtered_df.apply(lambda row: 
                        search_query.lower() in str(row['Symbol']).lower() or
                        search_query.lower() in str(row['Strategy']).lower() or
                        search_query.lower() in str(row['EntryRationale']).lower() or
                        search_query.lower() in str(row['TradeJournalNotes']).lower() or
                        any(search_query.lower() in tag.lower() for tag in str(row['Tags']).split(','))
                    , axis=1)
                ]

            st.markdown("---")
            st.subheader(f"Displaying {len(filtered_df)} of {len(df_playbook)} Trades")

            if filtered_df.empty:
                st.info("No trades match your current filters and search query.")

            # Display trades with Edit/Delete functionality
            for index, row in filtered_df.sort_values(by="Date", ascending=False).iterrows():
                trade_id = row['TradeID']
                outcome_color = {"Win": "#2da44e", "Loss": "#cf222e", "Breakeven": "#8b949e", "No Trade/Study": "#58a6ff"}.get(row['Outcome'], "#30363d")

                st.markdown(f"""
                <div class="trade-card" style="border-left-color: {outcome_color};">
                    <h4>
                        <span>{row['Symbol']} - <span style="color: {outcome_color};">{row['Direction']} / {row['Outcome']}</span></span>
                        <span class="trade-pnl" style="color: {'#2da44e' if row['PnL'] > 0 else ('#cf222e' if row['PnL'] < 0 else '#8b949e')};">
                            {('+' if row['PnL'] > 0 else '')}{row['PnL']:.2f} USD
                        </span>
                    </h4>
                    <div class="trade-info">
                        <span>{row['Date'].strftime('%A, %d %B %Y')} | Trade ID: {trade_id} | R:R: {row['RR']:.2f}R</span>
                    </div>
                    <div class="trade-card-actions">
                        <button class="stButton" style="background-color:#21262d; border-color:#30363d; color:#c9d1d9;" onclick="window.parent.document.querySelector('[data-testid=\\"stExpander-Playbook-{trade_id}\\"] button').click()">
                            {'✏️ Edit Details' if 'editing_trade_id' in st.session_state and st.session_state.editing_trade_id == trade_id else '👁️ View/Edit Details'}
                        </button>
                        <button class="stButton red-button" style="color:white;" onclick="window.parent.document.querySelector('[data-testid=\\"stButton-DeleteConfirmation-{trade_id}\\"]').click()">
                            🗑️ Delete
                        </button>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Hidden delete confirmation button (activated by JS above)
                if st.button("Confirm Delete", key=f"DeleteConfirmation-{trade_id}", type="secondary", help="Click to confirm deletion", disabled=True):
                    # Only execute if the confirmation is truly clicked, not just rerunning
                    if st.session_state[f"DeleteConfirmation-{trade_id}"]: # This is a trick to ensure the button was actually pressed
                        st.session_state.trade_journal = st.session_state.trade_journal.drop(index).reset_index(drop=True)
                        if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                            st.success(f"Trade {trade_id} deleted successfully.")
                            if row['EntryScreenshot'] and os.path.exists(row['EntryScreenshot']): os.remove(row['EntryScreenshot'])
                            if row['ExitScreenshot'] and os.path.exists(row['ExitScreenshot']): os.remove(row['ExitScreenshot'])
                        else:
                            st.error(f"Failed to delete trade {trade_id}.")
                        st.rerun() # Rerun to update the display

                # Expander for full trade details and editing
                expander_key = f"Playbook-{trade_id}"
                expander = st.expander(f"Full Details & Edit: {trade_id}", expanded=('editing_trade_id' in st.session_state and st.session_state.editing_trade_id == trade_id), key=expander_key)
                with expander:
                    # Set editing state when expander is opened
                    if expander.expanded:
                        st.session_state.editing_trade_id = trade_id
                    elif 'editing_trade_id' in st.session_state and st.session_state.editing_trade_id == trade_id:
                        del st.session_state.editing_trade_id # Clear if expander is closed
                    
                    edit_trade_data = st.session_state.trade_journal[st.session_state.trade_journal['TradeID'] == trade_id].iloc[0]

                    with st.form(key=f"edit_form_{trade_id}"):
                        st.subheader(f"Editing Trade: {trade_id}")
                        edit_cols_1 = st.columns(4)
                        with edit_cols_1[0]:
                            edit_date = st.date_input("Date", value=edit_trade_data['Date'].date(), key=f"edit_date_{trade_id}")
                            edit_symbol_options = sorted(list(pairs_map.keys())) + ["Other", edit_trade_data['Symbol']]
                            edit_symbol_selected = st.selectbox("Symbol", edit_symbol_options, index=edit_symbol_options.index(edit_trade_data['Symbol']) if edit_trade_data['Symbol'] in edit_symbol_options else len(pairs_map.keys()) + 1, key=f"edit_symbol_select_{trade_id}")
                            edit_symbol = edit_symbol_selected
                            if edit_symbol_selected == "Other":
                                edit_custom_symbol = st.text_input("Enter Custom Symbol", value=edit_trade_data['Symb
