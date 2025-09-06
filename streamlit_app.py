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
    initial_sidebar_state="expanded", # Changed to expanded for better navigation
    menu_items={
        'Get Help': 'https://www.google.com/search?q=streamlit+help',
        'Report a bug': "https://www.google.com/search?q=streamlit+bug+report",
        'About': "# This is a *pro* journal application."
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
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding: 1rem 2.5rem 2rem 2.5rem !important; /* Adjusted padding */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #e6edf3 !important; /* Lighter header color */
        font-weight: 600;
    }
    h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    h2 {
        font-size: 2rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
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
        padding: 1.2rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stMetric"]:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #8b949e;
        font-size: 0.9rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e6edf3;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* --- Tab Styling --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #161b22; /* Default tab background */
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0 20px; /* Reduced padding */
        transition: all 0.2s ease-in-out;
        color: #8b949e; /* Default tab text color */
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1f2a35; /* Darker on hover */
        color: #58a6ff;
        border-color: #58a6ff;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0d1117; /* Active tab background */
        border-bottom-color: #0d1117 !important; /* Hide bottom border */
        color: #58a6ff;
        border-color: #58a6ff;
        border-bottom-width: 3px;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #161b22; /* Content panel background */
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 2rem;
    }

    /* --- Styling for Markdown in Trade Playbook & General Markdown --- */
    .trade-notes-display {
        background-color: #0d1117; /* Changed to match app background for contrast */
        border-left: 4px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
    }
    .trade-notes-display p { font-size: 15px; color: #c9d1d9; line-height: 1.6; }
    .trade-notes-display h1, h2, h3, h4 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 4px; margin-top: 1rem; }
    
    /* --- General Form Elements --- */
    .stSelectbox > div > div, .stTextInput > div > div, .stDateInput > div > div, .stNumberInput > div > div {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        color: #c9d1d9;
    }
    .stSelectbox > div > div:focus-within, .stTextInput > div > div:focus-within, .stDateInput > div > div:focus-within, .stNumberInput > div > div:focus-within {
        border-color: #58a6ff;
        box-shadow: 0 0 0 1px #58a6ff;
    }
    label {
        color: #c9d1d9 !important;
        font-weight: 500;
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
    }
    .stButton > button:hover {
        background-color: #30363d;
        border-color: #58a6ff;
        color: #58a6ff;
    }
    .stButton > button[kind="primary"] {
        background-color: #238636; /* GitHub green */
        border-color: #238636;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #2da44e;
        border-color: #2da44e;
    }
    .stButton > button[kind="secondary"] {
        background-color: #21262d;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #30363d;
    }

    /* --- Sidebar Styling --- */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
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
    [data-testid="stSidebarNavLinkText"] {
        font-size: 1rem;
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
    st.session_state.xp = user_data['xp'] # Update session state immediately

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
    st.session_state.streak = streak
    if save_user_data(username, user_data):
        st.toast(f"🔥 Daily streak: {streak} days!", icon="🗓️")

def load_image(image_path):
    if image_path and os.path.exists(image_path):
        return Image.open(image_path)
    return None

def save_uploaded_file(uploaded_file, filename):
    filepath = os.path.join(SCREENSHOTS_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath

# =========================================================
# MOCK AUTHENTICATION & SESSION STATE SETUP
# =========================================================
if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = "pro_trader"
    # Check if user exists, if not, create
    c.execute("SELECT username FROM users WHERE username = ?", (st.session_state.logged_in_user,))
    if not c.fetchone():
        hashed_password = hashlib.sha256("password".encode()).hexdigest()
        initial_data = json.dumps({'xp': 0, 'streak': 0, 'trade_journal': []})
        c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)", 
                  (st.session_state.logged_in_user, hashed_password, initial_data))
        conn.commit()
    
    # Load initial user data into session state
    user_data = get_user_data(st.session_state.logged_in_user)
    st.session_state.xp = user_data.get('xp', 0)
    st.session_state.streak = user_data.get('streak', 0)

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
# PAGE LAYOUT FUNCTIONS
# =========================================================

# --- Sidebar ---
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
    st.page_link("app.py", label="Trading Chart", icon="📈") # This is a placeholder for multi-page apps, for now it links to itself
    st.page_link("app.py", label="Log New Trade", icon="📝")
    st.page_link("app.py", label="Trade Playbook", icon="📚")
    st.page_link("app.py", label="Analytics Dashboard", icon="📊")
    
    st.markdown("---")
    if st.button("Logout", help="End current session"):
        st.session_state.logged_in_user = None # In a real app, this would clear full session state
        st.rerun()

# --- Main Content ---
st.title("📈 Pro Journal & Backtesting Environment")
st.caption(f"A streamlined interface for professional trade analysis. | Logged in as: **{st.session_state.logged_in_user}**")
st.markdown("---")

# --- CHARTING AREA ---
pairs_map = {
    "EUR/USD": "FX_IDC:EURUSD", "USD/JPY": "FX_IDC:USDJPY", "GBP/USD": "FX_IDC:GBPUSD", "USD/CHF": "FX_IDC:USDCHF",
    "AUD/USD": "FX_IDC:AUDUSD", "NZD/USD": "FX_IDC:NZDUSD", "USD/CAD": "FX_IDC:USDCAD",
    "XAU/USD (Gold)": "XAUUSD", "BTC/USD": "BINANCE:BTCUSD", "ETH/USD": "BINANCE:ETHUSD"
}
pair = st.selectbox("Select Chart Pair", list(pairs_map.keys()), index=0, key="tv_pair")
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
        "ROC@tv-basic",
        "RSI@tv-basic",
        "Stochastic@tv-basic"
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
tab_entry, tab_playbook, tab_analytics = st.tabs(["**📝 Log New Trade**", "**📚 Trade Playbook**", "**📊 Analytics Dashboard**"])

# --- TAB 1: LOG NEW TRADE ---
with tab_entry:
    st.header("Log a New Trade")
    st.caption("Focus on a quick, essential entry. You can add detailed notes and screenshots later in the Playbook.")

    with st.form("trade_entry_form", clear_on_submit=True):
        st.markdown("##### ⚡ Quick Entry Details")
        cols_quick_entry = st.columns(4)
        
        with cols_quick_entry[0]:
            date_val = st.date_input("Trade Date", dt.date.today(), key="new_trade_date")
            symbol_options = sorted(list(pairs_map.keys())) + ["Other"]
            symbol_selected = st.selectbox("Symbol", symbol_options, index=symbol_options.index(pair) if pair in symbol_options else 0, key="new_trade_symbol_select")
            if symbol_selected == "Other":
                custom_symbol = st.text_input("Enter Custom Symbol", key="new_trade_custom_symbol").upper()
                symbol = custom_symbol if custom_symbol else "N/A"
            else:
                symbol = symbol_selected
            
        with cols_quick_entry[1]:
            direction = st.radio("Direction", ["Long", "Short"], horizontal=True, key="new_trade_direction")
            lots = st.number_input("Position Size (Lots)", min_value=0.01, max_value=1000.0, value=0.10, step=0.01, format="%.2f", key="new_trade_lots")
        
        with cols_quick_entry[2]:
            entry_price = st.number_input("Entry Price", min_value=0.0, value=0.0, step=0.00001, format="%.5f", key="new_trade_entry_price")
            stop_loss = st.number_input("Stop Loss", min_value=0.0, value=0.0, step=0.00001, format="%.5f", key="new_trade_stop_loss")
        
        with cols_quick_entry[3]:
            final_exit = st.number_input("Final Exit Price", min_value=0.0, value=0.0, step=0.00001, format="%.5f", key="new_trade_final_exit")
            outcome = st.selectbox("Outcome", ["Win", "Loss", "Breakeven", "No Trade/Study"], key="new_trade_outcome")
        
        with st.expander("📚 Add Strategy, Rationale & Tags (Optional)"):
            all_strategies = sorted(list(set(st.session_state.trade_journal['Strategy'].dropna().astype(str).str.strip())))
            strategy = st.selectbox("Strategy Used", options=[''] + all_strategies + ["New Strategy"], key="new_trade_strategy_select")
            if strategy == "New Strategy":
                strategy = st.text_input("Enter New Strategy Name", key="new_trade_custom_strategy")

            entry_rationale = st.text_area("Why did you enter this trade? (e.g., specific setup, market conditions)", height=100, key="new_trade_entry_rationale")
            
            all_tags = sorted(list(set(st.session_state.trade_journal['Tags'].str.split(',').explode().dropna().str.strip())))
            suggested_tags = ["Breakout", "Reversal", "Trend Follow", "Counter-Trend", "News Play", "FOMO", "Over-leveraged", "High Impact News", "Liquidity Grab"]
            unique_suggested_tags = sorted(list(set(all_tags + suggested_tags)))
            tags = st.multiselect("Trade Tags", options=unique_suggested_tags, key="new_trade_tags")

        submitted = st.form_submit_button("Save Trade", type="primary", use_container_width=True)
        
        if submitted:
            if entry_price == 0.0 or final_exit == 0.0:
                st.warning("Entry Price and Final Exit must be greater than 0 for PnL calculation.")
            if symbol == "N/A" and symbol_selected == "Other":
                st.warning("Please enter a custom symbol or select an existing one.")
            else:
                pnl, rr = 0.0, 0.0
                if outcome in ["Win", "Loss"] and entry_price > 0 and final_exit > 0:
                    pip_value = 0.0001 # Default for most FX pairs, adjust for JPY/XAU etc.
                    if "JPY" in symbol: pip_value = 0.01 # Special handling for JPY pairs
                    if "XAU" in symbol: pip_value = 0.01 # Placeholder, actual pip value might vary

                    price_diff = (final_exit - entry_price)
                    if direction == "Short": price_diff *= -1

                    # Simplified PnL calculation (needs to be more accurate based on instrument)
                    # For a standard lot (100,000 units), 1 pip = $10. For 0.1 lots, 1 pip = $1
                    pnl = (price_diff / pip_value) * lots * (10 if pip_value == 0.0001 else 100) / 100000 
                    # This PnL calculation is a rough estimate and needs precise instrument specific logic.
                    # A better way would be (final_exit - entry_price) * contract_size * lots * pip_value_per_unit

                    if stop_loss > 0 and entry_price > 0:
                        risk_per_unit = abs(entry_price - stop_loss)
                        if risk_per_unit > 0:
                            pnl_per_unit = abs(final_exit - entry_price)
                            rr = (pnl_per_unit / risk_per_unit) if pnl >= 0 else -(pnl_per_unit / risk_per_unit)
                    
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
                    "PnL": pnl,
                    "RR": rr,
                    "Strategy": strategy,
                    "Tags": ','.join(tags) if tags else '',
                    "EntryRationale": entry_rationale,
                    "TradeJournalNotes": '', # To be added in playbook
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
                    st.rerun()
                else:
                    st.error("Failed to save new trade.")


# --- TAB 2: TRADE PLAYBOOK ---
with tab_playbook:
    st.header("Your Trade Playbook")
    st.caption("Filter, review, and refine your past trades. Edit details or add screenshots.")
    
    df_playbook = st.session_state.trade_journal
    if df_playbook.empty:
        st.info("Your logged trades will appear here as playbook cards. Log your first trade to get started!")
    else:
        # Filtering & Search
        search_query = st.text_input("Search Trades (Symbol, Strategy, Rationale, Notes)", placeholder="e.g., EURUSD, Breakout, FOMO...", key="playbook_search")
        
        filter_cols = st.columns([1, 1, 1, 2])
        outcome_filter = filter_cols[0].multiselect("Filter Outcome", df_playbook['Outcome'].unique(), default=df_playbook['Outcome'].unique())
        symbol_filter = filter_cols[1].multiselect("Filter Symbol", df_playbook['Symbol'].unique(), default=df_playbook['Symbol'].unique())
        direction_filter = filter_cols[2].multiselect("Filter Direction", df_playbook['Direction'].unique(), default=df_playbook['Direction'].unique())
        
        all_tags = sorted(list(set(df_playbook['Tags'].str.split(',').explode().dropna().str.strip())))
        tag_filter = filter_cols[3].multiselect("Filter Tag", options=all_tags)
        
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
        if filtered_df.empty:
            st.info("No trades match your current filters and search query.")

        # Display trades with Edit/Delete functionality
        for index, row in filtered_df.sort_values(by="Date", ascending=False).iterrows():
            trade_id = row['TradeID']
            outcome_color = {"Win": "#2da44e", "Loss": "#cf222e", "Breakeven": "#8b949e", "No Trade/Study": "#58a6ff"}.get(row['Outcome'], "#30363d")

            with st.container(border=True): # Use Streamlit's native border for better theme integration
                header_cols = st.columns([0.7, 0.3])
                with header_cols[0]:
                    st.markdown(f"##### {row['Symbol']} - <span style='color: {outcome_color};'>{row['Direction']} / {row['Outcome']}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='color: #8b949e; font-size: 0.85rem;'>{row['Date'].strftime('%A, %d %B %Y')} | Trade ID: {trade_id}</span>", unsafe_allow_html=True)
                with header_cols[1]:
                    btn_cols = st.columns([1, 1])
                    if btn_cols[0].button("✏️ Edit", key=f"edit_{trade_id}", use_container_width=True):
                        st.session_state.editing_trade_id = trade_id
                        st.session_state.current_tab = "playbook" # Ensure we stay on this tab
                        st.rerun() # Rerun to show the edit form
                    if btn_cols[1].button("🗑️ Delete", key=f"delete_{trade_id}", use_container_width=True):
                        st.session_state.trade_journal = st.session_state.trade_journal.drop(index).reset_index(drop=True)
                        if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                            st.success(f"Trade {trade_id} deleted.")
                            # Clean up associated screenshots if any
                            if row['EntryScreenshot'] and os.path.exists(row['EntryScreenshot']): os.remove(row['EntryScreenshot'])
                            if row['ExitScreenshot'] and os.path.exists(row['ExitScreenshot']): os.remove(row['ExitScreenshot'])
                        else:
                            st.error(f"Failed to delete trade {trade_id}.")
                        st.rerun()

                if 'editing_trade_id' in st.session_state and st.session_state.editing_trade_id == trade_id:
                    st.markdown("---")
                    st.subheader(f"Edit Trade: {trade_id}")
                    edit_trade_data = st.session_state.trade_journal[st.session_state.trade_journal['TradeID'] == trade_id].iloc[0]

                    with st.form(key=f"edit_form_{trade_id}", clear_on_submit=False):
                        edit_cols_1 = st.columns(4)
                        with edit_cols_1[0]:
                            edit_date = st.date_input("Date", value=edit_trade_data['Date'], key=f"edit_date_{trade_id}")
                            edit_symbol_options = sorted(list(pairs_map.keys())) + ["Other", edit_trade_data['Symbol']]
                            edit_symbol_selected = st.selectbox("Symbol", edit_symbol_options, index=edit_symbol_options.index(edit_trade_data['Symbol']) if edit_trade_data['Symbol'] in edit_symbol_options else 0, key=f"edit_symbol_select_{trade_id}")
                            if edit_symbol_selected == "Other":
                                edit_custom_symbol = st.text_input("Enter Custom Symbol", value=edit_trade_data['Symbol'] if edit_trade_data['Symbol'] not in pairs_map.keys() else "", key=f"edit_custom_symbol_{trade_id}").upper()
                                edit_symbol = edit_custom_symbol if edit_custom_symbol else "N/A"
                            else:
                                edit_symbol = edit_symbol_selected
                        with edit_cols_1[1]:
                            edit_direction = st.radio("Direction", ["Long", "Short"], horizontal=True, index=["Long", "Short"].index(edit_trade_data['Direction']), key=f"edit_direction_{trade_id}")
                            edit_lots = st.number_input("Lots", value=edit_trade_data['Lots'], min_value=0.01, step=0.01, format="%.2f", key=f"edit_lots_{trade_id}")
                        with edit_cols_1[2]:
                            edit_entry_price = st.number_input("Entry Price", value=edit_trade_data['EntryPrice'], min_value=0.0, step=0.00001, format="%.5f", key=f"edit_entry_price_{trade_id}")
                            edit_stop_loss = st.number_input("Stop Loss", value=edit_trade_data['StopLoss'], min_value=0.0, step=0.00001, format="%.5f", key=f"edit_stop_loss_{trade_id}")
                        with edit_cols_1[3]:
                            edit_final_exit = st.number_input("Final Exit Price", value=edit_trade_data['FinalExit'], min_value=0.0, step=0.00001, format="%.5f", key=f"edit_final_exit_{trade_id}")
                            edit_outcome = st.selectbox("Outcome", ["Win", "Loss", "Breakeven", "No Trade/Study"], index=["Win", "Loss", "Breakeven", "No Trade/Study"].index(edit_trade_data['Outcome']), key=f"edit_outcome_{trade_id}")
                        
                        edit_cols_2 = st.columns(2)
                        with edit_cols_2[0]:
                            edit_all_strategies = sorted(list(set(st.session_state.trade_journal['Strategy'].dropna().astype(str).str.strip())))
                            edit_strategy = st.selectbox("Strategy Used", options=[''] + edit_all_strategies + ["New Strategy"], index=([edit_all_strategies.index(edit_trade_data['Strategy']) + 1] if edit_trade_data['Strategy'] in edit_all_strategies else [0])[0], key=f"edit_strategy_{trade_id}")
                            if edit_strategy == "New Strategy":
                                edit_strategy = st.text_input("Enter New Strategy Name", key=f"edit_custom_strategy_{trade_id}")
                            
                            edit_entry_rationale = st.text_area("Entry Rationale", value=edit_trade_data['EntryRationale'], height=100, key=f"edit_entry_rationale_{trade_id}")
                            
                            edit_all_tags = sorted(list(set(df_playbook['Tags'].str.split(',').explode().dropna().str.strip())))
                            current_tags = [tag.strip() for tag in str(edit_trade_data['Tags']).split(',') if tag.strip()]
                            edit_tags = st.multiselect("Tags", options=edit_all_tags, default=current_tags, key=f"edit_tags_{trade_id}")

                        with edit_cols_2[1]:
                            edit_notes = st.text_area("Detailed Trade Journal Notes (Markdown supported)", value=edit_trade_data['TradeJournalNotes'], height=200, key=f"edit_notes_{trade_id}")

                        st.markdown("##### 📸 Screenshots")
                        screenshot_cols = st.columns(2)
                        
                        entry_ss_path = edit_trade_data['EntryScreenshot']
                        exit_ss_path = edit_trade_data['ExitScreenshot']

                        with screenshot_cols[0]:
                            st.caption("Entry Screenshot")
                            if entry_ss_path and os.path.exists(entry_ss_path):
                                st.image(entry_ss_path, caption="Current Entry Screenshot", use_column_width=True)
                                if st.button("Delete Entry Screenshot", key=f"del_entry_ss_{trade_id}"):
                                    os.remove(entry_ss_path)
                                    edit_trade_data['EntryScreenshot'] = ''
                                    st.session_state.trade_journal.loc[st.session_state.trade_journal['TradeID'] == trade_id, 'EntryScreenshot'] = ''
                                    st.success("Entry screenshot deleted.")
                                    if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                                        st.rerun()
                            new_entry_ss = st.file_uploader("Upload New Entry Screenshot", type=["png", "jpg", "jpeg"], key=f"new_entry_ss_{trade_id}")
                        
                        with screenshot_cols[1]:
                            st.caption("Exit Screenshot")
                            if exit_ss_path and os.path.exists(exit_ss_path):
                                st.image(exit_ss_path, caption="Current Exit Screenshot", use_column_width=True)
                                if st.button("Delete Exit Screenshot", key=f"del_exit_ss_{trade_id}"):
                                    os.remove(exit_ss_path)
                                    edit_trade_data['ExitScreenshot'] = ''
                                    st.session_state.trade_journal.loc[st.session_state.trade_journal['TradeID'] == trade_id, 'ExitScreenshot'] = ''
                                    st.success("Exit screenshot deleted.")
                                    if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                                        st.rerun()
                            new_exit_ss = st.file_uploader("Upload New Exit Screenshot", type=["png", "jpg", "jpeg"], key=f"new_exit_ss_{trade_id}")


                        submit_edit_cols = st.columns(2)
                        if submit_edit_cols[0].form_submit_button("Update Trade", type="primary", use_container_width=True):
                            # Recalculate PnL and RR if prices changed
                            updated_pnl, updated_rr = 0.0, 0.0
                            if edit_outcome in ["Win", "Loss"] and edit_entry_price > 0 and edit_final_exit > 0:
                                pip_value = 0.0001
                                if "JPY" in edit_symbol: pip_value = 0.01
                                if "XAU" in edit_symbol: pip_value = 0.01
                                
                                price_diff = (edit_final_exit - edit_entry_price)
                                if edit_direction == "Short": price_diff *= -1

                                updated_pnl = (price_diff / pip_value) * edit_lots * (10 if pip_value == 0.0001 else 100) / 100000

                                if edit_stop_loss > 0 and edit_entry_price > 0:
                                    risk_per_unit = abs(edit_entry_price - edit_stop_loss)
                                    if risk_per_unit > 0:
                                        pnl_per_unit = abs(edit_final_exit - edit_entry_price)
                                        updated_rr = (pnl_per_unit / risk_per_unit) if updated_pnl >= 0 else -(pnl_per_unit / risk_per_unit)

                            # Handle screenshot uploads
                            entry_ss_path_final = edit_trade_data['EntryScreenshot']
                            if new_entry_ss is not None:
                                # Delete old screenshot if it exists and a new one is uploaded
                                if entry_ss_path and os.path.exists(entry_ss_path):
                                    os.remove(entry_ss_path)
                                entry_filename = f"{trade_id}_entry_{uuid.uuid4().hex[:4]}{os.path.splitext(new_entry_ss.name)[1]}"
                                entry_ss_path_final = save_uploaded_file(new_entry_ss, entry_filename)
                                st.toast("Entry screenshot uploaded!")

                            exit_ss_path_final = edit_trade_data['ExitScreenshot']
                            if new_exit_ss is not None:
                                # Delete old screenshot if it exists and a new one is uploaded
                                if exit_ss_path and os.path.exists(exit_ss_path):
                                    os.remove(exit_ss_path)
                                exit_filename = f"{trade_id}_exit_{uuid.uuid4().hex[:4]}{os.path.splitext(new_exit_ss.name)[1]}"
                                exit_ss_path_final = save_uploaded_file(new_exit_ss, exit_filename)
                                st.toast("Exit screenshot uploaded!")

                            # Update the DataFrame
                            trade_index = st.session_state.trade_journal[st.session_state.trade_journal['TradeID'] == trade_id].index[0]
                            st.session_state.trade_journal.loc[trade_index] = {
                                "TradeID": trade_id,
                                "Date": pd.to_datetime(edit_date),
                                "Symbol": edit_symbol,
                                "Direction": edit_direction,
                                "Outcome": edit_outcome,
                                "PnL": updated_pnl,
                                "RR": updated_rr,
                                "Strategy": edit_strategy,
                                "Tags": ','.join(edit_tags) if edit_tags else '',
                                "EntryPrice": edit_entry_price,
                                "StopLoss": edit_stop_loss,
                                "FinalExit": edit_final_exit,
                                "Lots": edit_lots,
                                "EntryRationale": edit_entry_rationale,
                                "TradeJournalNotes": edit_notes,
                                "EntryScreenshot": entry_ss_path_final,
                                "ExitScreenshot": exit_ss_path_final,
                            }
                            if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                                st.success(f"Trade {trade_id} updated successfully!")
                                del st.session_state.editing_trade_id # Clear editing state
                                st.rerun()
                            else:
                                st.error(f"Failed to update trade {trade_id}.")

                        if submit_edit_cols[1].form_submit_button("Cancel", use_container_width=True):
                            del st.session_state.editing_trade_id
                            st.rerun()
                else:
                    # Regular trade display
                    expander = st.expander(f"**Details & Performance** (PnL: ${row['PnL']:.2f}, R:R: {row['RR']:.2f}R)", expanded=False)
                    with expander:
                        metric_cols = st.columns(4)
                        metric_cols[0].metric("Entry Price", f"{row['EntryPrice']:.5f}")
                        metric_cols[1].metric("Stop Loss", f"{row['StopLoss']:.5f}")
                        metric_cols[2].metric("Final Exit", f"{row['FinalExit']:.5f}")
                        metric_cols[3].metric("Position Size", f"{row['Lots']:.2f} lots")
                        
                        st.markdown(f"**Strategy:** {row['Strategy'] if row['Strategy'] else 'N/A'}")
                        if row['EntryRationale']:
                            st.markdown(f"**Entry Rationale:** *{row['EntryRationale']}*")
                        if row['Tags']:
                            tags_list = [f" `{tag.strip()}`" for tag in str(row['Tags']).split(',') if tag.strip()]
                            st.markdown(f"**Tags:** {','.join(tags_list)}")
                        
                        if row['TradeJournalNotes']:
                            st.markdown("##### 📝 Detailed Notes")
                            st.markdown(f"<div class='trade-notes-display'>{row['TradeJournalNotes']}</div>", unsafe_allow_html=True)

                        st.markdown("---")
                        st.markdown("##### Screenshots")
                        screenshot_display_cols = st.columns(2)
                        with screenshot_display_cols[0]:
                            entry_img = load_image(row['EntryScreenshot'])
                            if entry_img:
                                st.image(entry_img, caption="Entry Screenshot", use_column_width=True)
                            else:
                                st.info("No Entry Screenshot uploaded.")
                        with screenshot_display_cols[1]:
                            exit_img = load_image(row['ExitScreenshot'])
                            if exit_img:
                                st.image(exit_img, caption="Exit Screenshot", use_column_width=True)
                            else:
                                st.info("No Exit Screenshot uploaded.")


# --- TAB 3: ANALYTICS DASHBOARD ---
with tab_analytics:
    st.header("Your Performance Dashboard")
    df_analytics = st.session_state.trade_journal[st.session_state.trade_journal['Outcome'].isin(['Win', 'Loss'])].copy()
    
    if df_analytics.empty:
        st.info("Complete at least one winning or losing trade to view your performance analytics.")
    else:
        # Date Range Filter for Analytics
        min_date = df_analytics['Date'].min().date() if not df_analytics.empty else dt.date.today()
        max_date = df_analytics['Date'].max().date() if not df_analytics.empty else dt.date.today()
        
        analytics_date_range = st.slider(
            "Select Date Range for Analytics",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key="analytics_date_range"
        )
        
        filtered_analytics_df = df_analytics[
            (df_analytics['Date'].dt.date >= analytics_date_range[0]) &
            (df_analytics['Date'].dt.date <= analytics_date_range[1])
        ].copy()

        if filtered_analytics_df.empty:
            st.warning("No trades found in the selected date range for analytics.")
        else:
            # High-Level KPIs
            total_pnl = filtered_analytics_df['PnL'].sum()
            total_trades = len(filtered_analytics_df)
            wins = filtered_analytics_df[filtered_analytics_df['Outcome'] == 'Win']
            losses = filtered_analytics_df[filtered_analytics_df['Outcome'] == 'Loss']
            
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            avg_win = wins['PnL'].mean() if not wins.empty else 0
            avg_loss = losses['PnL'].mean() if not losses.empty else 0
            profit_factor = wins['PnL'].sum() / abs(losses['PnL'].sum()) if not losses.empty and losses['PnL'].sum() != 0 else 0
            largest_win = wins['PnL'].max() if not wins.empty else 0
            largest_loss = losses['PnL'].min() if not losses.empty else 0

            # Calculate Max Drawdown
            if not filtered_analytics_df.empty:
                filtered_analytics_df.sort_values(by='Date', inplace=True)
                filtered_analytics_df['CumulativePnL'] = filtered_analytics_df['PnL'].cumsum()
                filtered_analytics_df['Peak'] = filtered_analytics_df['CumulativePnL'].cummax()
                filtered_analytics_df['Drawdown'] = filtered_analytics_df['CumulativePnL'] - filtered_analytics_df['Peak']
                max_drawdown = filtered_analytics_df['Drawdown'].min()
            else:
                max_drawdown = 0

            kpi_cols = st.columns(6)
            kpi_cols[0].metric("Net PnL ($)", f"${total_pnl:,.2f}", delta=f"{total_pnl:+.2f}")
            kpi_cols[1].metric("Win Rate", f"{win_rate:.1f}%")
            kpi_cols[2].metric("Profit Factor", f"{profit_factor:.2f}")
            kpi_cols[3].metric("Avg. Win/Loss ($)", f"${avg_win:,.2f} / ${abs(avg_loss):,.2f}")
            kpi_cols[4].metric("Largest Win ($)", f"${largest_win:,.2f}")
            kpi_cols[5].metric("Largest Loss ($)", f"${largest_loss:,.2f}")
            
            st.metric("Max Drawdown ($)", f"${max_drawdown:,.2f}") # Display Max Drawdown prominently
            st.markdown("---")

            # Visualizations
            chart_cols_1 = st.columns(2)
            with chart_cols_1[0]:
                st.subheader("Cumulative PnL (Equity Curve)")
                fig_equity = px.line(filtered_analytics_df, x='Date', y='CumulativePnL', title="Your Equity Curve", template="plotly_dark", markers=True)
                fig_equity.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", hovermode="x unified")
                fig_equity.update_traces(line_color='#58a6ff', marker=dict(color='#58a6ff', size=6))
                st.plotly_chart(fig_equity, use_container_width=True)
                
            with chart_cols_1[1]:
                st.subheader("Performance by Symbol")
                pnl_by_symbol = filtered_analytics_df.groupby('Symbol')['PnL'].sum().sort_values(ascending=False).reset_index()
                fig_pnl_symbol = px.bar(pnl_by_symbol, x='Symbol', y='PnL', title="Net PnL by Symbol", template="plotly_dark",
                                        color='PnL', color_continuous_scale=px.colors.sequential.RdYlGn)
                fig_pnl_symbol.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", showlegend=False)
                st.plotly_chart(fig_pnl_symbol, use_container_width=True)

            chart_cols_2 = st.columns(2)
            with chart_cols_2[0]:
                st.subheader("Performance by Strategy")
                if 'Strategy' in filtered_analytics_df.columns and not filtered_analytics_df['Strategy'].astype(str).str.strip().eq('').all():
                    pnl_by_strategy = filtered_analytics_df.groupby('Strategy')['PnL'].sum().sort_values(ascending=False).reset_index()
                    fig_pnl_strategy = px.bar(pnl_by_strategy, x='Strategy', y='PnL', title="Net PnL by Strategy", template="plotly_dark",
                                            color='PnL', color_continuous_scale=px.colors.sequential.RdYlGn)
                    fig_pnl_strategy.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", showlegend=False)
                    st.plotly_chart(fig_pnl_strategy, use_container_width=True)
                else:
                    st.info("No strategies logged or selected in the filtered range.")

            with chart_cols_2[1]:
                st.subheader("R-Multiple Distribution")
                if not filtered_analytics_df['RR'].isna().all():
                    fig_rr_dist = px.histogram(filtered_analytics_df, x='RR', title="R-Multiple Distribution", template="plotly_dark",
                                            nbins=20, color_discrete_sequence=['#58a6ff'])
                    fig_rr_dist.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22")
                    st.plotly_chart(fig_rr_dist, use_container_width=True)
                else:
                    st.info("No R-Multiples available (e.g., no stop loss or entry/exit prices).")
