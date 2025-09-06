# = =========================================================
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

# =========================================================
# PAGE CONFIGURATION & PROFESSIONAL UX STYLING
# =========================================================
st.set_page_config(page_title="Pro Journal | Zenvo", layout="wide", initial_sidebar_state="collapsed")

# Inject custom CSS for a professional, modern "Fintech Dark Mode"
st.markdown("""
<style>
    /* --- Main App Styling --- */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .block-container {
        padding: 1.5rem 2.5rem 2rem 2.5rem !important;
    }
    h1, h2, h3 {
        color: #c9d1d9 !important;
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
    }
    [data-testid="stMetric"]:hover {
        border-color: #58a6ff;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #8b949e;
    }

    /* --- Tab Styling --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0 24px;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #161b22;
        color: #58a6ff;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #161b22;
        border-color: #58a6ff;
    }

    /* --- Playbook Card & Editor Styling --- */
    .trade-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    .trade-notes-display {
        background-color: #0D1117;
        border-left: 4px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
        min-height: 100px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOGGING, DATABASE & HELPER FUNCTIONS
# =========================================================
logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_FILE = "zenvo_journal_pro.db"

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.datetime, dt.date)): return obj.isoformat()
        if pd.isna(obj) or np.isnan(obj): return None
        return super().default(obj)

@st.cache_resource
def connect_db():
    try:
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, data TEXT)''')
        conn.commit()
        return conn, c
    except Exception as e:
        st.error("Fatal Error: Could not connect to the database.")
        logging.critical(f"Failed to initialize SQLite database: {e}", exc_info=True)
        st.stop()

conn, c = connect_db()

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
        return False

# =========================================================
# MOCK AUTH & SESSION STATE SETUP
# =========================================================
if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = "pro_trader"
    user_data = get_user_data(st.session_state.logged_in_user)
    if not user_data:
        hashed_password = hashlib.sha256("password".encode()).hexdigest()
        initial_data = {'xp': 0, 'streak': 0, 'trade_journal': []}
        c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)",
                  (st.session_state.logged_in_user, hashed_password, json.dumps(initial_data)))
        conn.commit()

# =========================================================
# JOURNAL SCHEMA & ROBUST DATA MIGRATION
# =========================================================
journal_cols = [
    "TradeID", "Date", "Symbol", "Direction", "Outcome", "PnL", "RR",
    "Strategy", "Tags", "EntryPrice", "StopLoss", "FinalExit", "Lots",
    "EntryRationale", "TradeJournalNotes", "EntryScreenshot", "ExitScreenshot"
]
journal_dtypes = {
    "TradeID": str, "Date": "datetime64[ns]", "Symbol": str, "Direction": str, "Outcome": str,
    "PnL": float, "RR": float, "Strategy": str, "Tags": str, "EntryPrice": float,
    "StopLoss": float, "FinalExit": float, "Lots": float, "EntryRationale": str,
    "TradeJournalNotes": str, "EntryScreenshot": str, "ExitScreenshot": str
}

@st.cache_data
def load_and_migrate_journal(username):
    user_data = get_user_data(username)
    journal_data = user_data.get("trade_journal", [])
    df = pd.DataFrame(journal_data)

    legacy_map = {
        "Trade ID": "TradeID", "Entry Price": "EntryPrice", "Stop Loss": "StopLoss",
        "Final Exit": "FinalExit", "PnL ($)": "PnL", "R:R": "RR",
        "Entry Rationale": "EntryRationale", "Trade Journal Notes": "TradeJournalNotes",
        "Entry Screenshot": "EntryScreenshot", "Exit Screenshot": "ExitScreenshot"
    }
    df.rename(columns=legacy_map, inplace=True, errors='ignore')

    for col, dtype in journal_dtypes.items():
        if col not in df.columns:
            if dtype == str: df[col] = ''
            # CORRECTED: Convert dtype to string before using 'in' operator
            elif 'datetime' in str(dtype): df[col] = pd.NaT
            else: df[col] = 0.0
            
    df = df[journal_cols]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for col in ["PnL", "RR", "EntryPrice", "StopLoss", "FinalExit", "Lots"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    return df

st.session_state.trade_journal = load_and_migrate_journal(st.session_state.logged_in_user)

# =========================================================
# PAGE LAYOUT & HEADER
# =========================================================
st.title("📈 Pro Journal & Backtesting Environment")

main_cols = st.columns([6, 4])
with main_cols[0]:
    st.caption(f"A streamlined interface for professional trade analysis.")
with main_cols[1]:
    if st.button("📝 Log New Trade", use_container_width=True, type="primary"):
        st.session_state.log_trade_dialog = True
    st.caption(f"Logged in as: **{st.session_state.logged_in_user}**")

# =========================================================
# TRADE LOGGING DIALOG
# =========================================================
if st.session_state.get("log_trade_dialog"):
    with st.dialog("Log a Quick Trade", expanded=True):
        with st.form("quick_trade_form"):
            cols = st.columns(4)
            date_val = cols[0].date_input("Date", dt.date.today())
            symbol = cols[0].selectbox("Symbol", ["EUR/USD", "USD/JPY", "GBP/USD", "Other"])
            if symbol == "Other": symbol = cols[0].text_input("Custom Symbol")

            direction = cols[1].radio("Direction", ["Long", "Short"], horizontal=True)
            outcome = cols[1].selectbox("Outcome", ["Win", "Loss", "Breakeven", "Study"])

            entry = cols[2].number_input("Entry Price", value=0.0, step=0.00001, format="%.5f")
            stop = cols[2].number_input("Stop Loss", value=0.0, step=0.00001, format="%.5f")
            
            exit_price = cols[3].number_input("Final Exit", value=0.0, step=0.00001, format="%.5f")
            lots = cols[3].number_input("Lots", min_value=0.01, value=0.1, step=0.01)

            rationale = st.text_input("Entry Rationale (e.g., '1H Break of Structure')")

            if st.form_submit_button("Log Trade", use_container_width=True):
                risk = abs(entry - stop) if stop > 0 else 0
                pnl = ((exit_price - entry) if direction == "Long" else (entry - exit_price)) * lots * 100000 * 0.0001 if outcome in ["Win", "Loss"] else 0
                rr = (pnl / (risk * lots * 100000 * 0.0001)) if risk > 0 else 0
                
                new_trade = pd.DataFrame([{
                    "TradeID": f"TRD-{uuid.uuid4().hex[:6].upper()}", "Date": pd.to_datetime(date_val),
                    "Symbol": symbol, "Direction": direction, "Outcome": outcome, "Lots": lots,
                    "EntryPrice": entry, "StopLoss": stop, "FinalExit": exit_price,
                    "PnL": pnl, "RR": rr, "EntryRationale": rationale, "Strategy": "", "Tags": "", 
                    "TradeJournalNotes": "", "EntryScreenshot": "", "ExitScreenshot": ""
                }])
                
                st.session_state.trade_journal = pd.concat([st.session_state.trade_journal, new_trade], ignore_index=True)
                if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                     st.toast(f"Trade {new_trade['TradeID'].iloc[0]} logged!")
                st.session_state.log_trade_dialog = False
                st.rerun()
st.markdown("---")

# =========================================================
# MAIN INTERFACE: Chart and Tabs
# =========================================================
pairs_map = { "EUR/USD": "FX:EURUSD", "USD/JPY": "FX:USDJPY", "GBP/USD": "FX:GBPUSD"}
tv_symbol = pairs_map[st.selectbox("Select Chart Pair", list(pairs_map.keys()), index=0, key="tv_pair")]

tv_html = f"""<div id="tv_chart_container" style="height: 500px;"></div><script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{ "container_id": "tv_chart_container", "autosize": true, "symbol": "{tv_symbol}", "interval": "D", "timezone": "Etc/UTC", "theme": "dark", "style": "1" }});</script>"""
st.components.v1.html(tv_html, height=500)
st.markdown("---")

tab_playbook, tab_analytics = st.tabs(["**📚 Trade Playbook**", "**📊 Analytics Dashboard**"])

with tab_playbook:
    if st.session_state.trade_journal.empty:
        st.info("Your logged trades appear here. Click 'Log New Trade' to get started!")
    else:
        playbook_cols = st.columns([5, 7])
        with playbook_cols[0]:
            st.subheader("Trade History")
            # ... Filtering UI ...
            df_display = st.session_state.trade_journal.sort_values("Date", ascending=False)
            st.dataframe(df_display[["Date", "Symbol", "Direction", "Outcome", "PnL", "RR"]],
                         key="trade_selector", on_select="rerun", hide_index=True,
                         column_config={"Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD")},
                         use_container_width=True)

        with playbook_cols[1]:
            selection = st.session_state.get("trade_selector", {}).get("selection", {})
            if not selection or not selection.get("rows"):
                st.subheader("Select a Trade")
                st.info("Click a row in the Trade History to view and edit details.")
            else:
                selected_idx = selection["rows"][0]
                trade_id = df_display.iloc[selected_idx]['TradeID']
                trade_data_series = st.session_state.trade_journal.loc[st.session_state.trade_journal['TradeID'] == trade_id].iloc[0]

                st.subheader(f"Reviewing Trade: {trade_data_series['TradeID']}")
                # ... Editing Form as before ...

with tab_analytics:
    st.header("Your Performance Dashboard")
    df_analytics = st.session_state.trade_journal[st.session_state.trade_journal['Outcome'].isin(['Win', 'Loss'])].copy()
    
    if df_analytics.empty:
        st.info("Complete at least one winning or losing trade to view your performance analytics.")
    else:
        # High-Level KPIs
        total_pnl = df_analytics['PnL'].sum()
        wins = df_analytics[df_analytics['Outcome'] == 'Win']
        losses = df_analytics[df_analytics['Outcome'] == 'Loss'] # Correctly defined here
        win_rate = (len(wins) / len(df_analytics)) * 100 if len(df_analytics) > 0 else 0
        
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Net PnL ($)", f"${total_pnl:,.2f}", delta=f"{total_pnl:+.2f}")
        kpi_cols[1].metric("Win Rate", f"{win_rate:.1f}%")
        # ... Other metrics ...

        # ... Charts ...
