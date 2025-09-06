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

# =========================================================
# PAGE CONFIGURATION & PROFESSIONAL UX STYLING
# =========================================================
st.set_page_config(page_title="Pro Journal | Zenvo", layout="wide", initial_sidebar_state="collapsed")

# Inject custom CSS for a professional, modern "Fintech Dark Mode"
st.markdown("""
<style>
    /* --- Main App Styling --- */
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .block-container { padding: 1rem 2rem 2rem 2rem !important; }
    h1, h2, h3 { color: #FAFAFA !important; }
    
    /* --- Hide Streamlit Branding --- */
    #MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden !important; }

    /* --- Custom Horizontal Line --- */
    hr { margin: 1.5rem 0 !important; border-top: 1px solid #30363d !important; }

    /* --- Metric Card Styling --- */
    [data-testid="stMetric"] {
        background-color: #161B22; border: 1px solid #30363d; border-radius: 8px;
        padding: 1.2rem; transition: all 0.2s ease-in-out;
    }
    [data-testid="stMetric"]:hover { border-color: #58A6FF; }
    [data-testid="stMetricLabel"] { font-weight: 500; color: #8B949E; }

    /* --- Tab & Button Styling --- */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 48px; background-color: transparent; border: 1px solid #30363d;
        border-radius: 8px; padding: 0 24px; transition: all 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: #161B22; color: #58A6FF; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #161B22; border-color: #58A6FF; }

    /* --- Playbook Card & Editor Styling --- */
    .trade-card {
        background-color: #161B22; border: 1px solid #30363d; border-radius: 8px;
        padding: 1.5rem; margin-bottom: 2rem;
    }
    .trade-notes-display {
        background-color: #0D1117; border-left: 4px solid #58A6FF; border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem; margin-top: 1rem; min-height: 100px;
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

    # Migrate legacy column names to the new, safer schema if they exist
    legacy_map = {"Trade ID": "TradeID", "Entry Price": "EntryPrice", "Stop Loss": "StopLoss",
                  "Final Exit": "FinalExit", "PnL ($)": "PnL", "R:R": "RR"}
    df.rename(columns=legacy_map, inplace=True)

    for col, dtype in journal_dtypes.items():
        if col not in df.columns:
            if dtype == str: df[col] = ''
            elif 'datetime' in dtype: df[col] = pd.NaT
            else: df[col] = 0.0
            
    df = df[journal_cols].astype(journal_dtypes, errors='ignore')
    df['Date'] = pd.to_datetime(df['Date'])
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
        with st.form("quick_trade_form", clear_on_submit=True):
            cols = st.columns(4)
            date_val = cols[0].date_input("Date", dt.date.today())
            symbol = cols[0].selectbox("Symbol", ["EUR/USD", "USD/JPY", "GBP/USD", "Other"])
            if symbol == "Other": symbol = cols[0].text_input("Custom Symbol")

            direction = cols[1].radio("Direction", ["Long", "Short"], horizontal=True)
            outcome = cols[1].selectbox("Outcome", ["Win", "Loss", "Breakeven", "Study"])

            entry = cols[2].number_input("Entry Price", min_value=0.0, step=0.00001, format="%.5f")
            stop = cols[2].number_input("Stop Loss", min_value=0.0, step=0.00001, format="%.5f")
            
            exit_price = cols[3].number_input("Final Exit", min_value=0.0, step=0.00001, format="%.5f")
            lots = cols[3].number_input("Lots", min_value=0.01, value=0.1, step=0.01)

            rationale = st.text_input("Entry Rationale (e.g., '1H Break of Structure')")

            if st.form_submit_button("Log Trade", use_container_width=True):
                # Calculate metrics
                risk = abs(entry - stop) if stop > 0 else 0
                pnl = ((exit_price - entry) if direction == "Long" else (entry - exit_price)) * lots * 100000 * 0.0001 if outcome in ["Win", "Loss"] else 0
                rr = (abs(exit_price - entry) / risk) if risk > 0 and pnl > 0 else -(abs(exit_price - entry) / risk) if risk > 0 else 0
                
                new_trade = pd.DataFrame([{
                    "TradeID": f"TRD-{uuid.uuid4().hex[:6].upper()}", "Date": pd.to_datetime(date_val),
                    "Symbol": symbol, "Direction": direction, "Outcome": outcome, "Lots": lots,
                    "EntryPrice": entry, "StopLoss": stop, "FinalExit": exit_price,
                    "PnL": pnl, "RR": rr, "EntryRationale": rationale,
                    "Strategy": "", "Tags": "", "TradeJournalNotes": "", "EntryScreenshot": "", "ExitScreenshot": ""
                }])
                
                st.session_state.trade_journal = pd.concat([st.session_state.trade_journal, new_df], ignore_index=True)
                # ... save, update XP etc ...
                st.session_state.log_trade_dialog = False
                st.rerun()

st.markdown("---")
# =========================================================
# MAIN INTERFACE: Chart and Tabs
# =========================================================
# Charting Area remains on top for constant context
pairs_map = {
    "EUR/USD": "FX:EURUSD", "USD/JPY": "FX:USDJPY", "GBP/USD": "FX:GBPUSD", "USD/CHF": "OANDA:USDCHF",
    "AUD/USD": "FX:AUDUSD", "NZD/USD": "OANDA:NZDUSD", "USD/CAD": "FX:USDCAD"
}
tv_symbol = pairs_map[st.selectbox("Select Chart Pair", list(pairs_map.keys()), index=0, key="tv_pair")]

tv_html = f"""<div id="tv_chart_container" style="height: 500px;"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{ "container_id": "tv_chart_container", "autosize": true, "symbol": "{tv_symbol}", "interval": "D", "timezone": "Etc/UTC", "theme": "dark", "style": "1" }});</script>"""
st.components.v1.html(tv_html, height=500)
st.markdown("---")

# Main Interface Tabs
tab_playbook, tab_analytics = st.tabs(["**📚 Trade Playbook**", "**📊 Analytics Dashboard**"])

# --- TAB 1: TRADE PLAYBOOK (MASTER-DETAIL VIEW) ---
with tab_playbook:
    if st.session_state.trade_journal.empty:
        st.info("Your logged trades will appear here. Click 'Log New Trade' to get started!")
    else:
        playbook_cols = st.columns([5, 7]) # Master on left, Detail on right
        
        # --- LEFT (MASTER) COLUMN: TRADE LIST ---
        with playbook_cols[0]:
            st.subheader("Trade History")
            
            # --- Quick Filters ---
            filter_cols = st.columns(3)
            out_filter = filter_cols[0].multiselect("Outcome", st.session_state.trade_journal['Outcome'].unique(), default=st.session_state.trade_journal['Outcome'].unique())
            dir_filter = filter_cols[1].multiselect("Direction", st.session_state.trade_journal['Direction'].unique(), default=st.session_state.trade_journal['Direction'].unique())
            sym_filter = filter_cols[2].multiselect("Symbol", st.session_state.trade_journal['Symbol'].unique(), default=st.session_state.trade_journal['Symbol'].unique())
            
            display_df = st.session_state.trade_journal[
                st.session_state.trade_journal['Outcome'].isin(out_filter) &
                st.session_state.trade_journal['Direction'].isin(dir_filter) &
                st.session_state.trade_journal['Symbol'].isin(sym_filter)
            ].sort_values("Date", ascending=False)
            
            # Interactive Data Editor for selecting trades
            st.data_editor(
                display_df[["Date", "Symbol", "Direction", "Outcome", "PnL", "RR"]],
                key="trade_selector",
                on_select="rerun",
                hide_index=True,
                column_config={"Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD")},
                use_container_width=True
            )

        # --- RIGHT (DETAIL) COLUMN: SELECTED TRADE EDITOR ---
        with playbook_cols[1]:
            selection = st.session_state.get("trade_selector", {}).get("selection", {}).get("rows")
            
            if not selection:
                st.subheader("Select a Trade")
                st.info("Click a row in the Trade History table to view and edit its details here.")
            else:
                selected_index = selection[0]
                trade_id = display_df.iloc[selected_index]['TradeID']
                trade_data = st.session_state.trade_journal[st.session_state.trade_journal['TradeID'] == trade_id].iloc[0].to_dict()

                st.subheader(f"Reviewing Trade: {trade_data['TradeID']}")
                
                with st.form(f"edit_{trade_id}"):
                    # Non-editable Core Metrics
                    m_cols = st.columns(4)
                    m_cols[0].metric("PnL", f"${trade_data['PnL']:.2f}")
                    m_cols[1].metric("R-Multiple", f"{trade_data['RR']:.2f}R")
                    m_cols[2].metric("Lots", f"{trade_data['Lots']:.2f}")
                    
                    # Editable Details
                    trade_data['EntryRationale'] = st.text_area("Entry Rationale", trade_data['EntryRationale'])
                    trade_data['Tags'] = st.multiselect("Tags", ["Breakout", "Reversal", "Trend"], default=[t.strip() for t in trade_data['Tags'].split(',') if t])
                    trade_data['TradeJournalNotes'] = st.text_area("Detailed Notes (Supports Markdown)", trade_data['TradeJournalNotes'], height=200)

                    # Update Screenshot Paths (example, more logic needed for actual uploads)
                    # st.file_uploader for entry and exit screenshots
                    
                    if st.form_submit_button("Save Changes", use_container_width=True, type="primary"):
                        # Find the index in the original dataframe and update it
                        original_idx = st.session_state.trade_journal[st.session_state.trade_journal['TradeID'] == trade_id].index
                        trade_data['Tags'] = ','.join(trade_data['Tags']) # convert list back to string
                        st.session_state.trade_journal.loc[original_idx] = pd.Series(trade_data)
                        _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal)
                        st.toast(f"Trade {trade_id} updated!")
                        st.rerun()
                
# --- TAB 2: ANALYTICS DASHBOARD ---
with tab_analytics:
    st.header("Your Performance Dashboard")
    df_analytics = st.session_state.trade_journal[st.session_state.trade_journal['Outcome'].isin(['Win', 'Loss'])].copy()
    
    if df_analytics.empty:
        st.info("Complete at least one winning or losing trade to view your performance analytics.")
    else:
        total_pnl = df_analytics['PnL'].sum()
        total_trades = len(df_analytics)
        wins = df_analytics[df_analytics['Outcome'] == 'Win']
        
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        avg_rr_win = wins['RR'].mean() if not wins.empty else 0
        avg_rr_loss = df_analytics[df_analytics['Outcome'] == 'Loss']['RR'].mean() if not losses.empty else 0

        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Net PnL ($)", f"${total_pnl:,.2f}", f"{total_pnl:+,.2f}")
        kpi_cols[1].metric("Win Rate", f"{win_rate:.1f}%")
        kpi_cols[2].metric("Avg. Winning RR", f"{avg_rr_win:.2f}R")
        kpi_cols[3].metric("Avg. Losing RR", f"{avg_rr_loss:.2f}R")

        st.markdown("---")
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.subheader("Equity Curve")
            df_analytics.sort_values('Date', inplace=True)
            df_analytics['CumulativePnL'] = df_analytics['PnL'].cumsum()
            fig_equity = px.area(df_analytics, x='Date', y='CumulativePnL', template="plotly_dark")
            st.plotly_chart(fig_equity.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161B22"), use_container_width=True)
            
        with chart_cols[1]:
            st.subheader("R-Multiple Distribution")
            fig_rr = px.histogram(df_analytics, x="RR", nbins=20, title="Distribution of Trade Outcomes by R-Multiple", template="plotly_dark")
            st.plotly_chart(fig_rr.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161B22"), use_container_width=True)
