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
from PIL import Image
import base64
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# =========================================================
# PAGE CONFIGURATION & ENHANCED UX STYLING
# =========================================================
st.set_page_config(page_title="Pro Journal | Zenvo", layout="wide", initial_sidebar_state="collapsed")

# Inject custom CSS for a professional, modern dark theme
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

    /* --- Styling for Markdown in Trade Playbook --- */
    .trade-notes-display {
        background-color: #161b22;
        border-left: 4px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
    }
    .trade-notes-display p { font-size: 15px; color: #c9d1d9; line-height: 1.6; }
    .trade-notes-display h1, h2, h3, h4 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 4px; }
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
        if pd.isna(obj) or np.isnan(obj): return None
        return super().default(obj)

try:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, data TEXT)''')
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
        return False

def _ta_save_journal(username, journal_df):
    user_data = get_user_data(username)
    user_data["trade_journal"] = journal_df.to_dict('records')
    if save_user_data(username, user_data):
        logging.info(f"Journal saved for user {username}: {len(journal_df)} trades")
        return True
    st.error("Failed to save journal.")
    return False

def ta_update_xp(username, amount):
    user_data = get_user_data(username)
    user_data['xp'] = user_data.get('xp', 0) + amount
    st.toast(f"⭐ +{amount} XP Earned!", icon="🎉")
    save_user_data(username, user_data)

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
    save_user_data(username, user_data)


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

# =========================================================
# JOURNAL SCHEMA & ROBUST DATA MIGRATION
# =========================================================
journal_cols = [
    "Trade ID", "Date", "Symbol", "Direction", "Outcome", "PnL ($)", "R:R", 
    "Strategy", "Tags", "Entry Price", "Stop Loss", "Final Exit", "Lots",
    "Entry Rationale", "Trade Journal Notes", "Entry Screenshot", "Exit Screenshot"
]
journal_dtypes = {
    "Trade ID": str, "Date": "datetime64[ns]", "Symbol": str, "Direction": str, 
    "Outcome": str, "PnL ($)": float, "R:R": float, "Strategy": str, 
    "Tags": str, "Entry Price": float, "Stop Loss": float, "Final Exit": float, "Lots": float,
    "Entry Rationale": str, "Trade Journal Notes": str, 
    "Entry Screenshot": str, "Exit Screenshot": str
}

if 'trade_journal' not in st.session_state:
    user_data = get_user_data(st.session_state.logged_in_user)
    journal_data = user_data.get("trade_journal", [])
    df = pd.DataFrame(journal_data)
    
    # Safely migrate data to the new schema
    for col, dtype in journal_dtypes.items():
        if col not in df.columns:
            if dtype == str: df[col] = ''
            elif 'datetime' in str(dtype): df[col] = pd.NaT
            elif dtype == float: df[col] = 0.0
            else: df[col] = None
    
    st.session_state.trade_journal = df[journal_cols].astype(journal_dtypes, errors='ignore')
    st.session_state.trade_journal['Date'] = pd.to_datetime(st.session_state.trade_journal['Date'])


# =========================================================
# PAGE LAYOUT
# =========================================================
st.title("📈 Pro Journal & Backtesting Environment")
st.caption(f"A streamlined interface for professional trade analysis. | Logged in as: **{st.session_state.logged_in_user}**")
st.markdown("---")

# --- CHARTING AREA ---
pairs_map = {
    "EUR/USD": "FX:EURUSD", "USD/JPY": "FX:USDJPY", "GBP/USD": "FX:GBPUSD", "USD/CHF": "OANDA:USDCHF",
    "AUD/USD": "FX:AUDUSD", "NZD/USD": "OANDA:NZDUSD", "USD/CAD": "FX:USDCAD"
}
pair = st.selectbox("Select Chart Pair", list(pairs_map.keys()), index=0, key="tv_pair")
tv_symbol = pairs_map[pair]

tv_html = f"""
<div id="tradingview_widget"></div>
<script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
<script> new TradingView.widget({{
    "container_id": "tradingview_widget", "width": "100%", "height": 550, "symbol": "{tv_symbol}",
    "interval": "D", "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "en",
    "enable_publishing": false, "allow_symbol_change": true, "hide_side_toolbar": false, "autosize": true
}});</script>
"""
st.components.v1.html(tv_html, height=560)
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
        st.markdown("##### ⚡ 30-Second Journal Entry")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            date_val = st.date_input("Date", dt.date.today())
            symbol_options = list(pairs_map.keys()) + ["Other"]
            symbol = st.selectbox("Symbol", symbol_options, index=symbol_options.index(pair))
            if symbol == "Other": symbol = st.text_input("Custom Symbol")
        with col2:
            direction = st.radio("Direction", ["Long", "Short"], horizontal=True)
            lots = st.number_input("Size (Lots)", 0.01, 1000.0, 0.10, 0.01, "%.2f")
        with col3:
            entry_price = st.number_input("Entry Price", 0.0, None, 0.0, "%.5f")
            stop_loss = st.number_input("Stop Loss", 0.0, None, 0.0, "%.5f")
        with col4:
            final_exit = st.number_input("Final Exit Price", 0.0, None, 0.0, "%.5f")
            outcome = st.selectbox("Outcome", ["Win", "Loss", "Breakeven", "No Trade/Study"])
        
        with st.expander("Add Quick Rationale & Tags (Optional)"):
            entry_rationale = st.text_area("Why did you enter this trade?", height=100)
            all_tags = sorted(list(set(st.session_state.trade_journal['Tags'].str.split(',').explode().dropna().str.strip())))
            suggested_tags = ["Breakout", "Reversal", "Trend Follow", "Counter-Trend", "News Play", "FOMO", "Over-leveraged"]
            tags = st.multiselect("Trade Tags", options=sorted(list(set(all_tags + suggested_tags))))

        submitted = st.form_submit_button("Save Trade", type="primary", use_container_width=True)
        if submitted:
            pnl, rr = 0.0, 0.0
            risk_per_unit = abs(entry_price - stop_loss) if stop_loss > 0 else 0
            
            if outcome in ["Win", "Loss"]:
                pnl = ((final_exit - entry_price) if direction == "Long" else (entry_price - final_exit)) * lots * 100000 * 0.0001
            
            if risk_per_unit > 0:
                pnl_per_unit = abs(final_exit - entry_price)
                rr = pnl_per_unit / risk_per_unit if pnl > 0 else -(pnl_per_unit / risk_per_unit)

            new_trade_data = {
                "Trade ID": f"TRD-{uuid.uuid4().hex[:6].upper()}", "Date": pd.to_datetime(date_val),
                "Symbol": symbol, "Direction": direction, "Outcome": outcome,
                "Lots": lots, "Entry Price": entry_price, "Stop Loss": stop_loss, "Final Exit": final_exit,
                "PnL ($)": pnl, "R:R": rr,
                "Tags": ','.join(tags), "Entry Rationale": entry_rationale,
                "Strategy": '', "Trade Journal Notes": '', "Entry Screenshot": '', "Exit Screenshot": ''
            }
            new_df = pd.DataFrame([new_trade_data])
            st.session_state.trade_journal = pd.concat([st.session_state.trade_journal, new_df], ignore_index=True)
            
            if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                ta_update_xp(st.session_state.logged_in_user, 10)
                ta_update_streak(st.session_state.logged_in_user)
                st.success(f"Trade {new_trade_data['Trade ID']} logged successfully!")
            st.rerun()

# --- TAB 2: TRADE PLAYBOOK ---
with tab_playbook:
    st.header("Your Trade Playbook")
    df_playbook = st.session_state.trade_journal
    if df_playbook.empty:
        st.info("Your logged trades will appear here as playbook cards. Log your first trade to get started!")
    else:
        st.caption("Filter and review your past trades to refine your strategy and identify patterns.")
        
        # --- Filtering Controls ---
        filter_cols = st.columns([1, 1, 1, 2])
        with filter_cols[0]:
            outcome_filter = st.multiselect("Filter by Outcome", options=df_playbook['Outcome'].unique(), default=df_playbook['Outcome'].unique())
        with filter_cols[1]:
            symbol_filter = st.multiselect("Filter by Symbol", options=df_playbook['Symbol'].unique(), default=df_playbook['Symbol'].unique())
        with filter_cols[2]:
            direction_filter = st.multiselect("Filter by Direction", options=df_playbook['Direction'].unique(), default=df_playbook['Direction'].unique())
        with filter_cols[3]:
             tag_options = sorted(list(set(df_playbook['Tags'].str.split(',').explode().dropna().str.strip())))
             tag_filter = st.multiselect("Filter by Tag", options=tag_options)
        
        # Apply filters
        filtered_df = df_playbook[
            (df_playbook['Outcome'].isin(outcome_filter)) &
            (df_playbook['Symbol'].isin(symbol_filter)) &
            (df_playbook['Direction'].isin(direction_filter))
        ]
        if tag_filter:
            filtered_df = filtered_df[filtered_df['Tags'].apply(lambda x: any(tag in x for tag in tag_filter))]

        # --- Display Trade Cards ---
        for index, row in filtered_df.sort_values(by="Date", ascending=False).iterrows():
            outcome_color = {"Win": "#2da44e", "Loss": "#cf222e", "Breakeven": "#8b949e"}.get(row['Outcome'], "#30363d")
            
            with st.container():
                st.markdown(f"""
                <div style="border: 1px solid #30363d; border-left: 8px solid {outcome_color}; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem;">
                    <h4>{row['Symbol']} <span style="font-weight: 500; color: {outcome_color};"> {row['Direction']} / {row['Outcome']}</span></h4>
                    <span style="color: #8b949e;">{pd.to_datetime(row['Date']).strftime('%A, %d %B %Y')}</span>
                </div>
                """, unsafe_allow_html=True)

                cols = st.columns(3)
                cols[0].metric("Net PnL", f"${row['PnL ($)]:.2f}")
                cols[1].metric("R:R Multiple", f"{row['R:R']:.2f}R")
                cols[2].metric("Position Size", f"{row['Lots']:.2f} lots")
                
                if row['Entry Rationale']:
                    st.markdown(f"**Entry Rationale:** *{row['Entry Rationale']}*")
                if row['Tags']:
                    st.write(f"**Tags:** `{'`, `'.join(row['Tags'].split(','))}`")
                
                # Full details in an expander
                with st.expander("View Full Trade Details & Notes"):
                     # Display formatted notes
                    st.markdown("**Trade Journal Notes & Learnings**")
                    if row['Trade Journal Notes']:
                        st.markdown(f"<div class='trade-notes-display'>{row['Trade Journal Notes']}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No detailed notes for this trade yet.")
                     # You can add a text_area here to edit notes and a button to save
                
                st.markdown("---")

# --- TAB 3: ANALYTICS DASHBOARD ---
with tab_analytics:
    st.header("Your Performance Dashboard")
    df_analytics = st.session_state.trade_journal[st.session_state.trade_journal['Outcome'].isin(['Win', 'Loss'])]
    
    if df_analytics.empty:
        st.info("Complete at least one winning or losing trade to view your performance analytics.")
    else:
        # --- High-Level KPIs ---
        total_pnl = df_analytics['PnL ($)'].sum()
        total_trades = len(df_analytics)
        wins = df_analytics[df_analytics['Outcome'] == 'Win']
        losses = df_analytics[df_analytics['Outcome'] == 'Loss']
        
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        avg_win = wins['PnL ($)'].mean() if not wins.empty else 0
        avg_loss = losses['PnL ($)'].mean() if not losses.empty else 0
        profit_factor = wins['PnL ($)'].sum() / abs(losses['PnL ($)'].sum()) if not losses.empty and losses['PnL ($)'].sum() != 0 else float('inf')

        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Net PnL ($)", f"${total_pnl:,.2f}", delta=f"{total_pnl:+,.2f}")
        kpi_cols[1].metric("Win Rate (%)", f"{win_rate:.1f}%")
        kpi_cols[2].metric("Profit Factor", f"{profit_factor:.2f}")
        kpi_cols[3].metric("Avg Win / Avg Loss ($)", f"${avg_win:,.2f} / ${abs(avg_loss):,.2f}")
        
        st.markdown("---")

        # --- Visualizations ---
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.subheader("Cumulative PnL")
            df_analytics['Cumulative PnL'] = df_analytics['PnL ($)'].cumsum()
            fig_equity = px.line(df_analytics, x=df_analytics['Date'], y='Cumulative PnL', title="Your Equity Curve", template="plotly_dark")
            fig_equity.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22")
            st.plotly_chart(fig_equity, use_container_width=True)

        with chart_cols[1]:
            st.subheader("Performance by Symbol")
            pnl_by_symbol = df_analytics.groupby('Symbol')['PnL ($)'].sum().sort_values(ascending=False)
            fig_pnl_symbol = px.bar(pnl_by_symbol, title="Net PnL by Symbol", template="plotly_dark")
            fig_pnl_symbol.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", showlegend=False)
            st.plotly_chart(fig_pnl_symbol, use_container_width=True)
