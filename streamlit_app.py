An updated version of your Streamlit code is provided below. The following key features have been implemented as requested:

Trade Notes: Each trade in the "Trade Playbook" now has an expander section titled "Add/View Journal Notes & Actions" where users can write and save detailed notes using Markdown.

Remove Trade & XP Penalty: A "Delete Trade" button has been added within the new actions section. Clicking this button will remove the trade permanently from the journal and apply a -10 XP penalty to the user's account.

Custom Tag Creation: The "Log New Trade" form now includes a text input field allowing users to type in their own custom tags (comma-separated), in addition to selecting from the list of existing ones.

UI Refinement: The horizontal line that previously appeared within the content of each trade card has been removed to create a cleaner look. A separator is now used between each distinct trade card.

Here is the complete, modified code:

code
Python
download
content_copy
expand_less

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
    # This toast is for positive XP events, so we will create a custom one for penalties.
    if amount > 0:
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
# JOURNAL SCHEMA & ROBUST DATA MIGRATION (REVISED)
# =========================================================
# CLEANED UP SCHEMA with safe names for columns
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
        st.markdown("##### ⚡ Trade Entry Details")
        col1, col2, col3 = st.columns(3) # Adjusted to 3 columns

        with col1:
            date_val = st.date_input("Date", dt.date.today())
            symbol_options = list(pairs_map.keys()) + ["Other"]
            symbol = st.selectbox("Symbol", symbol_options, index=symbol_options.index(pair))
            if symbol == "Other": symbol = st.text_input("Custom Symbol")
        with col2:
            direction = st.radio("Direction", ["Long", "Short"], horizontal=True)
            lots = st.number_input("Size (Lots)", min_value=0.01, max_value=1000.0, value=0.10, step=0.01, format="%.2f")
        with col3:
            entry_price = st.number_input("Entry Price", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
            stop_loss = st.number_input("Stop Loss", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
        
        st.markdown("---")
        st.markdown("##### Trade Results & Metrics")
        res_col1, res_col2, res_col3 = st.columns(3) # New 3-column row for results

        with res_col1:
            final_exit = st.number_input("Final Exit Price", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
            outcome = st.selectbox("Outcome", ["Win", "Loss", "Breakeven", "No Trade/Study"])
        
        with res_col2:
            manual_pnl_input = st.number_input("Manual PnL ($)", value=0.0, format="%.2f", help="Enter the profit/loss amount manually.")
        
        with res_col3:
            manual_rr_input = st.number_input("Manual Risk:Reward (R)", value=0.0, format="%.2f", help="Enter the risk-to-reward ratio manually.")
        
        calculate_pnl_rr = st.checkbox("Calculate PnL/RR from Entry/Stop/Exit Prices", value=False, 
                                       help="Check this to automatically calculate PnL and R:R based on prices entered above, overriding manual inputs.")

        with st.expander("Add Quick Rationale & Tags (Optional)"):
            entry_rationale = st.text_area("Why did you enter this trade?", height=100)
            all_tags = sorted(list(set(st.session_state.trade_journal['Tags'].str.split(',').explode().dropna().str.strip())))
            suggested_tags = ["Breakout", "Reversal", "Trend Follow", "Counter-Trend", "News Play", "FOMO", "Over-leveraged"]
            tags = st.multiselect("Select Existing Tags", options=sorted(list(set(all_tags + suggested_tags))))
            custom_tags_input = st.text_input("Or Add New Tags (comma-separated)", placeholder="e.g., Scalp, Fed Meeting")


        submitted = st.form_submit_button("Save Trade", type="primary", use_container_width=True)
        if submitted:
            final_pnl, final_rr = 0.0, 0.0 # Initialize final values

            if calculate_pnl_rr:
                risk_per_unit = abs(entry_price - stop_loss) if stop_loss > 0 else 0
                
                if outcome in ["Win", "Loss"]:
                    pnl_calculated = ((final_exit - entry_price) if direction == "Long" else (entry_price - final_exit)) * lots * 100000 * 0.0001
                else:
                    pnl_calculated = 0.0

                if risk_per_unit > 0:
                    pnl_per_unit_abs = abs(final_exit - entry_price)
                    rr_calculated = (pnl_per_unit_abs / risk_per_unit) if pnl_calculated >= 0 else -(pnl_per_unit_abs / risk_per_unit)
                else:
                    rr_calculated = 0.0

                final_pnl = pnl_calculated
                final_rr = rr_calculated
            else:
                final_pnl = manual_pnl_input
                final_rr = manual_rr_input

            custom_tags_list = [tag.strip() for tag in custom_tags_input.split(',') if tag.strip()]
            final_tags_list = sorted(list(set(tags + custom_tags_list)))

            new_trade_data = {
                "TradeID": f"TRD-{uuid.uuid4().hex[:6].upper()}", "Date": pd.to_datetime(date_val),
                "Symbol": symbol, "Direction": direction, "Outcome": outcome,
                "Lots": lots, "EntryPrice": entry_price, "StopLoss": stop_loss, "FinalExit": final_exit,
                "PnL": final_pnl, "RR": final_rr, 
                "Tags": ','.join(final_tags_list), "EntryRationale": entry_rationale,
                "Strategy": '', "TradeJournalNotes": '', "EntryScreenshot": '', "ExitScreenshot": ''
            }
            new_df = pd.DataFrame([new_trade_data])
            st.session_state.trade_journal = pd.concat([st.session_state.trade_journal, new_df], ignore_index=True)
            
            if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                ta_update_xp(st.session_state.logged_in_user, 10)
                ta_update_streak(st.session_state.logged_in_user)
                st.success(f"Trade {new_trade_data['TradeID']} logged successfully!")
            st.rerun()

# --- TAB 2: TRADE PLAYBOOK ---
with tab_playbook:
    st.header("Your Trade Playbook")
    df_playbook = st.session_state.trade_journal
    if df_playbook.empty:
        st.info("Your logged trades will appear here as playbook cards. Log your first trade to get started!")
    else:
        st.caption("Filter and review your past trades to refine your strategy and identify patterns.")
        
        filter_cols = st.columns([1, 1, 1, 2])
        outcome_filter = filter_cols[0].multiselect("Filter Outcome", df_playbook['Outcome'].unique(), default=df_playbook['Outcome'].unique())
        symbol_filter = filter_cols[1].multiselect("Filter Symbol", df_playbook['Symbol'].unique(), default=df_playbook['Symbol'].unique())
        direction_filter = filter_cols[2].multiselect("Filter Direction", df_playbook['Direction'].unique(), default=df_playbook['Direction'].unique())
        tag_options = sorted(list(set(df_playbook['Tags'].str.split(',').explode().dropna().str.strip())))
        tag_filter = filter_cols[3].multiselect("Filter Tag", options=tag_options)
        
        filtered_df = df_playbook[
            (df_playbook['Outcome'].isin(outcome_filter)) &
            (df_playbook['Symbol'].isin(symbol_filter)) &
            (df_playbook['Direction'].isin(direction_filter))
        ]
        if tag_filter:
            filtered_df = filtered_df[filtered_df['Tags'].apply(lambda x: any(tag in str(x) for tag in tag_filter))]

        # A robust way to handle deletion while iterating
        if 'trade_to_delete' in st.session_state and st.session_state.trade_to_delete is not None:
            index_to_delete = st.session_state.trade_to_delete
            st.session_state.trade_journal.drop(index_to_delete, inplace=True)
            st.session_state.trade_journal.reset_index(drop=True, inplace=True)
            
            user_data = get_user_data(st.session_state.logged_in_user)
            user_data['xp'] = user_data.get('xp', 0) - 10
            save_user_data(st.session_state.logged_in_user, user_data)
            
            _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal)
            
            st.toast("Trade removed. -10 XP penalty.", icon="🗑️")
            del st.session_state['trade_to_delete']
            st.rerun()

        for index, row in filtered_df.sort_values(by="Date", ascending=False).iterrows():
            outcome_color = {"Win": "#2da44e", "Loss": "#cf222e", "Breakeven": "#8b949e"}.get(row['Outcome'], "#30363d")
            with st.container(border=True):
                st.markdown(f"""
                <div style="border-left: 6px solid {outcome_color}; padding-left: 10px;">
                    <h4 style="margin-bottom: 0.25rem;">{row['Symbol']} <span style="font-weight: 500;">{row['Direction']} / {row['Outcome']}</span></h4>
                    <span style="color: #8b949e; font-size: 0.9rem;">{row['Date'].strftime('%A, %d %B %Y')} | ID: {row['TradeID']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                metric_cols = st.columns(3)
                metric_cols[0].metric("Net PnL", f"${row['PnL']:.2f}")
                metric_cols[1].metric("R-Multiple", f"{row['RR']:.2f}R")
                metric_cols[2].metric("Position Size", f"{row['Lots']:.2f} lots")
                
                if row['EntryRationale']:
                    st.markdown(f"**Entry Rationale:** *{row['EntryRationale']}*")
                if row['Tags']:
                    tags_list = [f"`{tag.strip()}`" for tag in str(row['Tags']).split(',') if tag.strip()]
                    st.markdown(f"**Tags:** {', '.join(tags_list)}")

                with st.expander("Add/View Journal Notes & Actions"):
                    with st.form(key=f"notes_form_{row['TradeID']}"):
                        notes_content = st.text_area(
                            "Journal Notes (Supports Markdown)",
                            value=row['TradeJournalNotes'],
                            height=250,
                            label_visibility="collapsed"
                        )
                        if st.form_submit_button("Save Notes", use_container_width=True):
                            st.session_state.trade_journal.loc[index, 'TradeJournalNotes'] = notes_content
                            if _ta_save_journal(st.session_state.logged_in_user, st.session_state.trade_journal):
                                st.toast("Notes saved successfully!", icon="✅")
                            else:
                                st.error("Failed to save notes.")
                            st.rerun()
                    
                    st.markdown("---")
                    st.markdown("**⚠️ Danger Zone**")
                    if st.button("Delete Trade", key=f"delete_{row['TradeID']}", use_container_width=True, type="primary"):
                        st.session_state['trade_to_delete'] = index
                        st.rerun()

            st.markdown("---") # Separator between cards


# --- TAB 3: ANALYTICS DASHBOARD ---
with tab_analytics:
    st.header("Your Performance Dashboard")
    df_analytics = st.session_state.trade_journal[st.session_state.trade_journal['Outcome'].isin(['Win', 'Loss'])].copy()
    
    if df_analytics.empty:
        st.info("Complete at least one winning or losing trade to view your performance analytics.")
    else:
        # High-Level KPIs
        total_pnl = df_analytics['PnL'].sum()
        total_trades = len(df_analytics)
        wins = df_analytics[df_analytics['Outcome'] == 'Win']
        losses = df_analytics[df_analytics['Outcome'] == 'Loss']
        
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        avg_win = wins['PnL'].mean() if not wins.empty else 0
        avg_loss = losses['PnL'].mean() if not losses.empty else 0
        profit_factor = wins['PnL'].sum() / abs(losses['PnL'].sum()) if not losses.empty and losses['PnL'].sum() != 0 else 0

        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Net PnL ($)", f"${total_pnl:,.2f}", delta=f"{total_pnl:+.2f}")
        kpi_cols[1].metric("Win Rate", f"{win_rate:.1f}%")
        kpi_cols[2].metric("Profit Factor", f"{profit_factor:.2f}")
        kpi_cols[3].metric("Avg. Win / Loss ($)", f"${avg_win:,.2f} / ${abs(avg_loss):,.2f}")
        
        st.markdown("---")

        # Visualizations
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.subheader("Cumulative PnL")
            df_analytics['CumulativePnL'] = df_analytics['PnL'].cumsum()
            fig_equity = px.area(df_analytics, x='Date', y='CumulativePnL', title="Your Equity Curve", template="plotly_dark")
            fig_equity.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22")
            st.plotly_chart(fig_equity, use_container_width=True)
            
        with chart_cols[1]:
            st.subheader("Performance by Symbol")
            pnl_by_symbol = df_analytics.groupby('Symbol')['PnL'].sum().sort_values(ascending=False)
            fig_pnl_symbol = px.bar(pnl_by_symbol, title="Net PnL by Symbol", template="plotly_dark")
            fig_pnl_symbol.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", showlegend=False)
            st.plotly_chart(fig_pnl_symbol, use_container_width=True)
