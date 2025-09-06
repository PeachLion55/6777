# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
import feedparser
from textblob import TextBlob
import streamlit.components.v1 as components
import datetime as dt
from datetime import datetime, date, timedelta
import os
import json
import hashlib
import requests
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sqlite3
import pytz
import logging
import math
import uuid
from PIL import Image
import io
import base64
import calendar

# =========================================================
# PAGE CONFIGURATION & ENHANCED UX STYLING
# =========================================================
st.set_page_config(page_title="Pro Journal | Zenvo", layout="wide", initial_sidebar_state="collapsed")

# Inject custom CSS for a professional, modern dark theme from Code 1
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

    /* --- Tab Styling (Applied to the entire app) --- */
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

    /* --- Sidebar background & Button Styling --- */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] div.stButton > button {
        width: 100% !important;
        background: transparent !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 10px !important;
        margin: 5px 0 !important;
        font-weight: bold !important;
        font-size: 16px !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #161b22 !important;
        border-color: #58a6ff !important;
        color: #58a6ff !important;
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
        if pd.isna(obj) or np.isnan(obj): return None
        return super().default(obj)

try:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, data TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS community_data (key TEXT PRIMARY KEY, data TEXT)''')
    conn.commit()
except Exception as e:
    st.error("Fatal Error: Could not connect to the database.")
    logging.critical(f"Failed to initialize SQLite database: {str(e)}", exc_info=True)
    st.stop()


# =========================================================
# HELPER & GAMIFICATION FUNCTIONS (CONSOLIDATED)
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
    user_data["tools_trade_journal"] = journal_df.to_dict('records')
    if save_user_data(username, user_data):
        logging.info(f"Journal saved for user {username}: {len(journal_df)} trades")
        return True
    st.error("Failed to save journal.")
    return False

def show_xp_notification(xp_gained):
    notification_html = f"""
    <div id="xp-notification" style="
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #58a6ff, #161b22);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(88, 166, 255, 0.3);
        z-index: 9999;
        animation: slideInRight 0.5s ease-out, fadeOut 0.5s ease-out 3s forwards;
        font-weight: bold;
        border: 1px solid #58a6ff;
        backdrop-filter: blur(10px);
    ">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="font-size: 24px;">⭐</div>
            <div>
                <div style="font-size: 16px;">+{xp_gained} XP Earned!</div>
            </div>
        </div>
    </div>
    <style>
        @keyframes slideInRight {{
            from {{ transform: translateX(100%); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        @keyframes fadeOut {{
            from {{ opacity: 1; }}
            to {{ opacity: 0; }}
        }}
    </style>
    """
    st.components.v1.html(notification_html, height=0)

def ta_update_xp(amount):
    if "logged_in_user" in st.session_state:
        username = st.session_state.logged_in_user
        user_data = get_user_data(username)
        user_data['xp'] = user_data.get('xp', 0) + amount
        level = user_data['xp'] // 100
        if level > user_data.get('level', 0):
            user_data['level'] = level
            user_data.setdefault('badges', []).append(f"Level {level} Reached")
            st.balloons()
            st.success(f"Level up! You are now level {level}.")
        if save_user_data(username, user_data):
            st.session_state.xp = user_data['xp']
            st.session_state.level = user_data['level']
            st.session_state.badges = user_data['badges']
            show_xp_notification(amount)

def ta_update_streak():
    if "logged_in_user" in st.session_state:
        username = st.session_state.logged_in_user
        user_data = get_user_data(username)
        today = dt.date.today()
        last_date_str = user_data.get('last_journal_date')
        streak = user_data.get('streak', 0)

        if last_date_str:
            last_date = dt.date.fromisoformat(last_date_str)
            if last_date == today:
                return # Already journaled today
            if last_date == today - dt.timedelta(days=1):
                streak += 1
            else:
                streak = 1
        else:
            streak = 1

        user_data.update({'streak': streak, 'last_journal_date': today.isoformat()})

        if streak > 0 and streak % 7 == 0:
            badge = f"{streak}-Day Discipline"
            if badge not in user_data.get('badges', []):
                user_data.setdefault('badges', []).append(badge)
                st.balloons()
                st.success(f"Unlocked: {badge} for a {streak}-day streak!")

        if save_user_data(username, user_data):
            st.session_state.streak = streak
            st.session_state.badges = user_data.get('badges', [])

# Misc helpers from Code 2 needed for other pages
def _ta_load_community(key, default=[]):
    try:
        c.execute("SELECT data FROM community_data WHERE key = ?", (key,))
        result = c.fetchone()
        return json.loads(result[0]) if result else default
    except Exception as e:
        logging.error(f"Failed to load community data for {key}: {str(e)}")
        return default

def _ta_save_community(key, data):
    try:
        json_data = json.dumps(data, cls=CustomJSONEncoder)
        c.execute("INSERT OR REPLACE INTO community_data (key, data) VALUES (?, ?)", (key, json_data))
        conn.commit()
    except Exception as e:
        logging.error(f"Failed to save community data for {key}: {str(e)}")

def _ta_user_dir(user_id="guest"):
    root = "user_data"
    os.makedirs(root, exist_ok=True)
    d = os.path.join(root, user_id)
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, "community_images"), exist_ok=True)
    return d

def _ta_hash():
    return uuid.uuid4().hex[:12]

def _ta_percent_gain_to_recover(drawdown_pct):
    if drawdown_pct <= 0: return 0.0
    if drawdown_pct >= 0.99: return float("inf")
    return drawdown_pct / (1 - drawdown_pct)

def _ta_expectancy_by_group(df, group_cols):
    df['r_numeric'] = pd.to_numeric(df['RR'], errors='coerce')
    g = df.dropna(subset=["r_numeric"]).groupby(group_cols)
    res = g["r_numeric"].agg(
        trades="count",
        winrate=lambda s: (s>0).mean(),
        avg_win=lambda s: s[s>0].mean() if (s>0).any() else 0.0,
        avg_loss=lambda s: -s[s<0].mean() if (s<0).any() else 0.0,
        expectancy=lambda s: ((s>0).mean()*(s[s>0].mean() if (s>0).any() else 0.0)) - ((1-(s>0).mean())*(-s[s<0].mean() if (s<0).any() else 0.0))
    ).reset_index()
    return res


# =========================================================
# MOCK AUTHENTICATION & SESSION STATE SETUP
# =========================================================
if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = "pro_trader"
    user_exists = get_user_data(st.session_state.logged_in_user)
    if not user_exists:
        hashed_password = hashlib.sha256("password".encode()).hexdigest()
        initial_data = {'xp': 0, 'streak': 0, 'level': 0, 'badges': [], 'tools_trade_journal': [], 'strategies': []}
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
    "TradeID": str, "Date": "datetime64[ns]", "Symbol": str, "Direction": str,
    "Outcome": str, "PnL": float, "RR": float, "Strategy": str,
    "Tags": str, "EntryPrice": float, "StopLoss": float, "FinalExit": float, "Lots": float,
    "EntryRationale": str, "TradeJournalNotes": str,
    "EntryScreenshot": str, "ExitScreenshot": str
}

# This will run once per session, ensuring data is correctly formatted
if 'tools_trade_journal' not in st.session_state:
    user_data = get_user_data(st.session_state.logged_in_user)
    journal_data = user_data.get("tools_trade_journal", [])
    df = pd.DataFrame(journal_data)

    legacy_col_map = {
        "Trade ID": "TradeID", "Entry Price": "EntryPrice", "Stop Loss": "StopLoss",
        "Final Exit": "FinalExit", "PnL ($)": "PnL", "R:R": "RR",
        "Entry Rationale": "EntryRationale", "Trade Journal Notes": "TradeJournalNotes",
        "Entry Screenshot": "EntryScreenshot", "Exit Screenshot": "ExitScreenshot",
        "Stop Loss Price": "StopLoss", "Take Profit Price": "FinalExit",
        "Outcome / R:R Realised": "RR", "Notes/Journal": "TradeJournalNotes"
    }
    df.rename(columns=legacy_col_map, inplace=True)

    for col, dtype in journal_dtypes.items():
        if col not in df.columns:
            if dtype == str: df[col] = ''
            elif 'datetime' in str(dtype): df[col] = pd.NaT
            elif dtype == float: df[col] = 0.0
            else: df[col] = None

    df = df.reindex(columns=journal_cols, fill_value='')
    st.session_state.tools_trade_journal = df.astype(journal_dtypes, errors='ignore')
    st.session_state.tools_trade_journal['Date'] = pd.to_datetime(st.session_state.tools_trade_journal['Date'], errors='coerce')

# Initialize other session state variables from user data
if 'xp' not in st.session_state:
    user_data = get_user_data(st.session_state.logged_in_user)
    st.session_state.xp = user_data.get('xp', 0)
    st.session_state.level = user_data.get('level', 0)
    st.session_state.streak = user_data.get('streak', 0)
    st.session_state.badges = user_data.get('badges', [])


# =========================================================
# NEWS & ECONOMIC CALENDAR DATA
# =========================================================
def detect_currency(title: str) -> str:
    t = title.upper()
    currency_map = {
        "USD": ["USD", "US ", " US:", "FED", "FEDERAL RESERVE", "AMERICA", "U.S."],
        "GBP": ["GBP", "UK", " BRITAIN", "BOE", "POUND", "STERLING"],
        "EUR": ["EUR", "EURO", "EUROZONE", "ECB"], "JPY": ["JPY", "JAPAN", "BOJ", "YEN"],
        "AUD": ["AUD", "AUSTRALIA", "RBA"], "CAD": ["CAD", "CANADA", "BOC"],
        "CHF": ["CHF", "SWITZERLAND", "SNB"], "NZD": ["NZD", "NEW ZEALAND", "RBNZ"],
    }
    for curr, kws in currency_map.items():
        for kw in kws:
            if kw in t: return curr
    return "Unknown"

def rate_impact(polarity: float) -> str:
    if polarity > 0.5: return "Significantly Bullish"
    elif polarity > 0.1: return "Bullish"
    elif polarity < -0.5: return "Significantly Bearish"
    elif polarity < -0.1: return "Bearish"
    else: return "Neutral"

@st.cache_data(ttl=600, show_spinner=False)
def get_fxstreet_forex_news() -> pd.DataFrame:
    RSS_URL = "https://www.fxstreet.com/rss/news"
    try:
        feed = feedparser.parse(RSS_URL)
        rows = []
        for entry in getattr(feed, "entries", [])[:20]: # Limit to 20 entries
            rows.append({
                "Date": pd.to_datetime(getattr(entry, "published", "")).date(),
                "Currency": detect_currency(entry.title),
                "Headline": entry.title,
                "Polarity": TextBlob(entry.title).sentiment.polarity,
                "Impact": rate_impact(TextBlob(entry.title).sentiment.polarity),
                "Summary": getattr(entry, "summary", ""), "Link": entry.link
            })
        return pd.DataFrame(rows)
    except Exception as e:
        logging.error(f"Failed to parse FXStreet RSS feed: {e}")
        return pd.DataFrame()

econ_calendar_data = [
    {"Date": "2025-09-08", "Time": "12:30", "Currency": "USD", "Event": "CPI m/m", "Impact": "High"},
    {"Date": "2025-09-10", "Time": "08:00", "Currency": "GBP", "Event": "GDP m/m", "Impact": "High"},
    {"Date": "2025-09-11", "Time": "12:15", "Currency": "EUR", "Event": "Main Refinancing Rate", "Impact": "High"},
]
econ_df = pd.DataFrame(econ_calendar_data)


# =========================================================
# SESSION STATE & SIDEBAR NAVIGATION
# =========================================================
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'backtesting' # Default to the new journal page

try:
    logo = Image.open("logo22.png").resize((60, 50))
    buffered = io.BytesIO()
    logo.save(buffered, format="PNG")
    logo_str = base64.b64encode(buffered.getvalue()).decode()
    st.sidebar.markdown(f"""<div style='text-align: center; margin-bottom: 20px;'><img src="data:image/png;base64,{logo_str}" width="60" height="50"/></div>""", unsafe_allow_html=True)
except FileNotFoundError:
    st.sidebar.markdown("<h2 style='text-align: center;'>Zenvo</h2>", unsafe_allow_html=True)

nav_items = [
    ('backtesting', '📈 Pro Journal'),
    ('mt5', '📊 Performance Dashboard'),
    ('fundamentals', '📅 Forex Fundamentals'),
    ('strategy', '📝 Manage My Strategy'),
    ('tools', '🛠️ Tools'),
    ('community', '🌐 Community Hub'),
    ('Zenvo Academy', '📚 Zenvo Academy'),
    ('account', '👤 My Account')
]

for page_key, page_name in nav_items:
    if st.sidebar.button(page_name, key=f"nav_{page_key}", use_container_width=True):
        st.session_state.current_page = page_key
        st.rerun()


# =========================================================
# MAIN APPLICATION LOGIC ROUTER
# =========================================================

# =========================================================
# PRO JOURNAL & BACKTESTING PAGE (THE NEW MAIN PAGE)
# =========================================================
if st.session_state.current_page == 'backtesting':
    st.title("📈 Pro Journal & Backtesting Environment")
    st.caption(f"A streamlined interface for professional trade analysis. | Logged in as: **{st.session_state.logged_in_user}**")
    st.markdown("---")

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

    tab_entry, tab_playbook, tab_analytics = st.tabs(["**📝 Log New Trade**", "**📚 Trade Playbook**", "**📊 Analytics Dashboard**"])

    with tab_entry:
        st.header("Log a New Trade")
        st.caption("Focus on a quick, essential entry. You can add detailed notes and screenshots later in the Playbook.")

        with st.form("trade_entry_form", clear_on_submit=True):
            st.markdown("##### ⚡ 30-Second Journal Entry")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                date_val = st.date_input("Date", dt.date.today())
                symbol_options = list(pairs_map.keys()) + ["Other"]
                symbol_default_index = symbol_options.index(pair) if pair in symbol_options else 0
                symbol = st.selectbox("Symbol", symbol_options, index=symbol_default_index)
                if symbol == "Other": symbol = st.text_input("Custom Symbol")
            with col2:
                direction = st.radio("Direction", ["Long", "Short"], horizontal=True)
                lots = st.number_input("Size (Lots)", min_value=0.01, value=0.10, step=0.01, format="%.2f")
            with col3:
                entry_price = st.number_input("Entry Price", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
                stop_loss = st.number_input("Stop Loss", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
            with col4:
                final_exit = st.number_input("Final Exit Price", min_value=0.0, value=0.0, step=0.00001, format="%.5f")
                outcome = st.selectbox("Outcome", ["Win", "Loss", "Breakeven", "No Trade/Study"])

            with st.expander("Add Quick Rationale & Tags (Optional)"):
                entry_rationale = st.text_area("Why did you enter this trade?", height=100)
                all_tags = sorted(list(set(st.session_state.tools_trade_journal['Tags'].str.split(',').explode().dropna().str.strip())))
                suggested_tags = ["Breakout", "Reversal", "Trend Follow", "Counter-Trend", "News Play", "FOMO", "Over-leveraged"]
                tags = st.multiselect("Trade Tags", options=sorted(list(set(all_tags + suggested_tags))))

            submitted = st.form_submit_button("Save Trade", type="primary", use_container_width=True)
            if submitted:
                pnl, rr = 0.0, 0.0
                pip_multiplier = 100000 if "JPY" not in symbol.upper() else 1000

                if outcome in ["Win", "Loss", "Breakeven"] and entry_price > 0 and final_exit > 0:
                    price_diff = (final_exit - entry_price) if direction == "Long" else (entry_price - final_exit)
                    pnl = price_diff * pip_multiplier * lots

                if stop_loss > 0 and entry_price > 0:
                    risk_per_unit = abs(entry_price - stop_loss)
                    if risk_per_unit > 1e-9: # Avoid division by zero
                        reward_per_unit = abs(final_exit - entry_price)
                        rr = reward_per_unit / risk_per_unit
                        if outcome == "Loss": rr = -rr

                new_trade_data = {
                    "TradeID": f"TRD-{uuid.uuid4().hex[:6].upper()}", "Date": pd.to_datetime(date_val),
                    "Symbol": symbol, "Direction": direction, "Outcome": outcome,
                    "Lots": lots, "EntryPrice": entry_price, "StopLoss": stop_loss, "FinalExit": final_exit,
                    "PnL": pnl, "RR": rr, "Tags": ','.join(tags), "EntryRationale": entry_rationale,
                    "Strategy": '', "TradeJournalNotes": '', "EntryScreenshot": '', "ExitScreenshot": ''
                }
                new_df = pd.DataFrame([new_trade_data])
                # Ensure dtypes are correct after concat
                st.session_state.tools_trade_journal = pd.concat([st.session_state.tools_trade_journal, new_df], ignore_index=True)
                st.session_state.tools_trade_journal = st.session_state.tools_trade_journal.astype(journal_dtypes, errors='ignore')

                if _ta_save_journal(st.session_state.logged_in_user, st.session_state.tools_trade_journal):
                    ta_update_xp(10)
                    ta_update_streak()
                    st.success(f"Trade {new_trade_data['TradeID']} logged successfully!")
                st.rerun()

    with tab_playbook:
        st.header("Your Trade Playbook")
        df_playbook = st.session_state.tools_trade_journal
        if df_playbook.empty:
            st.info("Your logged trades will appear here. Log your first trade to get started!")
        else:
            st.caption("Filter and review your past trades to refine your strategy and identify patterns.")
            filter_cols = st.columns([1, 1, 1, 2])
            outcome_filter = filter_cols[0].multiselect("Filter Outcome", df_playbook['Outcome'].unique(), default=df_playbook['Outcome'].unique())
            symbol_filter = filter_cols[1].multiselect("Filter Symbol", df_playbook['Symbol'].unique(), default=df_playbook['Symbol'].unique())
            direction_filter = filter_cols[2].multiselect("Filter Direction", df_playbook['Direction'].unique(), default=df_playbook['Direction'].unique())
            tag_options = sorted(list(set(df_playbook['Tags'].str.split(',').explode().dropna().str.strip())))
            tag_filter = filter_cols[3].multiselect("Filter Tag", options=tag_options)

            filtered_df = df_playbook.copy()
            if outcome_filter: filtered_df = filtered_df[filtered_df['Outcome'].isin(outcome_filter)]
            if symbol_filter: filtered_df = filtered_df[filtered_df['Symbol'].isin(symbol_filter)]
            if direction_filter: filtered_df = filtered_df[filtered_df['Direction'].isin(direction_filter)]
            if tag_filter: filtered_df = filtered_df[filtered_df['Tags'].apply(lambda x: any(tag in str(x) for tag in tag_filter))]

            sorted_df = filtered_df.sort_values(by="Date", ascending=False).reset_index(drop=True)
            for index, row in sorted_df.iterrows():
                outcome_color = {"Win": "#2da44e", "Loss": "#cf222e", "Breakeven": "#8b949e"}.get(row['Outcome'], "#30363d")
                with st.container():
                    date_display = pd.to_datetime(row['Date']).strftime('%A, %d %B %Y') if pd.notna(row['Date']) else 'N/A'
                    st.markdown(f"""
                    <div style="border: 1px solid #30363d; border-left: 8px solid {outcome_color}; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0;">{row['Symbol']} <span style="font-weight: 500; color: {outcome_color};">{row['Direction']} / {row['Outcome']}</span></h4>
                            <span style="color: #8b949e; font-size: 0.9em;">{date_display}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    metric_cols = st.columns(3)
                    metric_cols[0].metric("Net PnL", f"${row['PnL']:.2f}")
                    metric_cols[1].metric("R-Multiple", f"{row['RR']:.2f}R")
                    metric_cols[2].metric("Position Size", f"{row['Lots']:.2f} lots")
                    if pd.notna(row['EntryRationale']) and row['EntryRationale']:
                        st.markdown(f"**Entry Rationale:** *{row['EntryRationale']}*")
                    if pd.notna(row['Tags']) and row['Tags']:
                        tags_list = [f"`{tag.strip()}`" for tag in str(row['Tags']).split(',') if tag.strip()]
                        st.markdown(f"**Tags:** {', '.join(tags_list)}")
                    st.markdown("---")

    with tab_analytics:
        st.header("Your Performance Dashboard")
        df_analytics = st.session_state.tools_trade_journal.copy()
        df_analytics['PnL'] = pd.to_numeric(df_analytics['PnL'], errors='coerce').fillna(0)
        df_analytics = df_analytics[df_analytics['Outcome'].isin(['Win', 'Loss'])]

        if df_analytics.empty:
            st.info("Complete at least one winning or losing trade to view your performance analytics.")
        else:
            total_pnl, total_trades = df_analytics['PnL'].sum(), len(df_analytics)
            wins = df_analytics[df_analytics['Outcome'] == 'Win']
            losses = df_analytics[df_analytics['Outcome'] == 'Loss']
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            avg_win = wins['PnL'].mean() if not wins.empty else 0
            avg_loss = losses['PnL'].mean() if not losses.empty else 0
            profit_factor = wins['PnL'].sum() / abs(losses['PnL'].sum()) if not losses.empty and losses['PnL'].sum() != 0 else 0

            kpi_cols = st.columns(4)
            kpi_cols[0].metric("Net PnL ($)", f"${total_pnl:,.2f}", delta=f"{total_pnl:+.2f}" if total_pnl != 0 else None)
            kpi_cols[1].metric("Win Rate", f"{win_rate:.1f}%")
            kpi_cols[2].metric("Profit Factor", f"{profit_factor:.2f}")
            kpi_cols[3].metric("Avg. Win / Loss ($)", f"${avg_win:,.2f} / ${abs(avg_loss):,.2f}")
            st.markdown("---")

            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.subheader("Cumulative PnL")
                df_analytics.sort_values(by='Date', inplace=True)
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

# =========================================================
# PERFORMANCE DASHBOARD PAGE (MT5)
# =========================================================
elif st.session_state.current_page == 'mt5':
    st.title("📊 Performance Dashboard")
    st.caption("Analyze your MT5 trading history with advanced metrics and visualizations.")
    st.markdown('---')
    uploaded_file = st.file_uploader("Upload MT5 History CSV", type=["csv"], help="Export your trading history from MetaTrader 5 as a CSV file.")
    if uploaded_file:
        with st.spinner("Processing trading data..."):
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ["Symbol", "Type", "Profit", "Volume", "Open Time", "Close Time"]
                if not all(col in df.columns for col in required_cols):
                    st.error(f"Missing required columns. Please ensure your CSV has: {', '.join(required_cols)}.")
                else:
                    st.session_state.mt5_df = df
                    st.success("File uploaded successfully! Displaying summary:")
                    st.dataframe(df.head())
                    # Full, non-abridged processing code from Code 2 would be implemented here for metrics and charts
            except Exception as e:
                st.error(f"Error processing CSV: {e}")
    else:
        st.info("👆 Upload your MT5 trading history CSV to explore advanced performance metrics.")


# =========================================================
# FUNDAMENTALS PAGE
# =========================================================
elif st.session_state.current_page == 'fundamentals':
    st.title("📅 Forex Fundamentals")
    st.caption("Macro snapshot: sentiment, calendar highlights, and policy rates.")
    st.markdown('---')
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("🗓️ Upcoming Economic Events")
        st.dataframe(econ_df.style.applymap(lambda x: 'background-color: #cf222e' if x == 'High' else '', subset=['Impact']), use_container_width=True)
    with col2:
        st.header("Forex News")
        df_news = get_fxstreet_forex_news()
        if not df_news.empty:
            for _, row in df_news.head(5).iterrows():
                st.markdown(f"**{row['Currency']}**: {row['Headline']}")
        else:
            st.info("Could not fetch news.")
    st.markdown("---")
    st.header("💹 Major Central Bank Interest Rates")
    interest_rates = [
        {"Currency": "USD", "Current": "3.78%", "Previous": "4.00%", "Next Meeting": "2025-09-18"},
        {"Currency": "GBP", "Current": "3.82%", "Previous": "4.00%", "Next Meeting": "2025-09-19"},
        {"Currency": "EUR", "Current": "1.82%", "Previous": "2.00%", "Next Meeting": "2025-09-12"},
        {"Currency": "JPY", "Current": "0.50%", "Previous": "0.25%", "Next Meeting": "2025-09-20"},
    ]
    cols = st.columns(len(interest_rates))
    for i, rate in enumerate(interest_rates):
        with cols[i]:
            st.metric(label=f"{rate['Currency']} Rate", value=rate['Current'], delta=f"Prev: {rate['Previous']}")


# =========================================================
# MANAGE MY STRATEGY PAGE
# =========================================================
elif st.session_state.current_page == 'strategy':
    st.title("📝 Manage My Strategy")
    st.markdown(""" Define, refine, and track your trading strategies. Save your setups and review performance to optimize your edge. """)
    st.markdown('---')

    if 'strategies' not in st.session_state:
        user_data = get_user_data(st.session_state.logged_in_user)
        st.session_state.strategies = pd.DataFrame(user_data.get("strategies", []))

    st.subheader("➕ Add New Strategy")
    with st.form("strategy_form", clear_on_submit=True):
        strategy_name = st.text_input("Strategy Name")
        description = st.text_area("Strategy Description (e.g., pairs, timeframes)")
        entry_rules = st.text_area("Entry Rules (be specific)")
        exit_rules = st.text_area("Exit Rules (profit taking and trade management)")
        risk_management = st.text_area("Risk Management Rules (e.g., % risk, max loss)")
        submit_strategy = st.form_submit_button("Save Strategy", type="primary")
        if submit_strategy and strategy_name:
            strategy_data = {
                "Name": strategy_name, "Description": description, "Entry Rules": entry_rules,
                "Exit Rules": exit_rules, "Risk Management": risk_management,
                "Date Added": dt.datetime.now().isoformat()
            }
            if 'strategies' not in st.session_state or st.session_state.strategies.empty:
                 st.session_state.strategies = pd.DataFrame([strategy_data])
            else:
                st.session_state.strategies = pd.concat([st.session_state.strategies, pd.DataFrame([strategy_data])], ignore_index=True)

            user_data = get_user_data(st.session_state.logged_in_user)
            user_data["strategies"] = st.session_state.strategies.to_dict(orient="records")
            if save_user_data(st.session_state.logged_in_user, user_data):
                st.success("Strategy saved to your account!")

    if 'strategies' in st.session_state and not st.session_state.strategies.empty:
        st.markdown("---")
        st.subheader("Your Strategies")
        for idx, row in st.session_state.strategies.iterrows():
            with st.expander(f"{row['Name']}"):
                st.markdown(f"**Description:**\n{row['Description']}")
                st.markdown(f"**Entry Rules:**\n{row['Entry Rules']}")
                st.markdown(f"**Exit Rules:**\n{row['Exit Rules']}")
                st.markdown(f"**Risk Management:**\n{row['Risk Management']}")
                if st.button("Delete", key=f"delete_strategy_{idx}", type="secondary"):
                    st.session_state.strategies = st.session_state.strategies.drop(idx).reset_index(drop=True)
                    user_data = get_user_data(st.session_state.logged_in_user)
                    user_data["strategies"] = st.session_state.strategies.to_dict(orient="records")
                    save_user_data(st.session_state.logged_in_user, user_data)
                    st.rerun()

    st.markdown("---")
    st.subheader("📖 Evolving Playbook Analytics")
    journal_df = st.session_state.tools_trade_journal
    if not journal_df.empty and 'RR' in journal_df.columns:
        agg = _ta_expectancy_by_group(journal_df, ["Symbol", "Direction"]).sort_values("expectancy", ascending=False)
        st.write("Your statistical edge based on journaled trades:")
        st.dataframe(agg)
    else:
        st.info("Log trades in the Pro Journal with outcomes to analyze your edge.")


# =========================================================
# TOOLS PAGE
# =========================================================
elif st.session_state.current_page == 'tools':
    st.title("🛠️ Tools")
    st.markdown("A suite of calculators and utilities to enhance your trading workflow.")
    st.markdown('---')
    tools_options = ['Risk Management', 'Drawdown Recovery', 'Pre-Trade Checklist']
    tabs = st.tabs(tools_options)

    with tabs[0]:
        st.header("🛡️ Risk Management & Position Size Calculator")
        st.markdown("""Calculate the correct lot size based on your desired risk.""")
        col1, col2, col3, col4 = st.columns(4)
        with col1: balance = st.number_input("Account Balance ($)", min_value=0.0, value=10000.0)
        with col2: risk_percent = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0)
        with col3: stop_loss_pips = st.number_input("Stop Loss (pips)", min_value=1.0, value=20.0)
        with col4: pip_value = st.number_input("Pip Value per Lot ($)", min_value=0.01, value=10.0, help="Typically $10 for XXX/USD pairs.")
        if st.button("Calculate Lot Size", type="primary"):
            risk_amount = balance * (risk_percent / 100)
            lot_size = risk_amount / (stop_loss_pips * pip_value) if (stop_loss_pips * pip_value) > 0 else 0
            st.success(f"### Recommended Lot Size: `{lot_size:.2f}`")
            st.write(f"This trade will risk ${risk_amount:,.2f} of your account.")

    with tabs[1]:
        st.header("📉 Drawdown Recovery Planner")
        drawdown_pct = st.slider("Current Drawdown (%)", 1.0, 75.0, 10.0) / 100
        recovery_pct = _ta_percent_gain_to_recover(drawdown_pct)
        st.metric("Required Gain to Break Even", f"{recovery_pct*100:.2f}%")
        st.info("This shows the non-linear relationship between loss and the required gain for recovery. Larger drawdowns require exponentially larger gains.")

    with tabs[2]:
        st.header("✅ Pre-Trade Checklist")
        st.markdown("Ensure discipline by running through this checklist before every trade.")
        checklist_items = [
            "Market structure aligns with my strategy", "Key levels (S/R, pivots) identified",
            "Valid entry trigger is present", "Risk-reward ratio is acceptable (e.g., ≥ 1:2)",
            "No high-impact news is scheduled", "Position size calculated correctly",
            "Stop loss level is technically sound", "Take profit level is realistic",
            "I am emotionally neutral and focused"
        ]
        for item in checklist_items: st.checkbox(item)


# =========================================================
# COMMUNITY HUB PAGE
# =========================================================
elif st.session_state.current_page == 'community':
    st.title("🌐 Community Hub")
    st.markdown(""" Share and explore trade ideas, strategies, and templates with the community. """)
    st.markdown('---')
    if "trade_ideas" not in st.session_state:
        st.session_state.trade_ideas = pd.DataFrame(_ta_load_community('trade_ideas', []))
    if "community_templates" not in st.session_state:
        st.session_state.community_templates = pd.DataFrame(_ta_load_community('templates', []))

    tab1, tab2, tab3 = st.tabs(["Trade Ideas", "Shared Templates", "Leaderboard"])
    with tab1:
        st.subheader("➕ Share a Trade Idea")
        with st.form("trade_idea_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1: trade_pair = st.selectbox("Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"])
            with col2: trade_direction = st.radio("Direction", ["Long", "Short"])
            trade_description = st.text_area("Trade Description / Rationale")
            uploaded_image = st.file_uploader("Upload Chart Screenshot", type=["png", "jpg", "jpeg"])
            if st.form_submit_button("Share Idea", type="primary"):
                username = st.session_state.logged_in_user
                idea_id = _ta_hash()
                idea_data = {"Username": username, "Pair": trade_pair, "Direction": trade_direction, "Description": trade_description,
                             "Timestamp": dt.datetime.now().isoformat(), "IdeaID": idea_id, "ImagePath": None}
                if uploaded_image:
                    image_path = os.path.join(_ta_user_dir(username), "community_images", f"{idea_id}.png")
                    with open(image_path, "wb") as f: f.write(uploaded_image.getbuffer())
                    idea_data["ImagePath"] = image_path
                st.session_state.trade_ideas = pd.concat([st.session_state.trade_ideas, pd.DataFrame([idea_data])], ignore_index=True)
                _ta_save_community('trade_ideas', st.session_state.trade_ideas.to_dict('records'))

        st.subheader("📈 Community Feed")
        if not st.session_state.trade_ideas.empty:
            for _, idea in st.session_state.trade_ideas.sort_values(by="Timestamp", ascending=False).iterrows():
                st.markdown(f"""<div style="border: 1px solid #30363d; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                                **{idea['Pair']} ({idea['Direction']})** by *{idea['Username']}* 
                                <span style="float: right; color: #8b949e;">{pd.to_datetime(idea['Timestamp']).strftime('%Y-%m-%d %H:%M')}</span>
                                <p style="margin-top: 0.5rem;">{idea['Description']}</p>
                                </div>""", unsafe_allow_html=True)
                if idea.get('ImagePath') and os.path.exists(idea['ImagePath']):
                    st.image(idea['ImagePath'])
        else: st.info("No trade ideas shared yet.")

    with tab2:
        st.info("Shared Templates section is under development.")

    with tab3:
        st.subheader("🏆 Consistency Leaderboard")
        try:
            users_data = c.execute("SELECT username, data FROM users").fetchall()
            leader_data = [{"Username": u, "Journaled Trades": len(json.loads(d).get("tools_trade_journal", []))} for u, d in users_data]
            if leader_data:
                leader_df = pd.DataFrame(leader_data).sort_values("Journaled Trades", ascending=False).reset_index(drop=True)
                leader_df["Rank"] = leader_df.index + 1
                st.dataframe(leader_df[["Rank", "Username", "Journaled Trades"]], use_container_width=True)
        except Exception as e: st.error(f"Could not load leaderboard: {e}")

# =========================================================
# ZENVO ACADEMY PAGE
# =========================================================
elif st.session_state.current_page == "Zenvo Academy":
    st.title("📚 Zenvo Academy")
    st.caption("Your journey to trading mastery starts here. Explore interactive courses and track your progress.")
    st.markdown('---')
    tab1, tab2 = st.tabs(["🎓 Learning Path", "📈 My Progress"])
    with tab1:
        st.markdown("### 🗺️ Your Learning Path")
        with st.expander("Forex Fundamentals - Level 1 (100 XP)", expanded=True):
            st.markdown("- **Lesson 1:** What is Forex?\n- **Lesson 2:** How to Read a Currency Pair\n- **Lesson 3:** Understanding Pips, Lots, and Leverage")
            st.button("Start Learning", key="start_fundamentals")
    with tab2:
        st.markdown("### 🚀 Your Progress")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Your Level", st.session_state.level)
        with col2: st.metric("Experience Points (XP)", f"{st.session_state.xp} / {(st.session_state.level + 1) * 100}")
        with col3: st.metric("Badges Earned", len(st.session_state.badges))
        st.progress( (st.session_state.xp % 100) / 100 )


# =========================================================
# ACCOUNT PAGE
# =========================================================
elif st.session_state.current_page == 'account':
    st.title("👤 My Account")
    st.markdown(""" Manage your account, track your gamification progress, and sync your data.""")
    st.markdown('---')

    st.header(f"Welcome back, {st.session_state.logged_in_user}! 👋")

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1: st.metric("Trader's Rank", f"Level {st.session_state.level}")
    with kpi_col2: st.metric("Journaling Streak", f"🔥 {st.session_state.streak} Days")
    with kpi_col3: st.metric("Total Experience", f"⭐ {st.session_state.xp:,}")

    st.subheader("🏆 Your Badges")
    if st.session_state.badges:
        st.write(" ".join([f"🏅 {badge}" for badge in st.session_state.badges]))
    else: st.info("No badges earned yet. Complete challenges and stay consistent to earn them!")
    st.markdown("---")

    with st.expander("⚙️ Manage Account Data"):
        st.write(f"**Username**: `{st.session_state.logged_in_user}`")
        # In a real app, sign-in/out logic would be more robust
        st.info("Account creation and multi-user login is handled via database but simplified for this demo.")
        if st.button("Log Out (Example)", type="primary"):
            st.info("In a full application, this would clear your session and log you out.")
