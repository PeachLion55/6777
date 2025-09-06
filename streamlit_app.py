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
import io
import base64
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# GLOBAL CSS & PAGE CONFIGURATION
# =========================================================
st.set_page_config(page_title="Zenvo Backtesting", layout="wide")

st.markdown(
    """
    <style>
    /* --- Global Horizontal Line Style --- */
    hr {
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        border-top: 1px solid #4d7171 !important;
        border-bottom: none !important;
        background-color: transparent !important;
        height: 1px !important;
    }
    /* Hide Streamlit UI elements */
    #MainMenu, footer, [data-testid="stDecoration"] {
        visibility: hidden !important;
    }
    .css-18e3th9, .css-1d391kg {
        padding-top: 1rem !important; /* Adjust top padding for the main content area */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Gridline background settings ---
grid_color = "#58b3b1"
grid_opacity = 0.16
grid_size = 40

r = int(grid_color[1:3], 16)
g = int(grid_color[3:5], 16)
b = int(grid_color[5:7], 16)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #000000;
        background-image:
            linear-gradient(rgba({r}, {g}, {b}, {grid_opacity}) 1px, transparent 1px),
            linear-gradient(90deg, rgba({r}, {g}, {b}, {grid_opacity}) 1px, transparent 1px);
        background-size: {grid_size}px {grid_size}px;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# LOGGING SETUP
# =========================================================
logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================================================
# DATABASE & HELPER FUNCTIONS
# =========================================================
DB_FILE = "users.db"

# Custom JSON encoder for handling special data types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.datetime, dt.date)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super().default(obj)

# --- Database Connection ---
try:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, data TEXT)''')
    conn.commit()
    logging.info("SQLite database initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize SQLite database: {str(e)}")
    st.error(f"Database initialization failed: {str(e)}")

# --- Helper Function to Save Journal ---
def _ta_save_journal(username, journal_df):
    try:
        c.execute("SELECT data FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        user_data = json.loads(result[0]) if result else {}
        user_data["tools_trade_journal"] = journal_df.replace({pd.NA: None, float('nan'): None}).to_dict('records')
        c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
        conn.commit()
        logging.info(f"Journal saved for user {username}: {len(journal_df)} trades")
        return True
    except Exception as e:
        logging.error(f"Failed to save journal for {username}: {str(e)}")
        st.error(f"Failed to save journal: {str(e)}")
        return False

# --- Gamification Helpers ---
def ta_update_xp(amount):
    if "logged_in_user" in st.session_state:
        username = st.session_state.logged_in_user
        c.execute("SELECT data FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        if result:
            user_data = json.loads(result[0]) if result[0] else {}
            user_data['xp'] = user_data.get('xp', 0) + amount
            st.toast(f"⭐ +{amount} XP Earned!")
            c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
            conn.commit()

def ta_update_streak():
    if "logged_in_user" in st.session_state:
        username = st.session_state.logged_in_user
        c.execute("SELECT data FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        if result:
            user_data = json.loads(result[0]) if result[0] else {}
            today = dt.date.today()
            last_date_str = user_data.get('last_journal_date')
            streak = user_data.get('streak', 0)
            
            if last_date_str:
                last_date = dt.date.fromisoformat(last_date_str)
                if last_date == today - dt.timedelta(days=1):
                    streak += 1
                elif last_date < today - dt.timedelta(days=1):
                    streak = 1
            else:
                streak = 1
                
            user_data['streak'] = streak
            user_data['last_journal_date'] = today.isoformat()
            c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
            conn.commit()
            st.session_state.streak = streak


# =========================================================
# BACKTESTING PAGE LOGIC
# =========================================================

# --- Authentication Mockup / User Check ---
# In a real multi-user app, this would be a full login system.
# For this standalone script, we will simulate a logged-in user.
if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = "default_user" # Simulate login for saving features
    # Check if the user exists, if not, create one
    c.execute("SELECT username FROM users WHERE username = ?", (st.session_state.logged_in_user,))
    if not c.fetchone():
        hashed_password = hashlib.sha256("password".encode()).hexdigest()
        initial_data = json.dumps({})
        c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)", 
                  (st.session_state.logged_in_user, hashed_password, initial_data))
        conn.commit()
        logging.info(f"Created default user: {st.session_state.logged_in_user}")

# --- Page Title and Header ---
st.title("📈 Backtesting")
st.caption("Live TradingView chart for backtesting and enhanced trading journal for tracking and analyzing trades.")
st.markdown('---')

# --- JOURNAL SCHEMA DEFINITION & INITIALIZATION ---
# Define journal columns and dtypes
journal_cols = [
    "Date", "Symbol", "Weekly Bias", "Daily Bias", "4H Structure", "1H Structure",
    "Positive Correlated Pair & Bias", "Potential Entry Points", "5min/15min Setup?",
    "Entry Conditions", "Planned R:R", "News Filter", "Alerts", "Concerns", "Emotions",
    "Confluence Score 1-7", "Outcome / R:R Realised", "Notes/Journal",
    "Entry Price", "Stop Loss Price", "Take Profit Price", "Lots"
]

journal_dtypes = {
    "Date": "datetime64[ns]", "Symbol": str, "Weekly Bias": str, "Daily Bias": str,
    "4H Structure": str, "1H Structure": str, "Positive Correlated Pair & Bias": str,
    "Potential Entry Points": str, "5min/15min Setup?": str, "Entry Conditions": str,
    "Planned R:R": str, "News Filter": str, "Alerts": str, "Concerns": str, "Emotions": str,
    "Confluence Score 1-7": float, "Outcome / R:R Realised": str, "Notes/Journal": str,
    "Entry Price": float, "Stop Loss Price": float, "Take Profit Price": float, "Lots": float
}

# Load user's journal from database or initialize a new one
if "tools_trade_journal" not in st.session_state:
    try:
        username = st.session_state.logged_in_user
        c.execute("SELECT data FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        user_data = json.loads(result[0]) if result and result[0] else {}
        journal_data = user_data.get("tools_trade_journal", [])
        
        # Load data, ensuring schema consistency
        current_journal = pd.DataFrame(journal_data)
        if not current_journal.empty:
            missing_cols = [col for col in journal_cols if col not in current_journal.columns]
            for col in missing_cols:
                current_journal[col] = pd.Series(dtype=journal_dtypes.get(col, object))
            st.session_state.tools_trade_journal = current_journal[journal_cols].astype(journal_dtypes, errors='ignore')
        else:
            st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)

        logging.info(f"Journal for {username} loaded with {len(st.session_state.tools_trade_journal)} trades.")
    except Exception as e:
        st.error("Failed to load your journal data.")
        logging.error(f"Error loading journal for {st.session_state.logged_in_user}: {e}")
        st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes)


# Initialize drawings in session_state
if "drawings" not in st.session_state:
    st.session_state.drawings = {}
    logging.info("Initialized st.session_state.drawings")

# --- CHART & DRAWING TOOLS ---
pairs_map = {
    "EUR/USD": "FX:EURUSD", "USD/JPY": "FX:USDJPY", "GBP/USD": "FX:GBPUSD", "USD/CHF": "OANDA:USDCHF",
    "AUD/USD": "FX:AUDUSD", "NZD/USD": "OANDA:NZDUSD", "USD/CAD": "CMCMARKETS:USDCAD", "EUR/GBP": "FX:EURGBP",
    "EUR/JPY": "FX:EURJPY", "GBP/JPY": "FX:GBPJPY", "AUD/JPY": "FX:AUDJPY", "AUD/NZD": "FX:AUDNZD",
    "AUD/CAD": "FX:AUDCAD", "AUD/CHF": "FX:AUDCHF", "CAD/JPY": "FX:CADJPY", "CHF/JPY": "FX:CHFJPY",
    "EUR/AUD": "FX:EURAUD", "EUR/CAD": "FX:EURCAD", "EUR/CHF": "FX:EURCHF", "GBP/AUD": "FX:GBPAUD",
    "GBP/CAD": "FX:GBPCAD", "GBP/CHF": "FX:GBPCHF", "NZD/JPY": "FX:NZDJPY", "NZD/CAD": "FX:NZDCAD",
    "NZD/CHF": "FX:NZDCHF", "CAD/CHF": "FX:CADCHF",
}
pair = st.selectbox("Select pair", list(pairs_map.keys()), index=0, key="tv_pair")
tv_symbol = pairs_map[pair]

# TradingView widget HTML
tv_html = f"""
<div id="tradingview_widget"></div>
<script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
<script type="text/javascript">
new TradingView.widget({{
    "container_id": "tradingview_widget", "width": "100%", "height": 600, "symbol": "{tv_symbol}",
    "interval": "D", "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "en",
    "enable_publishing": false, "allow_symbol_change": true, "studies": [], "show_popup_button": true,
    "popup_width": "1000", "popup_height": "650"
}});
</script>
"""
st.components.v1.html(tv_html, height=620, scrolling=False)

# Save, Load, and Refresh buttons for drawings
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Save Drawings", key="bt_save_drawings", use_container_width=True):
        # JavaScript logic would need to be implemented to get drawings from TradingView
        st.warning("Save drawing functionality requires custom JavaScript integration.")
with col2:
    if st.button("Load Drawings", key="bt_load_drawings", use_container_width=True):
        st.warning("Load drawing functionality requires custom JavaScript integration.")
with col3:
    st.info(f"Signed in as: **{st.session_state.logged_in_user}**")


# =========================================================
# TRADING JOURNAL SECTION
# =========================================================
st.markdown("### 📝 Trading Journal")
st.markdown("Log your trades with detailed analysis, track psychological factors, and review performance.")

tab_entry, tab_analytics, tab_history = st.tabs(["📝 Log Trade", "📊 Analytics", "📜 Trade History"])

# --- Log Trade Tab ---
with tab_entry:
    st.subheader("Log a New Trade")
    with st.form("trade_entry_form"):
        c1, c2 = st.columns(2)
        with c1:
            trade_date = st.date_input("Date", value=dt.datetime.now().date())
            symbol = st.selectbox("Symbol", list(pairs_map.keys()) + ["Other"], index=0)
            if symbol == "Other":
                symbol = st.text_input("Custom Symbol")
            weekly_bias = st.selectbox("Weekly Bias", ["Bullish", "Bearish", "Neutral"])
            daily_bias = st.selectbox("Daily Bias", ["Bullish", "Bearish", "Neutral"])
            entry_price = st.number_input("Entry Price", min_value=0.0, step=0.0001, format="%.5f")
            stop_loss_price = st.number_input("Stop Loss Price", min_value=0.0, step=0.0001, format="%.5f")
        with c2:
            take_profit_price = st.number_input("Take Profit Price", min_value=0.0, step=0.0001, format="%.5f")
            lots = st.number_input("Lots", min_value=0.01, step=0.01, format="%.2f")
            entry_conditions = st.text_area("Entry Conditions")
            emotions = st.selectbox("Emotions", ["Confident", "Anxious", "Fearful", "Excited", "Frustrated", "Neutral"])
            tags = st.multiselect("Tags", ["Setup: Breakout", "Setup: Reversal", "Mistake: Overtrading", "Mistake: No Stop Loss", "Emotion: FOMO", "Emotion: Revenge"])
            notes = st.text_area("Notes/Journal")
        
        submitted = st.form_submit_button("Save Trade", type="primary", use_container_width=True)
        if submitted:
            # Simple R:R calculation
            rr = 0.0
            if (entry_price - stop_loss_price) != 0:
                rr = abs(take_profit_price - entry_price) / abs(entry_price - stop_loss_price)

            new_trade = pd.DataFrame([{
                'Date': pd.to_datetime(trade_date), 'Symbol': symbol, 'Weekly Bias': weekly_bias, 'Daily Bias': daily_bias,
                'Entry Conditions': entry_conditions, 'Emotions': emotions, 'Notes/Journal': notes, 'Lots': lots,
                'Entry Price': entry_price, 'Stop Loss Price': stop_loss_price, 'Take Profit Price': take_profit_price,
                'Planned R:R': f"1:{rr:.2f}",
                # Fill other columns with defaults
                '4H Structure': '', '1H Structure': '', 'Positive Correlated Pair & Bias': '', 'Potential Entry Points': '', '5min/15min Setup?': '',
                'News Filter': '', 'Alerts': '', 'Concerns': '', 'Confluence Score 1-7': 0.0, 'Outcome / R:R Realised': '',
            }])
            
            st.session_state.tools_trade_journal = pd.concat([st.session_state.tools_trade_journal, new_trade], ignore_index=True)
            if _ta_save_journal(st.session_state.logged_in_user, st.session_state.tools_trade_journal):
                ta_update_xp(10)
                ta_update_streak()
                st.success("Trade saved and synced to your account!")
                logging.info(f"Trade logged and saved for user {st.session_state.logged_in_user}")
            else:
                st.error("Failed to save trade to your account. It has been saved in this session only.")

# --- Analytics Tab ---
with tab_analytics:
    st.subheader("Trade Analytics")
    df_journal = st.session_state.tools_trade_journal
    if not df_journal.empty:
        total_trades = len(df_journal)
        # Simplified metrics as Outcome is not pre-filled in this version
        st.metric("Total Trades Logged", total_trades)

        st.subheader("Trades by Symbol")
        symbol_counts = df_journal['Symbol'].value_counts().reset_index()
        symbol_counts.columns = ['Symbol', 'Number of Trades']
        fig = px.bar(symbol_counts, x='Symbol', y='Number of Trades', title="Trades Logged per Symbol")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Log some trades in the 'Log Trade' tab to see your analytics.")

# --- Trade History Tab ---
with tab_history:
    st.subheader("Your Trade History")
    if not st.session_state.tools_trade_journal.empty:
        st.dataframe(st.session_state.tools_trade_journal.sort_values(by="Date", ascending=False), use_container_width=True)
        # Export options
        st.subheader("Export Journal")
        csv = st.session_state.tools_trade_journal.to_csv(index=False).encode('utf-8')
        st.download_button("Download Journal as CSV", csv, "trade_journal.csv", "text/csv", use_container_width=True)
    else:
        st.info("Your logged trades will appear here.")
        
# --- Gamification and Leaderboard section ---
st.markdown('---')
st.subheader("🏅 Gamification & Leaderboard")
col_gam1, col_gam2 = st.columns(2)

with col_gam1:
    st.write("**30-Day Journaling Discipline Challenge**")
    streak = st.session_state.get('streak', 0)
    progress = min(streak / 30.0, 1.0)
    st.progress(progress, text=f"Current Streak: {streak} Day(s)")
    if progress >= 1.0:
        st.success("Challenge Completed! You've unlocked the Discipline Badge.")
        
with col_gam2:
    st.write("**Consistency Leaderboard**")
    try:
        users = c.execute("SELECT username, data FROM users").fetchall()
        leader_data = []
        for u, d in users:
            user_d = json.loads(d) if d and d != 'null' else {}
            trades = len(user_d.get("tools_trade_journal", []))
            leader_data.append({"Username": u, "Journaled Trades": trades})

        if leader_data:
            leader_df = pd.DataFrame(leader_data).sort_values("Journaled Trades", ascending=False).reset_index(drop=True)
            leader_df["Rank"] = leader_df.index + 1
            st.dataframe(leader_df[["Rank", "Username", "Journaled Trades"]], use_container_width=True, hide_index=True)
        else:
            st.info("No leaderboard data yet.")
    except Exception as e:
        st.error("Could not load leaderboard.")
        logging.error(f"Leaderboard error: {e}")
