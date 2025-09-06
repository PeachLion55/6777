import streamlit as st
import pandas as pd
import datetime as dt
import os
import json
import hashlib
import requests # Used for Price Alerts, but not critical for journal. Kept for completeness.
# from streamlit_autorefresh import st_autorefresh # Not used in Backtesting page itself, remove from imports for this isolated file
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sqlite3
import pytz
import logging
import math
import uuid
# import glob # Not explicitly used in the Backtesting page code provided, remove from imports
# import time # Not explicitly used in the Backtesting page code provided, remove from imports
from PIL import Image # For logo (not essential for this page but general setup)
import io
import base64
# import calendar # Not explicitly used in the Backtesting page code provided, remove from imports


# =========================================================
# GLOBAL CSS & GRIDLINE SETTINGS (Copied from main app)
# =========================================================
st.markdown(
    """
    <style>
    /* --- Global Horizontal Line Style --- */
    hr {
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        border-top: 1px solid #4d7171 !important;
        border-bottom: none !important; /* Remove any bottom border */
        background-color: transparent !important; /* Ensure no background color interferes */
        height: 1px !important; /* Set a specific height */
    }

    /* Hide Streamlit top-right menu */
    #MainMenu {visibility: hidden !important;}
    /* Hide Streamlit footer (bottom-left) */
    footer {visibility: hidden !important;}
    /* Hide the GitHub / Share banner (bottom-right) */
    [data-testid="stDecoration"] {display: none !important;}
    
    #/* Optional: remove extra padding/margin from main page */
    #.css-1d391kg {padding-top: 0rem !important;}
    #</style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Remove top padding and margins for main content */
    .css-18e3th9, .css-1d391kg {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
    }
    /* Optional: reduce padding inside Streamlit containers */
    .block-container {
    padding-top: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Gridline background settings --- 
grid_color = "#58b3b1" # gridline color
grid_opacity = 0.16 # 0.0 (transparent) to 1.0 (solid)
grid_size = 40 # distance between gridlines in px

# Convert HEX to RGB
r = int(grid_color[1:3], 16)
g = int(grid_color[3:5], 16)
b = int(grid_color[5:7], 16)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #000000; /* black background */
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
# For standalone test, logging to console might be sufficient.
# You can uncomment to log to a file if you create a debug.log file in the same directory.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# =========================================================
# TA_PRO HELPER FUNCTIONS (Essential subset)
# =========================================================

# Helper for file paths (ensures user data directories exist)
def _ta_user_dir(user_id="guest"):
    # Using 'user_data' for isolated test
    root = os.path.join(os.path.dirname(__file__), "user_data") 
    os.makedirs(root, exist_ok=True)
    d = os.path.join(root, user_id)
    os.makedirs(d, exist_ok=True)
    # Ensure specific subdirectories exist for journal_images
    os.makedirs(os.path.join(d, "journal_images"), exist_ok=True) # NEW for screenshots
    # os.makedirs(os.path.join(d, "community_images"), exist_ok=True) # Not strictly needed for this isolated page
    # os.makedirs(os.path.join(d, "playbooks"), exist_ok=True) # Not strictly needed for this isolated page
    return d

# Generates a short unique ID (hexadecimal)
def _ta_hash():
    return uuid.uuid4().hex[:12]

# Custom JSON encoder for SQLite storage
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.datetime, dt.date)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super().default(obj)

# Helper to save journal data to DB
# Modified to accept conn/c directly as not using global variables anymore in isolated test.
def _ta_save_journal(username, journal_df, conn, c):
    try:
        c.execute("SELECT data FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        user_data = json.loads(result[0]) if result and result[0] else {}
        user_data["tools_trade_journal"] = journal_df.replace({pd.NA: None, float('nan'): None}).to_dict('records')
        c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
        conn.commit()
        logging.info(f"Journal saved for user {username}: {len(journal_df)} trades")
        return True
    except Exception as e:
        logging.error(f"Failed to save journal for {username}: {str(e)}")
        st.error(f"Failed to save journal: {str(e)}")
        return False

# XP notification system (simplified/mocked for isolated file)
def show_xp_notification(xp_gained):
    st.toast(f"🎉 +{xp_gained} XP Earned!", icon="⭐")

# XP update function (integrates with DB for a logged in user)
def ta_update_xp(amount):
    if "logged_in_user" in st.session_state:
        username = st.session_state.logged_in_user
        try:
            # Need to use existing conn/c from database setup below
            c.execute("SELECT data FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            if result:
                user_data = json.loads(result[0])
                old_xp = user_data.get('xp', 0)
                user_data['xp'] = old_xp + amount
                level = user_data['xp'] // 100
                if level > user_data.get('level', 0):
                    user_data['level'] = level
                    user_data['badges'] = user_data.get('badges', []) + [f"Level {level}"]
                    st.balloons()
                    st.success(f"Level up! You are now level {level}.")
                c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
                conn.commit()
                st.session_state.xp = user_data['xp']
                st.session_state.level = user_data['level']
                st.session_state.badges = user_data['badges']
                
                show_xp_notification(amount)
        except Exception as e:
            logging.error(f"Failed to update XP for {username}: {str(e)}")
            # st.warning(f"Could not update XP: {str(e)}") # Might not show a warning in an isolated file depending on exact context
            
def ta_update_streak():
    if "logged_in_user" in st.session_state:
        username = st.session_state.logged_in_user
        try:
            c.execute("SELECT data FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            if result:
                user_data = json.loads(result[0])
                today = dt.date.today().isoformat()
                last_date = user_data.get('last_journal_date')
                streak = user_data.get('streak', 0)
                
                if last_date:
                    last = dt.date.fromisoformat(last_date)
                    if last == dt.date.fromisoformat(today) - dt.timedelta(days=1):
                        streak += 1
                    elif last < dt.date.fromisoformat(today) - dt.timedelta(days=1):
                        streak = 1
                else:
                    streak = 1
                    
                user_data['streak'] = streak
                user_data['last_journal_date'] = today
                
                if streak % 7 == 0 and streak > 0: # Only give badge after a non-zero streak and for multiples of 7
                    badge = "Discipline Badge"
                    if badge not in user_data.get('badges', []):
                        user_data['badges'] = user_data.get('badges', []) + [badge]
                        st.balloons()
                        st.success(f"Unlocked: {badge} for {streak} day streak!")
                
                c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
                conn.commit()
                st.session_state.streak = streak
                st.session_state.badges = user_data['badges']
        except Exception as e:
            logging.error(f"Failed to update streak for {username}: {str(e)}")
            # st.warning(f"Could not update streak: {str(e)}") # Might not show a warning


# =========================================================
# DATABASE CONNECTION (Standalone test specific)
# =========================================================
DB_FILE = "test_users.db" # Using a different DB file name for testing

try:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, data TEXT)''')
    # c.execute('''CREATE TABLE IF NOT EXISTS community_data (key TEXT PRIMARY KEY, data TEXT)''') # Not needed for this isolated page
    conn.commit()
    logging.info("SQLite database initialized successfully for isolated test.")
    
    # --- Simulate a logged-in user for testing ---
    # You can comment this out to test the "not logged in" messages.
    test_username = "test_user"
    c.execute("SELECT username FROM users WHERE username = ?", (test_username,))
    if c.fetchone() is None:
        hashed_password = hashlib.sha256("test_password".encode()).hexdigest()
        initial_data_user = json.dumps({"xp": 100, "level": 1, "badges": ["Starter"], "streak": 5, "drawings": {}, "tools_trade_journal": [], "strategies": [{"Name": "Trend Following", "Description": "Simple Trend", "Date Added": "2023-01-01"}]})
        c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)", (test_username, hashed_password, initial_data_user))
        conn.commit()
        logging.info(f"Test user '{test_username}' created and logged in.")
    
    st.session_state.logged_in_user = test_username
    # Initialize other user-specific session states if they depend on login (copied from main app's login success logic)
    c.execute("SELECT data FROM users WHERE username = ?", (test_username,))
    user_data = json.loads(c.fetchone()[0]) if c.fetchone() else {}
    st.session_state.drawings = user_data.get("drawings", {})
    st.session_state.strategies = pd.DataFrame(user_data.get("strategies", []))

except Exception as e:
    logging.error(f"Failed to initialize SQLite database for test: {str(e)}")
    st.error(f"Test database initialization failed: {str(e)}. Please ensure you have write access or run locally.")


# =========================================================
# PAGE CONFIGURATION (Copied from main app)
# =========================================================
st.set_page_config(page_title="Forex Dashboard - Backtesting Test", layout="wide")


# =========================================================
# JOURNAL & DRAWING INITIALIZATION (GLOBAL SECTION - Crucial for schema)
# =========================================================
# Define journal columns and dtypes (UPDATED GLOBAL DEFINITIONS FOR SIMPLER UX)
journal_cols = [
    "Trade ID", "Date", "Entry Time", "Exit Time", "Symbol", "Trade Type", "Lots",
    "Entry Price", "Stop Loss Price", "Take Profit Price", "Final Exit Price",
    "Win/Loss", "PnL ($)", "Pips", "Initial R", "Realized R",
    "Strategy Used",
    "HTF Bias & Structure", # Consolidated: Weekly/Daily Bias + 4H/1H Structure
    "Market State (HTF)", "News Event Impact",
    "Setup Details", # Consolidated: Setup Name, Indicators, Entry Trigger
    "Entry Rationale",  # Plain text/markdown
    "Exit Rationale",   # Plain text/markdown
    "Pre-Trade Mindset", "In-Trade Emotions", "Discipline Score 1-5",
    "Trade Journal Notes", # General notes (reflection, analysis, adjustments combined)
    "Entry Screenshot", "Exit Screenshot", # Local path string
    "Tags"
]

journal_dtypes = {
    "Trade ID": str, "Date": "datetime64[ns]", "Entry Time": "datetime64[ns]", "Exit Time": "datetime64[ns]", "Symbol": str,
    "Trade Type": str, "Lots": float,
    "Entry Price": float, "Stop Loss Price": float, "Take Profit Price": float, "Final Exit Price": float,
    "Win/Loss": str, "PnL ($)": float, "Pips": float, "Initial R": float, "Realized R": float,
    "Strategy Used": str,
    "HTF Bias & Structure": str,
    "Market State (HTF)": str, "News Event Impact": str,
    "Setup Details": str,
    "Entry Rationale": str,
    "Exit Rationale": str,
    "Pre-Trade Mindset": str,
    "In-Trade Emotions": str,
    "Discipline Score 1-5": float,
    "Trade Journal Notes": str,
    "Entry Screenshot": str, "Exit Screenshot": str,
    "Tags": str
}

# This robust initialization logic is key. It ensures the session state DataFrame always matches the current schema.
if "tools_trade_journal" not in st.session_state:
    st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes, errors='ignore')
    for col, dtype in journal_dtypes.items():
        if dtype == str:
            st.session_state.tools_trade_journal[col] = ''
        elif 'datetime' in str(dtype):
            st.session_state.tools_trade_journal[col] = pd.NaT
        elif dtype == float:
            st.session_state.tools_trade_journal[col] = 0.0
        elif dtype == bool:
            st.session_state.tools_trade_journal[col] = False
else:
    # Migrate existing journal data to new schema on app load if needed.
    current_journal_df = st.session_state.tools_trade_journal.copy()
    reindexed_journal = pd.DataFrame(index=current_journal_df.index, columns=journal_cols)

    for col in journal_cols:
        if col in current_journal_df.columns:
            reindexed_journal[col] = current_journal_df[col]
        # Handle migration for consolidated columns. This is a heuristic.
        elif col == "HTF Bias & Structure":
            weekly_bias = current_journal_df.get("Weekly Bias", "").astype(str).replace("None","")
            daily_bias = current_journal_df.get("Daily Bias", "").astype(str).replace("None","")
            h4_struct = current_journal_df.get("4H Structure", "").astype(str).replace("None","")
            h1_struct = current_journal_df.get("1H Structure", "").astype(str).replace("None","")
            
            parts = []
            if weekly_bias: parts.append(f"W:{weekly_bias}")
            if daily_bias: parts.append(f"D:{daily_bias}")
            if h4_struct: parts.append(f"4H:{h4_struct}")
            if h1_struct: parts.append(f"1H:{h1_struct}")
            reindexed_journal[col] = ", ".join(parts) if parts else ""
        elif col == "Setup Details":
            setup_name = current_journal_df.get("Setup Name", "").astype(str).replace("None","")
            indicators = current_journal_df.get("Indicators Used", "").astype(str).replace("None","")
            trigger = current_journal_df.get("Entry Trigger", "").astype(str).replace("None","")
            
            parts = []
            if setup_name: parts.append(f"Setup: {setup_name}")
            if indicators: parts.append(f"Indicators: {indicators}")
            if trigger: parts.append(f"Trigger: {trigger}")
            reindexed_journal[col] = ", ".join(parts) if parts else ""

        elif col == "Entry Rationale":
            val = current_journal_df.get("Reasons for Entry", current_journal_df.get("Entry Conditions", pd.Series('')))
            reindexed_journal[col] = val.apply(lambda x: json.loads(x).get('text', '') if isinstance(x, str) and x.strip().startswith('{') else x)
        elif col == "Exit Rationale":
            val = current_journal_df.get("Reasons for Exit", pd.Series(''))
            reindexed_journal[col] = val.apply(lambda x: json.loads(x).get('text', '') if isinstance(x, str) and x.strip().startswith('{') else x)
        elif col == "Trade Journal Notes":
            # Combine old "Notes/Journal", "Post-Trade Analysis", "Lessons Learned" etc.
            notes_old = current_journal_df.get("Notes/Journal", pd.Series(''))
            pta_old = current_journal_df.get("Post-Trade Analysis", pd.Series(''))
            ll_old = current_journal_df.get("Lessons Learned", pd.Series(''))
            
            # Helper to extract text from old JSON-formatted fields, if present
            def extract_text_from_old_json(cell_value):
                try:
                    if isinstance(cell_value, str) and cell_value.strip().startswith('{'):
                        return json.loads(cell_value).get('text', '').strip()
                    return cell_value.strip() if isinstance(cell_value, str) else ""
                except json.JSONDecodeError:
                    return cell_value.strip() if isinstance(cell_value, str) else ""

            combined_notes_series = []
            # Ensure proper iteration that matches length
            for i in current_journal_df.index:
                n_val = extract_text_from_old_json(notes_old.loc[i])
                p_val = extract_text_from_old_json(pta_old.loc[i])
                l_val = extract_text_from_old_json(ll_old.loc[i])

                parts = []
                if n_val: parts.append(f"**General Notes:**\n{n_val}")
                if p_val: parts.append(f"**Post-Trade Analysis:**\n{p_val}")
                if l_val: parts.append(f"**Lessons Learned:**\n{l_val}")
                combined_notes_series.append("\n\n---\n\n".join(parts))
            reindexed_journal[col] = combined_notes_series

        elif col == "Entry Screenshot":
            old_ss_link = current_journal_df.get("Entry Screenshot Link/Hash", "").astype(str).str.split(',', expand=True)
            reindexed_journal[col] = old_ss_link[0].fillna("") if 0 in old_ss_link.columns else ""
        elif col == "Exit Screenshot":
            old_ss_link = current_journal_df.get("Entry Screenshot Link/Hash", "").astype(str).str.split(',', expand=True)
            reindexed_journal[col] = old_ss_link[1].fillna("") if 1 in old_ss_link.columns else ""
        
        else: # For other cols not explicitly handled or old deprecated ones, initialize with default
            if journal_dtypes[col] == str: reindexed_journal[col] = ""
            elif 'datetime' in str(journal_dtypes[col]): reindexed_journal[col] = pd.NaT
            elif journal_dtypes[col] == float: reindexed_journal[col] = 0.0
            elif journal_dtypes[col] == bool: reindexed_journal[col] = False
            else: reindexed_journal[col] = np.nan

    for col, dtype in journal_dtypes.items():
        if dtype == str:
            reindexed_journal[col] = reindexed_journal[col].fillna('').astype(str)
        elif 'datetime' in str(dtype):
            reindexed_journal[col] = pd.to_datetime(reindexed_journal[col], errors='coerce')
        elif dtype == float:
            reindexed_journal[col] = pd.to_numeric(reindexed_journal[col], errors='coerce').fillna(0.0).astype(float)
        elif dtype == bool:
            reindexed_journal[col] = reindexed_journal[col].fillna(False).astype(bool)
        else:
            reindexed_journal[col] = reindexed_journal[col].astype(dtype, errors='ignore')

    st.session_state.tools_trade_journal = reindexed_journal[journal_cols] # Ensure column order
    
# Initialize temporary journal for form (remains the same)
if "temp_journal" not in st.session_state:
    st.session_state.temp_journal = None

# Custom CSS for rendering Markdown in text areas with Streamlit defaults.
st.markdown("""
    <style>
    /* This CSS ensures that text areas displayed with st.markdown(unsafe_allow_html=True) */
    /* use consistent styling when interpreting markdown like paragraphs and lists. */
    div.stText p, div.stExpander div.stText p {
        margin-bottom: 0.5rem; /* Better spacing for paragraphs in notes */
        line-height: 1.5;
        color: var(--text-color); /* Use Streamlit's default text color */
    }
    div.stText ul, div.stText ol, div.stExpander div.stText ul, div.stExpander div.stText ol {
        margin-top: 0;
        margin-bottom: 0.5rem;
        padding-left: 20px;
    }
    div.stText h1, div.stText h2, div.stText h3, div.stText h4, div.stText h5, div.stText h6,
    div.stExpander div.stText h1, div.stExpander div.stText h2, div.stExpander div.stText h3, 
    div.stExpander div.stText h4, div.stExpander div.stText h5, div.stExpander div.stText h6 {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(88, 179, 177, 0.5); /* Subtle line for headers */
        padding-bottom: 5px;
        color: var(--primary-color); /* Use Streamlit's primary color for headers */
    }
    </style>
    """, unsafe_allow_html=True)

# Simplified function to apply styling - now it just ensures rendering of plain Markdown.
def render_markdown_content(text_content):
    # We now rely on Streamlit's native Markdown rendering within a div for basic styling control
    return text_content # Directly pass the markdown content. Streamlit handles rendering


# =========================================================
# BACKTESTING PAGE CONTENT
# =========================================================
st.title("📈 Backtesting (Standalone Test)") # Title updated for clarity
st.caption("Live TradingView chart for backtesting and a simplified, user-friendly trading journal for tracking and analyzing trades.")
st.markdown('---')

# Pair selector & symbol map (remains unchanged)
pairs_map = {
    "EUR/USD": "FX:EURUSD", "USD/JPY": "FX:USDJPY", "GBP/USD": "FX:GBPUSD",
    "USD/CHF": "OANDA:USDCHF", "AUD/USD": "FX:AUDUSD", "NZD/USD": "OANDA:NZDUSD",
    "USD/CAD": "CMCMARKETS:USDCAD", "EUR/GBP": "FX:EURGBP", "EUR/JPY": "FX:EURJPY",
    "GBP/JPY": "FX:GBPJPY", "AUD/JPY": "FX:AUDJPY", "AUD/NZD": "FX:AUDNZD",
    "AUD/CAD": "FX:AUDCAD", "AUD/CHF": "FX:AUDCHF", "CAD/JPY": "FX:CADJPY",
    "CHF/JPY": "FX:CHFJPY", "EUR/AUD": "FX:EURAUD", "EUR/CAD": "FX:EURCAD",
    "EUR/CHF": "FX:EURCHF", "GBP/AUD": "FX:GBPAUD", "GBP/CAD": "FX:GBPCAD",
    "GBP/CHF": "FX:GBPCHF", "NZD/JPY": "FX:NZDJPY", "NZD/CAD": "FX:NZDCAD",
    "NZD/CHF": "FX:NZDCHF", "CAD/CHF": "FX:CADCHF",
}
pair = st.selectbox("Select pair", list(pairs_map.keys()), index=0, key="tv_pair")
tv_symbol = pairs_map[pair]

# Initialize drawings in session state if not present
if 'drawings' not in st.session_state:
    st.session_state.drawings = {}

# Load initial drawings if available (requires login)
if "logged_in_user" in st.session_state and pair not in st.session_state.drawings:
    username = st.session_state.logged_in_user
    logging.info(f"Loading drawings for user {username}, pair {pair}")
    try:
        # Use local 'c' for this standalone file
        c.execute("SELECT data FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        if result:
            user_data = json.loads(result[0])
            st.session_state.drawings[pair] = user_data.get("drawings", {}).get(pair, {})
            logging.info(f"Loaded drawings for {pair}: {st.session_state.drawings[pair]}")
        else:
            logging.warning(f"No data found for user {username}")
    except Exception as e:
        logging.error(f"Error loading drawings for {username}: {str(e)}")
        st.error(f"Failed to load drawings: {str(e)}")

# TradingView widget (remains unchanged)
tv_html = f"""
<div id="tradingview_widget"></div>
<script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
<script type="text/javascript">
new TradingView.widget({{
    "container_id": "tradingview_widget",
    "width": "100%",
    "height": 800,
    "symbol": "{tv_symbol}",
    "interval": "D",
    "timezone": "Etc/UTC",
    "theme": "dark",
    "style": "1",
    "locale": "en",
    "toolbar_bg": "#f1f3f6",
    "enable_publishing": false,
    "allow_symbol_change": true,
    "studies": [],
    "show_popup_button": true,
    "popup_width": "1000",
    "popup_height": "650"
}});
</script>
"""
st.components.v1.html(tv_html, height=820, scrolling=False)

# Save, Load, and Refresh buttons for drawings (minor update for journal sync)
if "logged_in_user" in st.session_state:
    col1_draw, col2_draw, col3_draw = st.columns([1, 1, 1])
    with col1_draw:
        if st.button("Save Drawings", key="bt_save_drawings"):
            logging.info(f"Save Drawings button clicked for pair {pair}")
            save_script = f"""
            <script>
            parent.window.postMessage({{action: 'save_drawings', pair: '{pair}'}}, '');
            </script>
            """
            st.components.v1.html(save_script, height=0)
            logging.info(f"Triggered save script for {pair}")
            st.session_state[f"bt_save_trigger_{pair}"] = True
    with col2_draw:
        if st.button("Load Drawings", key="bt_load_drawings"):
            username = st.session_state.logged_in_user
            logging.info(f"Load Drawings button clicked for user {username}, pair {pair}")
            try:
                # Use local 'c' for this standalone file
                c.execute("SELECT data FROM users WHERE username = ?", (username,))
                result = c.fetchone()
                if result:
                    user_data = json.loads(result[0])
                    st.session_state.drawings[pair] = user_data.get("drawings", {}).get(pair, {})
                    logging.info(f"Loaded drawings for {pair}: {st.session_state.drawings[pair]}")
                else:
                    st.info("No saved drawings for this pair.")
                    logging.info(f"No saved drawings found for {pair}")
            except Exception as e:
                st.error(f"Failed to load drawings: {str(e)}")
                logging.error(f"Error loading drawings for {username}: {str(e)}")
    with col3_draw:
        if st.button("Refresh Account (Drawings/Journal Sync)", key="bt_refresh_account"):
            username = st.session_state.logged_in_user
            logging.info(f"Refresh Account button clicked for user {username}")
            try:
                # Use local 'c' for this standalone file
                c.execute("SELECT data FROM users WHERE username = ?", (username,))
                result = c.fetchone()
                if result:
                    user_data = json.loads(result[0])
                    st.session_state.drawings = user_data.get("drawings", {})
                    
                    # Robust re-processing of journal data for new schema.
                    loaded_journal_raw = user_data.get("tools_trade_journal", [])
                    loaded_journal_df = pd.DataFrame(loaded_journal_raw)

                    # Create an empty, fully-structured DataFrame with all current journal_cols
                    master_journal_df = pd.DataFrame(columns=journal_cols)

                    # Copy existing data, fill missing with defaults based on dtype
                    for col in journal_cols:
                        if col in loaded_journal_df.columns:
                            master_journal_df[col] = loaded_journal_df[col]
                        else:
                            if journal_dtypes[col] == str:
                                master_journal_df[col] = ""
                            elif 'datetime' in str(journal_dtypes[col]):
                                master_journal_df[col] = pd.NaT # pandas Not-a-Time for missing datetimes
                            elif journal_dtypes[col] == float:
                                master_journal_df[col] = 0.0
                            elif journal_dtypes[col] == bool:
                                master_journal_df[col] = False
                            else:
                                master_journal_df[col] = np.nan

                    # Enforce dtypes and fill any remaining NaNs in string columns
                    for col, dtype in journal_dtypes.items():
                        if dtype == str:
                            master_journal_df[col] = master_journal_df[col].fillna('').astype(str)
                        elif 'datetime' in str(dtype):
                            master_journal_df[col] = pd.to_datetime(master_journal_df[col], errors='coerce')
                        elif dtype == float:
                            master_journal_df[col] = pd.to_numeric(master_journal_df[col], errors='coerce').fillna(0.0).astype(float)
                        elif dtype == bool:
                            master_journal_df[col] = master_journal_df[col].fillna(False).astype(bool)
                        else: # Fallback for other dtypes
                            master_journal_df[col] = master_journal_df[col].astype(dtype, errors='ignore')

                    st.session_state.tools_trade_journal = master_journal_df
                    
                    st.success("Account synced successfully!")
                    logging.info(f"Account synced for {username}. Drawings and Journal updated.")
                else:
                    st.error("Failed to sync account.")
                    logging.error(f"No user data found for {username}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to sync account: {str(e)}")
                logging.error(f"Error syncing account for {username}: {str(e)}")
    
    # Check for saved drawings from postMessage
    drawings_key = f"bt_drawings_key_{pair}"
    if drawings_key in st.session_state and st.session_state.get(f"bt_save_trigger_{pair}", False):
        content = st.session_state[drawings_key]
        logging.info(f"Received drawing content for {pair}: {content}")
        if content and isinstance(content, dict) and content:
            username = st.session_state.logged_in_user
            try:
                # Use local 'c' for this standalone file
                c.execute("SELECT data FROM users WHERE username = ?", (username,))
                result = c.fetchone()
                user_data = json.loads(result[0]) if result else {}
                user_data.setdefault("drawings", {})[pair] = content
                c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
                conn.commit()
                st.session_state.drawings[pair] = content
                st.success(f"Drawings for {pair} saved successfully!")
                logging.info(f"Drawings saved to database for {pair}: {content}")
            except Exception as e:
                st.error(f"Failed to save drawings: {str(e)}")
                logging.error(f"Database error saving drawings for {pair}: {str(e)}")
            finally:
                # Clean up the session state flags
                if drawings_key in st.session_state: del st.session_state[drawings_key]
                if f"bt_save_trigger_{pair}" in st.session_state: del st.session_state[f"bt_save_trigger_{pair}"]
        else:
            st.warning("No valid drawing content received. Ensure you have drawn on the chart.")
            logging.warning(f"No valid drawing content received for {pair}: {content}")
else:
    st.info("Sign in via the My Account tab to save/load drawings and trading journal. (Currently mocked in this test file)")
    logging.info("User not logged in, save/load drawings disabled")

st.markdown("### 📝 Simplified Trading Journal")
st.markdown(
    """
    Log your trades with essential details for tracking performance and learning.
    Use Markdown (e.g., `**bold**`, `*italic*`, `- list item`) in text fields for basic formatting.
    """
)

# Tabs for Journal Entry, Analytics, and History
tab_entry, tab_analytics, tab_history = st.tabs(["📝 Log Trade", "📊 Analytics", "📜 Trade History"])

# =========================================================
# LOG TRADE TAB (UX Friendly Redesign)
# =========================================================
with tab_entry:
    st.subheader("Log a New Trade (Essential Information)")
    # Pre-fill form if 'edit_trade_data' exists (from Trade History's Edit button)
    initial_data = st.session_state.get('edit_trade_data', {})
    is_editing = bool(initial_data)

    with st.form("trade_entry_form", clear_on_submit=not is_editing):
        # --- Section: General Trade Info ---
        st.markdown("### General Trade Info")
        
        cols_overview_1, cols_overview_2 = st.columns(2)
        with cols_overview_1:
            trade_id_input = st.text_input("Trade ID", value=initial_data.get("Trade ID", f"TRD-{_ta_hash()}"), disabled=is_editing, help="A unique identifier for your trade. Auto-generated if left empty.")
            
            trade_date_val = initial_data.get("Date", dt.datetime.now())
            entry_time_val = initial_data.get("Entry Time", dt.datetime.now())
            exit_time_val = initial_data.get("Exit Time", dt.datetime.now())

            trade_date = st.date_input("Date", value=trade_date_val.date() if isinstance(trade_date_val, dt.datetime) else trade_date_val, help="The calendar date of your trade.", key="trade_date_input")
            entry_time = st.time_input("Entry Time", value=entry_time_val.time() if isinstance(entry_time_val, dt.datetime) else entry_time_val, help="The time you entered the trade.", key="entry_time_input")
            exit_time = st.time_input("Exit Time", value=exit_time_val.time() if isinstance(exit_time_val, dt.datetime) else exit_time_val, help="The time you exited the trade.", key="exit_time_input")
        
        with cols_overview_2:
            symbol_options = list(pairs_map.keys()) + ["Other"]
            default_symbol = initial_data.get("Symbol", pair)
            default_symbol_idx = symbol_options.index(default_symbol) if default_symbol in symbol_options else (symbol_options.index("Other") if default_symbol != "" else 0)
            symbol = st.selectbox("Currency Pair / Asset", symbol_options, index=default_symbol_idx, help="The asset you traded.", key="symbol_input")
            if symbol == "Other":
                symbol = st.text_input("Specify Custom Asset", value=initial_data.get("Symbol", ""), help="Enter asset name if not in list.", key="custom_symbol_input")
            
            trade_type = st.radio("Trade Direction", ["Long", "Short", "Breakeven", "No-Trade (Study)"], horizontal=True, 
                                  index=["Long", "Short", "Breakeven", "No-Trade (Study)"].index(initial_data.get("Trade Type", "Long")), 
                                  help="Was this a buy, sell, a trade that broke even, or just a study/watch entry?", key="trade_type_input")
                                  
            lots = st.number_input("Position Size (Lots)", min_value=0.01, step=0.01, format="%.2f", value=float(initial_data.get("Lots", 0.1)), help="The size of your trade in lots.", key="lots_input")
            entry_price = st.number_input("Entry Price", min_value=0.0, step=0.00001, format="%.5f", value=float(initial_data.get("Entry Price", 0.0)), help="The price you entered the market.", key="entry_price_input")
            stop_loss_price = st.number_input("Stop Loss Price", min_value=0.0, step=0.00001, format="%.5f", value=float(initial_data.get("Stop Loss Price", 0.0)), help="The price where your stop loss was placed.", key="stop_loss_price_input")
            take_profit_price = st.number_input("Take Profit Price", min_value=0.0, step=0.00001, format="%.5f", value=float(initial_data.get("Take Profit Price", 0.0)), help="The price where your take profit was placed.", key="take_profit_price_input")
            final_exit_price = st.number_input("Final Exit Price", min_value=0.0, step=0.00001, format="%.5f", value=float(initial_data.get("Final Exit Price", 0.0)), help="The actual price your trade was closed.", key="final_exit_price_input")
        
        st.markdown("---")

        # --- Section: Optional Trade Details ---
        with st.expander("More Trade Details (Optional)", expanded=False):
            cols_more_details_1, cols_more_details_2 = st.columns(2)

            with cols_more_details_1:
                st.markdown("#### Context & Strategy")
                user_strategies = ["(Select One)"] + sorted([s['Name'] for s in st.session_state.strategies.to_dict('records')] if "strategies" in st.session_state and not st.session_state.strategies.empty else [])
                default_strategy_idx = user_strategies.index(initial_data.get("Strategy Used", "(Select One)")) if initial_data.get("Strategy Used", "(Select One)") in user_strategies else 0
                selected_strategy = st.selectbox("Strategy Used", options=user_strategies, index=default_strategy_idx, help="Link this trade to one of your defined strategies.", key="strategy_used_input")
                
                htf_bias_structure = st.text_area("Higher Timeframe Bias & Structure", value=initial_data.get("HTF Bias & Structure", ""), height=80,
                                                  help="E.g., 'Weekly Bullish, Daily Corrective, 4H Impulse'", key="htf_bias_structure_input")
                market_state = st.selectbox("Market State (HTF)", ["Trend (Bullish)", "Trend (Bearish)", "Range", "Complex Pullback", "Choppy", "Undefined"], 
                                            index=["Trend (Bullish)", "Trend (Bearish)", "Range", "Complex Pullback", "Choppy", "Undefined"].index(initial_data.get("Market State (HTF)", "Undefined")), help="Dominant market condition.", key="market_state_input")
                
                news_impact_options = ["High Impact (Positive)", "High Impact (Negative)", "Medium Impact", "Low Impact", "None"]
                initial_news_impact = initial_data.get("News Event Impact", "").split(',') if initial_data.get("News Event Impact") else []
                news_impact = st.multiselect("News Event Impact", news_impact_options, 
                                             default=[ni for ni in initial_news_impact if ni in news_impact_options], help="How upcoming or recent news events influenced your trade decision.", key="news_impact_input")
                
                setup_details = st.text_area("Setup Details & Entry Trigger", value=initial_data.get("Setup Details", ""), height=80,
                                             help="E.g., 'Double Bottom, RSI Divergence. Entry: Pin bar close on 5min.'", key="setup_details_input")

            with cols_more_details_2:
                st.markdown("#### Psychological & Visuals")
                pre_trade_mindset = st.text_area("Pre-Trade Mindset", value=initial_data.get("Pre-Trade Mindset", ""), height=80,
                                                 help="How were you feeling and what was your plan going into the trade?", key="pre_trade_mindset_input")
                
                in_trade_emotions_options = ["Confident", "Anxious", "Fearful", "Excited", "Frustrated", "Neutral", "FOMO", "Greedy", "Revenge", "Impulsive", "Disciplined", "Overconfident", "Patient", "Irritable"]
                initial_in_trade_emotions = initial_data.get("In-Trade Emotions", "").split(',') if initial_data.get("In-Trade Emotions") else []
                in_trade_emotions = st.multiselect(
                    "In-Trade Emotions",
                    in_trade_emotions_options, default=[e for e in initial_in_trade_emotions if e in in_trade_emotions_options],
                    help="Select emotions experienced during the trade.", key="in_trade_emotions_input"
                )
                discipline_score = st.slider("Discipline Score (1=Low, 5=High)", 1, 5, value=int(initial_data.get("Discipline Score 1-5", 3)), help="Rate your adherence to your plan and discipline.", key="discipline_score_input")

                user_journal_images_dir = os.path.join(_ta_user_dir(st.session_state.get("logged_in_user", "guest")), "journal_images")
                os.makedirs(user_journal_images_dir, exist_ok=True) 
                entry_ss_initial_val = initial_data.get("Entry Screenshot", "")
                exit_ss_initial_val = initial_data.get("Exit Screenshot", "")
                
                uploaded_entry_image = st.file_uploader("Upload Entry Screenshot (Optional)", type=["png", "jpg", "jpeg"], help="Upload an image of your chart at entry.", key="upload_entry_screenshot")
                if uploaded_entry_image:
                    image_filename = f"{trade_id_input}_entry_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.png" # Unique filename
                    image_file_path = os.path.join(user_journal_images_dir, image_filename)
                    with open(image_file_path, "wb") as f:
                        f.write(uploaded_entry_image.getbuffer())
                    entry_ss_initial_val = image_file_path # Update the value to be saved
                    st.success("Entry screenshot uploaded!")

                uploaded_exit_image = st.file_uploader("Upload Exit Screenshot (Optional)", type=["png", "jpg", "jpeg"], help="Upload an image of your chart at exit.", key="upload_exit_screenshot")
                if uploaded_exit_image:
                    image_filename = f"{trade_id_input}_exit_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.png" # Unique filename
                    image_file_path = os.path.join(user_journal_images_dir, image_filename)
                    with open(image_file_path, "wb") as f:
                        f.write(uploaded_exit_image.getbuffer())
                    exit_ss_initial_val = image_file_path # Update the value to be saved
                    st.success("Exit screenshot uploaded!")

                entry_screenshot_storage = st.text_input("Entry Screenshot (Saved Path)", value=entry_ss_initial_val, key="entry_screenshot_storage_hidden", label_visibility="hidden")
                exit_screenshot_storage = st.text_input("Exit Screenshot (Saved Path)", value=exit_ss_initial_val, key="exit_screenshot_storage_hidden", label_visibility="hidden")
        
        st.markdown("---")

        # --- Section: Trade Reflection & Notes ---
        with st.expander("Trade Reflection & Notes (Key Learnings)", expanded=True):
            st.markdown("Document your thoughts, reasons for entry/exit, and lessons learned using **Markdown**.")
            
            cols_notes_1, cols_notes_2 = st.columns(2)
            with cols_notes_1:
                entry_rationale = st.text_area("Entry Rationale", value=initial_data.get("Entry Rationale", ""), height=150, 
                                               help="Why did you enter this trade? (e.g., trend confirmation, divergence).", key="entry_rationale_input")
                exit_rationale = st.text_area("Exit Rationale", value=initial_data.get("Exit Rationale", ""), height=150, 
                                              help="Why did you exit the trade? (e.g., hit TP, hit SL, discretionary).", key="exit_rationale_input")
            
            with cols_notes_2:
                trade_journal_notes = st.text_area("Overall Trade Notes & Learning", value=initial_data.get("Trade Journal Notes", ""), height=250, 
                                                   help="Combine post-trade analysis, lessons learned, and any strategy adjustments here.", key="trade_journal_notes_input")
            
            current_tags_in_journal = sorted(list(set(st.session_state.tools_trade_journal['Tags'].str.split(',').explode().dropna().astype(str).str.strip())))
            suggested_tags = ["Strategy: Breakout", "Strategy: Reversal", "Mistake: FOMO", "Mistake: Overleveraged", 
                              "Emotion: Fear", "Emotion: Greed", "Session: London", "Session: New York"]
            all_tag_options = sorted(list(set(current_tags_in_journal + suggested_tags)))
            initial_tags = initial_data.get("Tags", "").split(',') if initial_data.get("Tags") else []
            tags = st.multiselect("Trade Tags", all_tag_options, 
                                  default=[t for t in initial_tags if t in all_tag_options], help="Categorize your trade for easier analysis.", key="tags_input")


        # Clear edit state after form is rendered with initial_data
        if is_editing:
            pass # The deletion will happen after successful form submission

        st.markdown("---")
        submit_button = st.form_submit_button(
            f"{'Update Trade' if is_editing else 'Save New Trade'}", 
            type="primary", 
            help=f"{'Click to update this trade log.' if is_editing else 'Click to save your new trade log.'}"
        )
    
        if submit_button:
            # --- Calculations for derived fields ---
            pip_multiplier = 0 # default, will be set below
            approx_pip_value_usd_per_lot = 0.0 # default, will be set below
            pip_scale = 0.0 # default, will be set below

            if "JPY" in symbol:
                pip_multiplier = 100
                approx_pip_value_usd_per_lot = 8.5 # Rough average for JPY pairs ($/std lot/pip)
                pip_scale = 0.01 
            else: # Most other pairs like EUR/USD, GBP/USD
                pip_multiplier = 10000 
                approx_pip_value_usd_per_lot = 10.0 # Rough average for non-JPY pairs ($/std lot/pip)
                pip_scale = 0.0001 
            
            trade_pips_gain = 0.0
            pnL_dollars = 0.0
            win_loss_status = "No-Trade (Study)" # Default for clarity if calculations fail or type is 'Study'

            if trade_type in ["Long", "Short"]:
                if entry_price > 0.0 and final_exit_price > 0.0:
                    if trade_type == "Long":
                        trade_pips_gain = (final_exit_price - entry_price) / pip_scale
                    elif trade_type == "Short":
                        trade_pips_gain = (entry_price - final_exit_price) / pip_scale
                    
                    pnL_dollars = trade_pips_gain * lots * (approx_pip_value_usd_per_lot)

                    if pnL_dollars > 0.0:
                        win_loss_status = "Win"
                    elif pnL_dollars < 0.0:
                        win_loss_status = "Loss"
                    else:
                        win_loss_status = "Breakeven"
                else:
                    win_loss_status = "Pending / Invalid Prices"
                    trade_pips_gain = 0.0
                    pnL_dollars = 0.0
            elif trade_type == "Breakeven":
                win_loss_status = "Breakeven"
                pnL_dollars = 0.0
                trade_pips_gain = 0.0
            
            initial_r_calc = 0.0
            if entry_price > 0.0 and stop_loss_price > 0.0 and take_profit_price > 0.0 and trade_type in ["Long", "Short"]:
                risk_per_unit = abs(entry_price - stop_loss_price)
                reward_per_unit = abs(take_profit_price - entry_price)
                if risk_per_unit > 0.0:
                    initial_r_calc = reward_per_unit / risk_per_unit
            
            realized_r_calc = 0.0
            if entry_price > 0.0 and stop_loss_price > 0.0 and final_exit_price > 0.0 and trade_type in ["Long", "Short"]:
                risk_per_unit_realized = abs(entry_price - stop_loss_price)
                realized_pnl_raw = final_exit_price - entry_price if trade_type == "Long" else entry_price - final_exit_price
                if risk_per_unit_realized > 0.0:
                    realized_r_calc = realized_pnl_raw / risk_per_unit_realized # Can be negative
            
            new_trade_data = {
                "Trade ID": trade_id_input,
                "Date": pd.to_datetime(trade_date),
                "Entry Time": pd.to_datetime(f"{trade_date} {entry_time}"),
                "Exit Time": pd.to_datetime(f"{trade_date} {exit_time}"),
                "Symbol": symbol,
                "Trade Type": trade_type,
                "Lots": lots,
                "Entry Price": entry_price,
                "Stop Loss Price": stop_loss_price,
                "Take Profit Price": take_profit_price,
                "Final Exit Price": final_exit_price,
                "Win/Loss": win_loss_status,
                "PnL ($)": pnL_dollars,
                "Pips": trade_pips_gain,
                "Initial R": initial_r_calc,
                "Realized R": realized_r_calc,
                "Strategy Used": selected_strategy if selected_strategy != "(Select One)" else "",
                "HTF Bias & Structure": htf_bias_structure,
                "Market State (HTF)": market_state,
                "News Event Impact": ','.join(news_impact),
                "Setup Details": setup_details,
                "Entry Rationale": entry_rationale,
                "Exit Rationale": exit_rationale,
                "Pre-Trade Mindset": pre_trade_mindset,
                "In-Trade Emotions": ','.join(in_trade_emotions),
                "Discipline Score 1-5": float(discipline_score),
                "Trade Journal Notes": trade_journal_notes,
                "Entry Screenshot": entry_screenshot_storage,
                "Exit Screenshot": exit_screenshot_storage,
                "Tags": ','.join(tags)
            }

            new_trade_df_row = pd.DataFrame([new_trade_data])
            for col, dtype in journal_dtypes.items():
                if col in new_trade_df_row.columns:
                    if dtype == str:
                        new_trade_df_row[col] = new_trade_df_row[col].fillna('').astype(str)
                    elif 'datetime' in str(dtype):
                        new_trade_df_row[col] = pd.to_datetime(new_trade_df_row[col], errors='coerce')
                    elif dtype == float:
                        new_trade_df_row[col] = pd.to_numeric(new_trade_df_row[col], errors='coerce').fillna(0.0).astype(float)
                    elif dtype == bool:
                        new_trade_df_row[col] = new_trade_df_row[col].fillna(False).astype(bool)

            if is_editing:
                trade_to_update_idx = st.session_state.tools_trade_journal[st.session_state.tools_trade_journal['Trade ID'] == trade_id_input].index
                if not trade_to_update_idx.empty:
                    for col in journal_cols:
                        st.session_state.tools_trade_journal.loc[trade_to_update_idx, col] = new_trade_df_row[col].iloc[0]
                    st.success(f"Trade {trade_id_input} updated successfully!")
                else:
                    st.error(f"Error: Trade with ID {trade_id_input} not found for update.")
            else:
                st.session_state.tools_trade_journal = pd.concat(
                    [st.session_state.tools_trade_journal, new_trade_df_row],
                    ignore_index=True
                ).astype(journal_dtypes, errors='ignore')
                st.success("New trade saved successfully!")
            
            # Save to database if user is logged in
            if 'logged_in_user' in st.session_state:
                username = st.session_state.logged_in_user
                # Using local 'conn' and 'c' here.
                if _ta_save_journal(username, st.session_state.tools_trade_journal, conn, c):
                    ta_update_xp(10) # 10 XP per trade log
                    ta_update_streak()
                    logging.info(f"Trade {'updated' if is_editing else 'logged'} and saved to database for user {username} with ID {trade_id_input}")
                else:
                    st.error("Failed to save trade to account. Saved locally only.")
            else:
                st.warning("Trade saved locally (not synced to account, please log in to save).")
                logging.info("Trade logged for anonymous user")
            
            if 'edit_trade_data' in st.session_state:
                del st.session_state['edit_trade_data']
            st.rerun()

    st.subheader("Recent Trades Overview")
    # Define a simplified column config for the overview table, for better readability
    overview_column_config = {
        "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
        "Trade ID": st.column_config.TextColumn("Trade ID", width="small"),
        "Symbol": st.column_config.TextColumn("Symbol"),
        "Trade Type": st.column_config.TextColumn("Type"),
        "Lots": st.column_config.NumberColumn("Lots", format="%.2f"),
        "PnL ($)": st.column_config.NumberColumn("P&L ($)", format="$%.2f", help="Profit and Loss in account currency"),
        "Realized R": st.column_config.NumberColumn("Realized R", format="%.2f", help="Risk-Reward multiple realized"),
        "Win/Loss": st.column_config.TextColumn("Outcome"),
        "Strategy Used": st.column_config.TextColumn("Strategy"),
        "Tags": st.column_config.TextColumn("Tags", width="small", help="Categorization tags")
    }
    
    cols_to_display = [col for col in overview_column_config.keys() if col in st.session_state.tools_trade_journal.columns]
    
    st.dataframe(st.session_state.tools_trade_journal[cols_to_display],
                    column_config=overview_column_config,
                    hide_index=True,
                    use_container_width=True)
    
    # Export options
    st.subheader("Export Journal")
    col_export1, col_export2 = st.columns(2)

    with col_export1:
        csv = st.session_state.tools_trade_journal.to_csv(index=False)
        st.download_button("Download CSV", csv, "trade_journal.csv", "text/csv")

    with col_export2:
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                try:
                    latex_content = """
                    \\documentclass{article}
                    \\usepackage{booktabs}
                    \\usepackage{geometry}
                    \\geometry{a4paper, margin=1in}
                    \\usepackage{pdflscape}
                    \\begin{document}
                    \\section*{Trade Journal Report}
                    \\begin{landscape}
                    \\begin{tabular}{llrrlll}
                    \\toprule
                    Date & Symbol & Entry Price & Exit Price & P&L ($) & Realized R & Tags \\\\
                    \\midrule
                    """
                    export_cols_for_pdf = ["Date", "Symbol", "Entry Price", "Final Exit Price", "PnL ($)", "Realized R", "Tags"]
                    temp_df = st.session_state.tools_trade_journal[[col for col in export_cols_for_pdf if col in st.session_state.tools_trade_journal.columns]].copy()
                    temp_df = temp_df.fillna({'Entry Price':0.0, 'Final Exit Price':0.0, 'PnL ($)':0.0, 'Realized R':0.0, 'Tags':''}).round({'Entry Price':5, 'Final Exit Price':5, 'PnL ($)':2, 'Realized R':2})
                    
                    for _, row in temp_df.iterrows():
                        date_str = row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else ''
                        sanitized_symbol = row['Symbol'].replace('_', '\\_').replace('&', '\\&')
                        sanitized_tags = row['Tags'].replace('_', '\\_').replace('&', '\\&')
                        latex_content += f"{date_str} & {sanitized_symbol} & {row['Entry Price']:.5f} & {row['Final Exit Price']:.5f} & {row['PnL ($)']:.2f} & {row['Realized R']:.2f} & {sanitized_tags} \\\\\n"
                
                    latex_content += """
                    \\bottomrule
                    \\end{tabular}
                    \\end{landscape}
                    \\end{document}
                    """
                
                    with open("trade_journal_report.tex", "w") as f:
                        f.write(latex_content)
                
                    import subprocess
                    process = subprocess.run(["latexmk", "-pdf", "trade_journal_report.tex"], check=True, capture_output=True, text=True, shell=True) 
                    if process.returncode != 0:
                        st.error(f"LaTeX compilation failed: {process.stderr}")
                        logging.error(f"LaTeX compilation failed: {process.stderr}", exc_info=True)
                        raise RuntimeError("LaTeX compilation failed")

                    with open("trade_journal_report.pdf", "rb") as f:
                        st.download_button("Download PDF Report", f, "trade_journal_report.pdf", "application/pdf")
                except FileNotFoundError:
                    st.error("LaTeX compilation tools not found. Please install a TeX distribution (e.g., MiKTeX on Windows, TeX Live on Linux/macOS) to generate PDFs.")
                    logging.error("LaTeX tools not found for PDF generation.")
                except subprocess.CalledProcessError as e:
                    st.error(f"PDF generation failed: LaTeX command returned an error. See logs for details.")
                    logging.error(f"LaTeX compilation failed: {e.stderr}", exc_info=True)
                except Exception as e:
                    st.error(f"PDF generation failed unexpectedly: {str(e)}")
                    logging.error(f"PDF generation error: {str(e)}", exc_info=True)


# =========================================================
# ANALYTICS TAB
# =========================================================
with tab_analytics:
    st.subheader("📊 Trade Analytics & Insights")
    st.markdown("Dive deeper into your performance with these comprehensive analytics.")
    
    if not st.session_state.tools_trade_journal.empty:
        df_analytics = st.session_state.tools_trade_journal.copy()
        df_analytics = df_analytics[df_analytics["Win/Loss"].isin(["Win", "Loss", "Breakeven", "Pending / Invalid Prices"])].copy()

        if df_analytics.empty:
            st.info("No completed trades to analyze. Log some trades first.")
        else:
            df_analytics["Date"] = pd.to_datetime(df_analytics["Date"])
            df_analytics['Strategy Used'] = df_analytics['Strategy Used'].replace('', 'N/A')
            
            # Filters
            st.markdown("---")
            st.markdown("#### Filter Your Analytics")
            col_filter_a, col_filter_b, col_filter_c, col_filter_d = st.columns(4)
            with col_filter_a:
                analytics_symbol_filter = st.multiselect(
                    "Filter by Symbol",
                    options=df_analytics['Symbol'].unique(),
                    default=df_analytics['Symbol'].unique(), key="analytics_symbol_filter"
                )
            with col_filter_b:
                analytics_trade_type_filter = st.multiselect(
                    "Filter by Trade Type",
                    options=df_analytics['Trade Type'].unique(),
                    default=df_analytics['Trade Type'].unique(), key="analytics_trade_type_filter"
                )
            with col_filter_c:
                tag_options_analytics = sorted(list(set(df_analytics['Tags'].str.split(',').explode().dropna().astype(str).str.strip())))
                analytics_tag_filter = st.multiselect("Filter by Tags", options=tag_options_analytics, key="analytics_tag_filter")
            with col_filter_d:
                strategy_options_analytics = sorted(df_analytics['Strategy Used'].unique())
                analytics_strategy_filter = st.multiselect("Filter by Strategy", options=strategy_options_analytics, default=strategy_options_analytics, key="analytics_strategy_filter")
            
            if analytics_symbol_filter:
                df_analytics = df_analytics[df_analytics['Symbol'].isin(analytics_symbol_filter)]
            if analytics_trade_type_filter:
                df_analytics = df_analytics[df_analytics['Trade Type'].isin(analytics_trade_type_filter)]
            if analytics_tag_filter:
                df_analytics = df_analytics[df_analytics['Tags'].apply(lambda x: any(tag in x.split(',') for tag in analytics_tag_filter) if isinstance(x, str) and x else False)]
            if analytics_strategy_filter:
                df_analytics = df_analytics[df_analytics['Strategy Used'].isin(analytics_strategy_filter)]

            if df_analytics.empty:
                st.warning("No trades match the current filter criteria.")
            else:
                # Metrics Section
                st.markdown("---")
                st.markdown("#### Key Performance Indicators")
                total_trades_ana = len(df_analytics)
                winning_trades_ana = df_analytics[df_analytics["Win/Loss"] == "Win"]
                losing_trades_ana = df_analytics[df_analytics["Win/Loss"] == "Loss"]
                
                net_profit_ana = df_analytics["PnL ($)"].sum()
                gross_profit_ana = winning_trades_ana["PnL ($)"].sum()
                gross_loss_ana = losing_trades_ana["PnL ($)"].sum() 

                win_rate_ana = (len(winning_trades_ana) / total_trades_ana * 100) if total_trades_ana > 0 else 0
                
                avg_win_ana = winning_trades_ana["PnL ($)"].mean() if not winning_trades_ana.empty else 0.0
                avg_loss_ana = losing_trades_ana["PnL ($)"].mean() if not losing_trades_ana.empty else 0.0

                expectancy_ana = ((len(winning_trades_ana) / total_trades_ana) * avg_win_ana) + ((len(losing_trades_ana) / total_trades_ana) * avg_loss_ana) if total_trades_ana > 0 else 0.0
                
                profit_factor_val = gross_profit_ana / abs(gross_loss_ana) if gross_loss_ana != 0 else (np.inf if gross_profit_ana > 0 else 0.0)

                col_metrics_ana1, col_metrics_ana2, col_metrics_ana3, col_metrics_ana4, col_metrics_ana5 = st.columns(5)
                col_metrics_ana1.metric("Net P&L", f"${net_profit_ana:,.2f}")
                col_metrics_ana2.metric("Total Trades", total_trades_ana)
                col_metrics_ana3.metric("Win Rate", f"{win_rate_ana:,.2f}%")
                col_metrics_ana4.metric("Avg Win", f"${avg_win_ana:,.2f}")
                col_metrics_ana5.metric("Avg Loss", f"${abs(avg_loss_ana):,.2f}") 

                col_metrics_ana6, col_metrics_ana7, col_metrics_ana8, col_metrics_ana9 = st.columns(4)
                col_metrics_ana6.metric("Profit Factor", f"{profit_factor_val:,.2f}" if profit_factor_val != np.inf else "∞")
                col_metrics_ana7.metric("Expectancy", f"${expectancy_ana:,.2f} per trade")
                
                longest_win_streak = 0
                current_win_streak = 0
                longest_loss_streak = 0
                current_loss_streak = 0

                for outcome in df_analytics['Win/Loss']:
                    if outcome == "Win":
                        current_win_streak += 1
                        longest_win_streak = max(longest_win_streak, current_win_streak)
                        current_loss_streak = 0 
                    elif outcome == "Loss":
                        current_loss_streak += 1
                        longest_loss_streak = max(longest_loss_streak, current_loss_streak)
                        current_win_streak = 0 
                    else: # Breakeven or other statuses reset both streaks
                        current_win_streak = 0
                        current_loss_streak = 0

                col_metrics_ana8.metric("Longest Win Streak", longest_win_streak)
                col_metrics_ana9.metric("Longest Loss Streak", longest_loss_streak)
                
                st.markdown("---")
                st.markdown("#### Performance Visualizations")

                # Equity Curve
                st.subheader("Equity Curve")
                df_analytics['Cumulative PnL'] = df_analytics["PnL ($)"].cumsum()
                fig_equity = px.line(df_analytics, x=df_analytics.index, y="Cumulative PnL",
                                     title="Equity Curve", labels={"index": "Trade Number"},
                                     color_discrete_sequence=['#58b3b1'])
                fig_equity.update_layout(hovermode="x unified", template="plotly_dark",
                                        xaxis_title="Trade Number", yaxis_title="Cumulative P&L ($)")
                st.plotly_chart(fig_equity, use_container_width=True)

                # Performance by Symbol
                st.subheader("Performance by Symbol")
                df_sym_perf = df_analytics.groupby('Symbol').agg(
                    Total_PnL=('PnL ($)', 'sum'),
                    Trades=('Trade ID', 'count'),
                    Wins=('Win/Loss', lambda x: (x == "Win").sum()),
                    Losses=('Win/Loss', lambda x: (x == "Loss").sum())
                ).reset_index()
                df_sym_perf['Win Rate (%)'] = (df_sym_perf['Wins'] / df_sym_perf['Trades']) * 100 if df_sym_perf['Trades'].sum() > 0 else 0.0
                df_sym_perf.sort_values("Total_PnL", ascending=False, inplace=True)

                fig_symbol = px.bar(df_sym_perf, x='Symbol', y='Total_PnL', color='Win Rate (%)',
                                    title='Total P&L by Symbol (Colored by Win Rate)',
                                    labels={'Total_PnL': 'Total P&L ($)', 'Win Rate (%)': 'Win Rate (%)'},
                                    color_continuous_scale=px.colors.sequential.Viridis)
                fig_symbol.update_layout(template="plotly_dark")
                st.plotly_chart(fig_symbol, use_container_width=True)

                # Performance by Strategy
                st.subheader("Performance by Strategy")
                df_strat_perf = df_analytics.groupby('Strategy Used').agg(
                    Total_PnL=('PnL ($)', 'sum'),
                    Trades=('Trade ID', 'count'),
                    Wins=('Win/Loss', lambda x: (x == "Win").sum()),
                    Losses=('Win/Loss', lambda x: (x == "Loss").sum())
                ).reset_index()
                df_strat_perf['Win Rate (%)'] = (df_strat_perf['Wins'] / df_strat_perf['Trades']) * 100 if df_strat_perf['Trades'].sum() > 0 else 0.0
                df_strat_perf.sort_values("Total_PnL", ascending=False, inplace=True)
                fig_strategy = px.bar(df_strat_perf, x='Strategy Used', y='Total_PnL', color='Win Rate (%)',
                                    title='Total P&L by Strategy Used (Colored by Win Rate)',
                                    labels={'Total_PnL': 'Total P&L ($)', 'Win Rate (%)': 'Win Rate (%)'},
                                    color_continuous_scale=px.colors.sequential.Plasma)
                fig_strategy.update_layout(template="plotly_dark")
                st.plotly_chart(fig_strategy, use_container_width=True)


                # R-Multiples Distribution
                st.subheader("Realized R-Multiples Distribution")
                df_analytics_r = df_analytics[df_analytics['Realized R'].notna() & (df_analytics['Realized R'] != 0.0)].copy() 
                if not df_analytics_r.empty:
                    fig_r_dist = px.histogram(df_analytics_r, x="Realized R", nbins=20,
                                            title="Distribution of Realized R-Multiples",
                                            labels={'Realized R': 'Realized R-Multiple'},
                                            color_discrete_sequence=['#4d7171'])
                    fig_r_dist.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_r_dist, use_container_width=True)
                else:
                    st.info("No trades with valid Realized R-multiples to display.")

                # Emotional Analysis
                st.subheader("Emotional Impact on Performance")
                if "In-Trade Emotions" in df_analytics.columns and not df_analytics["In-Trade Emotions"].str.strip().eq("").all():
                    df_emo_perf = df_analytics[df_analytics["In-Trade Emotions"] != ""].copy()
                    df_emo_perf["Emotion"] = df_emo_perf["In-Trade Emotions"].str.split(',').explode().str.strip()
                    df_emo_perf = df_emo_perf[df_emo_perf["Emotion"] != ""].copy() # Remove empty strings from exploded results
                    
                    if not df_emo_perf.empty:
                        df_emo_grouped = df_emo_perf.groupby("Emotion").agg(
                            Avg_PnL=('PnL ($)', 'mean'),
                            Trades=('Trade ID', 'count'),
                            Wins=('Win/Loss', lambda x: (x == "Win").sum()),
                            Losses=('Win/Loss', lambda x: (x == "Loss").sum())
                        ).reset_index()
                        df_emo_grouped['Win Rate (%)'] = (df_emo_grouped['Wins'] / df_emo_grouped['Trades']) * 100
                        df_emo_grouped.sort_values("Avg_PnL", ascending=False, inplace=True)
                        
                        fig_emotion = px.bar(df_emo_grouped, x='Emotion', y='Avg_PnL', color='Win Rate (%)',
                                            title='Average P&L by In-Trade Emotion',
                                            labels={'Avg_PnL': 'Average P&L ($)', 'Win Rate (%)': 'Win Rate (%)'},
                                            color_continuous_scale=px.colors.sequential.Magenta)
                        fig_emotion.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_emotion, use_container_width=True)
                    else:
                        st.info("No valid emotional data found after filtering. Log more specific emotions.")
                else:
                    st.info("No emotional data logged to analyze. Fill 'In-Trade Emotions' in the journal.")


                # Discipline Score correlation
                st.subheader("Discipline Score vs. P&L")
                df_discipline = df_analytics[df_analytics['Discipline Score 1-5'].notna() & (df_analytics['Discipline Score 1-5'] > 0)].copy()
                if not df_discipline.empty:
                    fig_discipline = px.scatter(df_discipline, x="Discipline Score 1-5", y="PnL ($)",
                                                trendline="ols", title="P&L vs. Discipline Score",
                                                labels={'Discipline Score 1-5': 'Discipline Score', 'PnL ($)': 'Profit/Loss ($)'},
                                                color_discrete_sequence=['#58b3b1'])
                    fig_discipline.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_discipline, use_container_width=True)
                else:
                    st.info("No discipline score data available to plot. Fill 'Discipline Score' in the journal.")

    else:
        st.info("No trades logged yet. Add trades in the 'Log Trade' tab to view analytics.")
        
# =========================================================
# TRADE HISTORY (REVIEW/REPLAY) TAB
# =========================================================
with tab_history:
    st.subheader("📜 Detailed Trade History & Review")
    st.markdown("Select a trade to review its detailed parameters, performance, and your comprehensive notes.")

    if not st.session_state.tools_trade_journal.empty:
        df_history = st.session_state.tools_trade_journal.sort_values(by="Date", ascending=False).reset_index(drop=True)
        
        trade_to_review_options = [
            f"{row['Date'].strftime('%Y-%m-%d %H:%M')} - {row['Symbol']} ({row['Trade ID']}) ({row['Win/Loss']}: ${row['PnL ($)']:.2f})"
            for _, row in df_history.iterrows()
        ]
        
        selected_trade_display = st.selectbox(
            "Select a Trade to Review/Edit",
            options=trade_to_review_options, 
            key="select_trade_to_review",
            help="Choose a logged trade from your journal to view or edit its details."
        )

        selected_trade_id = None
        try:
            # Assuming format "DATE - SYMBOL (TRADE_ID) (WIN/LOSS: $PNL)"
            # Find the index of the first '(' and last ')' that are part of Trade ID
            # It's in the second to last parenthesis.
            # Example: 2025-09-06 16:02 - EUR/USD (TRD-abcde12345) (Win: $100.00)
            # Find the string ' (TRD-...' then split and get the ID.
            id_start_idx = selected_trade_display.rfind('(') # Find the last '('.
            if id_start_idx != -1:
                id_end_idx = selected_trade_display.rfind(')', 0, id_start_idx) # Find ')' before it
                if id_end_idx != -1:
                    trade_id_section = selected_trade_display[id_end_idx + 1:id_start_idx].strip()
                    selected_trade_id = trade_id_section.strip('()')
                else: # Fallback if second-to-last parenthesis parsing fails, maybe it's the very first?
                     selected_trade_id_parts = selected_trade_display.split('(')
                     if len(selected_trade_id_parts) > 1:
                         selected_trade_id = selected_trade_id_parts[-2].strip(') ').strip()

            if not selected_trade_id: # Still not found, try a simpler split
                 selected_trade_id_parts = selected_trade_display.split('(')
                 if len(selected_trade_id_parts) > 1:
                    selected_trade_id = selected_trade_id_parts[-2].strip(') ').strip()

        except Exception as e:
            logging.error(f"Error parsing trade ID from display string '{selected_trade_display}': {e}", exc_info=True)
            st.error("Could not parse Trade ID from the selected entry. Please select another trade.")

        selected_trade_row = None
        trade_idx_in_df = None # Initialize to None

        if selected_trade_id:
            matching_rows = df_history[df_history['Trade ID'] == selected_trade_id]
            if not matching_rows.empty:
                selected_trade_row = matching_rows.iloc[0]
                trade_idx_in_df = matching_rows.index[0] # Original index for update/delete
            else:
                st.warning(f"Trade with ID `{selected_trade_id}` not found in journal. It might have been deleted.")

        if selected_trade_row is not None:
            st.markdown(f"### Reviewing Trade: `{selected_trade_row['Trade ID']}`")
            st.markdown("---")

            # --- Section: Summary Overview ---
            with st.expander("📊 Trade Summary", expanded=True):
                cols_summary_r1, cols_summary_r2, cols_summary_r3 = st.columns(3)
                cols_summary_r1.metric("Date", selected_trade_row['Date'].strftime('%Y-%m-%d'))
                cols_summary_r2.metric("Symbol", selected_trade_row['Symbol'])
                cols_summary_r3.metric("Strategy Used", selected_trade_row['Strategy Used'] if selected_trade_row['Strategy Used'] else "N/A")

                cols_summary_r4, cols_summary_r5, cols_summary_r6 = st.columns(3)
                cols_summary_r4.metric("Trade Type", selected_trade_row['Trade Type'])
                cols_summary_r5.metric("Outcome", selected_trade_row['Win/Loss'])
                delta_pnl = f"{selected_trade_row['Pips']:.2f} pips" if selected_trade_row['Pips'] != 0 else None
                delta_color_pnl = "normal" if selected_trade_row['PnL ($)'] != 0 else "off"

                cols_summary_r6.metric("P&L ($)", f"${selected_trade_row['PnL ($)']:.2f}",
                                        delta=delta_pnl,
                                        delta_color=delta_color_pnl)
                
                cols_summary_r7, cols_summary_r8 = st.columns(2)
                cols_summary_r7.metric("Initial R", f"{selected_trade_row['Initial R']:.2f}")
                cols_summary_r8.metric("Realized R", f"{selected_trade_row['Realized R']:.2f}")


            # --- Section: Detailed Execution ---
            with st.expander("📈 Execution Details"):
                cols_exec_time, cols_pricing_ex, cols_lots_type = st.columns(3)
                cols_exec_time.write(f"**Entry Time:** {selected_trade_row['Entry Time'].strftime('%H:%M:%S') if pd.notna(selected_trade_row['Entry Time']) else 'N/A'}")
                cols_exec_time.write(f"**Exit Time:** {selected_trade_row['Exit Time'].strftime('%H:%M:%S') if pd.notna(selected_trade_row['Exit Time']) else 'N/A'}")

                cols_pricing_ex.write(f"**Entry Price:** {selected_trade_row['Entry Price']:.5f}")
                cols_pricing_ex.write(f"**Stop Loss Price:** {selected_trade_row['Stop Loss Price']:.5f}")
                cols_pricing_ex.write(f"**Take Profit Price:** {selected_trade_row['Take Profit Price']:.5f}")
                cols_pricing_ex.write(f"**Final Exit Price:** {selected_trade_row['Final Exit Price']:.5f}")
                
                cols_lots_type.write(f"**Lots:** {selected_trade_row['Lots']:.2f}")
                # These were old columns. Removed now, so just fallback to N/A
                cols_lots_type.write(f"**Original Order Type:** {selected_trade_row.get('Order Type', 'N/A')}")
                cols_lots_type.write(f"**Original Partial Exits:** {'Yes' if selected_trade_row.get('Partial Exits', False) else 'No'}")


            # --- Section: Market & Strategy Context ---
            with st.expander("🌍 Market & Strategy Context"):
                st.write(f"**Higher Timeframe Bias & Structure:** {selected_trade_row['HTF Bias & Structure']}")
                st.write(f"**Market State (HTF):** {selected_trade_row['Market State (HTF)']}")
                st.write(f"**News Event Impact:** {selected_trade_row['News Event Impact'].replace(',', ', ')}")
                st.write(f"**Setup Details & Entry Trigger:** {selected_trade_row['Setup Details']}")
                

            # --- Section: Rationale & Visuals ---
            with st.expander("📝 Rationale & Visuals"):
                st.markdown("##### Entry Rationale:")
                st.markdown(render_markdown_content(selected_trade_row["Entry Rationale"]), unsafe_allow_html=True)
                
                st.markdown("##### Exit Rationale:")
                st.markdown(render_markdown_content(selected_trade_row["Exit Rationale"]), unsafe_allow_html=True)
                
                entry_screenshot_link = selected_trade_row['Entry Screenshot'] if pd.notna(selected_trade_row['Entry Screenshot']) else ""
                exit_screenshot_link = selected_trade_row['Exit Screenshot'] if pd.notna(selected_trade_row['Exit Screenshot']) else ""
                
                if entry_screenshot_link or exit_screenshot_link:
                    st.markdown("---")
                    st.markdown("#### Screenshots")
                    col_screens_1, col_screens_2 = st.columns(2)
                    if entry_screenshot_link:
                        with col_screens_1:
                            if os.path.exists(entry_screenshot_link):
                                st.image(entry_screenshot_link, caption="Entry Screenshot", use_column_width=True)
                            else:
                                st.info(f"Entry screenshot referenced, but file not found: `{os.path.basename(entry_screenshot_link)}`")
                    if exit_screenshot_link:
                        with col_screens_2:
                            if os.path.exists(exit_screenshot_link):
                                st.image(exit_screenshot_link, caption="Exit Screenshot", use_column_width=True)
                            else:
                                st.info(f"Exit screenshot referenced, but file not found: `{os.path.basename(exit_screenshot_link)}`")
                else:
                    st.info("No screenshots linked to this trade.")
            
            # --- Section: Psychological Factors ---
            with st.expander("🧠 Psychological Factors"):
                st.write(f"**Pre-Trade Mindset:** {selected_trade_row['Pre-Trade Mindset']}")
                st.write(f"**In-Trade Emotions:** {selected_trade_row['In-Trade Emotions'].replace(',', ', ')}")
                # Emotional Triggers field from prior complex version. Keep fallback.
                st.write(f"**Emotional Triggers (Original):** {selected_trade_row.get('Emotional Triggers', 'N/A')}")
                st.write(f"**Discipline Score:** {selected_trade_row['Discipline Score 1-5']:.0f}/5")

            # --- Section: Combined Journal Notes & Tags ---
            with st.expander("📚 Comprehensive Journal Notes & Tags"):
                st.markdown("##### Trade Notes:")
                st.markdown(render_markdown_content(selected_trade_row["Trade Journal Notes"]), unsafe_allow_html=True)

                st.write(f"**Tags:** {selected_trade_row['Tags'].replace(',', ', ')}")

            st.markdown("---")
            col_review_buttons_final = st.columns(2)
            with col_review_buttons_final[0]:
                if st.button("Edit This Trade", key=f"edit_trade_history_{selected_trade_row['Trade ID']}", type="primary"):
                    # Prepare data for pre-filling, convert datetime to Python objects expected by date_input/time_input
                    trade_data_to_edit = selected_trade_row.to_dict()
                    # Convert pandas NaT to None or Python None
                    for key in ["Date", "Entry Time", "Exit Time"]:
                        if pd.isna(trade_data_to_edit.get(key)):
                            trade_data_to_edit[key] = None
                        elif isinstance(trade_data_to_edit.get(key), dt.datetime):
                            if key == "Date": trade_data_to_edit[key] = trade_data_to_edit[key].date()
                            if key == "Entry Time" or key == "Exit Time": trade_data_to_edit[key] = trade_data_to_edit[key].time()

                    st.session_state.edit_trade_data = trade_data_to_edit

                    st.info(f"Form pre-filled for Trade ID `{selected_trade_row['Trade ID']}`. Please go to the **Log Trade** tab to modify.")
                    st.rerun() 

            with col_review_buttons_final[1]:
                if st.button("Delete This Trade", key=f"delete_trade_history_{selected_trade_row['Trade ID']}"):
                    if trade_idx_in_df is not None: # Ensure a valid index was found
                        st.session_state.tools_trade_journal = st.session_state.tools_trade_journal.drop(index=trade_idx_in_df).reset_index(drop=True)
                        if 'logged_in_user' in st.session_state:
                            _ta_save_journal(st.session_state.logged_in_user, st.session_state.tools_trade_journal, conn, c)
                        st.success("Trade deleted successfully!")
                        st.rerun()
                    else:
                        st.error("Cannot delete trade: Original index not found.")
        else: # This 'else' clause handles the case where selected_trade_row is None after all checks.
            st.info("Select a trade from the dropdown above to view its details.")

    else:
        st.info("No trades logged yet. Add trades in the 'Log Trade' tab.")
            
# Challenge Mode (remains unchanged)
st.markdown("---")
st.subheader("🏅 Challenge Mode")
st.write("30-Day Journaling Discipline Challenge - Gain 300 XP for completing, XP can be exchanged for gift cards!")
streak = st.session_state.get('streak', 0)
progress = min(streak / 30.0, 1.0)
st.progress(progress)
if progress >= 1.0 and st.session_state.get('challenge_30day_completed', False) == False:
    st.success("Challenge completed! Great job on your consistency.")
    if 'logged_in_user' in st.session_state:
        ta_update_xp(300) # Bonus XP for completion
    st.session_state.challenge_30day_completed = True # Ensure XP is only given once
elif progress >= 1.0:
    st.info("You've already completed this challenge!")

# Leaderboard / Self-Competition (remains largely unchanged)
st.subheader("🏆 Leaderboard - Consistency")
if 'logged_in_user' in st.session_state: # Only attempt to load leaderboard if we have a connection
    users_from_db = c.execute("SELECT username, data FROM users").fetchall()
    leader_data = []
    for u, d in users_from_db:
        user_d = json.loads(d) if d else {}
        journal_entries = user_d.get("tools_trade_journal", [])
        if isinstance(journal_entries, list): # Check if it's a list of dicts (actual rows)
            trade_count = sum(1 for entry in journal_entries if entry.get("Win/Loss") in ["Win", "Loss", "Breakeven", "Pending / Invalid Prices"])
        else: # Fallback for any unexpected/older formats
            trade_count = 0
            logging.warning(f"Journal data for user {u} is not in expected list format. Entries will not be counted.")
        leader_data.append({"Username": u, "Journaled Trades": trade_count})
    if leader_data:
        leader_df = pd.DataFrame(leader_data).sort_values("Journaled Trades", ascending=False).reset_index(drop=True)
        leader_df["Rank"] = leader_df.index + 1
        st.dataframe(leader_df[["Rank", "Username", "Journaled Trades"]], hide_index=True)
    else:
        st.info("No leaderboard data yet. Log some trades!")
else:
    st.info("Log in to view the Leaderboard. (Mocked login enabled in this test file.)")
