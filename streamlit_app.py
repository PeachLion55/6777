import streamlit as st
import pandas as pd
import datetime as dt
import os
import json
import hashlib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sqlite3
import logging
import uuid
import re

# =========================================================
# PAGE CONFIGURATION & GLOBAL CSS
# =========================================================
st.set_page_config(page_title="Forex Dashboard - Backtesting", layout="wide")

st.markdown(
    """
    <style>
    /* --- Global Styles --- */
    hr {
        margin-top: 1.5rem !important; margin-bottom: 1.5rem !important;
        border-top: 1px solid #4d7171 !important; border-bottom: none !important;
        background-color: transparent !important; height: 1px !important;
    }
    #MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden !important; }
    .css-18e3th9, .css-1d391kg, .block-container { padding-top: 1rem !important; margin-top: 0rem !important; }
    
    /* --- Markdown in Text Areas Display --- */
    div.stText p, div.stExpander div.stText p {
         margin-bottom: 0.5rem; line-height: 1.5; color: var(--text-color);
    }
    div.stText ul, div.stText ol, div.stExpander div.stText ul, div.stExpander div.stText ol {
        margin-top: 0; margin-bottom: 0.5rem; padding-left: 20px;
    }
    div.stText h1, h2, h3, h4, h5, h6,
    div.stExpander div.stText h1, h2, h3, h4, h5, h6 {
        margin-top: 1rem; margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(88, 179, 177, 0.5); padding-bottom: 5px;
        color: var(--primary-color);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Gridline Background Settings ---
grid_color = "#58b3b1"; grid_opacity = 0.16; grid_size = 40
r, g, b = int(grid_color[1:3], 16), int(grid_color[3:5], 16), int(grid_color[5:7], 16)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #000000;
        background-image:
        linear-gradient(rgba({r},{g},{b},{grid_opacity}) 1px, transparent 1px),
        linear-gradient(90deg, rgba({r},{g},{b},{grid_opacity}) 1px, transparent 1px);
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _ta_user_dir(user_id="guest"):
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    root = os.path.join(script_dir, "user_data")
    os.makedirs(root, exist_ok=True)
    d = os.path.join(root, user_id)
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(d, "journal_images"), exist_ok=True)
    return d

def _ta_hash():
    return uuid.uuid4().hex[:12]

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.datetime, dt.date)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super().default(obj)

def _ta_save_journal(username, journal_df, conn, c):
    try:
        c.execute("SELECT data FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        user_data = json.loads(result[0]) if result and result[0] else {}
        user_data["tools_trade_journal"] = journal_df.replace({pd.NA: None, np.nan: None}).to_dict('records')
        c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
        conn.commit()
        logging.info(f"Journal saved for user {username}: {len(journal_df)} trades")
        return True
    except Exception as e:
        logging.error(f"Failed to save journal for {username}: {str(e)}", exc_info=True)
        st.error(f"Failed to save journal: {str(e)}")
        return False

def show_xp_notification(xp_gained):
    st.toast(f"🎉 +{xp_gained} XP Earned!", icon="⭐")

def ta_update_xp(amount):
    if "logged_in_user" in st.session_state and 'conn' in globals() and 'c' in globals():
        username = st.session_state.logged_in_user
        try:
            c.execute("SELECT data FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            if result:
                user_data = json.loads(result[0])
                user_data['xp'] = user_data.get('xp', 0) + amount
                if (user_data['xp'] // 100) > user_data.get('level', 0):
                    user_data['level'] = user_data['xp'] // 100
                    user_data['badges'] = user_data.get('badges', []) + [f"Level {user_data['level']}"]
                    st.balloons()
                c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
                conn.commit()
                st.session_state.xp = user_data.get('xp')
                show_xp_notification(amount)
        except Exception as e:
            logging.error(f"Failed to update XP for {username}: {str(e)}")
            
def ta_update_streak():
    if "logged_in_user" in st.session_state and 'conn' in globals() and 'c' in globals():
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
                    if dt.date.fromisoformat(last_date) == dt.date.fromisoformat(today) - dt.timedelta(days=1):
                        streak += 1
                    elif dt.date.fromisoformat(last_date) < dt.date.fromisoformat(today) - dt.timedelta(days=1):
                        streak = 1
                else: streak = 1
                user_data['streak'], user_data['last_journal_date'] = streak, today
                c.execute("UPDATE users SET data = ? WHERE username = ?", (json.dumps(user_data, cls=CustomJSONEncoder), username))
                conn.commit()
                st.session_state.streak = streak
        except Exception as e:
            logging.error(f"Failed to update streak for {username}: {str(e)}")


# =========================================================
# DATABASE CONNECTION (Standalone test setup)
# =========================================================
DB_FILE = "test_users_simplified.db"
conn = None; c = None

try:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, data TEXT)''')
    conn.commit()
    logging.info("SQLite database initialized for standalone test.")
    
    test_username = "test_user_journal"
    st.session_state.logged_in_user = test_username 

    c.execute("SELECT username FROM users WHERE username = ?", (test_username,))
    if c.fetchone() is None:
        hashed_password = hashlib.sha256("test_password".encode()).hexdigest()
        initial_user_data = {
            "xp": 0, "level": 0, "badges": [], "streak": 0, "drawings": {}, 
            "tools_trade_journal": [], 
            "strategies": [{"Name": "Trend Following", "Description": "Simple Trend Following"}]
        }
        c.execute("INSERT INTO users (username, password, data) VALUES (?, ?, ?)", (test_username, hashed_password, json.dumps(initial_user_data)))
        conn.commit()
        logging.info(f"Test user '{test_username}' created and (mock) logged in.")
    
    c.execute("SELECT data FROM users WHERE username = ?", (test_username,))
    user_data_result = c.fetchone()
    if user_data_result:
        user_data_loaded = json.loads(user_data_result[0])
        st.session_state.drawings = user_data_loaded.get("drawings", {})
        st.session_state.strategies = pd.DataFrame(user_data_loaded.get("strategies", []))

except Exception as e:
    logging.error(f"Failed to initialize SQLite database for test: {str(e)}", exc_info=True)
    st.error(f"Test database initialization failed.")


# =========================================================
# JOURNAL SCHEMA & DATA MIGRATION
# =========================================================

journal_cols = [
    "Trade ID", "Date", "Entry Time", "Exit Time", "Symbol", "Trade Type", "Lots",
    "Entry Price", "Stop Loss Price", "Take Profit Price", "Final Exit Price",
    "Win/Loss", "PnL ($)", "Pips", "Initial R", "Realized R",
    "Strategy Used", "HTF Context", "Market State (HTF)", "News Event Impact",
    "Trade Setup", "Entry Rationale", "Exit Rationale",
    "Pre-Trade Mindset", "In-Trade Emotions", "Discipline Score 1-5",
    "Trade Journal Notes", "Entry Screenshot", "Exit Screenshot", "Tags"
]

journal_dtypes = {
    "Trade ID": str, "Date": "datetime64[ns]", "Entry Time": "datetime64[ns]", "Exit Time": "datetime64[ns]", "Symbol": str,
    "Trade Type": str, "Lots": float, "Entry Price": float, "Stop Loss Price": float, 
    "Take Profit Price": float, "Final Exit Price": float, "Win/Loss": str, "PnL ($)": float, 
    "Pips": float, "Initial R": float, "Realized R": float, "Strategy Used": str, "HTF Context": str,
    "Market State (HTF)": str, "News Event Impact": str, "Trade Setup": str, "Entry Rationale": str, 
    "Exit Rationale": str, "Pre-Trade Mindset": str, "In-Trade Emotions": str, 
    "Discipline Score 1-5": float, "Trade Journal Notes": str, "Entry Screenshot": str, 
    "Exit Screenshot": str, "Tags": str
}

if "tools_trade_journal" not in st.session_state:
    st.session_state.tools_trade_journal = pd.DataFrame(columns=journal_cols).astype(journal_dtypes, errors='ignore')
else:
    # Robust migration logic
    current_df = st.session_state.tools_trade_journal.copy()
    migrated_df = pd.DataFrame(index=current_df.index)

    for col in journal_cols:
        if col in current_df.columns:
            migrated_df[col] = current_df[col]
        else: # Handle newly added columns with defaults
            dtype = journal_dtypes.get(col, str)
            if dtype == str: migrated_df[col] = ""
            elif 'datetime' in str(dtype): migrated_df[col] = pd.NaT
            elif dtype == float: migrated_df[col] = 0.0
            elif dtype == bool: migrated_df[col] = False
            else: migrated_df[col] = None

    for col, dtype in journal_dtypes.items():
        try:
            if dtype == str: migrated_df[col] = migrated_df[col].fillna('').astype(str)
            elif 'datetime' in str(dtype): migrated_df[col] = pd.to_datetime(migrated_df[col], errors='coerce')
            elif dtype == float: migrated_df[col] = pd.to_numeric(migrated_df[col], errors='coerce').fillna(0.0).astype(float)
            elif dtype == bool: migrated_df[col] = migrated_df[col].fillna(False).astype(bool)
            else: migrated_df[col] = migrated_df[col].astype(dtype, errors='ignore')
        except Exception as e:
             logging.error(f"Error casting column '{col}' to '{dtype}': {e}")
    
    st.session_state.tools_trade_journal = migrated_df[journal_cols]

def render_markdown_content(text_content):
    return text_content # Directly pass the markdown content.


# =========================================================
# BACKTESTING PAGE CONTENT
# =========================================================
st.title("📈 Backtesting") 
st.caption("Live TradingView chart for backtesting and a simplified, user-friendly trading journal for tracking and analyzing trades.")
st.markdown('---')

pairs_map = {
    "EUR/USD": "FX:EURUSD", "USD/JPY": "FX:USDJPY", "GBP/USD": "FX:GBPUSD", "USD/CHF": "OANDA:USDCHF", 
    "AUD/USD": "FX:AUDUSD", "NZD/USD": "OANDA:NZDUSD", "USD/CAD": "CMCMARKETS:USDCAD",
}
pair = st.selectbox("Select pair", list(pairs_map.keys()), index=0, key="tv_pair")
tv_symbol = pairs_map[pair]

tv_html = f"""
<div id="tradingview_widget"></div>
<script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
<script type="text/javascript">
new TradingView.widget({{
    "container_id": "tradingview_widget", "width": "100%", "height": 650, "symbol": "{tv_symbol}",
    "interval": "D", "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "en",
    "enable_publishing": false, "allow_symbol_change": true, "show_popup_button": true
}});
</script>
"""
st.components.v1.html(tv_html, height=670, scrolling=False)

if "logged_in_user" in st.session_state:
    st.empty() # Placeholder for future drawing buttons

st.markdown("### 📝 Simplified Trading Journal")
st.markdown("Log your trades with essential details. Use Markdown (e.g., `**bold**`) in text fields for basic formatting.")

# =========================================================
# TRADING JOURNAL TABS
# =========================================================

tab_entry, tab_analytics, tab_history = st.tabs(["📝 Log Trade", "📊 Analytics", "📜 Trade History"])

# =========================================================
# LOG TRADE TAB
# =========================================================
with tab_entry:
    st.header("Log a New Trade")
    st.caption("Focus on the essentials. You can add more context later if needed.")

    initial_data = st.session_state.get('edit_trade_data', {})
    is_editing = bool(initial_data)

    with st.form("trade_entry_form", clear_on_submit=not is_editing):
        
        # --- ESSENTIALS ---
        st.markdown("##### Trade Essentials")
        cols_essentials_1, cols_essentials_2 = st.columns(2)
        with cols_essentials_1:
            trade_id = initial_data.get("Trade ID", f"TRD-{_ta_hash()}")
            st.text_input("Trade ID", value=trade_id, disabled=True)
            
            date_val = initial_data.get("Date", dt.date.today())
            trade_date = st.date_input("Date", value=date_val, key="trade_date_input")

            symbol_options = list(pairs_map.keys()) + ["Other"]
            default_symbol = initial_data.get("Symbol", pair)
            default_symbol_idx = symbol_options.index(default_symbol) if default_symbol in symbol_options else symbol_options.index("Other")
            symbol = st.selectbox("Symbol", symbol_options, index=default_symbol_idx, key="symbol_input")
            if symbol == "Other":
                symbol = st.text_input("Custom Symbol", value=initial_data.get("Symbol", ""), key="custom_symbol_input")
            
            trade_type = st.radio("Direction", ["Long", "Short"], horizontal=True, 
                                  index=["Long", "Short"].index(initial_data.get("Trade Type", "Long")), key="trade_type_input")
        
        with cols_essentials_2:
            lots = st.number_input("Size (Lots)", min_value=0.01, step=0.01, format="%.2f", value=float(initial_data.get("Lots", 0.1)), key="lots_input")
            entry_price = st.number_input("Entry Price", min_value=0.0, step=0.00001, format="%.5f", value=float(initial_data.get("Entry Price", 0.0)), key="entry_price_input")
            stop_loss_price = st.number_input("Stop Loss Price", min_value=0.0, step=0.00001, format="%.5f", value=float(initial_data.get("Stop Loss Price", 0.0)), key="stop_loss_input")
            final_exit_price = st.number_input("Final Exit Price", min_value=0.0, step=0.00001, format="%.5f", value=float(initial_data.get("Final Exit Price", 0.0)), key="final_exit_input")
            win_loss_options = ["Win", "Loss", "Breakeven"]
            win_loss = st.selectbox("Outcome", options=win_loss_options, index=win_loss_options.index(initial_data.get("Win/Loss", "Win")), key="win_loss_input")
        
        # --- RATIONALE & REFLECTION ---
        st.markdown("##### Rationale & Reflection (Use Markdown for formatting)")
        entry_rationale = st.text_area("Entry Rationale", value=initial_data.get("Entry Rationale", ""), height=120, 
                                       help="Why did you enter this trade?", key="entry_rationale_input")
        trade_journal_notes = st.text_area("Overall Notes & Learnings", value=initial_data.get("Trade Journal Notes", ""), height=150, 
                                           help="How did the trade play out? What were the key takeaways?", key="trade_journal_notes_input")

        # --- MORE DETAILS (OPTIONAL) ---
        with st.expander("More Details (Optional)"):
            cols_advanced_1, cols_advanced_2 = st.columns(2)
            with cols_advanced_1:
                st.markdown("###### Context & Strategy")
                user_strategies = ["(None)"] + sorted([s['Name'] for s in st.session_state.strategies.to_dict('records')])
                default_strategy = initial_data.get("Strategy Used", "(None)")
                default_strategy_idx = user_strategies.index(default_strategy) if default_strategy in user_strategies else 0
                selected_strategy = st.selectbox("Strategy Used", options=user_strategies, index=default_strategy_idx, key="strategy_used_input_adv") 
                
                htf_context = st.text_area("Higher Timeframe Context", value=initial_data.get("HTF Context", ""), height=80, key="htf_context_input_adv")
                
                current_tags = sorted(list(set(st.session_state.tools_trade_journal['Tags'].str.split(',').explode().dropna().str.strip())))
                suggested_tags = ["Breakout", "Reversal", "FOMO", "Overleveraged", "London Session", "NY Session"]
                all_tag_options = sorted(list(set(current_tags + suggested_tags)))
                initial_tags = initial_data.get("Tags", "").split(',') if initial_data.get("Tags") else []
                tags = st.multiselect("Trade Tags", all_tag_options, default=[t for t in initial_tags if t in all_tag_options], key="tags_input_adv")

            with cols_advanced_2:
                st.markdown("###### Psychology & Visuals")
                in_trade_emotions_options = ["Confident", "Anxious", "Fearful", "Excited", "Frustrated", "Neutral", "FOMO", "Greedy", "Patient", "Bored"]
                initial_emotions = initial_data.get("In-Trade Emotions", "").split(',') if initial_data.get("In-Trade Emotions") else []
                in_trade_emotions = st.multiselect("In-Trade Emotions", in_trade_emotions_options, default=[e for e in initial_emotions if e in in_trade_emotions_options], key="in_trade_emotions_input_adv")
                discipline_score = st.slider("Discipline Score (1-5)", 1, 5, value=int(initial_data.get("Discipline Score 1-5", 3)), key="discipline_score_input_adv")
                
                user_images_dir = _ta_user_dir(st.session_state.get("logged_in_user", "guest"))
                
                entry_ss_path = st.text_input("Entry Screenshot Path", value=initial_data.get("Entry Screenshot", ""), key="entry_ss_path", label_visibility="collapsed")
                exit_ss_path = st.text_input("Exit Screenshot Path", value=initial_data.get("Exit Screenshot", ""), key="exit_ss_path", label_visibility="collapsed")
                
                uploaded_entry_image = st.file_uploader("Upload Entry Screenshot", type=["png", "jpg", "jpeg"], key="upload_entry_screenshot_adv")
                if uploaded_entry_image:
                    image_filename = f"{trade_id}_entry.png"
                    entry_ss_path = os.path.join(user_images_dir, "journal_images", image_filename)
                    with open(entry_ss_path, "wb") as f: f.write(uploaded_entry_image.getbuffer())
                    st.success("Entry screenshot uploaded!")

                uploaded_exit_image = st.file_uploader("Upload Exit Screenshot", type=["png", "jpg", "jpeg"], key="upload_exit_screenshot_adv")
                if uploaded_exit_image:
                    image_filename = f"{trade_id}_exit.png"
                    exit_ss_path = os.path.join(user_images_dir, "journal_images", image_filename)
                    with open(exit_ss_path, "wb") as f: f.write(uploaded_exit_image.getbuffer())
                    st.success("Exit screenshot uploaded!")
        
        # --- SUBMISSION ---
        st.markdown("---")
        submit_button = st.form_submit_button(f"{'Update Trade' if is_editing else 'Save New Trade'}", type="primary")

        if submit_button:
            if "JPY" in symbol: pip_scale = 0.01; approx_pip_value_usd_per_lot = 8.5
            else: pip_scale = 0.0001; approx_pip_value_usd_per_lot = 10.0
            
            pips, pnl, initial_r, realized_r = 0.0, 0.0, 0.0, 0.0

            if win_loss in ["Win", "Loss"]:
                if trade_type == "Long": pips = (final_exit_price - entry_price) / pip_scale
                else: pips = (entry_price - final_exit_price) / pip_scale
                pnl = pips * lots * approx_pip_value_usd_per_lot

            risk_per_unit = abs(entry_price - stop_loss_price)
            if risk_per_unit > 0:
                reward_per_unit = abs(take_profit_price - entry_price) if st.session_state.take_profit_price > 0 else 0
                initial_r = reward_per_unit / risk_per_unit
                realized_pnl_raw = final_exit_price - entry_price if trade_type == "Long" else entry_price - final_exit_price
                realized_r = realized_pnl_raw / risk_per_unit

            new_trade_data = {
                "Trade ID": trade_id, "Date": pd.to_datetime(trade_date), "Entry Time": pd.to_datetime(f"{trade_date} {st.session_state.entry_time_input}"),
                "Exit Time": pd.to_datetime(f"{trade_date} {st.session_state.exit_time_input}"), "Symbol": symbol, "Trade Type": trade_type, "Lots": lots,
                "Entry Price": entry_price, "Stop Loss Price": stop_loss_price, "Take Profit Price": st.session_state.take_profit_price,
                "Final Exit Price": final_exit_price, "Win/Loss": win_loss, "PnL ($)": pnl, "Pips": pips,
                "Initial R": initial_r, "Realized R": realized_r, "Strategy Used": selected_strategy if selected_strategy != "(None)" else "",
                "HTF Context": st.session_state.htf_context_input_adv, "Market State (HTF)": st.session_state.market_state_input_adv, 
                "News Event Impact": ','.join(st.session_state.news_impact_input_adv), "Trade Setup": st.session_state.trade_setup_input_adv,
                "Entry Rationale": entry_rationale, "Exit Rationale": exit_rationale, "Pre-Trade Mindset": st.session_state.pre_trade_mindset_input_adv,
                "In-Trade Emotions": ','.join(st.session_state.in_trade_emotions_input_adv), "Discipline Score 1-5": float(discipline_score),
                "Trade Journal Notes": trade_journal_notes, "Entry Screenshot": st.session_state.entry_ss_path, "Exit Screenshot": st.session_state.exit_ss_path,
                "Tags": ','.join(tags)
            }

            new_trade_df = pd.DataFrame([new_trade_data]).astype(journal_dtypes, errors='ignore')
            
            if is_editing:
                idx = st.session_state.tools_trade_journal[st.session_state.tools_trade_journal['Trade ID'] == trade_id].index
                st.session_state.tools_trade_journal.loc[idx] = new_trade_df.iloc[0].values
                st.success(f"Trade {trade_id} updated!")
            else:
                st.session_state.tools_trade_journal = pd.concat([st.session_state.tools_trade_journal, new_trade_df], ignore_index=True)
                st.success("New trade saved!")
            
            if 'logged_in_user' in st.session_state:
                if _ta_save_journal(st.session_state.logged_in_user, st.session_state.tools_trade_journal, conn, c):
                    if not is_editing: ta_update_xp(10); ta_update_streak()
            
            if 'edit_trade_data' in st.session_state: del st.session_state['edit_trade_data']
            st.rerun()

    # --- Analytics & History Tabs ---
    with tab_analytics:
        st.header("📊 Trade Analytics & Insights")
        if not st.session_state.tools_trade_journal.empty:
            df_analytics = st.session_state.tools_trade_journal.copy()
            df_analytics = df_analytics[df_analytics["Win/Loss"].isin(["Win", "Loss", "Breakeven"])].copy()
            if not df_analytics.empty:
                # [Analytics logic adapted for simplified schema if needed]
                st.dataframe(df_analytics)
            else: st.info("No trades to analyze. Log some wins, losses, or breakevens.")
        else: st.info("No trades logged yet.")
            
    with tab_history:
        st.header("📜 Detailed Trade History & Review")
        if not st.session_state.tools_trade_journal.empty:
            df_history = st.session_state.tools_trade_journal.sort_values(by="Date", ascending=False).reset_index(drop=True)
            
            # [Trade History selection & display logic from previous answer would go here]
            st.dataframe(df_history)
        else:
            st.info("No trades logged yet.")
            

# ... (Challenge Mode & Leaderboard code remains the same, placed after tabs) ...
st.markdown("---")
# Gamification features
if "logged_in_user" in st.session_state:
    st.subheader("🏅 Your Progress")
    cols_gamify_1, cols_gamify_2 = st.columns(2)
    with cols_gamify_1:
        st.metric("Journaling Streak", f"{st.session_state.get('streak', 0)} Days")
        progress = min(st.session_state.get('streak', 0) / 30.0, 1.0)
        st.progress(progress, text="30-Day Discipline Challenge")
        if progress >= 1.0: st.success("Challenge Completed!")
    with cols_gamify_2:
        # Leaderboard
        users_from_db = c.execute("SELECT username, data FROM users").fetchall()
        leader_data = []
        for u, d in users_from_db:
            user_d = json.loads(d) if d else {}
            journal_entries = user_d.get("tools_trade_journal", [])
            if isinstance(journal_entries, list): 
                trade_count = sum(1 for entry in journal_entries if entry.get("Win/Loss") in ["Win", "Loss", "Breakeven"])
            else: trade_count = 0
            leader_data.append({"Username": u, "Trades": trade_count})
        if leader_data:
            leader_df = pd.DataFrame(leader_data).sort_values("Trades", ascending=False).reset_index(drop=True)
            leader_df["Rank"] = leader_df.index + 1
            st.dataframe(leader_df[["Rank", "Username", "Trades"]], hide_index=True)
        else:
            st.info("No leaderboard data yet. Log some trades!")
else:
    st.info("Log in to view the Leaderboard. (Mocked login enabled in this test file.)")
