"""
app.py — Premium Financial Fraud Detection Dashboard
Multi-page Streamlit application with 6 analytical sections.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.preprocessing   import load_raw_data, get_dataset_stats, preprocess
from src.models          import train_all, predict_transaction
from src.explainability  import compute_shap_values, shap_summary_fig, shap_bar_fig, shap_single_transaction
from src.utils           import COLORS, risk_score, format_currency, KEY_FEATURES
from src.alerts          import send_email_alert, send_whatsapp_alert, build_fraud_email, build_alert_summary
from src.auth            import register_user, login_user, save_user_settings, load_user_settings, change_password

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI · Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# AUTH GATE — Show login/register if not logged in
# ══════════════════════════════════════════════════════════════════════════════

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    # Full-page login / register UI
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    .block-container { max-width: 480px !important; margin: 80px auto !important; }
    .auth-card {
        background: #12122a;
        border: 1px solid #2a2a4a;
        border-radius: 20px;
        padding: 40px 36px;
        box-shadow: 0 8px 40px rgba(0,0,0,.6);
    }
    .auth-title {
        font-size: 28px; font-weight: 800;
        background: linear-gradient(135deg,#e63946,#ffd700);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .auth-sub { color:#8888aa; font-size:13px; margin-bottom:28px; }

    /* ── All input boxes: dark background, bright text ── */
    input, textarea, [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
        background-color: #1a1a35 !important;
        color: #f0f0f0 !important;
        border: 1px solid #2a2a4a !important;
        border-radius: 8px !important;
        caret-color: #f0f0f0 !important;
    }
    /* Placeholder text */
    input::placeholder, textarea::placeholder { color: #6666aa !important; opacity: 1 !important; }
    /* Labels above every widget */
    label, .stTextInput > label, .stTextArea > label,
    .stNumberInput > label, .stSelectbox > label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] { color: #c8c8e0 !important; font-weight: 600 !important; }
    /* Select / dropdown */
    [data-baseweb="select"] div, [data-baseweb="select"] span { color: #f0f0f0 !important; background-color: #1a1a35 !important; }
    /* Focus ring */
    input:focus, textarea:focus { border-color: #e63946 !important; box-shadow: 0 0 0 2px rgba(230,57,70,.25) !important; }
    .stButton > button { border-radius: 10px !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;margin-bottom:32px;'>
        <div style='font-size:56px;'>&#x1F6E1;&#xFE0F;</div>
        <div class='auth-title'>FraudShield AI</div>
        <div class='auth-sub'>Financial Fraud Detection System</div>
    </div>
    """, unsafe_allow_html=True)

    auth_tab1, auth_tab2 = st.tabs(["  🔐  Login  ", "  📄  Create Account  "])

    # ── Login Tab ────────────────────────────────────────────────────
    with auth_tab1:
        with st.form("login_form"):
            login_user_input = st.text_input("👤 Username",    placeholder="Enter your username")
            login_pass_input = st.text_input("🔑 Password",    placeholder="Enter your password", type="password")
            login_btn = st.form_submit_button("🔐 Login", use_container_width=True, type="primary")

        if login_btn:
            result = login_user(login_user_input.strip(), login_pass_input)
            if result["success"]:
                st.session_state["logged_in"]  = True
                st.session_state["auth_user"]  = result["user"]
                # Restore saved user settings
                saved = load_user_settings(result["user"]["username"])
                for k, v in saved.items():
                    st.session_state[k] = v
                st.success(f"✅ Welcome back, {result['user']['full_name'] or result['user']['username']}!")
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")

    # ── Register Tab ───────────────────────────────────────────────────
    with auth_tab2:
        with st.form("register_form"):
            reg_name  = st.text_input("👤 Full Name",    placeholder="Your full name")
            reg_uname = st.text_input("💼 Username",      placeholder="Choose a username (min 3 chars)")
            reg_email = st.text_input("📧 Email",         placeholder="your@email.com")
            reg_pass  = st.text_input("🔒 Password",     placeholder="Min 6 characters", type="password")
            reg_pass2 = st.text_input("🔄 Confirm Password", placeholder="Repeat password", type="password")
            reg_btn   = st.form_submit_button("✨ Create Account", use_container_width=True, type="primary")

        if reg_btn:
            if reg_pass != reg_pass2:
                st.error("❌ Passwords do not match.")
            else:
                result = register_user(reg_uname.strip(), reg_pass, reg_name.strip(), reg_email.strip())
                if result["success"]:
                    st.success("✅ Account created! Switch to the Login tab to sign in.")
                else:
                    st.error(f"❌ {result['error']}")

    st.stop()   # Stop rendering the rest of the app until logged in

# ── User is logged in — persist settings on each run ──────────────────────────
_auth_user = st.session_state.get("auth_user", {"username": "guest"})

def _autosave_settings():
    """Save current session alert configs to the user's DB record."""
    settings = {}
    for k in ("email_config", "whatsapp_config", "alert_history"):
        if k in st.session_state:
            settings[k] = st.session_state[k]
    save_user_settings(_auth_user["username"], settings)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* ── App background ── */
.stApp { background: #0a0a1a; color: #f0f0f0; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d20 0%, #090918 100%);
    border-right: 1px solid #2a2a4a;
}

/* ── Sidebar title ── */
.sidebar-title {
    font-size: 22px; font-weight: 800;
    color: #f0f0f0;
    text-align: center; padding: 16px 0 8px;
    letter-spacing: 0.5px;
}
.sidebar-subtitle {
    font-size: 11px; color: #8888aa; text-align: center;
    margin-bottom: 24px; letter-spacing: 1.5px; text-transform: uppercase;
}

/* ── Nav buttons ── */
[data-testid="stSidebar"] .stButton>button {
    width: 100%; background: transparent; border: 1px solid #2a2a4a;
    color: #8888aa; border-radius: 10px; padding: 12px 16px;
    font-size: 14px; font-weight: 500; text-align: left;
    margin-bottom: 6px; transition: all 0.25s ease; cursor: pointer;
}
[data-testid="stSidebar"] .stButton>button:hover {
    background: linear-gradient(135deg, rgba(230,57,70,.15), rgba(255,215,0,.05));
    border-color: #e63946; color: #f0f0f0;
    box-shadow: 0 0 12px rgba(230,57,70,.3);
}

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(135deg, #12122a 0%, #1a1a35 100%);
    border: 1px solid #2a2a4a; border-radius: 16px;
    padding: 24px 20px; text-align: center;
    transition: transform .3s ease, box-shadow .3s ease;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top:0; left:0; right:0; height: 3px;
}
.kpi-red::before   { background: linear-gradient(90deg,#e63946,#ff6b6b); }
.kpi-gold::before  { background: linear-gradient(90deg,#ffd700,#ffaa00); }
.kpi-blue::before  { background: linear-gradient(90deg,#4cc9f0,#4361ee); }
.kpi-green::before { background: linear-gradient(90deg,#06d6a0,#00b4d8); }
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 12px 32px rgba(0,0,0,.5); }
.kpi-value { font-size: 32px; font-weight: 800; margin: 10px 0 4px; }
.kpi-label { font-size: 12px; color: #8888aa; letter-spacing:1.2px; text-transform:uppercase; }
.kpi-icon  { font-size: 28px; }
.kpi-delta { font-size:11px; margin-top:6px; }

/* ── Section headers ── */
.section-header {
    font-size: 26px; font-weight: 700;
    background: linear-gradient(135deg, #f0f0f0, #8888aa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.section-subheader { font-size:13px; color:#8888aa; margin-bottom:28px; }

/* ── Table ── */
.styled-table { border-radius:12px; overflow:hidden; }
[data-testid="stDataFrame"] { border-radius:12px; border:1px solid #2a2a4a; }

/* ── Badges ── */
.badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:11px; font-weight:600; letter-spacing:0.5px;
}
.badge-red   { background:rgba(230,57,70,.2);  color:#e63946; border:1px solid #e63946; }
.badge-green { background:rgba(6,214,160,.2);  color:#06d6a0; border:1px solid #06d6a0; }
.badge-gold  { background:rgba(255,215,0,.2);  color:#ffd700; border:1px solid #ffd700; }

/* ── Divider ── */
hr { border-color: #2a2a4a !important; }

/* ── Progress bars ── */
.stProgress > div > div > div { background: linear-gradient(90deg,#e63946,#ffd700); }

/* ── Metric delta ── */
[data-testid="stMetricDelta"] { font-size:12px; }

/* ── Gauge container ── */
.gauge-container {
    background: linear-gradient(135deg,#12122a,#1a1a35);
    border:1px solid #2a2a4a; border-radius:16px; padding:20px;
    text-align:center;
}

/* ── Alert box ── */
.alert-critical {
    background: rgba(230,57,70,.12); border:1px solid #e63946;
    border-left-width:4px; border-radius:10px; padding:14px 18px;
    color:#f0f0f0; margin-bottom:12px;
}
.alert-safe {
    background: rgba(6,214,160,.1); border:1px solid #06d6a0;
    border-left-width:4px; border-radius:10px; padding:14px 18px;
    color:#f0f0f0; margin-bottom:12px;
}

/* ── Selectbox / Slider labels ── */
label { color: #c8c8e0 !important; font-size:13px !important; }

/* ── Streamlit metric values ── */
[data-testid="stMetricValue"] { color: #f0f0f0 !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #c8c8e0 !important; }
[data-testid="stMetricDelta"] { font-size:12px !important; }

/* ── Streamlit widget text ── */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color: #c8c8e0 !important; }
[data-baseweb="select"] * { color: #f0f0f0 !important; }
[data-testid="stSelectbox"] label { color: #c8c8e0 !important; }

/* ── Expander label ── */
[data-testid="stExpander"] summary p { color: #f0f0f0 !important; font-weight: 600; }

/* ── Info/Success boxes ── */
[data-testid="stAlert"] { color: #f0f0f0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cached data loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_data():
    # If user uploaded a custom dataset, use that
    if "uploaded_df" in st.session_state and st.session_state["uploaded_df"] is not None:
        df = st.session_state["uploaded_df"]
    else:
        df = load_raw_data()
    stats = get_dataset_stats(df)
    return df, stats

@st.cache_resource(show_spinner=False)
def get_preprocessed():
    return preprocess()

@st.cache_resource(show_spinner=False)
def get_models():
    X_train, X_test, y_train, y_test, feature_names, _ = get_preprocessed()
    return train_all(X_train, X_test, y_train, y_test, feature_names)


# ══════════════════════════════════════════════════════════════════════════════
# Plotly helpers
# ══════════════════════════════════════════════════════════════════════════════

PTPL = dict(
    paper_bgcolor="#0a0a1a", plot_bgcolor="#12122a",
    font=dict(color="#f0f0f0", family="Inter"),
    xaxis=dict(
        gridcolor="#2a2a4a", linecolor="#2a2a4a",
        tickfont=dict(color="#c8c8e0", size=12),
        title_font=dict(color="#e0e0f0", size=13),
    ),
    yaxis=dict(
        gridcolor="#2a2a4a", linecolor="#2a2a4a",
        tickfont=dict(color="#c8c8e0", size=12),
        title_font=dict(color="#e0e0f0", size=13),
    ),
)

def apply_dark_theme(fig, title="", height=400):
    fig.update_layout(
        **PTPL,
        title=dict(text=title, font=dict(size=16, color="#f0f0f0")),
        height=height, margin=dict(l=20, r=20, t=48, b=20),
        legend=dict(
            bgcolor="#12122a", bordercolor="#2a2a4a",
            font=dict(color="#f0f0f0", size=13),
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar Navigation
# ══════════════════════════════════════════════════════════════════════════════

PAGES = {
    "🏠  Executive Summary":    "home",
    "🔍  Data Explorer":        "eda",
    "🤖  Model Arena":          "models",
    "⚡  Live Simulator":       "simulator",
    "🧠  AI Explainability":    "shap",
    "🚨  Fraud Alert Center":   "alerts",
    "⚙️  Settings & Alerts":   "settings",
}

with st.sidebar:
    st.markdown('<div class="sidebar-title">🛡️ FraudShield AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Financial Fraud Detection System</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── User profile badge ───────────────────────────────────────────────────
    _u = st.session_state.get("auth_user", {})
    _initials = (_u.get("full_name") or _u.get("username","?"))[:2].upper()
    _last = _u.get("last_login","")[:16].replace("T"," ") if _u.get("last_login") else "First login"
    st.markdown(f"""
    <div style='background:#1a1a35;border:1px solid #2a2a4a;border-radius:12px;
                padding:12px 14px;margin-bottom:12px;display:flex;align-items:center;gap:12px;'>
        <div style='background:linear-gradient(135deg,#e63946,#ffd700);
                    border-radius:50%;width:38px;height:38px;display:flex;
                    align-items:center;justify-content:center;
                    font-weight:800;font-size:14px;color:#000;flex-shrink:0;'>{_initials}</div>
        <div>
            <div style='color:#f0f0f0;font-weight:700;font-size:13px;'>{_u.get("username","Guest")}</div>
            <div style='color:#4a4a6a;font-size:10px;'>Last login: {_last}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
        _autosave_settings()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown("---")

    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    for label, key in PAGES.items():
        if st.button(label, key=f"nav_{key}"):
            st.session_state["page"] = key

    st.markdown("---")

    # ── Dataset Uploader ────────────────────────────────────────────────────
    with st.expander("📂 Upload Financial Dataset", expanded=False):
        st.markdown(
            "<div style='color:#c8c8e0;font-size:12px;margin-bottom:8px;'>"
            "Upload <b>any</b> financial CSV or Excel file — columns are auto-detected.</div>",
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(
            "Choose file", type=["csv", "xlsx", "xls"],
            key="csv_uploader", label_visibility="collapsed"
        )
        if uploaded_file is not None:
            try:
                fname = uploaded_file.name.lower()
                if fname.endswith((".xlsx", ".xls")):
                    upload_df = pd.read_excel(uploaded_file)
                else:
                    upload_df = pd.read_csv(uploaded_file)

                st.markdown(f"<div style='color:#4cc9f0;font-size:11px;'>📄 {uploaded_file.name} — {len(upload_df):,} rows × {upload_df.shape[1]} cols</div>", unsafe_allow_html=True)

                cols = list(upload_df.columns)

                # Auto-detect fraud/label column
                fraud_candidates = [c for c in cols if c.lower() in
                    ("class","fraud","is_fraud","label","isFraud","target",
                     "fraudulent","fraud_label","fraud_flag","Class")]
                default_fraud = fraud_candidates[0] if fraud_candidates else cols[-1]

                # Auto-detect amount column
                amt_candidates = [c for c in cols if "amount" in c.lower() or "amt" in c.lower()]
                default_amt = amt_candidates[0] if amt_candidates else cols[0]

                col_f, col_a = st.columns(2)
                with col_f:
                    fraud_col = st.selectbox("🎯 Fraud Label Column", cols,
                        index=cols.index(default_fraud), key="fraud_col_sel")
                with col_a:
                    amount_col = st.selectbox("💰 Amount Column", cols,
                        index=cols.index(default_amt), key="amount_col_sel")

                if st.button("✅ Load Dataset", key="load_dataset", use_container_width=True):
                    # Standardise column names so rest of app works
                    mapped = upload_df.rename(columns={fraud_col: "Class", amount_col: "Amount"})
                    if "Time" not in mapped.columns:
                        mapped["Time"] = range(len(mapped))
                    # Ensure Class is binary 0/1
                    mapped["Class"] = pd.to_numeric(mapped["Class"], errors="coerce").fillna(0).astype(int)
                    mapped["Class"] = (mapped["Class"] != 0).astype(int)

                    st.session_state["uploaded_df"] = mapped
                    get_data.clear()
                    n_fraud = int(mapped["Class"].sum())
                    n_norm  = len(mapped) - n_fraud
                    st.success(f"✅ Loaded! Fraud: {n_fraud:,} | Normal: {n_norm:,}")
            except Exception as e:
                st.error(f"Error reading file: {e}")

        if st.session_state.get("uploaded_df") is not None:
            udf = st.session_state["uploaded_df"]
            st.markdown(
                f"<div style='color:#06d6a0;font-size:11px;'>"
                f"✅ Active: {len(udf):,} rows · {int(udf['Class'].sum())} fraud</div>",
                unsafe_allow_html=True
            )
            if st.button("🔄 Reset to Default Dataset", key="reset_dataset"):
                st.session_state["uploaded_df"] = None
                get_data.clear()
                st.rerun()

    st.markdown("""
    <div style='color:#4a4a6a;font-size:11px;text-align:center;padding:8px 0;'>
        Dataset: Kaggle Credit Card Fraud<br>
        284,807 Transactions · 492 Fraudulent<br>
        <span style='color:#e63946'>©</span> FraudShield AI v2.0
    </div>""", unsafe_allow_html=True)

page = st.session_state["page"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Executive Summary
# ══════════════════════════════════════════════════════════════════════════════

if page == "home":
    st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Real-time overview of fraud metrics across the entire portfolio</div>', unsafe_allow_html=True)

    with st.spinner("Loading dataset..."):
        df, stats = get_data()

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card kpi-blue">
            <div class="kpi-icon">💳</div>
            <div class="kpi-value" style="color:#4cc9f0">{stats['total']:,}</div>
            <div class="kpi-label">Total Transactions</div>
            <div class="kpi-delta" style="color:#4cc9f0">Full dataset analyzed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card kpi-red">
            <div class="kpi-icon">🚨</div>
            <div class="kpi-value" style="color:#e63946">{stats['fraud']:,}</div>
            <div class="kpi-label">Fraudulent Transactions</div>
            <div class="kpi-delta" style="color:#e63946">{stats['fraud_rate']}% of all transactions</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card kpi-gold">
            <div class="kpi-icon">💰</div>
            <div class="kpi-value" style="color:#ffd700">{format_currency(stats['amount_at_risk'])}</div>
            <div class="kpi-label">Amount at Risk</div>
            <div class="kpi-delta" style="color:#ffd700">Avg fraud: {format_currency(stats['avg_fraud_amount'])}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card kpi-green">
            <div class="kpi-icon">✅</div>
            <div class="kpi-value" style="color:#06d6a0">{stats['normal']:,}</div>
            <div class="kpi-label">Legitimate Transactions</div>
            <div class="kpi-delta" style="color:#06d6a0">Avg: {format_currency(stats['avg_normal_amount'])}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Class Distribution + Amount Box Plot ───────────────────────────────────
    col_left, col_right = st.columns([1, 1.6])

    with col_left:
        labels  = ["Legitimate", "Fraudulent"]
        values  = [stats["normal"], stats["fraud"]]
        colors  = ["#4cc9f0", "#e63946"]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.6, marker_colors=colors,
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Pct: %{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(text=f"<b>{stats['fraud_rate']}%</b><br>Fraud", x=0.5, y=0.5,
            font_size=18, font_color="#e63946", showarrow=False)
        apply_dark_theme(fig_pie, "Transaction Class Distribution", height=360)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig_box = go.Figure()
        fill_map = {"#4cc9f0": "rgba(76,201,240,0.15)", "#e63946": "rgba(230,57,70,0.15)"}
        for cls, color, name in [(0,"#4cc9f0","Legitimate"),(1,"#e63946","Fraudulent")]:
            d = sample[sample["Class"]==cls]["Amount"]
            fig_box.add_trace(go.Box(
                y=d, name=name, marker_color=color, line_color=color,
                boxmean='sd', fillcolor=fill_map[color],
            ))
        apply_dark_theme(fig_box, "Transaction Amount Distribution by Class", height=360)
        fig_box.update_yaxes(title="Amount (USD)")
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Fraud Over Time ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🕐 Fraud Activity Over Time**")
    df_time = df.copy()
    df_time["hour"] = (df_time["Time"] // 3600).astype(int) % 48
    agg = df_time.groupby(["hour","Class"]).size().reset_index(name="count")
    fraud_time  = agg[agg["Class"]==1]
    normal_time = agg[agg["Class"]==0]

    fig_time = make_subplots(specs=[[{"secondary_y": True}]])
    fig_time.add_trace(go.Scatter(
        x=normal_time["hour"], y=normal_time["count"],
        name="Legitimate", fill="tozeroy",
        fillcolor="rgba(76,201,240,0.1)", line_color="#4cc9f0",
    ), secondary_y=False)
    fig_time.add_trace(go.Scatter(
        x=fraud_time["hour"], y=fraud_time["count"],
        name="Fraudulent", fill="tozeroy",
        fillcolor="rgba(230,57,70,0.3)", line_color="#e63946",
        line_width=2,
    ), secondary_y=True)
    fig_time.update_layout(**PTPL, height=280, margin=dict(l=20,r=20,t=20,b=20),
        legend=dict(bgcolor="#12122a", bordercolor="#2a2a4a"),
    )
    fig_time.update_xaxes(title="Hour of Day", gridcolor="#2a2a4a",
        tickfont=dict(color="#c8c8e0", size=12), title_font=dict(color="#e0e0f0"))
    fig_time.update_yaxes(title_text="Legitimate Count", secondary_y=False,
        gridcolor="#2a2a4a", tickfont=dict(color="#c8c8e0"), title_font=dict(color="#4cc9f0"))
    fig_time.update_yaxes(title_text="Fraud Count", secondary_y=True,
        gridcolor="#2a2a4a", tickfont=dict(color="#e63946"), title_font=dict(color="#e63946"))
    st.plotly_chart(fig_time, use_container_width=True)

    # ── Risk Summary Banner ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div class="alert-critical">
        <strong>⚠️ Risk Assessment:</strong> {stats['fraud']:,} fraudulent transactions were detected,
        totalling <strong>{format_currency(stats['amount_at_risk'])}</strong> in potential losses.
        The fraud rate of <strong>{stats['fraud_rate']}%</strong> is consistent with industry benchmarks
        (typical range: 0.1–0.3%). Immediate model deployment is recommended to prevent future losses.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════

elif page == "eda":
    st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Explore the raw dataset, feature distributions, and correlation patterns</div>', unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        df, stats = get_data()

    # ── Dataset Overview ───────────────────────────────────────────────────────
    with st.expander("📋 Dataset Overview", expanded=True):
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Records",   f"{len(df):,}")
        col_b.metric("Features",        f"{df.shape[1]-1}")
        col_c.metric("Missing Values",  f"{df.isnull().sum().sum()}")
        col_d.metric("Memory Usage",    f"{df.memory_usage(deep=True).sum()/1e6:.1f} MB")
        st.dataframe(df.describe().style.format("{:.4f}").background_gradient(
            cmap="RdYlGn", axis=0, subset=df.describe().columns[:10]),
            use_container_width=True)

    st.markdown("---")

    # ── Correlation Heatmap ────────────────────────────────────────────────────
    col1, col2 = st.columns([1.8, 1])
    with col1:
        corr_cols = [c for c in df.columns if c.startswith("V")][:15] + ["Amount","Class"]
        corr = df[corr_cols].corr(numeric_only=True)
        fig_heat = px.imshow(
            corr, color_continuous_scale=[(0,"#4cc9f0"),(0.5,"#0a0a1a"),(1,"#e63946")],
            zmin=-1, zmax=1, aspect="auto",
            labels=dict(color="Correlation"),
        )
        apply_dark_theme(fig_heat, "Feature Correlation Matrix", height=500)
        fig_heat.update_coloraxes(
            colorbar=dict(
                bgcolor="#0a0a1a",
                tickfont=dict(color="#c8c8e0", size=12),
                title_font=dict(color="#e0e0f0"),
            )
        )
        fig_heat.update_xaxes(tickfont=dict(color="#c8c8e0", size=11), tickangle=-45)
        fig_heat.update_yaxes(tickfont=dict(color="#c8c8e0", size=11))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        # Fraud correlation with Class
        corr_class = df.corr(numeric_only=True)["Class"].drop("Class").abs().sort_values(ascending=False).head(12)
        fig_bar = go.Figure(go.Bar(
            x=corr_class.values, y=corr_class.index,
            orientation="h",
            marker=dict(
                color=corr_class.values,
                colorscale=[[0,"#4cc9f0"],[0.5,"#ffd700"],[1,"#e63946"]],
                showscale=False,
            ),
        ))
        apply_dark_theme(fig_bar, "Top Features Correlated with Fraud", height=500)
        fig_bar.update_xaxes(title="Absolute Correlation")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Feature Distribution Viewer ────────────────────────────────────────────
    st.markdown("**📊 Feature Distribution Explorer**")
    feat_options = [c for c in df.columns if c != "Class"]
    selected_feat = st.selectbox("Select a feature to explore:", feat_options, index=feat_options.index("Amount"))

    c_a, c_b = st.columns(2)
    with c_a:
        sample = df.sample(min(10000, len(df)), random_state=42)
        fig_hist = px.histogram(
            sample, x=selected_feat, color="Class",
            color_discrete_map={0:"#4cc9f0", 1:"#e63946"},
            opacity=0.8, nbins=60, barmode="overlay",
            labels={"Class": "Transaction Type", selected_feat: selected_feat},
        )
        apply_dark_theme(fig_hist, f"Distribution of {selected_feat}", height=360)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c_b:
        fill_map2 = {"#4cc9f0": "rgba(76,201,240,0.15)", "#e63946": "rgba(230,57,70,0.15)"}
        fig_violin = go.Figure()
        for cls, color, name in [(0,"#4cc9f0","Legitimate"),(1,"#e63946","Fraudulent")]:
            d = sample[sample["Class"]==cls][selected_feat]
            fig_violin.add_trace(go.Violin(
                y=d, name=name, box_visible=True, meanline_visible=True,
                line_color=color, fillcolor=fill_map2[color],
                spanmode="hard",
            ))
        apply_dark_theme(fig_violin, f"Violin Plot: {selected_feat} by Class", height=360)
        st.plotly_chart(fig_violin, use_container_width=True)

    # ── Amount Analysis ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**💸 Transaction Amount Analysis**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Legitimate Amount", format_currency(df[df["Class"]==0]["Amount"].max()))
    c2.metric("Max Fraud Amount",      format_currency(df[df["Class"]==1]["Amount"].max()))
    c3.metric("Median Fraud Amount",   format_currency(df[df["Class"]==1]["Amount"].median()))

    # Amount buckets
    df["AmountBucket"] = pd.cut(df["Amount"],
        bins=[0,10,50,200,500,2000,df["Amount"].max()+1],
        labels=["<$10","$10-50","$50-200","$200-500","$500-2K",">$2K"])
    bucket_fraud = df[df["Class"]==1].groupby("AmountBucket", observed=True).size().reset_index(name="fraud_count")
    bucket_all   = df.groupby("AmountBucket", observed=True).size().reset_index(name="total_count")
    bucket = bucket_fraud.merge(bucket_all, on="AmountBucket")
    bucket["fraud_pct"] = (bucket["fraud_count"] / bucket["total_count"] * 100).round(2)

    fig_bucket = go.Figure()
    fig_bucket.add_trace(go.Bar(x=bucket["AmountBucket"].astype(str),
        y=bucket["fraud_pct"], name="Fraud Rate %",
        marker_color="#e63946", marker_opacity=0.85))
    fig_bucket.add_trace(go.Bar(x=bucket["AmountBucket"].astype(str),
        y=bucket["fraud_count"], name="Fraud Count",
        marker_color="#ffd700", marker_opacity=0.7, yaxis="y2"))
    fig_bucket.update_layout(
        paper_bgcolor="#0a0a1a", plot_bgcolor="#12122a",
        font=dict(color="#f0f0f0", family="Inter"),
        height=330, barmode="overlay",
        yaxis2=dict(overlaying="y", side="right", gridcolor="#2a2a4a",
            tickfont=dict(color="#ffd700"), title_font=dict(color="#ffd700")),
        margin=dict(l=20,r=20,t=40,b=20), title="Fraud Distribution by Transaction Amount Bucket",
        title_font=dict(size=16, color="#f0f0f0"),
        legend=dict(bgcolor="#12122a", bordercolor="#2a2a4a", font=dict(color="#f0f0f0")),
        xaxis=dict(gridcolor="#2a2a4a", linecolor="#2a2a4a",
            tickfont=dict(color="#c8c8e0", size=12), title_font=dict(color="#e0e0f0")),
        yaxis=dict(gridcolor="#2a2a4a", linecolor="#2a2a4a",
            tickfont=dict(color="#c8c8e0", size=12), title_font=dict(color="#e0e0f0")),
    )
    st.plotly_chart(fig_bucket, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Arena
# ══════════════════════════════════════════════════════════════════════════════

elif page == "models":
    st.markdown('<div class="section-header">Model Arena</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Train, compare, and evaluate 4 machine learning models side-by-side</div>', unsafe_allow_html=True)

    train_btn = st.button("🚀 Train All Models", type="primary")

    if train_btn:
        st.session_state["models_trained"] = True

    if st.session_state.get("models_trained", False) or not st.session_state.get("models_trained") is False:
        with st.spinner("⚙️ Preprocessing data (SMOTE + scaling) — this may take 1–2 minutes on first run..."):
            X_train, X_test, y_train, y_test, feature_names, _ = get_preprocessed()

        with st.spinner("🤖 Training models and computing metrics..."):
            models, metrics_df, curves = get_models()

        st.success("✅ All models trained and evaluated!")
        st.markdown("---")

        # ── Metrics Table ──────────────────────────────────────────────────────
        st.markdown("### 📊 Performance Metrics Comparison")

        display_cols = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC","PR-AUC"]
        styled = metrics_df[display_cols].style\
            .background_gradient(cmap="RdYlGn", axis=0)\
            .format("{:.2f}%")\
            .set_properties(**{"text-align":"center"})\
            .set_table_attributes('class="styled-table"')

        st.dataframe(metrics_df[display_cols].style.format("{:.2f}%")
            .highlight_max(axis=0, props="background-color:rgba(6,214,160,.25);color:#06d6a0;font-weight:700")
            .highlight_min(axis=0, props="background-color:rgba(230,57,70,.1);color:#e63946;"),
            use_container_width=True)

        # Best model badge
        best_model = metrics_df["F1 Score"].idxmax()
        st.markdown(f"""
        <div style="text-align:center;margin:12px 0;">
            <span class="badge badge-gold">🏆 Best Model: {best_model} &nbsp;|&nbsp;
            F1: {metrics_df.loc[best_model,'F1 Score']:.2f}% &nbsp;|&nbsp;
            ROC-AUC: {metrics_df.loc[best_model,'ROC-AUC']:.2f}%</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── ROC + PR Curves ────────────────────────────────────────────────────
        col_roc, col_pr = st.columns(2)
        model_colors = {"Logistic Regression":"#4cc9f0","Random Forest":"#06d6a0",
                        "XGBoost":"#ffd700","Isolation Forest":"#a78bfa"}

        with col_roc:
            fig_roc = go.Figure()
            fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                line=dict(dash="dash", color="#4a4a6a", width=1))
            for name, data in curves.items():
                auc = metrics_df.loc[name,"ROC-AUC"]
                fig_roc.add_trace(go.Scatter(
                    x=data["fpr"], y=data["tpr"], name=f"{name} ({auc:.1f}%)",
                    line=dict(color=model_colors[name], width=2.5),
                    mode="lines",
                ))
            apply_dark_theme(fig_roc, "ROC Curves — All Models", height=420)
            fig_roc.update_xaxes(title="False Positive Rate", range=[0,1])
            fig_roc.update_yaxes(title="True Positive Rate",  range=[0,1])
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_pr:
            fig_pr = go.Figure()
            for name, data in curves.items():
                pr_auc = metrics_df.loc[name,"PR-AUC"]
                fig_pr.add_trace(go.Scatter(
                    x=data["rec"], y=data["prec"], name=f"{name} ({pr_auc:.1f}%)",
                    line=dict(color=model_colors[name], width=2.5),
                    mode="lines",
                ))
            apply_dark_theme(fig_pr, "Precision-Recall Curves — All Models", height=420)
            fig_pr.update_xaxes(title="Recall", range=[0,1])
            fig_pr.update_yaxes(title="Precision", range=[0,1])
            st.plotly_chart(fig_pr, use_container_width=True)

        st.markdown("---")

        # ── Confusion Matrices ─────────────────────────────────────────────────
        st.markdown("### 🎯 Confusion Matrices")
        cols_cm = st.columns(4)
        for idx, (name, data) in enumerate(curves.items()):
            cm = data["cm"]
            with cols_cm[idx]:
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual"),
                    x=["Normal","Fraud"], y=["Normal","Fraud"],
                    text_auto=True,
                    color_continuous_scale=[(0,"#12122a"),(0.5,"#4361ee"),(1,"#e63946")],
                )
                apply_dark_theme(fig_cm, name, height=260)
                fig_cm.update_traces(textfont_size=14, textfont_color="#ffffff")
                fig_cm.update_coloraxes(showscale=False)
                fig_cm.update_xaxes(tickfont=dict(color="#c8c8e0", size=12), title_font=dict(color="#e0e0f0"))
                fig_cm.update_yaxes(tickfont=dict(color="#c8c8e0", size=12), title_font=dict(color="#e0e0f0"))
                st.plotly_chart(fig_cm, use_container_width=True)

                tp, fp = int(cm[1,1]), int(cm[0,1])
                tn, fn = int(cm[0,0]), int(cm[1,0])
                st.markdown(f"""
                <div style='font-size:11px;color:#8888aa;text-align:center;'>
                    TP: <span style='color:#06d6a0'>{tp:,}</span> &nbsp;
                    FP: <span style='color:#e63946'>{fp:,}</span> &nbsp;
                    TN: <span style='color:#4cc9f0'>{tn:,}</span> &nbsp;
                    FN: <span style='color:#ffd700'>{fn:,}</span>
                </div>""", unsafe_allow_html=True)

        # ── Radar Chart ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🕸️ Model Performance Radar")
        radar_metrics = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC","PR-AUC"]
        fig_radar = go.Figure()
        radar_fill_map = {
            "Logistic Regression": "rgba(76,201,240,0.10)",
            "Random Forest":       "rgba(6,214,160,0.10)",
            "XGBoost":             "rgba(255,215,0,0.10)",
            "Isolation Forest":    "rgba(167,139,250,0.10)",
        }
        for name in metrics_df.index:
            vals = metrics_df.loc[name, radar_metrics].tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=radar_metrics + [radar_metrics[0]],
                fill="toself", name=name, line_color=model_colors[name],
                fillcolor=radar_fill_map[name],
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#12122a",
                radialaxis=dict(visible=True, range=[60,102], gridcolor="#2a2a4a", tickfont_color="#8888aa"),
                angularaxis=dict(gridcolor="#2a2a4a", tickfont_color="#f0f0f0"),
            ),
            paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a", height=480,
            font=dict(color="#f0f0f0", family="Inter"),
            title=dict(text="Model Comparison Radar", font=dict(size=16, color="#f0f0f0")),
            legend=dict(bgcolor="#12122a", bordercolor="#2a2a4a"),
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    else:
        st.info("👆 Click **Train All Models** to begin training and evaluation.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Live Simulator
# ══════════════════════════════════════════════════════════════════════════════

elif page == "simulator":
    st.markdown('<div class="section-header">Live Transaction Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Simulate any transaction and get real-time fraud probability predictions from all models</div>', unsafe_allow_html=True)

    with st.spinner("Loading models..."):
        X_train, X_test, y_train, y_test, feature_names, _ = get_preprocessed()
        models, metrics_df, curves = get_models()

    # ── Reference transaction stats for slider bounds ──────────────────────────
    df_raw, _ = get_data()

    # ── Quick-fill presets ─────────────────────────────────────────────────────
    col_pre, _ = st.columns([2, 3])
    with col_pre:
        preset = st.selectbox("🎲 Load Preset Transaction:",
            ["Custom", "Typical Legitimate", "Suspicious High-Value", "Classic Fraud Pattern"])

    PRESETS = {
        "Typical Legitimate":    {"Amount": 50.0,   "V14": 0.3,  "V17": 0.2,  "V12": 0.1, "V10": 0.2, "V4": 0.3, "V11": 0.5, "V3": 0.2, "V7": 0.1},
        "Suspicious High-Value": {"Amount": 5000.0, "V14": -1.2, "V17": -0.8, "V12": -0.6,"V10":-0.5,"V4": 0.8, "V11": -0.3,"V3": -0.4,"V7": -0.3},
        "Classic Fraud Pattern": {"Amount": 1.0,    "V14": -8.5, "V17": -5.4, "V12": -6.2,"V10":-5.8,"V4": 3.2, "V11": -4.1,"V3": -5.2,"V7": -4.8},
        "Custom": None,
    }

    st.markdown("---")
    _, center, _ = st.columns([0.05, 0.9, 0.05])

    with center:
        st.markdown("### ⚙️ Configure Transaction Features")
        col_sliders_a, col_sliders_b = st.columns(2)

        input_vals = {}
        preset_vals = PRESETS.get(preset) or {}

        key_feat_a = KEY_FEATURES[:5]
        key_feat_b = KEY_FEATURES[5:]

        with col_sliders_a:
            for feat in key_feat_a:
                mn = float(df_raw[feat].min()) if feat in df_raw else -10.0
                mx = float(df_raw[feat].max()) if feat in df_raw else 10.0
                default = float(preset_vals.get(feat, 0.0))
                input_vals[feat] = st.slider(f"{feat}", min_value=mn, max_value=mx, value=default, step=0.01, key=f"sldr_{feat}")
        with col_sliders_b:
            for feat in key_feat_b:
                mn = float(df_raw[feat].min()) if feat in df_raw else -10.0
                mx = float(df_raw[feat].max()) if feat in df_raw else 10000.0
                default = float(preset_vals.get(feat, 0.0))
                input_vals[feat] = st.slider(f"{feat}", min_value=mn, max_value=mx, value=default, step=0.01 if feat != "Amount" else 1.0, key=f"sldr_{feat}")

        # Fill remaining features with 0
        full_input = {f: 0.0 for f in feature_names}
        full_input.update(input_vals)
        # Handle scaled_Amount and scaled_Time
        full_input["scaled_Amount"] = (full_input.get("Amount", 0) - 88.35) / 250.12
        full_input["scaled_Time"]   = 0.0
        if "Amount" in full_input: del full_input["Amount"]
        if "Time"   in full_input: del full_input["Time"]

        input_array = np.array([full_input.get(f, 0.0) for f in feature_names]).reshape(1, -1)

        predict_btn = st.button("⚡ Predict Fraud Probability", type="primary", use_container_width=True)

    if predict_btn:
        probs = predict_transaction(models, feature_names, input_array)
        avg_prob = np.mean(list(probs.values()))
        risk = risk_score(avg_prob)

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        # ── Gauge ──────────────────────────────────────────────────────────────
        col_gauge, col_breakdown = st.columns([1, 1.5])
        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_prob * 100,
                number={"suffix": "%", "font": {"size": 52, "color": risk["color"]}},
                delta={"reference": 50, "increasing": {"color": "#e63946"}, "decreasing": {"color": "#06d6a0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8888aa", "tickfont": {"color": "#8888aa"}},
                    "bar": {"color": risk["color"], "thickness": 0.25},
                    "bgcolor": "#12122a",
                    "steps": [
                        {"range": [0, 25],  "color": "rgba(6,214,160,.15)"},
                        {"range": [25, 50], "color": "rgba(255,215,0,.1)"},
                        {"range": [50, 75], "color": "rgba(255,107,53,.15)"},
                        {"range": [75, 100],"color": "rgba(230,57,70,.2)"},
                    ],
                    "threshold": {"line": {"color": risk["color"], "width": 3}, "value": avg_prob * 100},
                },
                title={"text": f"Fraud Risk Score<br><span style='font-size:18px'>{risk['icon']} {risk['level']}</span>",
                       "font": {"size": 18, "color": "#f0f0f0"}},
            ))
            fig_gauge.update_layout(paper_bgcolor="#0a0a1a", height=320, margin=dict(l=30,r=30,t=60,b=20),
                font=dict(color="#f0f0f0", family="Inter"))
            st.plotly_chart(fig_gauge, use_container_width=True)

            if avg_prob >= 0.5:
                st.markdown(f'<div class="alert-critical">🚨 <strong>HIGH FRAUD RISK</strong> — This transaction should be flagged for review. Avg probability: <strong>{avg_prob:.1%}</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-safe">✅ <strong>TRANSACTION APPEARS LEGITIMATE</strong> — Avg probability: <strong>{avg_prob:.1%}</strong></div>', unsafe_allow_html=True)

        with col_breakdown:
            # Per-model bars
            fig_models = go.Figure()
            names_sorted = sorted(probs, key=probs.get, reverse=True)
            colors_bar = ["#e63946" if p >= 0.5 else "#06d6a0" for p in [probs[n] for n in names_sorted]]
            fig_models.add_trace(go.Bar(
                x=[probs[n]*100 for n in names_sorted], y=names_sorted,
                orientation="h", marker_color=colors_bar, marker_opacity=0.85,
                text=[f"{probs[n]:.1%}" for n in names_sorted], textposition="outside",
                textfont_color="#f0f0f0",
            ))
            apply_dark_theme(fig_models, "Fraud Probability by Model", height=320)
            fig_models.update_xaxes(title="Fraud Probability (%)", range=[0, 110])
            fig_models.add_vline(x=50, line_dash="dash", line_color="#ffd700",
                annotation_text="Decision Threshold (50%)",
                annotation_font_color="#ffd700", annotation_font_size=11)
            st.plotly_chart(fig_models, use_container_width=True)

        # ── SHAP Single Transaction ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🧠 Why did the model predict this? (SHAP Explanation)")
        try:
            best_tree_model_name = "XGBoost" if "XGBoost" in models else "Random Forest"
            shap_data = compute_shap_values(models[best_tree_model_name], X_test, best_tree_model_name, n_samples=300)
            explainer = shap_data["explainer"]
            shap_df = shap_single_transaction(explainer, input_array, feature_names)

            fig_shap = go.Figure(go.Bar(
                x=shap_df["SHAP Value"], y=shap_df["Feature"], orientation="h",
                marker_color=["#e63946" if v > 0 else "#06d6a0" for v in shap_df["SHAP Value"]],
                text=shap_df["Direction"], textposition="outside", textfont_color="#8888aa",
            ))
            apply_dark_theme(fig_shap, f"Feature Contributions — {best_tree_model_name}", height=380)
            fig_shap.update_xaxes(title="SHAP Value (positive = increases fraud risk)")
            fig_shap.add_vline(x=0, line_color="#8888aa", line_width=1)
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — AI Explainability
# ══════════════════════════════════════════════════════════════════════════════

elif page == "shap":
    st.markdown('<div class="section-header">AI Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Understand what drives every prediction — built with SHAP (SHapley Additive exPlanations)</div>', unsafe_allow_html=True)

    with st.spinner("Loading models and computing SHAP values (first run may take ~1 min)..."):
        X_train, X_test, y_train, y_test, feature_names, _ = get_preprocessed()
        models, metrics_df, curves = get_models()

    model_choice = st.selectbox("🤖 Select Model for SHAP Analysis:", ["XGBoost", "Random Forest"])
    selected_model = models[model_choice]

    with st.spinner(f"Computing SHAP values for {model_choice}..."):
        shap_data = compute_shap_values(selected_model, X_test, model_choice, n_samples=500)

    st.markdown("---")

    # ── SHAP Summary (beeswarm) ────────────────────────────────────────────────
    st.markdown("### 🐝 SHAP Beeswarm Plot — Feature Impact Distribution")
    st.markdown("""
    <div style='color:#8888aa;font-size:13px;margin-bottom:16px;'>
    Each dot is one transaction. <span style='color:#e63946'>Red</span> = high feature value,
    <span style='color:#4cc9f0'>Blue</span> = low feature value.
    Dots to the right increase fraud risk; dots to the left decrease it.
    </div>""", unsafe_allow_html=True)

    fig_beeswarm = shap_summary_fig(shap_data, feature_names)
    st.pyplot(fig_beeswarm, use_container_width=True)

    st.markdown("---")

    # ── SHAP Bar Chart ─────────────────────────────────────────────────────────
    col_bar, col_insight = st.columns([1.4, 1])
    with col_bar:
        st.markdown("### 📊 Mean Absolute SHAP Values")
        fig_shapbar = shap_bar_fig(shap_data, feature_names)
        st.pyplot(fig_shapbar, use_container_width=True)

    with col_insight:
        st.markdown("### 💡 Key Insights")
        shap_vals = np.abs(shap_data["shap_values"]).mean(axis=0)
        top5 = pd.Series(shap_vals, index=feature_names).nlargest(5)
        for i, (feat, val) in enumerate(top5.items(), 1):
            rank_color = ["#ffd700","#c0c0c0","#cd7f32","#8888aa","#8888aa"][i-1]
            st.markdown(f"""
            <div style='background:#12122a;border:1px solid #2a2a4a;border-radius:10px;
                        padding:14px;margin-bottom:10px;border-left:3px solid {rank_color};'>
                <div style='color:{rank_color};font-weight:700;font-size:13px;'>#{i} {feat}</div>
                <div style='color:#8888aa;font-size:12px;margin-top:4px;'>
                    Mean |SHAP|: <span style='color:#f0f0f0'>{val:.4f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='alert-critical' style='margin-top:16px;'>
            <strong>⚠️ Model Analysis:</strong><br>
            <strong>{top5.index[0]}</strong> is the most influential feature for {model_choice}.
            Transactions with extreme values in this feature are most likely to be flagged as fraudulent.
        </div>""", unsafe_allow_html=True)

    # ── Feature interaction note ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    > **ℹ️ Note on SHAP Values**: The V1–V28 features in this dataset are PCA-transformed for privacy. 
    Despite being anonymized, the SHAP analysis reveals which transformed dimensions the model relies on most heavily — 
    providing full transparency into the model's decision-making process. In a real deployment, 
    these would correspond to features like merchant category, transaction frequency, and geographic anomaly scores.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Fraud Alert Center
# ══════════════════════════════════════════════════════════════════════════════

elif page == "alerts":
    st.markdown('<div class="section-header">Fraud Alert Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Review, filter, and investigate all flagged suspicious transactions</div>', unsafe_allow_html=True)

    with st.spinner("Loading data and models..."):
        df, stats = get_data()
        X_train, X_test, y_train, y_test, feature_names, _ = get_preprocessed()
        models, metrics_df, curves = get_models()

    # Use XGBoost scores on test set
    best_model = models["XGBoost"]
    y_scores = curves["XGBoost"]["y_scores"]
    y_pred   = curves["XGBoost"]["y_pred"]

    # Build alert dataframe from test set
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["True_Label"]  = y_test.values
    test_df["Fraud_Score"] = y_scores
    test_df["Predicted"]   = y_pred
    test_df["scaled_Amount_display"] = test_df.get("scaled_Amount", 0) * 250.12 + 88.35

    flagged = test_df[test_df["Fraud_Score"] >= 0.3].copy()
    flagged["Risk Level"] = flagged["Fraud_Score"].apply(
        lambda p: risk_score(p)["level"])
    flagged["Fraud Score %"] = (flagged["Fraud_Score"] * 100).round(2)
    flagged["Amount (Est.)"] = flagged["scaled_Amount_display"].apply(format_currency)
    flagged["Actual"]        = flagged["True_Label"].map({1:"🔴 FRAUD", 0:"✅ Normal"})
    flagged["Prediction"]    = flagged["Predicted"].map({1:"🔴 FRAUD", 0:"✅ Normal"})
    flagged = flagged.sort_values("Fraud_Score", ascending=False).reset_index(drop=True)

    # ── Summary KPIs ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚨 Total Alerts",    f"{len(flagged):,}")
    c2.metric("✅ True Positives",  f"{int((flagged['True_Label']==1).sum()):,}")
    c3.metric("⚠️ False Positives",f"{int((flagged['True_Label']==0).sum()):,}")
    c4.metric("💯 Precision",       f"{int((flagged['True_Label']==1).sum())/max(len(flagged),1)*100:.1f}%")
    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        risk_filter = st.multiselect("Filter by Risk Level:",
            ["CRITICAL","HIGH","MEDIUM"],
            default=["CRITICAL","HIGH","MEDIUM"])
    with col_f2:
        label_filter = st.selectbox("Filter by Actual Label:", ["All", "Fraud Only", "Normal Only"])
    with col_f3:
        min_score = st.slider("Minimum Fraud Score:", 0, 100, 30, 5)

    filtered = flagged[
        (flagged["Risk Level"].isin(risk_filter)) &
        (flagged["Fraud Score %"] >= min_score)
    ]
    if label_filter == "Fraud Only":
        filtered = filtered[filtered["True_Label"] == 1]
    elif label_filter == "Normal Only":
        filtered = filtered[filtered["True_Label"] == 0]

    # ── Styled Alerts Table ────────────────────────────────────────────────────
    st.markdown(f"**🔍 Showing {len(filtered):,} alerts** matching your filters")

    display_cols = ["Fraud Score %","Risk Level","Amount (Est.)","Actual","Prediction"]
    display_df = filtered[display_cols].head(200).copy()

    def color_risk(val):
        colors_map = {"CRITICAL":"color:#e63946;font-weight:700",
                      "HIGH":    "color:#ff6b35;font-weight:700",
                      "MEDIUM":  "color:#ffd700;font-weight:600"}
        return colors_map.get(val, "")

    st.dataframe(display_df.style.applymap(color_risk, subset=["Risk Level"])
        .background_gradient(subset=["Fraud Score %"], cmap="RdYlGn_r", vmin=0, vmax=100)
        .format({"Fraud Score %": "{:.1f}%"}),
        use_container_width=True, height=420)

    # ── Download + Send Alert ────────────────────────────────────────────────────
    col_dl, col_alert = st.columns([1, 1])
    with col_dl:
        csv_export = filtered[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Fraud Alert Report (CSV)",
            data=csv_export,
            file_name="fraud_alerts_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_alert:
        if st.button("📧 Send Batch Email Alert", type="primary", use_container_width=True):
            cfg = st.session_state.get("email_config", {})
            if not cfg.get("sender_email"):
                st.warning("⚠️ Configure email credentials in ⚙️ Settings & Alerts first.")
            else:
                n_critical = int((filtered["Risk Level"]=="CRITICAL").sum())
                n_high     = int((filtered["Risk Level"]=="HIGH").sum())
                amt = format_currency(filtered["scaled_Amount_display"].sum() if "scaled_Amount_display" in filtered.columns else 0)
                html_body  = build_alert_summary(n_critical, n_high, len(filtered), amt)
                result = send_email_alert(
                    smtp_server=cfg.get("smtp_server","smtp.gmail.com"),
                    smtp_port=int(cfg.get("smtp_port", 465)),
                    sender_email=cfg["sender_email"],
                    sender_password=cfg["sender_password"],
                    recipient_email=cfg["recipient_email"],
                    subject=f"🚨 FraudShield AI — {len(filtered)} Fraud Alerts Detected",
                    body_html=html_body,
                )
                if result["success"]:
                    st.success("✅ Batch alert email sent successfully!")
                else:
                    st.error(f"❌ Email failed: {result['error']}")

    st.markdown("---")

    # ── Score Distribution of Flagged ──────────────────────────────────────────
    col_dist, col_donut = st.columns([1.6, 1])
    with col_dist:
        fig_hist = px.histogram(
            flagged, x="Fraud Score %", color="Actual",
            color_discrete_map={"🔴 FRAUD":"#e63946","✅ Normal":"#4cc9f0"},
            nbins=40, opacity=0.8, barmode="overlay",
        )
        apply_dark_theme(fig_hist, "Fraud Score Distribution of Flagged Transactions", height=340)
        fig_hist.update_xaxes(title="Fraud Score (%)")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_donut:
        risk_counts = flagged["Risk Level"].value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=risk_counts.index, values=risk_counts.values,
            hole=0.55,
            marker_colors=["#e63946","#ff6b35","#ffd700"],
        ))
        apply_dark_theme(fig_donut, "Alert Risk Level Breakdown", height=340)
        st.plotly_chart(fig_donut, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Settings & Alerts
# ══════════════════════════════════════════════════════════════════════════════

elif page == "settings":
    st.markdown('<div class="section-header">⚙️ Settings & Alerts</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Configure Email notifications for fraud alerts</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📧 Email Alerts", "🧪 Test & History"])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1 — Email
    # ────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div style='background:#12122a;border:1px solid #2a2a4a;border-radius:12px;
                    padding:18px 22px;margin-bottom:20px;border-left:4px solid #4cc9f0;'>
            <div style='color:#4cc9f0;font-weight:700;font-size:14px;margin-bottom:8px;'>💡 Gmail Setup Guide</div>
            <ol style='color:#c8c8e0;font-size:13px;margin:0;padding-left:18px;line-height:1.8;'>
                <li>Go to your Google Account → Security → <b>2-Step Verification</b> (enable it)</li>
                <li>Then go to <b>App Passwords</b> → Select App: Mail → Generate</li>
                <li>Copy the 16-character App Password and paste it below</li>
            </ol>
            <div style='color:#ffd700;font-size:12px;margin-top:10px;'>
                ⚠️ Never use your main Gmail password. Use an App Password only.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            smtp_server    = st.text_input("🌐 SMTP Server",        value="smtp.gmail.com",  key="smtp_server")
            sender_email   = st.text_input("📤 Sender Email",       value="",               key="sender_email",   placeholder="your@gmail.com")
            recipient_email= st.text_input("📥 Recipient Email",    value="",               key="recipient_email", placeholder="alert@example.com")
        with col_e2:
            smtp_port      = st.number_input("🔒 SMTP Port",          value=465,             key="smtp_port",      step=1)
            sender_password= st.text_input("🔑 App Password",        value="",               key="sender_password", type="password", placeholder="xxxx xxxx xxxx xxxx")
            alert_threshold= st.slider("⚠️ Alert Threshold (%)",   min_value=30, max_value=95, value=75, step=5,
                                       key="alert_threshold",
                                       help="Only send alerts for fraud scores above this threshold")

        if st.button("💾 Save Email Configuration", type="primary", key="save_email"):
            st.session_state["email_config"] = {
                "smtp_server":     smtp_server,
                "smtp_port":       smtp_port,
                "sender_email":    sender_email,
                "sender_password": sender_password,
                "recipient_email": recipient_email,
                "threshold":       alert_threshold,
            }
            _autosave_settings()
            st.success("✅ Email configuration saved to your account!")

        # Status badge
        if st.session_state.get("email_config", {}).get("sender_email"):
            cfg = st.session_state["email_config"]
            st.markdown(f"""
            <div style='background:rgba(6,214,160,.1);border:1px solid #06d6a0;border-radius:10px;
                        padding:12px 16px;margin-top:12px;'>
                <span style='color:#06d6a0;font-weight:700;'>✅ Email Configured</span>
                <span style='color:#8888aa;font-size:12px;margin-left:12px;'>
                    {cfg['sender_email']} → {cfg['recipient_email']} · Threshold: {cfg['threshold']}%
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:rgba(230,57,70,.08);border:1px solid #e63946;border-radius:10px;
                        padding:12px 16px;margin-top:12px;'>
                <span style='color:#e63946;font-weight:700;'>❌ Not Configured</span>
                <span style='color:#8888aa;font-size:12px;margin-left:12px;'>Enter credentials above and click Save</span>
            </div>""", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2 — Test & History
    # ────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 🧪 Send Test Alert")

        # Test Email (full width)
        st.markdown("""
        <div style='background:#12122a;border:1px solid #2a2a4a;border-radius:12px;padding:18px;max-width:480px;'>
            <div style='color:#4cc9f0;font-size:14px;font-weight:700;margin-bottom:12px;'>📧 Test Email Alert</div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Send Test Email", use_container_width=True, key="test_email"):
                cfg = st.session_state.get("email_config", {})
                if not cfg.get("sender_email"):
                    st.error("❌ Save your email config in the Email tab first.")
                else:
                    sample_data = {"Transaction": "TEST-001", "Amount": "$250.00", "Risk": "CRITICAL", "Score": "92%"}
                    html_body   = build_fraud_email(sample_data, 0.92, "CRITICAL")
                    result = send_email_alert(
                        smtp_server=cfg["smtp_server"], smtp_port=int(cfg["smtp_port"]),
                        sender_email=cfg["sender_email"], sender_password=cfg["sender_password"],
                        recipient_email=cfg["recipient_email"],
                        subject="🧪 FraudShield AI — Test Alert (Do Not Action)",
                        body_html=html_body,
                    )
                    if result["success"]:
                        st.success(f"✅ Test email sent to {cfg['recipient_email']}!")
                        if "alert_history" not in st.session_state:
                            st.session_state["alert_history"] = []
                        st.session_state["alert_history"].append({"Type":"Email","Status":"✅ Sent","Target":cfg['recipient_email'],"Score":"TEST"})
                    else:
                        st.error(f"❌ {result['error']}")
        st.markdown("</div>", unsafe_allow_html=True)


        # Alert History
        st.markdown("---")
        st.markdown("### 📖 Alert History (this session)")
        history = st.session_state.get("alert_history", [])
        if history:
            hist_df = pd.DataFrame(history[::-1])  # newest first
            st.dataframe(hist_df, use_container_width=True, height=250)
        else:
            st.markdown("<div style='color:#8888aa;text-align:center;padding:24px;'>No alerts sent in this session yet.</div>", unsafe_allow_html=True)
