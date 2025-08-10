# AI Threat Detection — Live Demo (Streamlit)
# Author: CyberDudeBivash (Cybersecurity & AI Agents Masterclass)
# Run: streamlit run app.py

import time
import random
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import streamlit as st

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="AI Threat Detection — Live Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Threat Detection — Live Demo")
st.caption("Presented by **CyberDudeBivash** · LinkedIn Live Masterclass")

# -----------------------------
# Utility: Synthetic Event Generator
# -----------------------------
USERS = ["alice", "bob", "charlie", "diana", "eve", "frank", "grace", "heidi"]
CITIES = [
    ("Delhi", 28.6139, 77.2090),
    ("Mumbai", 19.0760, 72.8777),
    ("Bengaluru", 12.9716, 77.5946),
    ("Hyderabad", 17.3850, 78.4867),
    ("Kolkata", 22.5726, 88.3639),
    ("London", 51.5074, -0.1278),
    ("New York", 40.7128, -74.0060),
    ("Singapore", 1.3521, 103.8198),
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "curl/8.4.0",
    "python-requests/2.32",
]

MFA_METHODS = ["totp", "push", "sms", "webauthn", "none"]
SERVICES = ["vpn", "okta", "github", "salesforce", "s3", "gdrive", "jira"]

LOCAL_BREACHED_USERS = {"eve", "frank"}  # pretend these are known leaked credentials

def ipify(n):
    return ".".join(str(int(x)) for x in np.clip(np.random.normal(100, 50, n), 1, 254))

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.pi/180
    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p)*np.cos(lat2*p)*(1 - np.cos((lon2-lon1)*p))/2
    return 2*R*np.arcsin(np.sqrt(a))

def ua_entropy(ua: str) -> float:
    # quick proxy for randomness/complexity of UA string
    counts = {}
    for ch in ua:
        counts[ch] = counts.get(ch, 0) + 1
    probs = np.array(list(counts.values()), dtype=float)
    probs = probs / probs.sum()
    return float(-(probs * np.log2(probs)).sum())

def generate_event(now: datetime):
    src_city = random.choice(CITIES)
    dst_city = random.choice(CITIES)
    user = random.choice(USERS)
    service = random.choice(SERVICES)

    # base benign distributions
    bytes_sent = max(50, int(abs(np.random.normal(1500, 1200))))
    duration = max(1, abs(np.random.normal(5, 4)))  # seconds
    failed_logins = max(0, int(abs(np.random.normal(0.2, 0.7)) + 0.1) - 1)
    ua = random.choice(USER_AGENTS)
    mfa = random.choice(MFA_METHODS)
    geo_km = haversine_km(src_city[1], src_city[2], dst_city[1], dst_city[2])

    # inject anomalies sometimes
    anomaly_flag = 0
    if random.random() < 0.06:
        anomaly_flag = 1
        choice = random.choice(["exfil", "bruteforce", "impossible_travel", "mfa_bypass"])
        if choice == "exfil":
            bytes_sent = int(abs(np.random.normal(2.5e7, 5e6)))  # ~25MB burst
        elif choice == "bruteforce":
            failed_logins = int(abs(np.random.normal(25, 5)))
        elif choice == "impossible_travel":
            geo_km = int(abs(np.random.normal(8000, 500)))  # very far
        elif choice == "mfa_bypass":
            mfa = "none"
            ua = "python-requests/2.31 (bot)"

    event = {
        "timestamp": now.isoformat(timespec="seconds"),
        "user": user,
        "service": service,
        "src_ip": ipify(1)[0],
        "dst_ip": ipify(1)[0],
        "src_city": src_city[0],
        "dst_city": dst_city[0],
        "bytes": bytes_sent,
        "duration_s": duration,
        "failed_logins": failed_logins,
        "ua": ua,
        "mfa": mfa,
        "geo_km": geo_km,
        "anomaly_injected": anomaly_flag,
    }
    return event

# -----------------------------
# Feature Engineering
# -----------------------------
NUMERIC_FEATURES = ["bytes", "duration_s", "failed_logins", "geo_km", "ua_entropy", "mfa_none_flag", "breached_user_flag"]

def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ua_entropy"] = df["ua"].apply(ua_entropy)
    df["mfa_none_flag"] = (df["mfa"] == "none").astype(int)
    df["breached_user_flag"] = df["user"].isin(LOCAL_BREACHED_USERS).astype(int)
    return df

# -----------------------------
# Simple Rule Engine (complements ML)
# -----------------------------
def rule_scores(row):
    score = 0
    reasons = []
    if row["failed_logins"] >= 10:
        score += 0.35
        reasons.append("High failed logins")
    if row["geo_km"] >= 5000:
        score += 0.35
        reasons.append("Impossible travel")
    if row["mfa"] == "none":
        score += 0.25
        reasons.append("MFA bypass")
    if row["bytes"] >= 10_000_000:
        score += 0.25
        reasons.append("Large data egress")
    if row.get("breached_user_flag", 0) == 1:
        score += 0.2
        reasons.append("Known leaked user")
    return min(score, 1.0), reasons

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("Controls")
    event_rate = st.slider("Events per second", 1, 20, 6)
    threshold = st.slider("Anomaly threshold (lower = stricter)", -0.5, 0.5, -0.1, 0.01)
    contamination = st.slider("Model contamination (expected anomaly %)", 0.01, 0.10, 0.04, 0.01)
    autoupdate = st.checkbox("Auto-update model on new benign batches", value=True)
    st.markdown("---")
    st.caption("Tip: Start the stream, watch the **Risk Heatmap** and **Top Anomalies** fill in real-time.")

# -----------------------------
# State Init
# -----------------------------
if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=3000)
if "model" not in st.session_state:
    st.session_state.model = None
if "fitted" not in st.session_state:
    st.session_state.fitted = False
if "benign_pool" not in st.session_state:
    st.session_state.benign_pool = []  # store obviously benign for retraining
if "events_processed" not in st.session_state:
    st.session_state.events_processed = 0
if "anomalies_detected" not in st.session_state:
    st.session_state.anomalies_detected = 0

# -----------------------------
# Model (Isolation Forest)
# -----------------------------
def fit_model(df_feat: pd.DataFrame):
    model = IsolationForest(
        n_estimators=250,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(df_feat[NUMERIC_FEATURES])
    return model

# Bootstrap baseline on first run
if not st.session_state.fitted:
    now = datetime.utcnow()
    base = [generate_event(now - timedelta(seconds=i)) for i in range(1200)]
    df0 = pd.DataFrame(base)
    df0 = featurize(df0)
    st.session_state.model = fit_model(df0)
    st.session_state.fitted = True

# -----------------------------
# Layout: KPIs
# -----------------------------
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Events Processed", f"{st.session_state.events_processed:,}")
kpi2.metric("Anomalies Detected", f"{st.session_state.anomalies_detected:,}")
if st.session_state.events_processed > 0:
    rate = (st.session_state.anomalies_detected / st.session_state.events_processed) * 100
else:
    rate = 0.0
kpi3.metric("Anomaly Rate", f"{rate:.2f}%")

# -----------------------------
# Stream Controls
# -----------------------------
start = st.button("▶️ Start Stream", type="primary")
stop_placeholder = st.empty()

# Data displays
log_placeholder = st.empty()
table_placeholder = st.empty()
chart_placeholder = st.empty()

# -----------------------------
# Streaming Loop (runs while Start pressed)
# -----------------------------
def stream_events():
    model = st.session_state.model
    while True:
        now = datetime.utcnow()
        batch = [generate_event(now) for _ in range(event_rate)]
        df = pd.DataFrame(batch)
        df = featurize(df)

        # ML anomaly score (higher = more normal in sklearn; decision_function -> higher normal)
        # We'll invert to a "risk" score so higher = more risky
        scores = model.decision_function(df[NUMERIC_FEATURES])
        risk_ml = -scores  # higher is more anomalous
        df["risk_ml"] = risk_ml

        # Simple rules
        r_scores = []
        r_reasons = []
        for _, row in df.iterrows():
            sc, rs = rule_scores(row)
            r_scores.append(sc)
            r_reasons.append(", ".join(rs) if rs else "")
        df["risk_rules"] = r_scores
        df["rules_reasons"] = r_reasons

        # Blend risk (simple weighted fusion)
        df["risk_blend"] = (
        0.7 * (df["risk_ml"] - df["risk_ml"].min()) /
        ((df["risk_ml"].max() - df["risk_ml"].min()) + 1e-6)
        + 0.3 * df["risk_rules"]
    )

        # Track metrics
        st.session_state.events_processed += len(df)
        anomalies = df[df["risk_ml"] >= -threshold]  # since risk_ml is -decision_function
        st.session_state.anomalies_detected += len(anomalies)

        # Keep recent buffer for charts/tables
        st.session_state.buffer.extend(df.to_dict(orient="records"))

        # Append to benign pool (low risk)
        benign = df[(df["risk_blend"] < 0.2) & (df["failed_logins"] < 5) & (df["bytes"] < 5_000_000)]
        st.session_state.benign_pool.append(benign[NUMERIC_FEATURES])
        if autoupdate and len(st.session_state.benign_pool) >= 10:
            pool = pd.concat(st.session_state.benign_pool[-10:], ignore_index=True)
            st.session_state.model = fit_model(pool)
            model = st.session_state.model

        # Live log (last 12)
        df_display = df.sort_values("risk_blend", ascending=False).head(12)
        df_display = df_display[["timestamp","user","service","src_city","dst_city","bytes","failed_logins","mfa","risk_ml","risk_rules","risk_blend","rules_reasons"]]
        log_placeholder.dataframe(df_display, hide_index=True, use_container_width=True)

        # Top anomalies table (last 1500 events)
        buf_df = pd.DataFrame(list(st.session_state.buffer))[-1500:]
        if not buf_df.empty:
            top = buf_df.sort_values("risk_blend", ascending=False).head(25)
            table_placeholder.dataframe(top[["timestamp","user","service","src_city","dst_city","bytes","failed_logins","mfa","risk_blend","rules_reasons"]], hide_index=True, use_container_width=True)

            # Simple chart: rolling average risk by service
            agg = buf_df.groupby("service")["risk_blend"].mean().sort_values(ascending=False)
            chart_placeholder.bar_chart(agg)

        # Stop check
        time.sleep(1.0)

if start:
    with stop_placeholder:
        st.info("Streaming... Click **Stop** in the top-right of your browser tab or press **Rerun** to halt.")
    stream_events()
else:
    st.warning("Click **Start Stream** to begin generating events and detecting anomalies in real-time.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("🛡️ Built for the **Cybersecurity & AI Agents Masterclass** · © CyberDudeBivash")
