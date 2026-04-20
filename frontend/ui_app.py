import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import time

# ── Configuration ──────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Revenue Cycle Copilot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102,126,234,0.6) !important;
    }

    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    }

    .metric-card:hover { transform: translateY(-3px); }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #a78bfa;
        margin: 8px 0;
    }

    .metric-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .glass-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    h1, h2, h3 { color: #f0f0f0 !important; }

    .risk-high   { color: #fc8181; font-weight: 700; }
    .risk-medium { color: #f6ad55; font-weight: 700; }
    .risk-low    { color: #68d391; font-weight: 700; }

    .stTextArea textarea, .stTextInput input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: white !important;
        border-radius: 10px !important;
    }

    [data-testid="stMetricValue"] { color: #a78bfa !important; }

    .stDataFrame { border-radius: 12px; overflow: hidden; }

    div[data-testid="stAlert"] {
        border-radius: 10px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ───────────────────────────────────────────

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def process_text_claim(text_input: str):
    with st.spinner("🤖 Running multi-agent pipeline..."):
        try:
            r = requests.post(
                f"{API_URL}/claims/generate",
                json={"transcribed_text": text_input},
                timeout=120
            )
            if r.status_code == 200:
                return r.json()
            else:
                st.error(f"Pipeline error: {r.status_code} — {r.text[:200]}")
                return None
        except requests.exceptions.ConnectionError:
            st.error("❌ Backend unreachable. Start FastAPI: `uvicorn app.main:app --reload`")
            return None
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out — LLM inference may be slow. Try again.")
            return None


def process_voice_audio(audio_bytes):
    with st.spinner("🎙️ Transcribing audio..."):
        try:
            files = {"audio_file": ("recording.wav", audio_bytes, "audio/wav")}
            r = requests.post(f"{API_URL}/voice/process", files=files, timeout=60)
            if r.status_code == 200:
                return r.json().get("text", "")
            st.error("Audio transcription failed.")
            return ""
        except Exception as e:
            st.error(f"Voice error: {e}")
            return ""


def risk_badge(prob):
    if prob is None:
        return "—"
    if prob > 0.65:
        return f'<span class="risk-high">🔴 HIGH ({prob:.0%})</span>'
    elif prob > 0.45:
        return f'<span class="risk-medium">🟡 MED ({prob:.0%})</span>'
    else:
        return f'<span class="risk-low">🟢 LOW ({prob:.0%})</span>'


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 Revenue Cycle")
    st.markdown("**Copilot v1.0**")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🎙️ Copilot Workflow", "📊 Analytics Dashboard", "📋 Claims List", "🕵️ Agent Trace Logs"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    # Live backend status check
    health = api_get("/health")
    if health:
        st.success("🟢 Backend Online")
    else:
        st.error("🔴 Backend Offline")

    st.caption("Powered by XGBoost + RAG + LLaMA-3")


# ══════════════════════════════════════════════════════════════
#  PAGE 1: COPILOT WORKFLOW
# ══════════════════════════════════════════════════════════════
if page == "🎙️ Copilot Workflow":
    st.title("🏥 Clinical Copilot")
    st.markdown("Dictate or type a clinical note — the AI pipeline extracts, codes, validates, and scores denial risk automatically.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎙️ Voice Dictation")
        audio_value = st.audio_input("Record your clinical findings")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⌨️ Text Input")
        text_value = st.text_area(
            "Type clinical notes here...",
            placeholder="e.g. Patient John, 45 years old, presents with hypertension. Ordered ECG for symptom evaluation.",
            height=130
        )
        submit_text = st.button("🚀 Generate Claim", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    claim_result = None

    if audio_value is not None:
        st.success("✅ Audio captured!")
        if st.button("📡 Process Audio & Generate Claim", use_container_width=True):
            transcribed = process_voice_audio(audio_value.getvalue())
            if transcribed:
                st.info(f"**Transcribed:** {transcribed}")
                claim_result = process_text_claim(transcribed)

    elif submit_text and text_value:
        claim_result = process_text_claim(text_value)

    if claim_result:
        st.markdown("---")
        st.header("📄 Claim Result")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Patient",   claim_result.get("patient_name", "N/A"))
        m2.metric("Claim ID",  f"#{claim_result.get('claim_id', 'N/A')}")
        m3.metric("Status",    str(claim_result.get("status", "")).upper())

        risk = claim_result.get("risk_score", 0.0) * 100
        m4.metric("Denial Risk", f"{risk:.1f}%")

        col_gauge, col_info = st.columns([1, 1])

        with col_gauge:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            color = "#fc8181" if risk > 65 else "#f6ad55" if risk > 45 else "#68d391"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk,
                number={"suffix": "%", "font": {"color": color}},
                title={"text": "Denial Risk Score", "font": {"color": "#f0f0f0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#666"},
                    "bar": {"color": color},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 45],  "color": "rgba(104,211,145,0.15)"},
                        {"range": [45, 65], "color": "rgba(246,173,85,0.15)"},
                        {"range": [65, 100],"color": "rgba(252,129,129,0.15)"},
                    ],
                }
            ))
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f0f0f0"}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_info:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            final = claim_result.get("final_claim", {})
            st.markdown(f"**ICD-10:** `{final.get('icd_code', 'N/A')}`")
            st.markdown(f"**CPT:** `{final.get('cpt_code', 'N/A')}`")
            st.markdown(f"**Diagnosis:** {final.get('diagnosis', 'N/A')}")
            st.markdown(f"**Procedure:** {final.get('procedure', 'N/A')}")
            st.markdown("---")
            st.markdown(f"**AI Explanation:** {claim_result.get('explanation', '')}")
            st.markdown('</div>', unsafe_allow_html=True)

        if claim_result.get("correction_suggestions"):
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.warning("⚠️ **Suggested Corrections Before Submission:**")
            for sug in claim_result.get("correction_suggestions", []):
                st.markdown(f"• {sug}")
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2: ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    st.markdown("Real-time revenue cycle performance from the database.")

    data = api_get("/analytics/summary")

    if not data:
        st.warning("⚠️ No analytics data yet. Process some claims first, or check if the backend is running.")
        st.info("Start backend: `cd backend && uvicorn app.main:app --reload`")
    else:
        summary = data.get("summary", {})

        # ── Top KPI Cards ─────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Claims</div><div class="metric-value">{summary.get("total_claims", 0)}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Patients</div><div class="metric-value">{summary.get("total_patients", 0)}</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Valid Claims</div><div class="metric-value">{summary.get("valid_claims", 0)}</div></div>', unsafe_allow_html=True)
        with k4:
            denial_rate = summary.get("denial_rate", 0)
            color = "#fc8181" if denial_rate > 30 else "#f6ad55" if denial_rate > 15 else "#68d391"
            st.markdown(f'<div class="metric-card"><div class="metric-label">Denial Rate</div><div class="metric-value" style="color:{color}">{denial_rate}%</div></div>', unsafe_allow_html=True)
        with k5:
            avg_prob = summary.get("avg_denial_prob", 0)
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Risk Score</div><div class="metric-value">{avg_prob:.2f}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([2, 1])

        with col_left:
            # ── Claims Trend ──────────────────────────────────
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📈 Claims Over Time")
            trend = data.get("trend", [])
            if trend:
                df_trend = pd.DataFrame(trend)
                fig = px.area(
                    df_trend, x="date", y="claims",
                    title="", template="plotly_dark",
                    color_discrete_sequence=["#667eea"]
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f0f0f0"},
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=250
                )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trend data yet — process some claims to see the chart.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            # ── Risk Distribution Donut ───────────────────────
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🎯 Risk Distribution")
            high   = summary.get("high_risk_claims", 0)
            medium = summary.get("medium_risk_claims", 0)
            low    = summary.get("low_risk_claims", 0)
            if high + medium + low > 0:
                fig2 = go.Figure(go.Pie(
                    labels=["High Risk", "Medium Risk", "Low Risk"],
                    values=[high, medium, low],
                    hole=0.6,
                    marker_colors=["#fc8181", "#f6ad55", "#68d391"],
                ))
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f0f0f0"},
                    height=250,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=True,
                    legend=dict(font=dict(color="#f0f0f0"))
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No risk data yet.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Agent Usage ───────────────────────────────────────
        agent_stats = data.get("agent_stats", [])
        if agent_stats:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🤖 Agent Call Distribution")
            df_agents = pd.DataFrame(agent_stats)
            fig3 = px.bar(
                df_agents, x="agent", y="calls",
                template="plotly_dark",
                color_discrete_sequence=["#764ba2"]
            )
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f0f0f0"},
                height=250, margin=dict(l=10, r=10, t=10, b=10)
            )
            fig3.update_xaxes(showgrid=False)
            fig3.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Recent Claims Table ───────────────────────────────
        recent = data.get("recent_claims", [])
        if recent:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🕐 Recent Claims")
            df_recent = pd.DataFrame(recent)
            if "denial_probability" in df_recent.columns:
                df_recent["risk"] = df_recent["denial_probability"].apply(
                    lambda p: "🔴 HIGH" if p and p > 0.65 else ("🟡 MED" if p and p > 0.45 else "🟢 LOW")
                )
            st.dataframe(
                df_recent[["claim_id","patient_name","diagnosis","icd_code","cpt_code","status","risk"]],
                use_container_width=True,
                hide_index=True
            )
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3: CLAIMS LIST
# ══════════════════════════════════════════════════════════════
elif page == "📋 Claims List":
    st.title("📋 All Claims")
    st.markdown("Browse all processed claims with denial risk scores.")

    data = api_get("/claims/")

    if not data:
        st.warning("No claims found. Process a claim first from the Copilot Workflow page.")
    else:
        claims = data.get("claims", [])
        st.markdown(f"**Total claims:** {data.get('total', 0)}")

        if claims:
            df = pd.DataFrame(claims)

            # Search filter
            search = st.text_input("🔍 Search by patient name or diagnosis", "")
            if search:
                df = df[
                    df["patient_name"].str.contains(search, case=False, na=False) |
                    df["diagnosis"].str.contains(search, case=False, na=False)
                ]

            if "denial_probability" in df.columns:
                df["Risk"] = df["denial_probability"].apply(
                    lambda p: "🔴 HIGH" if p and p > 0.65 else ("🟡 MED" if p and p > 0.45 else "🟢 LOW")
                )

            display_cols = ["claim_id", "patient_name", "diagnosis", "icd_code", "cpt_code", "status", "denial_probability", "Risk"]
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

            # Claim detail viewer
            st.markdown("---")
            claim_id = st.number_input("View Claim Detail (enter Claim ID):", min_value=1, step=1)
            if st.button("🔍 Load Claim Detail"):
                detail = api_get(f"/claims/{int(claim_id)}")
                if detail:
                    st.json(detail)
                else:
                    st.error(f"Claim #{claim_id} not found.")


# ══════════════════════════════════════════════════════════════
#  PAGE 4: AGENT TRACE LOGS
# ══════════════════════════════════════════════════════════════
elif page == "🕵️ Agent Trace Logs":
    st.title("🕵️ Multi-Agent Execution Trace")
    st.markdown("Full transparency into how the LangGraph pipeline processed each claim. Zero black-box AI.")

    claim_id_input = st.number_input("Enter Claim ID to inspect:", min_value=1, step=1)

    if st.button("🔍 Load Agent Trace", use_container_width=True):
        detail = api_get(f"/claims/{int(claim_id_input)}")

        if not detail:
            st.error(f"Claim #{claim_id_input} not found.")
        else:
            # Claim summary
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader(f"Claim #{detail['claim_id']} — {detail['patient']['name']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Diagnosis", detail["claim"]["diagnosis"])
            c2.metric("ICD-10",    detail["claim"]["icd_code"])
            c3.metric("CPT",       detail["claim"]["cpt_code"])

            if detail.get("denial"):
                prob = detail["denial"]["probability"]
                st.metric("Denial Probability", f"{prob:.1%}" if prob else "N/A")
                st.markdown(f"**Reason:** {detail['denial']['reason']}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Agent logs
            logs = detail.get("agent_logs", [])
            if logs:
                st.subheader(f"🔎 {len(logs)} Agent Steps")
                for i, log in enumerate(logs, 1):
                    with st.expander(f"Step {i}: {log['agent']}", expanded=(i == 1)):
                        col_in, col_out = st.columns(2)
                        with col_in:
                            st.markdown("**Input:**")
                            st.json(log.get("input", {}))
                        with col_out:
                            st.markdown("**Output:**")
                            st.json(log.get("output", {}))
            else:
                st.info("No agent logs found for this claim.")
