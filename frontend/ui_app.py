import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from utils import (
    api_get, process_text_claim, process_voice_audio,
    risk_color, risk_label
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="MedClaim AI — Revenue Cycle Copilot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREMIUM CSS — Inspired by Linear, Vercel, and Stripe dashboards
# Color System:
#   Background:  #0f1117 (deep space) → #161b22 (card surfaces)
#   Primary:     #6366f1 (indigo) → #818cf8 (light indigo)
#   Accent:      #06b6d4 (cyan)
#   Success:     #34d399   Warning: #fbbf24   Danger: #f87171
#   Text:        #f1f5f9 (primary)  #94a3b8 (muted)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Reset ─────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #f1f5f9;
    }
    .stApp {
        background: #0f1117;
    }

    /* ── Hide Streamlit chrome ────────────────────── */
    #MainMenu, header, footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Sidebar ──────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1117 0%, #131720 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.08);
        padding-top: 1rem;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500 !important;
        letter-spacing: 0.01em;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        padding: 0.6rem 1rem !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        margin-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: rgba(99, 102, 241, 0.08) !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[aria-checked="true"] {
        background: rgba(99, 102, 241, 0.12) !important;
        border-left: 3px solid #6366f1 !important;
    }

    /* ── Card Component ───────────────────────────── */
    .card {
        background: #161b22;
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .card:hover {
        border-color: rgba(99, 102, 241, 0.15);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.05);
    }

    /* ── KPI Stat Card ────────────────────────────── */
    .kpi {
        background: linear-gradient(135deg, #161b22 0%, #1a1f2e 100%);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: left;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #f1f5f9;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 6px;
    }

    /* ── Status Badge ─────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-valid   { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
    .badge-invalid { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.2); }
    .badge-pending { background: rgba(251,191,36,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }
    .badge-low     { background: rgba(52,211,153,0.12);  color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
    .badge-medium  { background: rgba(251,191,36,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }
    .badge-high    { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.2); }

    /* ── Section Title ────────────────────────────── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }

    /* ── Data Detail Row ──────────────────────────── */
    .detail-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .detail-key {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
    }
    .detail-val {
        font-size: 0.85rem;
        color: #e2e8f0;
        font-weight: 600;
        font-family: 'JetBrains Mono', 'SF Mono', monospace;
    }

    /* ── Buttons ──────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.2rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(99,102,241,0.15) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.25) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Inputs ───────────────────────────────────── */
    .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox select {
        background: #1a1f2e !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
    }

    /* ── Expander ─────────────────────────────────── */
    .streamlit-expanderHeader {
        background: #161b22 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }

    /* ── Brand Divider ────────────────────────────── */
    .brand-line {
        height: 2px;
        background: linear-gradient(90deg, #6366f1, #06b6d4, transparent);
        border: none;
        margin: 0 0 24px 0;
        border-radius: 2px;
    }

    /* ── Tabs ─────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #161b22;
        border-radius: 12px;
        padding: 6px;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.05rem !important;
        padding: 12px 24px !important;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.15) !important;
        color: #818cf8 !important;
    }

    /* ── Scrollbar ────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.4); }

    /* ── Dataframe ────────────────────────────────── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR NAVIGATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 24px 0;">
        <div style="font-size: 1.4rem; font-weight: 800; letter-spacing: -0.03em; color: #f1f5f9;">
            MedClaim <span style="color: #6366f1;">AI</span>
        </div>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px; letter-spacing: 0.05em; text-transform: uppercase;">
            Revenue Cycle Copilot
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="brand-line"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Copilot", "Dashboard", "Claims", "Agent Trace"],
        label_visibility="collapsed",
    )

    st.markdown("<br>" * 3, unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 16px; background: rgba(99,102,241,0.06); border-radius: 10px; border: 1px solid rgba(99,102,241,0.1);">
        <div style="font-size: 0.75rem; font-weight: 600; color: #818cf8; margin-bottom: 6px;">System Status</div>
        <div style="font-size: 0.8rem; color: #94a3b8;">Backend: <span style="color: #34d399;">Online</span></div>
        <div style="font-size: 0.8rem; color: #94a3b8;">Model: LLaMA-3 via Groq</div>
        <div style="font-size: 0.8rem; color: #94a3b8;">Pipeline: LangGraph</div>
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: COPILOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "Copilot":
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span style="font-size: 1.8rem; font-weight: 800; letter-spacing: -0.03em; color: #f1f5f9;">
            Clinical Copilot
        </span>
    </div>
    <div style="font-size: 0.95rem; color: #64748b; margin-bottom: 24px;">
        Input a clinical note to extract diagnoses, assign codes, validate, and predict denial risk — all in one pass.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="brand-line"></div>', unsafe_allow_html=True)

    # ── Input Section ──
    tab_text, tab_voice = st.tabs(["Text Input", "Voice Dictation"])

    with tab_text:
        text_value = st.text_area(
            "Clinical Note",
            placeholder="Paste a full clinical note here — SUBJECTIVE, OBJECTIVE, ASSESSMENT & PLAN sections...",
            height=250,
            label_visibility="collapsed",
        )
        col_btn, col_spacer = st.columns([1, 3])
        with col_btn:
            submit_text = st.button("Generate Claim", use_container_width=True)

    with tab_voice:
        audio_value = st.audio_input("Record clinical findings", label_visibility="collapsed")
        if audio_value is not None:
            st.success("Audio captured. Click below to process.")
        voice_submit = st.button("Process Audio", use_container_width=True)

    # ── Process Claim ──
    claim_result = None

    if submit_text and text_value:
        claim_result = process_text_claim(text_value)
    elif voice_submit and audio_value is not None:
        transcribed = process_voice_audio(audio_value.getvalue())
        if transcribed:
            st.info(f"**Transcription:** {transcribed}")
            claim_result = process_text_claim(transcribed)

    if claim_result:
        st.session_state["latest_claim"] = claim_result

    # ── Display Result ──
    if st.session_state.get("latest_claim"):
        claim = st.session_state["latest_claim"]
        final = claim.get("final_claim", {})
        risk_val = claim.get("risk_score", 0) or 0
        risk_pct = risk_val * 100
        status_raw = str(claim.get("status", "")).upper()
        badge_class = "badge-valid" if status_raw == "VALID" else "badge-invalid" if status_raw == "INVALID" else "badge-pending"
        r_class = "badge-high" if risk_pct > 65 else "badge-medium" if risk_pct > 45 else "badge-low"

        st.markdown("---")

        # ── Top KPI Row ──
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class="kpi">
                <div class="kpi-label">Patient</div>
                <div class="kpi-value" style="font-size:1.3rem;">{claim.get("patient_name","Unknown")}</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="kpi">
                <div class="kpi-label">Claim ID</div>
                <div class="kpi-value">#{claim.get("claim_id","—")}</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="kpi">
                <div class="kpi-label">Status</div>
                <div style="margin-top:8px;"><span class="badge {badge_class}">{status_raw}</span></div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="kpi">
                <div class="kpi-label">Denial Risk</div>
                <div class="kpi-value" style="color:{risk_color(risk_val)};">{risk_pct:.1f}%</div>
                <div class="kpi-sub"><span class="badge {r_class}">{risk_label(risk_val)}</span></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Detail Columns ──
        col_detail, col_gauge = st.columns([3, 2])

        with col_detail:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Coding & Diagnosis</div>', unsafe_allow_html=True)

            rows = [
                ("ICD-10 Code",   final.get("icd_code", "N/A")),
                ("CPT Code(s)",   final.get("cpt_code", "N/A")),
                ("Diagnosis",     final.get("diagnosis", "N/A")),
                ("Procedure",     final.get("procedure", "N/A")),
            ]
            for key, val in rows:
                st.markdown(f"""<div class="detail-row">
                    <span class="detail-key">{key}</span>
                    <span class="detail-val">{val}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # ── AI Explanation ──
            explanation = claim.get("explanation", "")
            if explanation:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">AI Explanation</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:0.9rem; color:#94a3b8; line-height:1.7;">{explanation}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with col_gauge:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Risk Assessment</div>', unsafe_allow_html=True)
            rc = risk_color(risk_val)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                number={"suffix": "%", "font": {"color": rc, "size": 36, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#1e293b", "tickfont": {"color": "#475569"}},
                    "bar": {"color": rc, "thickness": 0.7},
                    "bgcolor": "#1e293b",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 45],  "color": "rgba(52,211,153,0.06)"},
                        {"range": [45, 65], "color": "rgba(251,191,36,0.06)"},
                        {"range": [65, 100],"color": "rgba(248,113,113,0.06)"},
                    ],
                }
            ))
            fig.update_layout(
                height=220,
                margin=dict(l=30, r=30, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#94a3b8", "family": "Inter"}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Corrections ──
        corrections = claim.get("correction_suggestions", [])
        if corrections:
            st.markdown('<div class="card" style="border-color: rgba(251,191,36,0.15);">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="color:#fbbf24;">Suggested Corrections</div>', unsafe_allow_html=True)
            for sug in corrections:
                st.markdown(f'<div style="padding:6px 0; font-size:0.9rem; color:#e2e8f0;">&#x2022; {sug}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Agent Trace (inline) ──
        logs = claim.get("agent_logs", [])
        if logs:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Agent Execution Trace</div>', unsafe_allow_html=True)
            for i, log in enumerate(logs, 1):
                agent_name = log.get("agent_name", log.get("agent", "Unknown"))
                with st.expander(f"Step {i} — {agent_name}", expanded=False):
                    c_in, c_out = st.columns(2)
                    with c_in:
                        st.markdown("**Input**")
                        inp = log.get("input", "")
                        if isinstance(inp, (dict, list)):
                            st.json(inp)
                        else:
                            st.code(str(inp), language="text")
                    with c_out:
                        st.markdown("**Output**")
                        out = log.get("output", "")
                        if isinstance(out, (dict, list)):
                            st.json(out)
                        else:
                            st.code(str(out), language="text")
            st.markdown('</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "Dashboard":
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span style="font-size: 1.8rem; font-weight: 800; letter-spacing: -0.03em; color: #f1f5f9;">
            Analytics Dashboard
        </span>
    </div>
    <div style="font-size: 0.95rem; color: #64748b; margin-bottom: 24px;">
        Revenue cycle performance metrics and claim analytics at a glance.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="brand-line"></div>', unsafe_allow_html=True)

    data = api_get("/analytics/summary")

    if not data:
        st.markdown("""<div class="card" style="text-align:center; padding: 60px 20px;">
            <div style="font-size: 1.1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 8px;">No Analytics Data Yet</div>
            <div style="font-size: 0.9rem; color: #64748b;">Process some claims from the Copilot page to populate the dashboard.</div>
        </div>""", unsafe_allow_html=True)
    else:
        summary = data.get("summary", {})

        # ── KPI Row ──
        k1, k2, k3, k4, k5 = st.columns(5)
        kpi_data = [
            ("Total Claims",  summary.get("total_claims", 0),  None),
            ("Patients",      summary.get("total_patients", 0), None),
            ("Valid",         summary.get("valid_claims", 0),   "#34d399"),
            ("Denial Rate",   f"{summary.get('denial_rate', 0)}%", "#f87171" if summary.get("denial_rate",0) > 30 else "#fbbf24" if summary.get("denial_rate",0) > 15 else "#34d399"),
            ("Avg Risk",      f"{summary.get('avg_denial_prob', 0):.2f}", None),
        ]
        for col, (label, value, color) in zip([k1,k2,k3,k4,k5], kpi_data):
            with col:
                color_style = f'color:{color};' if color else ''
                st.markdown(f"""<div class="kpi">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="{color_style}">{value}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_chart, col_donut = st.columns([5, 3])

        with col_chart:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Claims Over Time</div>', unsafe_allow_html=True)
            trend = data.get("trend", [])
            if trend:
                df_trend = pd.DataFrame(trend)
                fig = px.area(
                    df_trend, x="date", y="claims",
                    template="plotly_dark",
                    color_discrete_sequence=["#6366f1"]
                )
                fig.update_traces(
                    fill='tozeroy',
                    fillcolor='rgba(99,102,241,0.08)',
                    line=dict(width=2)
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#94a3b8", "family": "Inter"},
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=280,
                    xaxis=dict(showgrid=False, showline=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", showline=False),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown('<div style="text-align:center; padding:40px; color:#64748b;">No trend data yet.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_donut:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
            high = summary.get("high_risk_claims", 0)
            medium = summary.get("medium_risk_claims", 0)
            low = summary.get("low_risk_claims", 0)
            if high + medium + low > 0:
                fig2 = go.Figure(go.Pie(
                    labels=["High", "Medium", "Low"],
                    values=[high, medium, low],
                    hole=0.65,
                    marker_colors=["#f87171", "#fbbf24", "#34d399"],
                    textfont=dict(color="#e2e8f0", size=12, family="Inter"),
                ))
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#94a3b8", "family": "Inter"},
                    height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=True,
                    legend=dict(font=dict(color="#94a3b8", size=11), orientation="h", y=-0.05)
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.markdown('<div style="text-align:center; padding:40px; color:#64748b;">No risk data yet.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Recent Claims Table ──
        recent = data.get("recent_claims", [])
        if recent:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Recent Claims</div>', unsafe_allow_html=True)
            df_recent = pd.DataFrame(recent)
            if "denial_probability" in df_recent.columns:
                df_recent["Risk"] = df_recent["denial_probability"].apply(
                    lambda p: "HIGH" if p and p > 0.65 else ("MEDIUM" if p and p > 0.45 else "LOW")
                )
            display_cols = [c for c in ["claim_id","patient_name","diagnosis","icd_code","cpt_code","status","Risk"] if c in df_recent.columns]
            st.dataframe(df_recent[display_cols], use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: CLAIMS LIST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "Claims":
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span style="font-size: 1.8rem; font-weight: 800; letter-spacing: -0.03em; color: #f1f5f9;">
            Claims Registry
        </span>
    </div>
    <div style="font-size: 0.95rem; color: #64748b; margin-bottom: 24px;">
        Browse, search, and inspect all processed claims.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="brand-line"></div>', unsafe_allow_html=True)

    data = api_get("/claims/")

    if not data:
        st.markdown("""<div class="card" style="text-align:center; padding: 60px 20px;">
            <div style="font-size: 1.1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 8px;">No Claims Found</div>
            <div style="font-size: 0.9rem; color: #64748b;">Process a clinical note from the Copilot page first.</div>
        </div>""", unsafe_allow_html=True)
    else:
        claims = data.get("claims", [])
        total = data.get("total", 0)

        st.markdown(f"""<div class="kpi" style="display:inline-block; margin-bottom:16px; padding:12px 20px;">
            <span class="kpi-label" style="margin:0;">Total Claims</span>
            <span class="kpi-value" style="font-size:1.3rem; margin-left:12px;">{total}</span>
        </div>""", unsafe_allow_html=True)

        if claims:
            df = pd.DataFrame(claims)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            search = st.text_input("Search by patient or diagnosis", "", label_visibility="collapsed", placeholder="Search by patient name or diagnosis...")
            if search:
                df = df[
                    df.apply(lambda row: search.lower() in str(row.get("patient_name","")).lower() or search.lower() in str(row.get("diagnosis","")).lower(), axis=1)
                ]

            if "denial_probability" in df.columns:
                df["Risk"] = df["denial_probability"].apply(
                    lambda p: "HIGH" if pd.notnull(p) and p > 0.65 else ("MEDIUM" if pd.notnull(p) and p > 0.45 else "LOW")
                )

            display_cols = [c for c in ["claim_id", "patient_name", "diagnosis", "icd_code", "cpt_code", "status", "denial_probability", "Risk"] if c in df.columns]
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Claim Detail Viewer ──
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Claim Inspector</div>', unsafe_allow_html=True)
            col_id, col_btn = st.columns([1, 1])
            with col_id:
                claim_id = st.number_input("Claim ID", min_value=1, step=1, label_visibility="collapsed", value=1)
            with col_btn:
                load_detail = st.button("Load Detail", use_container_width=True)
            if load_detail:
                detail = api_get(f"/claims/{int(claim_id)}")
                if detail:
                    st.json(detail)
                else:
                    st.error(f"Claim #{claim_id} not found.")
            st.markdown('</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: AGENT TRACE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "Agent Trace":
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span style="font-size: 1.8rem; font-weight: 800; letter-spacing: -0.03em; color: #f1f5f9;">
            Multi-Agent Trace
        </span>
    </div>
    <div style="font-size: 0.95rem; color: #64748b; margin-bottom: 24px;">
        Inspect step-by-step execution of the LangGraph pipeline for full transparency.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="brand-line"></div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    col_id, col_btn = st.columns([1, 1])
    with col_id:
        trace_claim_id = st.number_input("Claim ID to inspect", min_value=1, step=1, value=1)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        load_trace = st.button("Load Trace", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if load_trace:
        detail = api_get(f"/claims/{int(trace_claim_id)}")

        if not detail:
            st.error(f"Claim #{trace_claim_id} not found.")
        else:
            # ── Claim Summary ──
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Claim Summary</div>', unsafe_allow_html=True)

            patient = detail.get("patient", {})
            claim_data = detail.get("claim", {})
            denial = detail.get("denial", {})

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="kpi">
                    <div class="kpi-label">Patient</div>
                    <div class="kpi-value" style="font-size:1.2rem;">{patient.get("name","Unknown")}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="kpi">
                    <div class="kpi-label">ICD-10 / CPT</div>
                    <div class="kpi-value" style="font-size:1.2rem;">{claim_data.get("icd_code","—")} / {claim_data.get("cpt_code","—")}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                prob = denial.get("probability")
                prob_str = f"{prob:.1%}" if prob else "N/A"
                st.markdown(f"""<div class="kpi">
                    <div class="kpi-label">Denial Risk</div>
                    <div class="kpi-value" style="font-size:1.2rem; color:{risk_color(prob)};">{prob_str}</div>
                </div>""", unsafe_allow_html=True)

            if denial.get("reason"):
                st.markdown(f'<div style="margin-top:16px; font-size:0.85rem; color:#94a3b8;"><strong style="color:#e2e8f0;">Reason:</strong> {denial["reason"]}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # ── Agent Steps ──
            logs = detail.get("agent_logs", [])
            if logs:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<div class="section-title">{len(logs)} Pipeline Steps</div>', unsafe_allow_html=True)
                for i, log in enumerate(logs, 1):
                    agent_name = log.get("agent", log.get("agent_name", "Unknown"))
                    with st.expander(f"Step {i} — {agent_name}", expanded=(i == 1)):
                        c_in, c_out = st.columns(2)
                        with c_in:
                            st.markdown("**Input**")
                            st.json(log.get("input", {}))
                        with c_out:
                            st.markdown("**Output**")
                            st.json(log.get("output", {}))
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""<div class="card" style="text-align:center; padding: 40px;">
                    <div style="color:#64748b;">No agent logs recorded for this claim.</div>
                </div>""", unsafe_allow_html=True)
