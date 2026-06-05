import os
import requests
import streamlit as st

# Render auto-sets API_URL from render.yaml; local dev falls back to localhost
_raw_url = os.environ.get('API_URL', 'http://localhost:8000')
# Render's fromService gives "host:port" — ensure it has a scheme
if _raw_url and not _raw_url.startswith('http'):
    _raw_url = f"https://{_raw_url}"
API_URL = _raw_url.rstrip('/')

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=30)  # 30s for Render cold starts
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

def process_text_claim(text_input: str):
    with st.spinner("Processing through multi-agent pipeline..."):
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
            st.error(f"Backend unreachable at {API_URL}. Please verify the backend service is running.")
            return None
        except requests.exceptions.Timeout:
            st.error("Request timed out. The LLM inference may be slow — please try again.")
            return None

def process_voice_audio(audio_bytes):
    with st.spinner("Transcribing audio..."):
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

def risk_color(prob):
    if prob is None: return "#94a3b8"
    if prob > 0.65: return "#f87171"
    if prob > 0.45: return "#fbbf24"
    return "#34d399"

def risk_label(prob):
    if prob is None: return "N/A"
    if prob > 0.65: return "HIGH"
    if prob > 0.45: return "MEDIUM"
    return "LOW"
