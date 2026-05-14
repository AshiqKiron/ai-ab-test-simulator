# === 1. IMPORTS (TOP OF FILE) ===
import streamlit as st
import pandas as pd
import gspread
import time
from groq import Groq  # ← Required for Groq client
# ... other imports

# === 2. PAGE CONFIG ===
st.set_page_config(page_title="AI A/B Prompt Simulator", layout="wide")
st.title("🔬 AI Feature A/B Prompt Simulator")

# === 3. CACHED CONNECTIONS (HELPER FUNCTIONS) ===
@st.cache_resource
def get_gsheet():
    creds = st.secrets["gsheets"]
    gc = gspread.service_account_from_dict(creds)
    return gc.open_by_key(st.secrets["sheet_id"])

@st.cache_resource
def get_llm_client():
    if "groq_key" in st.secrets:
        return Groq(api_key=st.secrets["groq_key"]), "groq"
    else:
        from openai import OpenAI
        return OpenAI(api_key=st.secrets["openai_key"]), "openai"

# === 4. SIDEBAR TEST CODE (PLACE HERE) ===
with st.sidebar:
    st.header("🔐 System Health Check")
    
    # Test Google Sheets
    if st.button("Test Google Sheets"):
        try:
            sh = get_gsheet()
            st.success(f"✅ Sheet: {sh.title}")
            tabs = [w.title for w in sh.workheets()]
            st.write("Tabs:", tabs)
        except Exception as e:
            st.error(f"❌ Sheets: {type(e).__name__}")
            st.code(str(e))
            if hasattr(e, 'response'):
                st.json(e.response.json())
    
    # Test Groq LLM
    if st.button("Test Groq LLM"):
        try:
            client, provider = get_llm_client()
            model = st.secrets.get("groq_model", "llama-3.1-8b-instant")
            st.info(f"Provider: {provider} | Model: {model}")
            
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply ONLY: OK"}],
                max_tokens=5,
                temperature=0
            )
            output = resp.choices[0].message.content.strip()
            st.success(f"✅ Response: `{output}`")
        except Exception as e:
            st.error(f"❌ LLM: {type(e).__name__}")
            st.code(str(e))
            if "decommissioned" in str(e):
                st.warning("💡 Update `groq_model` in Secrets to 'llama-3.1-8b-instant'")

    # Optional: Show config values (redacted)
    st.divider()
    st.caption("🔑 Config")
    st.caption(f"Groq key: `gsk_...{st.secrets.get('groq_key', '')[-4:]}`")
    st.caption(f"Sheet ID: `{st.secrets.get('sheet_id', '')[:10]}...`")

# === 5. MAIN APP LOGIC (BELOW SIDEBAR) ===
# ... your load_data(), run_llm(), A/B test UI, etc.
