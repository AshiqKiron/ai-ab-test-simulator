# === IMPORTS ===
import streamlit as st
import pandas as pd
import gspread
from openai import OpenAI

# At top of app.py (after imports)
import streamlit as st
from groq import Groq  # ← This works ONLY if 'groq' is in requirements.txt

@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["groq_key"])  # ← Key from Secrets, not code

# In sidebar
with st.sidebar:
    if st.button("Test Groq"):
        try:
            client = get_groq_client()
            resp = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "OK"}],
                max_tokens=5
            )
            st.success(f"✅ {resp.choices[0].message.content}")
        except Exception as e:
            st.error(f"❌ {e}")

# === CONFIG ===
st.set_page_config(page_title="AI A/B Test Simulator", layout="wide")
st.title("🔬 AI Prompt A/B Simulator")



# === CACHED CONNECTIONS ===
@st.cache_resource
def get_gsheet():
    creds = st.secrets["gsheets"]
    gc = gspread.service_account_from_dict(creds)
    return gc.open_by_key(st.secrets["sheet_id"])

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["openai_key"])

# === DEBUG PANEL (Sidebar) ===
with st.sidebar:
    st.header("🔐 Connection Test")
    if st.button("Test Google Sheets"):
        try:
            sh = get_gsheet()
            st.success(f"✅ Sheet: {sh.title}")
            st.write("Tabs:", [w.title for w in sh.worksheets()])
        except Exception as e:
            st.error(f"❌ {type(e).__name__}")
            st.code(str(e))
            if hasattr(e, 'response'):
                st.json(e.response.json())
    
    if st.button("Test OpenAI"):
        try:
            client = get_openai_client()
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "OK"}],
                max_tokens=3
            )
            st.success(f"✅ Response: {resp.choices[0].message.content}")
        except Exception as e:
            st.error(f"❌ {type(e).__name__}: {e}")

# === MAIN APP ===
st.info("👈 Use sidebar to test connections first")
st.write("### Next: Load your test cases and prompts from Google Sheets")
# ... rest of your A/B logic here
