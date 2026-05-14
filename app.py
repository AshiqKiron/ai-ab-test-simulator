import streamlit as st
import pandas as pd
import gspread
import json
import time
import numpy as np
from scipy import stats
from datetime import datetime
from groq import Groq
from openai import OpenAI

# ⚠️ Must be the first Streamlit command
st.set_page_config(page_title="AI A/B Prompt Simulator", layout="wide")
st.title("🔬 AI Feature A/B Prompt Simulator")

# ==========================
# 🔐 CACHED CONNECTIONS
# ==========================
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
        return OpenAI(api_key=st.secrets["openai_key"]), "openai"

# ==========================
# 📋 SIDEBAR HEALTH CHECK
# ==========================
with st.sidebar:
    st.header("🔐 System Health Check")
    
    if st.button("Test Google Sheets"):
        try:
            sh = get_gsheet()
            st.success(f"✅ Sheet: {sh.title}")
            try:
                tabs = [w.title for w in sh.worksheets()]
            except AttributeError:
                tabs = [w.title for w in sh.list_worksheets()]
            st.write("Tabs found:", tabs)
            
            required = ["Test_Cases", "Prompts", "Rubric", "Results"]
            missing = [t for t in required if t not in tabs]
            if missing:
                st.warning(f"⚠️ Missing tabs: {missing}")
        except Exception as e:
            st.error(f"❌ Sheets: {type(e).__name__}")
            st.code(str(e))
            if hasattr(e, 'response'):
                st.json(e.response.json())

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
            st.success(f"✅ Response: `{resp.choices[0].message.content.strip()}`")
        except Exception as e:
            st.error(f"❌ LLM: {type(e).__name__}")
            st.code(str(e))

    st.divider()
    st.caption("🔑 Config")
    st.caption(f"Groq key: `gsk_...{st.secrets.get('groq_key', '')[-4:]}`")
    st.caption(f"Sheet ID: `{st.secrets.get('sheet_id', '')[:10]}...`")

# ==========================
# 📦 DATA & LOGIC FUNCTIONS
# ==========================
def load_data():
    sh = get_gsheet()
    
    def safe_get_records(sheet_name):
        try:
            ws = sh.worksheet(sheet_name)
            values = ws.get_all_values()
            if not values or len(values) < 2:
                st.warning(f"⚠️ '{sheet_name}' tab is empty or missing data. Add headers + 1 row.")
                return []
            return ws.get_all_records()
        except Exception as e:
            st.error(f"❌ Failed to load '{sheet_name}': {e}")
            return []

    tests = safe_get_records("Test_Cases")
    prompts = safe_get_records("Prompts")
    rubric = safe_get_records("Rubric")
    
    return pd.DataFrame(tests), pd.DataFrame(prompts), pd.DataFrame(rubric)

def run_llm(system_prompt: str, user_prompt: str, test_input: str, model: str = None) -> dict:
    client, provider = get_llm_client()
    if provider == "groq" and model is None:
        model = st.secrets.get("groq_model", "llama-3.1-8b-instant")
    elif provider == "openai" and model is None:
        model = "gpt-3.5-turbo"

    full_user = user_prompt.replace("{input}", test_input)
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": full_user}],
        temperature=0.3,
        max_tokens=500
    )
    latency = (time.time() - start) * 1000
    return {"output": response.choices[0].message.content.strip(), "latency_ms": round(latency, 1)}

def score_output(llm_output: str, rubric_df: pd.DataFrame) -> dict:
    if rubric_df.empty or "Dimension" not in rubric_df.columns:
        return {"scores": {}, "total": 3.0}
    
    judge_prompt = f"""Rate this output 1-5 on these dimensions:
{rubric_df[['Dimension', 'Weight', 'Description']].to_markdown(index=False)}
Output ONLY a JSON: {{"Dimension_Name": score, ...}}
Output to score: {llm_output}"""
    
    try:
        client, provider = get_llm_client()
        model = "llama-3.1-8b-instant" if provider == "groq" else "gpt-3.5-turbo"
        res = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":judge_prompt}],
            temperature=0.0,
            max_tokens=200
        )
        raw = res.choices[0].message.content.strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        scores = json.loads(cleaned)
        
        weights = dict(zip(rubric_df["Dimension"], rubric_df["Weight"]))
        total = sum(scores.get(d, 3) * weights.get(d, 1) for d in weights) / sum(weights.values())
        return {"scores": scores, "total": round(total, 2)}
    except Exception:
        return {"scores": {}, "total": 3.0}

# ==========================
# 🖥️ MAIN APP UI
# ==========================
tests_df, prompts_df, rubric_df = load_data()

# Graceful fallback if sheet tabs are still empty
if prompts_df.empty or tests_df.empty:
    st.warning("⚠️ No test cases or prompts found. Please populate 'Test_Cases' and 'Prompts' tabs in Google Sheets with headers and at least 1 row of data.")
else:
    col1, col2 = st.columns(2)
    with col1:
        prompt_a = st.selectbox("Prompt A", prompts_df["Prompt_ID"].tolist())
    with col2:
        prompt_b = st.selectbox("Prompt B", prompts_df["Prompt_ID"].tolist())

    if st.button("🚀 Run A/B Test", type="primary"):
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        progress = st.progress(0)
        results = []
        
        for i, row in tests_df.iterrows():
            pa = prompts_df[prompts_df["Prompt_ID"] == prompt_a].iloc[0]
            pb = prompts_df[prompts_df["Prompt_ID"] == prompt_b].iloc[0]
            
            out_a = run_llm(pa["System_Prompt"], pa["User_Prompt_Template"], row["Input_Text"])
            out_b = run_llm(pb["System_Prompt"], pb["User_Prompt_Template"], row["Input_Text"])
            
            score_a = score_output(out_a["output"], rubric_df)
            score_b = score_output(out_b["output"], rubric_df)
            
            results.append({
                "Run_ID": run_id, "Test_ID": row["ID"],
                "Prompt_A": prompt_a, "Output_A": out_a["output"], "Score_A": score_a["total"],
                "Prompt_B": prompt_b, "Output_B": out_b["output"], "Score_B": score_b["total"],
                "Latency_A": out_a["latency_ms"], "Latency_B": out_b["latency_ms"],
                "Winner": "A" if score_a["total"] > score_b["total"] else ("Tie" if score_a["total"] == score_b["total"] else "B")
            })
            progress.progress((i + 1) / len(tests_df))
        
        res_df = pd.DataFrame(results)
        st.dataframe(res_df)
        
        # 📊 Statistics
        delta = res_df["Score_A"] - res_df["Score_B"]
        t_stat, p_val = stats.ttest_1samp(delta, 0)
        win_rate_a = (res_df["Score_A"] > res_df["Score_B"]).mean()
        
        st.subheader("📊 A/B Results Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Score Δ", f"{delta.mean():.2f}")
        c2.metric("P-Value", f"{p_val:.3f}")
        c3.metric("Prompt A Win Rate", f"{win_rate_a:.1%}")
        c4.metric("Avg Latency Δ", f"{(res_df['Latency_A'] - res_df['Latency_B']).mean():.0f}ms")
        
        # 💾 Save to Google Sheets
        sh = get_gsheet()
        sheet = sh.worksheet("Results")
        sheet.update([list(res_df.columns)] + res_df.values.tolist())
        st.success("✅ Results saved to Google Sheets")
