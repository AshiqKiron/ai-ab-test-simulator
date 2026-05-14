
import streamlit as st
import pandas as pd
import gspread
import json
import time
import numpy as np
from scipy import stats
from openai import OpenAI
from datetime import datetime

try:
    import gspread, json
    gc = gspread.service_account_from_dict(st.secrets["gsheets"])
    sh = gc.open_by_key(st.secrets["sheet_id"])
except Exception as e:
    st.error("🚫 Google Auth Failed")
    st.code(f"Error type: {type(e).__name__}\n\n{str(e)}", language="text")
    if hasattr(e, 'response'):
        st.json(e.response.json())  # Shows exact Google API error
    st.stop()
    

# --- 1. CONFIG & SECRETS ---
@st.cache_resource
def get_gsheet():
    try:
        creds = st.secrets["gsheets"]
        # Debug: Print redacted key info (never log full private_key)
        st.write(f"🔑 Service Account: {creds.get('client_email', 'MISSING')[:30]}...")
        st.write(f"📄 Sheet ID: {st.secrets.get('sheet_id', 'MISSING')}")
        
        gc = gspread.service_account_from_dict(creds)
        sh = gc.open_by_key(st.secrets["sheet_id"])
        st.success(f"✅ Connected to Sheet: {sh.title}")
        return sh
    except gspread.exceptions.APIError as e:
        st.error(f"🚫 Google API Error: {e}")
        st.code(str(e.response.json()), language="json")
        raise
    except Exception as e:
        st.error(f"🚫 Unexpected Error: {type(e).__name__}: {e}")
        raise

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["openai_key"])

# --- 2. PROMPT & TEST DATA LOADER ---
def load_data():
    sh = get_gsheet()
    tests = sh.worksheet("Test_Cases").get_all_records()
    prompts = sh.worksheet("Prompts").get_all_records()
    rubric = sh.worksheet("Rubric").get_all_records()
    return pd.DataFrame(tests), pd.DataFrame(prompts), pd.DataFrame(rubric)

# --- 3. LLM CALL ENGINE ---
def run_llm(system_prompt: str, user_prompt: str, test_input: str, model: str = "gpt-3.5-turbo") -> dict:
    client = get_openai_client()
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

# --- 4. SCORING ENGINE (LLM-as-Judge + Rule Fallback) ---
def score_output(llm_output: str, rubric_df: pd.DataFrame, expected: str = "") -> dict:
    # Simplified: Weighted average of dimensions using LLM judge prompt
    judge_prompt = f"""Rate this output 1-5 on these dimensions:
{rubric_df[['Dimension', 'Weight', 'Description']].to_markdown()}
Output ONLY a JSON: {{"Dimension_Name": score, ...}}
Output to score: {llm_output}"""
    
    try:
        client = get_openai_client()
        res = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role":"user","content":judge_prompt}], temperature=0.0)
        scores = json.loads(res.choices[0].message.content)
        weights = dict(zip(rubric_df["Dimension"], rubric_df["Weight"]))
        total = sum(scores.get(d, 3) * weights.get(d, 1) for d in weights) / sum(weights.values())
        return {"scores": scores, "total": round(total, 2)}
    except:
        return {"scores": {}, "total": 3.0}  # Fallback median

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="AI A/B Prompt Simulator", layout="wide")
st.title("🔬 AI Feature A/B Prompt Simulator")

tests_df, prompts_df, rubric_df = load_data()

col1, col2 = st.columns(2)
with col1:
    prompt_a = st.selectbox("Prompt A", prompts_df["Prompt_ID"])
with col2:
    prompt_b = st.selectbox("Prompt B", prompts_df["Prompt_ID"])

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
            "Winner": "A" if score_a["total"] > score_b["total"] else "B"
        })
        progress.progress((i + 1) / len(tests_df))
    
    res_df = pd.DataFrame(results)
    st.dataframe(res_df)
    
    # --- METRICS & STATISTICS ---
    delta = res_df["Score_A"] - res_df["Score_B"]
    t_stat, p_val = stats.ttest_1samp(delta, 0)
    win_rate_a = (res_df["Score_A"] > res_df["Score_B"]).mean()
    
    st.subheader("📊 A/B Results Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Score Δ", f"{delta.mean():.2f}")
    c2.metric("P-Value", f"{p_val:.3f}")
    c3.metric("Prompt A Win Rate", f"{win_rate_a:.1%}")
    c4.metric("Avg Latency Δ", f"{(res_df['Latency_A'] - res_df['Latency_B']).mean():.0f}ms")
    
    # Save to Sheets
    sh = get_gsheet()
    sheet = sh.worksheet("Results")
    sheet.update([list(res_df.columns)] + res_df.values.tolist())
    st.success("✅ Results saved to Google Sheets")
