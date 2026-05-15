# 🔬 AI Feature A/B Prompt Simulator

A free, open-source tool for Product Managers, Engineers, and AI Researchers to **scientifically evaluate prompt variants** before shipping. Uses LLM-as-judge methodology, statistical significance testing, and Google Sheets for persistent tracking.

🌐 **Live Demo**: [Link](https://ai-ab-test-simulator-hquzyjhpezn87wwbmskzmj.streamlit.app/)  
📦 **Stack**: Streamlit + Python + Google Sheets + Groq/OpenAI + SciPy  
🆓 **Cost**: 100% free tier friendly (no credit card required)

---

## 🎯 What This Project Does

Instead of guessing which prompt works better, this simulator:
1. Runs **parallel A/B generations** across a curated test suite
2. Scores outputs using a **customizable rubric** + LLM-as-judge
3. Calculates **statistical significance** (p-value, win rate, latency Δ)
4. Auto-logs results to Google Sheets for longitudinal tracking

Perfect for: Customer support replies, marketing copy, code generation, UX microcopy, and any AI-driven feature where prompt consistency matters.

---

## 🏗️ Architecture & Stack

```
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Streamlit UI │◄─────►│ Python Backend │◄─────►│ LLM APIs │
│ (Dashboard + │ │ (Caching, Stats, │ │ (Groq / OpenAI) │
│ Controls) │ │ Data Processing)│ └─────────────────┘
└────────┬────────┘ └────────┬─────────┘
│ │
▼ ▼
┌─────────────────┐ ┌──────────────────┐
│ Streamlit Secrets│ │ Google Sheets API│
│ (Encrypted Keys) │ │ (Data + Results) │
└─────────────────┘ └──────────────────┘
```


| Component | Purpose |
|-----------|---------|
| **Streamlit** | Interactive UI, dashboard metrics, progress tracking |
| **Google Sheets** | Test cases, prompts, rubric config, persistent results storage |
| **Groq / OpenAI** | Dual role: (1) Generate prompt outputs, (2) Judge & score them |
| **SciPy + Pandas** | Statistical testing (`t-test`), win rate, data manipulation |
| `@st.cache_data` | Rate-limit protection (5-min cache) for free-tier APIs |


## ⚙️ How It Works (Step-by-Step)
1. **Load Data**: App pulls `Test_Cases`, `Prompts`, and `Rubric` from Sheets
2. **Select Variants**: Pick Prompt A & Prompt B from dropdowns
3. **Run Pipeline**:
   - Injects `{input}` into each prompt
   - Calls LLM for A & B in parallel
   - Passes outputs + rubric to an LLM "grader"
   - Calculates weighted score per dimension
4. **Compute Stats**: Runs paired t-test, win rate, latency delta
5. **Visualize & Save**: Shows dashboard → writes `Results` tab to Sheets


## 🔍 Behind the Scenes

### 🧠 Scoring Engine
The app doesn't use hardcoded rules. It uses **LLM-as-Judge**:
```python
judge_prompt = f"""Rate this output 1-5 on: {rubric}
Output ONLY JSON: {{"Dimension": score, ...}}
Output: {llm_response}"""
```
---

The judge returns a JSON object. The app multiplies each score by your rubric Weight, normalizes, and produces a total score (1.0–5.0).
📊 Statistics
- **Paired t-test:** Compares A vs B scores across identical inputs
- **p-value < 0.05:** >95% confidence the winner isn't due to luck
- **Win Rate:** % of test cases where one prompt scored higher
- **Latency Δ:** Measures speed trade-off (critical for UX)

---

**⚡ Caching & Rate Limits**
Free tiers limit API calls (Groq: ~30 RPM, Sheets: 60 reads/min).
The app uses @st.cache_data(ttl=300) to cache sheet loads for 5 minutes, and includes a manual 🔄 Refresh button to control quota usage.

## 📊 How to Read the Dashboard

| Metric | Meaning | Decision Threshold |
|--------|---------|-------------------|
| **Avg Score Δ** | Quality difference (A − B) | `≥ +0.2` = meaningful improvement |
| **P-Value** | Statistical confidence | `< 0.05` = confident to ship |
| **Prompt A Win Rate** | Consistency across cases | `> 60%` = reliable advantage |
| **Avg Latency Δ** | Speed impact (ms) | `> +100ms` = potential UX risk |

✅ **Ship if:** `p < 0.05` AND `Δ > 0.2` AND `Latency Δ < 100ms`  
⏸️ **Hold if:** `p > 0.10` → Add 20+ more test cases or adjust rubric  
⚠️ **Reject if:** Winner is `+0.5` but `Latency Δ = +300ms` → Quality gain not worth UX cost

---

## ⚖️ Trade-offs & Limitations

| Area | Trade-off | Mitigation |
|------|-----------|------------|
| **Free Tier Limits** | ~30 LLM calls/min, 60 sheet reads/min | Use ≤10 test cases/run, rely on caching, batch exports |
| **LLM-as-Judge Bias** | ~80% human alignment; struggles with nuance/sarcasm | Calibrate rubric with 5 manual scores; spot-check 10% |
| **Prompt Brittleness** | Small wording changes can shift scores drastically | Test across edge cases; add tone/format guardrails |
| **Latency vs Quality** | Better prompts often use more tokens/time | Track `Latency Δ` metric; set SLA thresholds |
| **Manual Curation** | Garbage test cases = garbage insights | Use diverse, realistic inputs; include adversarial cases |

---

## 🚀 Quick Start Guide

### 1. Deploy (2 Minutes)
1. Click **Use this template** → Create GitHub repo
2. Go to [Streamlit Cloud](https://share.streamlit.io) → **New App** → Connect repo
3. Add secrets in `⚙️ Settings → Secrets` (see below)

### 2. Required Secrets

```groq_key = "gsk_xxxxx..."  # Free: https://console.groq.com
# OR openai_key = "sk-proj-..."
sheet_id = "1abc123..."    # From Google Sheets URL
groq_model = "llama-3.1-8b-instant"

[gsheets]
type = "service_account"
project_id = "your-project"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "sa@project.iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
```

---

## 3. Setup Google Sheets
Create 4 tabs with exact headers:
- **Test_Cases: **ID, Input_Text, Expected_Outcome, Ground_Truth_Score
- **Prompts:** Prompt_ID, System_Prompt, User_Prompt_Template
- **Rubric:** Dimension, Weight, Description
- **Results:** (leave empty)
🔗 Share sheet with your service account client_email as Editor.

---

**4. Run Your First Test**
- Click 🔄 Refresh Data in sidebar
- Select Prompt A & B
- Click 🚀 Run A/B Test
- Review dashboard → Results auto-save to Sheets

---

## 🛠️ Customization Tips

| Goal | How To |
|------|--------|
| **Add new scoring dimension** | Add row to `Rubric` tab → App auto-weights |
| **Test different models** | Change `groq_model` in Secrets or swap to OpenAI key |
| **Increase test rigor** | Add 20+ edge cases to `Test_Cases` (adversarial, long-form, multilingual) |
| **Export results** | Filter `Results` tab by `Run_ID` → Download as CSV |
| **Adjust judge strictness** | Edit `score_output()` temperature (currently `0.0` for consistency) |

---

## 📜 License & Contributing

MIT License. Built for PMs, devs, and AI teams who believe prompt engineering deserves the same rigor as A/B testing UI.
