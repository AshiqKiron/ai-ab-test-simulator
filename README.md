# 🔬 AI A/B Prompt Simulator
Simulate how prompt versions affect LLM output quality using Google Sheets + OpenAI API + Streamlit.

## 🚀 Quick Deploy
1. Fork & clone repo
2. Create Google Sheet with 4 tabs (see Phase 2)
3. Get OpenAI API key & Google Service Account JSON
4. Deploy to Streamlit Cloud → Add secrets
5. Run A/B tests directly in browser

## 📊 PM Framework Included
- Success metrics: Win rate, p-value, latency delta
- Eval design: LLM-as-judge + rubric weighting + calibration steps
- Tradeoff comms: Quality vs cost vs speed dashboard

## 💡 Free Tier Tips
- Use `gpt-3.5-turbo` or `groq/llama-3-8b` for $0 runs
- Cache repeated test cases via `@st.cache_data`
- Limit test sets to 50 cases per run to avoid rate limits

## 📝 License
MIT
