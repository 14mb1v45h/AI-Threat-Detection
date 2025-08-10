# AI Threat Detection — Live Demo
**Presenter:** CyberDudeBivash  
**Format:** Streamlit web app (local)

## Features
- Synthetic sign-in & network events generator (users, IPs, geo-distance, MFA, bytes)
- **IsolationForest** anomaly detection (unsupervised ML)
- Complementary **rule engine** (impossible travel, MFA bypass, brute force, data exfil)
- Real-time KPIs, risk table, and bar chart
- Auto-retraining on obviously benign batches

## Quickstart
```bash
# 1) Create venv (recommended)
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

## Live Demo Script
1. Open the app in your browser (Streamlit auto-opens).  
2. Click **Start Stream**. Events begin flowing (6 eps by default).  
3. Increase **Events per second** to 10–15 for drama.  
4. Show **Top Anomalies** table; point to **rules_reasons** explaining each flag.  
5. Toggle **Auto-update model** and mention continuous learning.  
6. Wrap up with: “Identity is the new perimeter; AI is a force multiplier. Adopt Zero Trust.”

## Notes
- This app uses synthetic data only (safe for live demos).
- You can tune contamination & threshold from the sidebar for different anomaly volumes.
