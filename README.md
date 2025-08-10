# 📊 FinanceBench AI

FinanceBench AI is an **AI-powered financial report analysis tool** that transforms raw corporate filings into **actionable insights** in seconds.  
It extracts key metrics, detects anomalies, performs sentiment analysis, and fact-checks important statements — enabling **faster and smarter investment decisions**.

---

## 🚀 Features

- **📄 PDF Report Upload** – Supports financial reports in PDF format.
- **🔍 Metric Extraction** – Automatically parses and structures financial metrics.
- **📈 Anomaly Detection** – Flags unusual changes in performance indicators.
- **🗣 Sentiment Analysis** – Uses FinBERT for finance-specific sentiment scoring.
- **✅ Financial Forecasting** – Provide better decision on making trade by forecasting value.
- **💻 Interactive Dashboard** – Built with Streamlit for a smooth user experience.

---

## 🛠 Tech Stack

- **Language:** Python
- **Libraries & Tools:**
  - `pdfplumber` – PDF parsing
  - `pandas`, `numpy` – Data processing
  - Hugging Face Transformers (`FinBERT`) – Sentiment analysis
  - ML models (`RandomForestClassifier`) (predictiona and forecasting)
  - Streamlit – UI dashboard

---

---

## ⚡ Installation

```bash
# Clone the repo
git clone https://github.com/Roshan-Kumar-Chaudhary/fin_doc_gpt.git
cd fin_doc_gpt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

# Install dependencies
pip install -r requirements.txt


# Run the Streamlit app
streamlit run app.py

```
