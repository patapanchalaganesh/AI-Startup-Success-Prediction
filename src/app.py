import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

# Safe import for joblib (supports standalone joblib or sklearn internal joblib)
try:
    import joblib
except ImportError:
    from sklearn.utils import _joblib as joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="AI Startup Success Predictor", page_icon="🚀", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stMetric { background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 15px; }
    .stButton>button {
        background: linear-gradient(135deg, #f97316 0%, #dc2626 100%);
        color: white; font-weight: bold; border-radius: 8px; height: 50px; width: 100%; border: none;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, 'models', 'user_best_model.pkl')
    prep_path = os.path.join(BASE_DIR, 'models', 'user_preprocessor.pkl')

    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        model_path = 'models/user_best_model.pkl'
        prep_path = 'models/user_preprocessor.pkl'

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)

    possible_csvs = [
        os.path.join(BASE_DIR, 'data', 'startup_funding.csv'),
        os.path.join(BASE_DIR, 'data', 'indian_startup_funding.csv'),
        'data/startup_funding.csv',
        'data/indian_startup_funding.csv',
        'startup_funding.csv'
    ]
    csv_path = None
    for p in possible_csvs:
        if os.path.exists(p):
            csv_path = p
            break

    df = pd.read_csv(csv_path)
    return model, preprocessor, df


try:
    model, preprocessor, raw_df = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False

st.title("🚀 AI Startup Success Predictor App")
st.markdown("Predict **5-Year Survival & Growth Probability** of Indian startups using Machine Learning.")

st.markdown("---")

if not assets_loaded:
    st.error("⚠️ Model files missing! Please run `python src/train_model.py` first.")
    st.stop()

top_investors = ['Sequoia', 'SoftBank', 'Tiger Global', 'Accel', 'SAIF Partners', 'Naspers', 'Matrix', 'Blume']


def get_inv_score(inv_str):
    if not inv_str: return 1.0
    score = 1.0
    for top in top_investors:
        if top.lower() in str(inv_str).lower(): score += 2.0
    return score


st.sidebar.header("🕹️ Selection Mode")
mode = st.sidebar.radio("Choose Mode:", ["Select Real Startup", "Custom Startup Predictor"])

if mode == "Select Real Startup":
    st.subheader("🏢 Select Startup from Dataset")
    startup_list = sorted(raw_df['Startup Name'].dropna().unique())
    selected = st.selectbox("Choose a Startup:", startup_list)
    row = raw_df[raw_df['Startup Name'] == selected].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Startup", str(row['Startup Name']))
    c2.metric("Sector", str(row['Industry Vertical']))
    c3.metric("Location", str(row['City  Location']))
    c4.metric("Funding", str(row['Amount in USD']))

    amount_raw = str(row['Amount in USD']).replace(',', '').replace('$', '')
    try:
        funding_val = float(amount_raw)
    except:
        funding_val = 5000000.0

    inv_score = get_inv_score(str(row['Investors Name']))
    city_str = str(row['City  Location'])
    clean_city = 'Bengaluru' if 'Bengaluru' in city_str else (
        'Delhi NCR' if any(c in city_str for c in ['Gurgaon', 'Delhi', 'Noida']) else 'Mumbai')
    ind_str = str(row['Industry Vertical'])
    clean_ind = 'E-Commerce & EdTech' if any(w in ind_str for w in ['E-Commerce', 'EdTech']) else (
        'FinTech & Finance' if 'Fin' in ind_str else 'SaaS & Tech')
    type_str = str(row['InvestmentnType'])
    clean_inv_type = 'Series A-D' if 'Series' in type_str else 'Seed / Pre-Series A'

    input_df = pd.DataFrame([{
        'Amount_USD': funding_val,
        'Clean_City': clean_city,
        'Clean_Industry': clean_ind,
        'Clean_Investment_Type': clean_inv_type,
        'Investor_Score': inv_score
    }])
else:
    st.subheader("🛠️ Input Custom Startup Parameters")
    c1, c2 = st.columns(2)
    with c1:
        funding_val = st.number_input("Total Funding Amount ($ USD)", 10000.0, 2000000000.0, 15000000.0, 500000.0)
        clean_city = st.selectbox("City Location", ["Bengaluru", "Delhi NCR", "Mumbai", "Hyderabad", "Pune", "Other"])
        clean_ind = st.selectbox("Industry Sector", ["E-Commerce & EdTech", "FinTech & Finance", "Healthcare & Health",
                                                     "Logistics & Mobility", "SaaS & Tech", "Consumer Goods & Others"])
    with c2:
        clean_inv_type = st.selectbox("Investment Stage",
                                      ["Seed / Pre-Series A", "Series A-D", "Late Stage PE / Expansion", "Debt Funding",
                                       "Venture / Other"])
        lead_investor = st.text_input("Lead Investor Name", "Sequoia Capital & Tiger Global")
        inv_score = get_inv_score(lead_investor)

    input_df = pd.DataFrame([{
        'Amount_USD': funding_val,
        'Clean_City': clean_city,
        'Clean_Industry': clean_ind,
        'Clean_Investment_Type': clean_inv_type,
        'Investor_Score': inv_score
    }])

if st.button("🚀 Predict Startup Success Probability"):
    transformed = preprocessor.transform(input_df)
    pred_class = model.predict(transformed)[0]
    probs = model.predict_proba(transformed)[0]
    success_pct = probs[1] * 100

    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=success_pct,
            title={'text': "5-Year Survival Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#f97316"},
                'steps': [
                    {'range': [0, 45], 'color': '#ef4444'},
                    {'range': [45, 75], 'color': '#f59e0b'},
                    {'range': [75, 100], 'color': '#10b981'}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if pred_class == 1:
            st.success(f"### 🎉 HIGH SURVIVAL LIKELIHOOD ({success_pct:.1f}%)")
            st.markdown("✅ Strong market standing and VC funding backing.")
        else:
            st.warning(f"### ⚠️ EARLY STAGE RISK ({100 - success_pct:.1f}%)")
            st.markdown("❌ Requires additional follow-on growth funding.")