import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

# Helper function for investor score
top_investors = ['Sequoia', 'SoftBank', 'Tiger Global', 'Accel', 'SAIF Partners', 'Naspers', 'Matrix', 'Blume']


def get_inv_score(inv_str):
    if not inv_str or pd.isna(inv_str): return 1.0
    score = 1.0
    for top in top_investors:
        if top.lower() in str(inv_str).lower(): score += 2.0
    return score


# Train Model On-The-Fly if .pkl is missing
def train_model_on_the_fly(raw_df):
    df = raw_df.copy()
    df.columns = df.columns.str.strip()

    amt_col = next((c for c in df.columns if 'amount' in c.lower() or 'usd' in c.lower()), df.columns[8])
    loc_col = next((c for c in df.columns if 'city' in c.lower() or 'location' in c.lower()), df.columns[5])
    sec_col = next((c for c in df.columns if 'industry' in c.lower() or 'vertical' in c.lower()), df.columns[3])
    typ_col = next((c for c in df.columns if 'investment' in c.lower() or 'type' in c.lower()), df.columns[7])
    inv_col = next((c for c in df.columns if 'investor' in c.lower()), df.columns[6])

    def clean_amount(val):
        if pd.isna(val): return np.nan
        val = str(val).strip().replace(',', '').replace('$', '').replace('+', '')
        if val.lower() in ['undisclosed', 'unknown', 'n/a', 'nan', '']: return np.nan
        try:
            return float(val)
        except:
            return np.nan

    df['Amount_USD'] = df[amt_col].apply(clean_amount)
    df['Amount_USD'] = df['Amount_USD'].fillna(df['Amount_USD'].median())

    def clean_city(city):
        if pd.isna(city): return 'Other'
        city = str(city).strip()
        if 'Bengaluru' in city or 'Kormangala' in city: return 'Bengaluru'
        if any(c in city for c in ['Gurgaon', 'Gurugram', 'Delhi', 'Noida', 'Faridabad']): return 'Delhi NCR'
        if 'Mumbai' in city or 'Chembur' in city: return 'Mumbai'
        if 'Hyderabad' in city or 'Taramani' in city: return 'Hyderabad'
        if 'Pune' in city: return 'Pune'
        return 'Other'

    df['Clean_City'] = df[loc_col].apply(clean_city)

    def clean_industry(ind):
        if pd.isna(ind): return 'Other'
        ind = str(ind).strip()
        if any(w in ind for w in
               ['E-Commerce', 'Ecommerce', 'Retail', 'E-Tech', 'EdTech', 'Education']): return 'E-Commerce & EdTech'
        if any(w in ind for w in ['FinTech', 'Finance', 'Accounting', 'NBFC']): return 'FinTech & Finance'
        if any(w in ind for w in ['Healthcare', 'Health', 'Medicine']): return 'Healthcare & Health'
        if any(
            w in ind for w in ['Logistics', 'Transportation', 'Transport', 'Automotive']): return 'Logistics & Mobility'
        if any(
            w in ind for w in ['Software', 'SaaS', 'Technology', 'IT', 'Deep-Tech', 'AI', 'IoT']): return 'SaaS & Tech'
        return 'Consumer Goods & Others'

    df['Clean_Industry'] = df[sec_col].apply(clean_industry)

    def clean_inv_type(inv):
        if pd.isna(inv): return 'Seed'
        inv = str(inv).strip()
        if any(w in inv for w in ['Seed', 'Pre-series', 'Angel']): return 'Seed / Pre-Series A'
        if any(w in inv for w in ['Series A', 'Series B', 'Series C', 'Series D']): return 'Series A-D'
        if any(w in inv for w in ['Private Equity', 'Corporate', 'Series E', 'Series F', 'Series G',
                                  'Series H']): return 'Late Stage PE / Expansion'
        if 'Debt' in inv: return 'Debt Funding'
        return 'Venture / Other'

    df['Clean_Investment_Type'] = df[typ_col].apply(clean_inv_type)
    df['Investor_Score'] = df[inv_col].apply(get_inv_score)
    df['Startup_Status'] = np.where((df['Amount_USD'] >= 10000000) | (
        df['Clean_Investment_Type'].isin(['Series A-D', 'Late Stage PE / Expansion'])), 1, 0)

    X = df[['Amount_USD', 'Clean_City', 'Clean_Industry', 'Clean_Investment_Type', 'Investor_Score']]
    y = df['Startup_Status']

    categorical_cols = ['Clean_City', 'Clean_Industry', 'Clean_Investment_Type']
    numerical_cols = ['Amount_USD', 'Investor_Score']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    X_prep = preprocessor.fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_prep, y)

    return model, preprocessor


@st.cache_resource
def load_assets():
    possible_csvs = [
        os.path.join(BASE_DIR, 'data', 'startup_funding.csv'),
        os.path.join(BASE_DIR, '..', 'data', 'startup_funding.csv'),
        os.path.join(BASE_DIR, 'data', 'indian_startup_funding.csv'),
        'data/startup_funding.csv',
        'data/indian_startup_funding.csv',
        'startup_funding.csv'
    ]
    csv_path = next((p for p in possible_csvs if os.path.exists(p)), None)
    if not csv_path:
        raise FileNotFoundError("Dataset CSV file not found!")

    raw_df = pd.read_csv(csv_path)
    raw_df.columns = raw_df.columns.str.strip()

    possible_models = [
        os.path.join(BASE_DIR, 'models', 'user_best_model.pkl'),
        os.path.join(BASE_DIR, '..', 'models', 'user_best_model.pkl'),
        'models/user_best_model.pkl',
        'user_best_model.pkl'
    ]
    possible_preps = [
        os.path.join(BASE_DIR, 'models', 'user_preprocessor.pkl'),
        os.path.join(BASE_DIR, '..', 'models', 'user_preprocessor.pkl'),
        'models/user_preprocessor.pkl',
        'user_preprocessor.pkl'
    ]

    model_path = next((p for p in possible_models if os.path.exists(p)), None)
    prep_path = next((p for p in possible_preps if os.path.exists(p)), None)

    if model_path and prep_path:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(prep_path, 'rb') as f:
                preprocessor = pickle.load(f)
        except Exception:
            model, preprocessor = train_model_on_the_fly(raw_df)
    else:
        model, preprocessor = train_model_on_the_fly(raw_df)

    return model, preprocessor, raw_df


# Execute Asset Loading
try:
    model, preprocessor, raw_df = load_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

st.title("🚀 AI Startup Success Predictor App")
st.markdown("Predict **5-Year Survival & Growth Probability** of Indian startups using Machine Learning.")

st.markdown("---")

name_col = 'Startup Name' if 'Startup Name' in raw_df.columns else raw_df.columns[2]

st.sidebar.header("🕹️ Selection Mode")
mode = st.sidebar.radio("Choose Mode:", ["Select Real Startup", "Custom Startup Predictor"])

if mode == "Select Real Startup":
    st.subheader("🏢 Select Startup from Dataset")
    startup_list = sorted(raw_df[name_col].dropna().astype(str).unique())
    selected = st.selectbox("Choose a Startup:", startup_list)
    row = raw_df[raw_df[name_col] == selected].iloc[0]

    sec_val = str(row.get('Industry Vertical', 'Technology'))
    loc_val = str(row.get('City  Location', 'Bengaluru'))
    amt_val = str(row.get('Amount in USD', '5000000'))
    inv_val = str(row.get('Investors Name', 'Sequoia'))
    typ_val = str(row.get('InvestmentnType', 'Series A'))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Startup", selected)
    c2.metric("Sector", sec_val)
    c3.metric("Location", loc_val)
    c4.metric("Funding", amt_val)

    amount_raw = amt_val.replace(',', '').replace('$', '').strip()
    try:
        funding_val = float(amount_raw)
    except:
        funding_val = 5000000.0

    inv_score = get_inv_score(inv_val)
    clean_city = 'Bengaluru' if 'Bengaluru' in loc_val else (
        'Delhi NCR' if any(c in loc_val for c in ['Gurgaon', 'Delhi', 'Noida']) else 'Mumbai')
    clean_ind = 'E-Commerce & EdTech' if any(w in sec_val for w in ['E-Commerce', 'EdTech']) else (
        'FinTech & Finance' if 'Fin' in sec_val else 'SaaS & Tech')
    clean_inv_type = 'Series A-D' if 'Series' in typ_val else 'Seed / Pre-Series A'

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