import pandas as pd
import numpy as np
import os

# Automatically locate the Project Root Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def preprocess_startup_data(filename='startup_funding.csv'):
    print("--- Step 1: Data Preprocessing ---")

    # Try finding the CSV file in multiple possible paths
    possible_paths = [
        os.path.join(BASE_DIR, 'data', filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, 'data', 'indian_startup_funding.csv'),
        filename,
        os.path.join('data', filename)
    ]

    csv_path = None
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = p
            break

    if not csv_path:
        raise FileNotFoundError(f"Could not find '{filename}' in project data folder!")

    print(f"Loading raw dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Clean Amount in USD
    def clean_amount(val):
        if pd.isna(val): return np.nan
        val = str(val).strip().replace(',', '').replace('$', '').replace('+', '')
        if val.lower() in ['undisclosed', 'unknown', 'n/a', 'nan', '']:
            return np.nan
        try:
            return float(val)
        except:
            return np.nan

    df['Amount_USD'] = df['Amount in USD'].apply(clean_amount)
    median_amount = df['Amount_USD'].median()
    df['Amount_USD'] = df['Amount_USD'].fillna(median_amount)

    # Clean City Location
    def clean_city(city):
        if pd.isna(city): return 'Other'
        city = str(city).strip()
        if 'Bengaluru' in city or 'Kormangala' in city: return 'Bengaluru'
        if any(c in city for c in ['Gurgaon', 'Gurugram', 'Delhi', 'Noida', 'Faridabad']): return 'Delhi NCR'
        if 'Mumbai' in city or 'Chembur' in city: return 'Mumbai'
        if 'Hyderabad' in city or 'Taramani' in city: return 'Hyderabad'
        if 'Pune' in city: return 'Pune'
        return 'Other'

    df['Clean_City'] = df['City  Location'].apply(clean_city)

    # Clean Industry Vertical
    def clean_industry(ind):
        if pd.isna(ind): return 'Other'
        ind = str(ind).strip()
        if any(w in ind for w in ['E-Commerce', 'Ecommerce', 'Retail', 'E-Tech', 'EdTech', 'Education']):
            return 'E-Commerce & EdTech'
        if any(w in ind for w in ['FinTech', 'Finance', 'Accounting', 'NBFC']):
            return 'FinTech & Finance'
        if any(w in ind for w in ['Healthcare', 'Health', 'Medicine']):
            return 'Healthcare & Health'
        if any(w in ind for w in ['Logistics', 'Transportation', 'Transport', 'Automotive']):
            return 'Logistics & Mobility'
        if any(w in ind for w in ['Software', 'SaaS', 'Technology', 'IT', 'Deep-Tech', 'AI', 'IoT']):
            return 'SaaS & Tech'
        return 'Consumer Goods & Others'

    df['Clean_Industry'] = df['Industry Vertical'].apply(clean_industry)

    # Clean Investment Type
    def clean_inv_type(inv):
        if pd.isna(inv): return 'Seed'
        inv = str(inv).strip()
        if any(w in inv for w in ['Seed', 'Pre-series', 'Angel']): return 'Seed / Pre-Series A'
        if any(w in inv for w in ['Series A', 'Series B', 'Series C', 'Series D']): return 'Series A-D'
        if any(w in inv for w in ['Private Equity', 'Corporate', 'Series E', 'Series F', 'Series G', 'Series H']):
            return 'Late Stage PE / Expansion'
        if 'Debt' in inv: return 'Debt Funding'
        return 'Venture / Other'

    df['Clean_Investment_Type'] = df['InvestmentnType'].apply(clean_inv_type)

    # Investor Score Calculation
    top_investors = ['Sequoia', 'SoftBank', 'Tiger Global', 'Accel', 'SAIF Partners', 'Naspers', 'Matrix', 'Blume']

    def calc_investor_score(inv_name):
        if pd.isna(inv_name): return 1.0
        score = 1.0
        for top in top_investors:
            if top.lower() in str(inv_name).lower():
                score += 2.0
        return score

    df['Investor_Score'] = df['Investors Name'].apply(calc_investor_score)

    # Target Label: Startup Success
    df['Startup_Status'] = np.where(
        (df['Amount_USD'] >= 10000000) | (
            df['Clean_Investment_Type'].isin(['Series A-D', 'Late Stage PE / Expansion'])),
        1, 0
    )

    out_dir = os.path.join(BASE_DIR, 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'processed_data.csv')
    df.to_csv(out_path, index=False)
    print(f"Processed dataset successfully saved to {out_path} with {len(df)} rows!")
    return df


if __name__ == "__main__":
    preprocess_startup_data()