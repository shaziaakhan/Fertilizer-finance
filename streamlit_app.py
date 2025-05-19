import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from snowflake.snowpark import Session

# Set Streamlit page config
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Function to establish a single Snowflake session
@st.cache_resource
def get_snowflake_session():
    sf_options = {
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"],
    }
    return Session.builder.configs(sf_options).create()

# Initialize the session only once
session = get_snowflake_session()

st.title('🌾 AgroEconomix™ – Predictive Analytics for Fertilizer Cost Efficiency')

@st.cache_data
def load_data():
    query = "SELECT * FROM FERTILIZER"
    return session.sql(query).to_pandas()

def prepare_data(df):
    # Clean and format
    for col in ['Budget Cost/MT', 'Actual Cost/MT', 'BUDGET_PRODUCTION',
                'BUDGET_VALUE', 'Budget Rate/MT']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.rename(columns={
        'BUDGET_PRODUCTION': 'Budget Production',
        'BUDGET_VALUE': 'Budget Value',
        'SEASON': 'Season',
        'REGION': 'Region',
        'DATE': 'Date'
    }, inplace=True)

    df['Cost_Efficiency_Ratio'] = df['Budget Cost/MT'] / df['Actual Cost/MT']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Cost_Efficiency_Ratio'], inplace=True)

    cat_cols = ['PL_COP: Applicable', 'PL_COP: Applicable: PL', 'PL_COP: Applicable: SKU',
                'M2_MATERIAL_TYPE', 'Region', 'Season']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    return df.dropna(), label_encoders

def train_model(df):
    features = ['PL_COP: Applicable', 'PL_COP: Applicable: PL', 'PL_COP: Applicable: SKU',
                'M2_MATERIAL_TYPE', 'Region', 'Season', 'Month', 'Day',
                'Budget Production', 'Budget Value', 'Budget Rate/MT']
    X = df[features]
    y = df['Cost_Efficiency_Ratio']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, features

try:
    df = load_data()
    df, label_encoders = prepare_data(df)
    model, scaler, features = train_model(df)

    st.markdown("---")
    st.subheader("🧩 Step 1: Select Fertilizer Details")

    input_data = {}

    # SKU
    sku_options = pd.Series(label_encoders['PL_COP: Applicable: SKU'].inverse_transform(df['PL_COP: Applicable: SKU'].unique()))
    selected_sku = st.selectbox("Select SKU (Material)", sku_options)
    input_data['PL_COP: Applicable: SKU'] = selected_sku

    sku_encoded = label_encoders['PL_COP: Applicable: SKU'].transform([selected_sku])[0]
    valid_pls = df[df['PL_COP: Applicable: SKU'] == sku_encoded]['PL_COP: Applicable: PL'].unique()
    pl_options = pd.Series(label_encoders['PL_COP: Applicable: PL'].inverse_transform(valid_pls))
    selected_pl = st.selectbox("Select PL", pl_options)
    input_data['PL_COP: Applicable: PL'] = selected_pl

    pl_encoded = label_encoders['PL_COP: Applicable: PL'].transform([selected_pl])[0]
    valid_cops = df[df['PL_COP: Applicable: PL'] == pl_encoded]['PL_COP: Applicable'].unique()
    cop_options = pd.Series(label_encoders['PL_COP: Applicable'].inverse_transform(valid_cops))
    selected_cop = st.selectbox("Select COP", cop_options)
    input_data['PL_COP: Applicable'] = selected_cop

    # Material type
    mat_options = pd.Series(label_encoders['M2_MATERIAL_TYPE'].inverse_transform(df['M2_MATERIAL_TYPE'].unique()))
    selected_mat = st.selectbox("Select Material Type", mat_options)
    input_data['M2_MATERIAL_TYPE'] = selected_mat

    st.markdown("---")
    st.subheader("🗓️ Step 2: Select Date, Region & Season")

    selected_date = st.date_input("📅 Select Date")
    input_data['Month'] = selected_date.month
    input_data['Day'] = selected_date.day

    region_options = pd.Series(label_encoders['Region'].inverse_transform(df['Region'].unique()))
    selected_region = st.selectbox("🌍 Select Region", region_options)
    input_data['Region'] = selected_region

    season_options = pd.Series(label_encoders['Season'].inverse_transform(df['Season'].unique()))
    selected_season = st.selectbox("☀️ Select Season", season_options)
    input_data['Season'] = selected_season

    st.markdown("---")
    st.subheader("💼 Step 3: Enter Budget Details")

    col1, col2 = st.columns(2)
    with col1:
        input_data['Budget Production'] = st.number_input("📦 Budget Production (MT)", min_value=0.0, step=100.0)
        input_data['Budget Rate/MT'] = st.number_input("💰 Budget Rate/MT", min_value=0.0, step=100.0)
    with col2:
        input_data['Budget Value'] = st.number_input("📊 Budget Value (₹)", min_value=0.0, step=10000.0)
        input_data['Budget Cost/MT'] = st.number_input("💵 Budget Cost/MT (₹)", min_value=0.0, step=100.0)

    if st.button("🚀 Predict Actual Cost/MT"):
        input_df = pd.DataFrame([input_data])
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))
        input_scaled = scaler.transform(input_df[features])
        predicted_efficiency = model.predict(input_scaled)[0]
        predicted_actual_cost = input_data['Budget Cost/MT'] / predicted_efficiency
        deviation = ((predicted_actual_cost - input_data['Budget Cost/MT']) / input_data['Budget Cost/MT']) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"💵 **Budget Cost/MT**: ₹{input_data['Budget Cost/MT']:,.2f}")
        with col2:
            st.info(f"📈 **Predicted Actual Cost/MT**: ₹{predicted_actual_cost:,.2f}")

        st.markdown("### Business Insight:")
        
        if deviation < 0:
            st.success(f"🟢 Profit Expected: Actual cost is **{abs(deviation):.2f}% lower** than the budgeted cost.")
            st.markdown("💡 _Great job! The cost control strategy is effective. You may consider reallocating the surplus to high-priority areas or negotiating better future rates._")
        elif deviation > 0:
            st.error(f"🔴 Loss Expected: Actual cost is **{deviation:.2f}% higher** than the budgeted cost.")
            st.markdown("⚠️ _Cost overrun detected. Investigate reasons such as raw material inflation, supply chain delays, or poor vendor terms._")
        else:
            st.info("🔄 No Deviation: Actual cost matches the budgeted cost exactly.")
            st.markdown("✅ _You're on track. Continue monitoring closely to maintain performance._")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")

