import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd
# Set up the Streamlit page
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
# Initialize Snowflake session
# session = get_active_session()
from snowflake.snowpark import Session
import streamlit as st

# Snowflake credentials (from secrets.toml or environment variables)
sf_options = {
    "account": st.secrets["snowflake"]["account"],
    "user": st.secrets["snowflake"]["user"],
    "password": st.secrets["snowflake"]["password"],
    "warehouse": st.secrets["snowflake"]["warehouse"],
    "database": st.secrets["snowflake"]["database"],
    "schema": st.secrets["snowflake"]["schema"],
}

# Initialize the Snowflake session
session = Session.builder.configs(sf_options).create()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from snowflake.snowpark.context import get_active_session

@st.cache_data
def load_data(_session):
    query = "SELECT * FROM FERTILIZER"
    return _session.sql(query).to_pandas()

def prepare_data(df):
    df['Budget Cost/MT'] = pd.to_numeric(df['Budget Cost/MT'], errors='coerce')
    df['Actual Cost/MT'] = pd.to_numeric(df['Actual Cost/MT'], errors='coerce')
    df['Budget Production'] = pd.to_numeric(df['Budget Production'], errors='coerce')
    df['Budget Value'] = pd.to_numeric(df['Budget Value'], errors='coerce')
    df['Budget Rate/MT'] = pd.to_numeric(df['Budget Rate/MT'], errors='coerce')
    
    df['Cost_Efficiency_Ratio'] = df['Budget Cost/MT'] / df['Actual Cost/MT']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Cost_Efficiency_Ratio'], inplace=True)
    
    categorical_cols = ['PL_COP: Applicable', 'PL_COP: Applicable: PL', 
                       'PL_COP: Applicable: SKU', 'M2_Material Type']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df, label_encoders

def train_model(df):
    features = ['PL_COP: Applicable', 'PL_COP: Applicable: PL', 
                'PL_COP: Applicable: SKU', 'M2_Material Type',
                'Budget Production', 'Budget Value', 'Budget Rate/MT']
    
    X = df[features]
    y = df['Cost_Efficiency_Ratio']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, features

st.title('Fertilizer Cost Prediction')

try:
    session = get_active_session()
    df = load_data(session)
    df, label_encoders = prepare_data(df)
    model, scaler, features = train_model(df)

    st.subheader('Enter Budget Details')
    
    input_data = {}
    
    categorical_cols = ['PL_COP: Applicable', 'PL_COP: Applicable: PL', 
                       'PL_COP: Applicable: SKU', 'M2_Material Type']
    for col in categorical_cols:
        unique_values = pd.Series(label_encoders[col].inverse_transform(df[col].unique()))
        input_data[col] = st.selectbox(f'Select {col}', unique_values)
    
    numerical_features = ['Budget Production', 'Budget Value', 'Budget Rate/MT', 'Budget Cost/MT']
    
    for feature in numerical_features:
        default_value = float(df[feature].mean()) if feature in df.columns else 0.0
        input_data[feature] = st.number_input(
            f'Enter {feature}',
            value=default_value,
            step=0.01
        )
    
    if st.button('Predict Actual Cost'):
        input_df = pd.DataFrame([input_data])
        
        for col, le in label_encoders.items():
            input_df[col] = le.transform(input_df[col].astype(str))
        
        prediction_features = [col for col in features if col in input_df.columns]
        input_scaled = scaler.transform(input_df[prediction_features])
        predicted_efficiency = model.predict(input_scaled)[0]
        predicted_actual_cost = input_data['Budget Cost/MT'] / predicted_efficiency
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Budget Cost/MT', f"${input_data['Budget Cost/MT']:.2f}")
        with col2:
            st.metric('Predicted Actual Cost/MT', f"${predicted_actual_cost:.2f}")
        
        deviation = ((predicted_actual_cost - input_data['Budget Cost/MT']) / 
                    input_data['Budget Cost/MT'] * 100)
        st.write(f'Predicted deviation from budget: {deviation:.2f}%')

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Debug info:")
    if 'df' in locals():
        st.write("Features:", features)
        st.write("Input columns:", input_df.columns.tolist() if 'input_df' in locals() else "No input data")
      
