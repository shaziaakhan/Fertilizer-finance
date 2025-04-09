import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
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
st.title("Fertilizer Production Dashboard in Snowflake")

# Fetch Data from Snowflake
@st.cache_data
def load_data():
    query = "SELECT * FROM TEST_DB.PUBLIC.FERTILIZER"
    df = session.sql(query).to_pandas()
    return df

df = load_data()

# Convert Number Columns to Numeric
numeric_cols = ["ACTUAL_PRODUCTION", "BUDGET_PRODUCTION", "ACTUAL_VALUE", "BUDGET_VALUE", "Budget Rate/MT", "Actual Rate/MT"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Sidebar Filters
st.sidebar.header("Filter Data")
material_type = st.sidebar.selectbox("Select Material Type", df["M2_MATERIAL_TYPE"].unique())
filtered_df = df[df["M2_MATERIAL_TYPE"] == material_type]

# Visualization 1: Actual vs. Budget Production
st.subheader("Actual vs Budget Production")
fig1 = px.bar(filtered_df, x="M2_MATERIAL_TYPE", y=["ACTUAL_PRODUCTION", "BUDGET_PRODUCTION"],
              barmode="group", title="Actual vs Budget Production", color_discrete_sequence=["#636EFA", "#EF553B"])
st.plotly_chart(fig1)
st.write("**Interpretation:** This graph compares the actual and budgeted production for the selected material type. A gap between the bars indicates deviations from the expected production targets.")

# Visualization 2: Actual vs Budget Value
st.subheader("Actual vs Budget Value")
fig2 = px.bar(filtered_df, x="M2_MATERIAL_TYPE", y=["ACTUAL_VALUE", "BUDGET_VALUE"],
              barmode="group", title="Actual vs Budget Value", color_discrete_sequence=["#00CC96", "#AB63FA"])
st.plotly_chart(fig2)
st.write("**Interpretation:** This chart shows the actual vs. budgeted value for the selected material type. Discrepancies suggest areas where financial expectations were not met.")

# Visualization 3: Actual vs Budget Rate per MT
st.subheader("Actual vs Budget Rate per MT")
fig3 = px.bar(filtered_df, x="M2_MATERIAL_TYPE", y=["Actual Rate/MT", "Budget Rate/MT"],
              barmode="group", title="Actual vs Budget Rate per MT",
              color_discrete_sequence=["#FF5733", "#33FF57"])
st.plotly_chart(fig3)
st.write("**Interpretation:** If the actual rate per metric ton is higher than the budgeted rate, this might indicate inefficiencies or increased costs.")

# 4th Visualization: Distribution of Actual Production
st.subheader("Distribution of Actual Production")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(filtered_df["ACTUAL_PRODUCTION"], bins=20, kde=True, color="#FF5733", ax=ax)
ax.set_title("Distribution of Actual Production", fontsize=14)
st.pyplot(fig)
st.write("**Interpretation:** This histogram provides an overview of the distribution of actual production values. A right-skewed distribution might indicate a few high-production outliers.")

# 5th Visualization: Contour Plot of Actual vs Budget Production
st.subheader("Contour Plot: Actual vs Budget Production")
fig5, ax5 = plt.subplots(figsize=(8, 6))
x = filtered_df["ACTUAL_PRODUCTION"].dropna()
y = filtered_df["BUDGET_PRODUCTION"].dropna()
sns.kdeplot(x=x, y=y, cmap="Blues", fill=True, ax=ax5)
ax5.set_xlabel("ACTUAL_PRODUCTION")
ax5.set_ylabel("BUDGET_PRODUCTION")
ax5.set_title("Density Contour of Actual vs Budget Production")
st.pyplot(fig5)
st.write("**Interpretation:** The contour plot visualizes density regions for actual vs budgeted production. Darker areas indicate where values are more concentrated, revealing patterns in production performance.")
