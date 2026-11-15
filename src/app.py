import os
import io
import boto3
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
BUCKET_NAME = "renewable-energy-raw-data-ashwin"

# Page Configuration
st.set_page_config(
    page_title="Renewable Energy Forecasting Dashboard",
    page_icon="⚡",
    layout="wide"
)

@st.cache_resource
def load_models():
    wind_model, wind_features = joblib.load("models/wind_forecaster.pkl")
    solar_model, solar_features = joblib.load("models/solar_forecaster.pkl")
    return wind_model, wind_features, solar_model, solar_features

@st.cache_data(ttl=600)
def load_gold_data():
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    prefix = "gold/weather/"
    paginator = s3.get_paginator('list_objects_v2')
    
    dfs = []
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            key = obj['Key']
            if not key.endswith('.parquet'):
                continue
            response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            buffer = io.BytesIO(response['Body'].read())
            dfs.append(pd.read_parquet(buffer))
            
    if not dfs:
        raise ValueError("No gold parquet files found in S3!")
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
    full_df = full_df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    return full_df

# Load models and data
try:
    wind_model, wind_features, solar_model, solar_features = load_models()
    df = load_gold_data()
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

# Dashboard Title & Sidebar
st.title("⚡ Renewable Energy 24-Hour Forecasting Dashboard")
st.markdown("Production-grade data pipeline and machine learning forecasting for wind and solar power generation.")

st.sidebar.header("Dashboard Controls")
energy_source = st.sidebar.selectbox(
    "Select Energy Source",
    ["Wind Power", "Solar Power"]
)

time_window = st.sidebar.selectbox(
    "Select Timeline View",
    ["Last 7 Days (168 Hours)", "Last 14 Days (336 Hours)", "Last 30 Days (720 Hours)", "Full Dataset View"]
)

# Select appropriate model and target column based on sidebar choice
if energy_source == "Wind Power":
    model = wind_model
    features = wind_features
    target_col = "wind_power_index"
    model_rmse = 0.3933
    model_mae = 0.3614
else:
    model = solar_model
    features = solar_features
    target_col = "solar_power_index"
    model_rmse = 0.0777
    model_mae = 0.0420

# Main Layout: Metric Cards
st.subheader(f"📊 {energy_source} Performance & Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Historical Records", f"{len(df):,}")
col2.metric("Latest Index Value", f"{df[target_col].iloc[-1]:.3f}")
col3.metric("Model Test RMSE", f"{model_rmse:.4f}")
col4.metric("Model Test MAE", f"{model_mae:.4f}")

st.divider()

# Generate Predictions for the Dataset
df_eval = df.copy()
X_pred = df_eval[features]
df_eval['predicted_index'] = model.predict(X_pred)

# Filter data based on selected time window
if "7 Days" in time_window:
    chart_data = df_eval[['timestamp', target_col, 'predicted_index']].tail(168).set_index('timestamp')
elif "14 Days" in time_window:
    chart_data = df_eval[['timestamp', target_col, 'predicted_index']].tail(336).set_index('timestamp')
elif "30 Days" in time_window:
    chart_data = df_eval[['timestamp', target_col, 'predicted_index']].tail(720).set_index('timestamp')
else:
    chart_data = df_eval[['timestamp', target_col, 'predicted_index']].set_index('timestamp')

chart_data.columns = ['Actual Power Index', 'Predicted 24h Forecast']

st.subheader("📈 Historical vs 24-Hour Ahead Forecast Timeline")
st.markdown(f"Displaying **{time_window.lower()}** of generation data and LightGBM predictions.")

st.line_chart(chart_data)

# Tabs for structured organization
tab1, tab2 = st.tabs(["🔍 Recent Data Table", "⚙️ Model & Pipeline Info"])

with tab1:
    st.markdown("### Recent Hourly Records & Predictions")
    st.dataframe(df_eval[['timestamp', target_col, 'predicted_index']].tail(50), width='stretch')

with tab2:
    st.markdown("### Architecture & Engineering Details")
    st.markdown("""
    - **Data Pipeline:** Bronze (Raw S3 Parquet) $\rightarrow$ Silver (Cleaned) $\rightarrow$ Gold (Engineered Features & Physics-based Indices).
    - **Machine Learning Model:** LightGBM Regressor trained with chronological 80/20 time-series split to prevent data leakage.
    - **Forecasting Horizon:** 24-Hour Ahead Day-Ahead prediction utilizing lag features, rolling statistics, and cyclical time encodings.
    """)

# Footer
st.markdown("---")
st.caption("Built with AWS S3, LightGBM, and Streamlit | Renewable Energy Analytics Platform")