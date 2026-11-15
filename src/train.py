import os
import io
import boto3
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
BUCKET_NAME = "renewable-energy-raw-data-ashwin"

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

def load_gold_data():
    s3 = get_s3_client()
    prefix = "gold/weather/"
    paginator = s3.get_paginator('list_objects_v2')
    
    dfs = []
    print("Loading Gold layer features from S3...")
    
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            key = obj['Key']
            if not key.endswith('.parquet'):
                continue
            response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            buffer = io.BytesIO(response['Body'].read())
            df_part = pd.read_parquet(buffer)
            dfs.append(df_part)
            
    if not dfs:
        raise ValueError("No gold parquet files found in S3! Make sure src/features.py ran successfully.")
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sort_values('timestamp').reset_index(drop=True)
    return full_df

def train_wind_forecaster():
    df = load_gold_data()
    
    # Define forecasting horizon: Predict wind_power_index 24 hours ahead
    horizon_hours = 24
    df['target_wind_power'] = df['wind_power_index'].shift(-horizon_hours)
    
    # Drop rows where target is NaN (the final 24 hours of the dataset)
    df = df.dropna(subset=['target_wind_power']).reset_index(drop=True)
    
    # Define feature columns (exclude timestamps, target variables, and raw labels to prevent leakage)
    exclude_cols = ['timestamp', 'target_wind_power', 'wind_power_index', 'solar_power_index']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Features used for training ({len(feature_cols)}): {feature_cols}")
    
    X = df[feature_cols]
    y = df['target_wind_power']
    
    # Chronological Train/Test Split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set size: {X_train.shape[0]} rows")
    print(f"Test set size:     {X_test.shape[0]} rows")
    
    # Initialize LightGBM Regressor
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining LightGBM model for 24-hour ahead wind power forecasting...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # Evaluate on the unseen Test Set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n==============================")
    print(" MODEL EVALUATION RESULTS ")
    print("==============================")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):     {mae:.4f}")
    print("==============================\n")
    
    return model, feature_cols

if __name__ == "__main__":
    train_wind_forecaster()