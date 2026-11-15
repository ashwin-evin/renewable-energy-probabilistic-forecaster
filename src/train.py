import os
import io
import boto3
import joblib
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

def train_and_evaluate_model(df, target_col, model_name):
    print(f"\n--- Training 24-Hour Ahead Forecast Model for: {target_col} ---")
    
    # Copy dataframe to avoid mutating shared state
    data = df.copy()
    
    # Create 24-hour ahead target variable
    data['target'] = data[target_col].shift(-24)
    data = data.dropna(subset=['target']).reset_index(drop=True)
    
    # Define features (exclude timestamps and all potential targets/raw indices to prevent leakage)
    exclude_cols = ['timestamp', 'target', 'wind_power_index', 'solar_power_index']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_cols]
    y = data['target']
    
    # Chronological Train/Test Split (80% train, 20% test)
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    
    # Initialize LightGBM Regressor
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_X=X_test,
        eval_y=y_test,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Evaluate performance
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"[{model_name}] Results:")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MAE:  {mae:.4f}")
    
    # Save model artifact locally
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_name}.pkl"
    joblib.dump((model, feature_cols), model_path)
    print(f"Saved model artifact to: {model_path}")
    
    return model, rmse, mae

def main():
    df = load_gold_data()
    
    # 1. Train Wind Power Forecaster
    train_and_evaluate_model(df, target_col='wind_power_index', model_name='wind_forecaster')
    
    # 2. Train Solar Power Forecaster
    train_and_evaluate_model(df, target_col='solar_power_index', model_name='solar_forecaster')
    
    print("\nAll models successfully trained, evaluated, and saved to disk!")

if __name__ == "__main__":
    main()