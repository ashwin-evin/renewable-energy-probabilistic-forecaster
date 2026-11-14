import os
import io
import boto3
import numpy as np
import pandas as pd
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

def calculate_wind_power_index(wind_speed):
    """
    Converts raw wind speed (m/s) into a normalized wind power index (0 to 1) 
    using standard turbine operational physics:
    - Cut-in speed: 3.0 m/s (below this, 0 power)
    - Rated speed: 12.0 m/s (reaches max capacity of 1.0)
    - Cut-out speed: 25.0 m/s (above this, safety shutdown to 0 power)
    - Cubic scaling between cut-in and rated speed (P proportional to v^3)
    """
    power = np.zeros_like(wind_speed, dtype=float)
    
    # Operational zone (cut-in <= v <= rated)
    mask_normal = (wind_speed >= 3.0) & (wind_speed <= 12.0)
    power[mask_normal] = ((wind_speed[mask_normal] - 3.0) / (12.0 - 3.0)) ** 3
    
    # Rated capacity zone (rated < v <= cut-out)
    mask_rated = (wind_speed > 12.0) & (wind_speed <= 25.0)
    power[mask_rated] = 1.0
    
    # Above cut-out speed (shutdown)
    mask_shutdown = wind_speed > 25.0
    power[mask_shutdown] = 0.0
    
    return power

def calculate_solar_power_index(shortwave_rad, temp_2m):
    """
    Converts shortwave radiation (W/m²) and temperature into a normalized 
    solar power index (0 to 1), accounting for panel temperature efficiency loss.
    """
    # 1. Base conversion: Normalize irradiance against peak clear-sky standard (1000 W/m²)
    power = shortwave_rad / 1000.0
    power = np.clip(power, 0.0, 1.0)
    
    # 2. Temperature Derating: Panels lose efficiency (~0.4% per °C) when overheating
    cell_temp = temp_2m + (shortwave_rad / 800.0) * 20.0
    temp_coefficient = -0.004
    derate_factor = 1.0 + temp_coefficient * np.maximum(0, cell_temp - 25.0)
    derate_factor = np.clip(derate_factor, 0.6, 1.0)
    
    final_solar_power = power * derate_factor
    return np.clip(final_solar_power, 0.0, 1.0)

def engineer_features(df):
    """
    Applies temporal, physical, and momentum-based feature engineering.
    """
    # Ensure dataframe is sorted chronologically
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1. Physics Features: Wind & Solar Power Indices
    if 'wind_speed_10m' in df.columns:
        df['wind_power_index'] = calculate_wind_power_index(df['wind_speed_10m'].values)
        
    if 'shortwave_radiation' in df.columns and 'temperature_2m' in df.columns:
        df['solar_power_index'] = calculate_solar_power_index(
            df['shortwave_radiation'].values, 
            df['temperature_2m'].values
        )
    
    # 2. Cyclical Time Encodings (Sine / Cosine mapping)
    hour = df['timestamp'].dt.hour
    month = df['timestamp'].dt.month
    
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    
    df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    
    # 3. Lag Features (Weather Momentum / History)
    lag_cols = ['wind_speed_10m', 'temperature_2m', 'shortwave_radiation']
    existing_lags = [col for col in lag_cols if col in df.columns]
    
    for col in existing_lags:
        df[f'{col}_lag_1h'] = df[col].shift(1)
        df[f'{col}_lag_3h'] = df[col].shift(3)
        df[f'{col}_lag_24h'] = df[col].shift(24) # Persistence model (yesterday same hour)
        
    # 4. Rolling Windows (Smoothing short-term volatility)
    for col in existing_lags:
        df[f'{col}_rolling_mean_3h'] = df[col].rolling(window=3, min_periods=1).mean()
        df[f'{col}_rolling_mean_6h'] = df[col].rolling(window=6, min_periods=1).mean()
        
    # Drop initial rows that have NaN values resulting from lags
    df = df.dropna().reset_index(drop=True)
    
    return df

def run_feature_pipeline():
    s3 = get_s3_client()
    prefix = "silver/weather/"
    paginator = s3.get_paginator('list_objects_v2')
    
    print("Scanning S3 Silver layer for cleaned Parquet files...")
    
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            
            if not key.endswith('.parquet'):
                continue
                
            print(f"Engineering features for: {key}")
            
            # Read Parquet file from S3 into Pandas
            response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            buffer = io.BytesIO(response['Body'].read())
            df = pd.read_parquet(buffer)
            
            # Apply feature engineering
            df_enriched = engineer_features(df)
            
            # Construct Gold partition path
            gold_key = key.replace('silver/', 'gold/').replace('data.parquet', 'features.parquet')
            
            # Serialize enriched DataFrame to Parquet in-memory
            out_buffer = io.BytesIO()
            df_enriched.to_parquet(out_buffer, engine='pyarrow', index=False)
            out_buffer.seek(0)
            
            print(f"Uploading Gold features to s3://{BUCKET_NAME}/{gold_key}...")
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=gold_key,
                Body=out_buffer.getvalue(),
                ContentType='application/octet-stream'
            )
            
    print("Feature engineering complete! Gold layer successfully updated.")

if __name__ == "__main__":
    print("Starting Gold layer feature engineering pipeline...")
    run_feature_pipeline()