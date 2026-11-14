import os
import io
import json
import boto3
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

def process_raw_to_silver():
    s3 = get_s3_client()
    prefix = "raw/weather/"
    paginator = s3.get_paginator('list_objects_v2')
    
    print("Scanning S3 raw bucket for weather data...")
    
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            
            # Only process JSON files
            if not key.endswith('.json'):
                continue
                
            print(f"Processing raw file: {key}")
            
            # Read JSON payload from S3
            response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            raw_content = response['Body'].read().decode('utf-8')
            data = json.loads(raw_content)
            
            # Extract hourly data block
            if 'hourly' not in data:
                print(f"Skipping {key}: No 'hourly' key found.")
                continue
                
            hourly_data = data['hourly']
            
            # Flatten parallel arrays into a Pandas DataFrame
            df = pd.DataFrame(hourly_data)
            
            # Clean and standardize timestamps
            if 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
                df = df.drop(columns=['time'])
                # Reorder columns so timestamp leads
                cols = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
                df = df[cols]
            
            # Construct the matching Silver partition path for Parquet output
            # Example: raw/weather/source=open_meteo_archive/year=2023/month=01/archive_2023_01.json
            # Becomes: silver/weather/source=open_meteo_archive/year=2023/month=01/data.parquet
            path_parts = key.split('/')
            if len(path_parts) >= 6:
                source_folder = path_parts[2]
                year_folder = path_parts[3]
                month_folder = path_parts[4]
                silver_key = f"silver/weather/{source_folder}/{year_folder}/{month_folder}/data.parquet"
            else:
                silver_key = key.replace('raw/', 'silver/').replace('.json', '.parquet')
            
            # Serialize DataFrame to Parquet format in-memory (avoids local disk writes)
            buffer = io.BytesIO()
            df.to_parquet(buffer, engine='pyarrow', index=False)
            buffer.seek(0)
            
            print(f"Uploading Parquet to s3://{BUCKET_NAME}/{silver_key}...")
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=silver_key,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream'
            )
            
    print("Transformation complete! All data successfully promoted to the Silver layer.")

if __name__ == "__main__":
    process_raw_to_silver()