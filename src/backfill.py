import os
import json
import calendar
from datetime import datetime
import requests
import boto3
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
BUCKET_NAME = "renewable-energy-raw-data-ashwin"

def fetch_monthly_weather(lat=51.5074, lon=-0.1278, start_date=None, end_date=None):
    """
    Fetches hourly historical weather data from Open-Meteo Archive API for a specific date range.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation,direct_radiation"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def upload_to_s3(data_dict, year, month):
    """
    Uploads the monthly payload to S3 using a consistent Hive-style partition structure.
    """
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    month_str = f"{month:02d}"
    file_name = f"archive_{year}_{month_str}.json"
    s3_key = f"raw/weather/source=open_meteo_archive/year={year}/month={month_str}/{file_name}"
    
    json_data = json.dumps(data_dict, indent=2)
    
    print(f"Uploading to s3://{BUCKET_NAME}/{s3_key}...")
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json_data,
        ContentType='application/json'
    )

def run_backfill():
    # Loop through years 2023 to 2025
    for year in range(2023, 2026):
        for month in range(1, 13):
            # Calculate the first and last day of the current month
            last_day = calendar.monthrange(year, month)[1]
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year}-{month:02d}-{last_day}"
            
            print(f"Fetching archive data for {start_date} to {end_date}...")
            try:
                data = fetch_monthly_weather(start_date=start_date, end_date=end_date)
                upload_to_s3(data, year, month)
            except Exception as e:
                print(f"Failed for {year}-{month:02d}: {e}")

if __name__ == "__main__":
    print("Starting historical backfill process...")
    run_backfill()
    print("All historical partitions uploaded successfully!")