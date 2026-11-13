import os
import json
from datetime import datetime
import requests
import boto3
from dotenv import load_dotenv

# 1. Load local environment variables from .env
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
BUCKET_NAME = "renewable-energy-raw-data-ashwin"

def fetch_weather_data(lat=51.5074, lon=-0.1278):
    """
    Fetches hourly weather data from Open-Meteo API for given coordinates.
    Default coordinates set to a central UK reference point.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation,direct_radiation",
        "forecast_days": 3
    }
    
    print(f"Fetching weather data for Lat: {lat}, Lon: {lon}...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def upload_to_s3(data_dict):
    """
    Uploads the JSON payload to S3 using a Hive-style partitioned path structure.
    """
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    # Construct Hive-style partition folders using current UTC date
    now = datetime.utcnow()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    file_name = f"weather_forecast_{date_str}.json"
    s3_key = f"raw/weather/source=open_meteo/year={year}/month={month}/{file_name}"
    
    # Convert dictionary to JSON string bytes
    json_data = json.dumps(data_dict, indent=2)
    
    print(f"Uploading data to s3://{BUCKET_NAME}/{s3_key}...")
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json_data,
        ContentType='application/json'
    )
    print("Upload completed successfully!")

if __name__ == "__main__":
    try:
        weather_data = fetch_weather_data()
        upload_to_s3(weather_data)
    except Exception as e:
        print(f"Pipeline execution failed: {e}")