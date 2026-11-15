import os
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")

def check_aws_status():
    print(f"Checking AWS connection in region: {AWS_REGION}...")
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        # List all buckets accessible by these credentials
        response = s3.list_buckets()
        print("\nSuccessfully connected to AWS!")
        print("Buckets found in this account:")
        
        bucket_found = False
        for bucket in response.get('Buckets', []):
            name = bucket['Name']
            print(f" - {name}")
            if name == "renewable-energy-raw-data-ashwin":
                bucket_found = True
                
        if bucket_found:
            print("\nSUCCESS: Your project bucket 'renewable-energy-raw-data-ashwin' exists and is accessible!")
        else:
            print("\nWARNING: Connected successfully, but 'renewable-energy-raw-data-ashwin' was not found in the list.")
            
    except Exception as e:
        print(f"\nERROR connecting to AWS: {e}")

if __name__ == "__main__":
    check_aws_status()