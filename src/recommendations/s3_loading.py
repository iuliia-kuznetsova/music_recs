'''
    Recommendation system outputs upload to S3 bucket

    This module provides functionality to upload data and recommendations to S3 bucket.

    Input:
    - local_path - path to local file
    - s3_key - S3 key (path in bucket)

    Output:
    - True if upload successful, False otherwise

    Usage:
    python -m src.recommendations.s3_loading --upload-data-to-s3
    python -m src.recommendations.s3_loading --upload-recommendations-to-s3
'''

# ---------- Imports ---------- #
import os
import logging

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# ---------- Load environment variables ---------- #
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
load_dotenv(os.path.join(project_root, '.env'))

class Config:
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    S3_SERVICE_NAME = 's3'
    S3_ENDPOINT_URL = 'https://storage.yandexcloud.net'
    BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
    
    # Validate required environment variables
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME]):
        missing = []
        if not AWS_ACCESS_KEY_ID: missing.append('AWS_ACCESS_KEY_ID')
        if not AWS_SECRET_ACCESS_KEY: missing.append('AWS_SECRET_ACCESS_KEY')
        if not BUCKET_NAME: missing.append('S3_BUCKET_NAME')
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}. "
                        f"Please create a .env file with these variables.")

# ---------- Logging setup ---------- #
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ---------- S3 Client ---------- #
def get_session():
    session = boto3.session.Session()
    return session.client(
        service_name=Config.S3_SERVICE_NAME,
        endpoint_url=Config.S3_ENDPOINT_URL,
        aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
    )

# ---------- Upload to S3 ---------- #
def upload_to_s3(local_path: str, s3_key: str) -> bool:
    '''
        Upload a file to S3 bucket.
        
        Args:
            local_path: Path to local file
            s3_key: S3 key (path in bucket)
            
        Returns:
            True if upload successful, False otherwise
    '''
    
    s3_client = get_session()
    if not s3_client:
        logger.warning('S3 client not initialized, skipping upload')
        return False
    
    try:
        s3_client.upload_file(local_path, Config.BUCKET_NAME, s3_key)
        logger.info(f'Uploaded {local_path} to s3://{Config.BUCKET_NAME}/{s3_key}')
        return True
    except ClientError as e:
        logger.error(f'Error uploading {local_path} to S3: {e}')
        return False
    except Exception as e:
        logger.error(f'Unexpected error uploading to S3: {e}')
        return False

# ---------- Upload data files ---------- #
def upload_data_to_s3(local_path: str, filename: str) -> bool:
    '''
        Upload data file to S3 recsys/data/ prefix.
    '''
    s3_key = f'recsys/data/{filename}'
    return upload_to_s3(local_path, s3_key)

# ---------- Upload recommendation files ---------- #
def upload_recommendations_to_s3(local_path: str, filename: str) -> bool:
    '''
        Upload recommendation file to S3 recsys/recommendations/ prefix.
    '''
    s3_key = f'recsys/recommendations/{filename}'
    return upload_to_s3(local_path, s3_key)

# ---------- All exports ---------- #
__all__ = ['upload_to_s3', 'upload_data_to_s3', 'upload_recommendations_to_s3']

