import os
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Load .env from project root
project_root = os.path.join(os.path.dirname(__file__), '..')
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


def get_session():
    session = boto3.session.Session()
    return session.client(
        service_name=Config.S3_SERVICE_NAME,
        endpoint_url=Config.S3_ENDPOINT_URL,
        aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
    )

def test_read_access(s3):
    try:
        response = s3.list_objects(Bucket=Config.BUCKET_NAME)
        if 'Contents' in response:
            print("Содержимое бакета:")
            for obj in response['Contents']:
                print(f" - {obj['Key']}")
        else:
            print("Бакет пустой")
        return True
    except ClientError as e:
        print(f"Ошибка доступа: {e}")
        return False

def test_write_access(s3):
    test_content = "test content"
    test_key = "test-file.txt"

    try:
        s3.put_object(
            Bucket=Config.BUCKET_NAME,
            Key=test_key,
            Body=test_content
        )
        print(f"Файл {test_key} успешно записан")
        return True
    except ClientError as e:
        print(f"Ошибка записи: {e}")
        return False

if __name__ == "__main__":
    s3 = get_session()

    print("Проверка чтения...")
    if test_read_access(s3):
        print("Чтение: OK")

    print("\nПроверка записи...")
    if test_write_access(s3):
        print("Запись: OK")