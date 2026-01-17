import os
import boto3
from boto3.s3.transfer import TransferConfig

os.environ["R2_ACCESS_KEY_ID"] = "340c0bb76a91a8a3a60707492e1a6c88"
os.environ["R2_SECRET_ACCESS_KEY"] = "28e08b66d45b63ce6dce7459b6bfef1c29516e97876806167bf864343f0c7962"

ACCOUNT_ID = "c70915b9956d035709662a8f1c8b3cdd"
ENDPOINT = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"
BUCKET = "quantum-yield-slim"

# R2 object key (source in Cloudflare R2)
R2_KEY = "yyyy/daily/yyyy_parquet.zip" #### <<<<<<<<<<<<< change yyyy to 2025 or 2026

# Local destination file path
LOCAL_FILE = "/yyyyy_parquet.zip" ####### <<<<<<<<<<<<<< input your local folder path

# Helpful error if env vars missing
for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
    if not os.environ.get(k):
        raise RuntimeError(f"Missing env var: {k}")

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)

# Multipart download config (good for multi-GB files)
config = TransferConfig(
    multipart_threshold=100 * 1024 * 1024,  # 100 MB
    multipart_chunksize=100 * 1024 * 1024,  # 100 MB
    max_concurrency=4,
    use_threads=True,
)

print(f"Downloading s3://{BUCKET}/{R2_KEY} -> {LOCAL_FILE}")
s3.download_file(
    Bucket=BUCKET,
    Key=R2_KEY,
    Filename=LOCAL_FILE,
    Config=config,
)
print("Download complete.")
