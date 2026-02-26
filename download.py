import os
from tqdm import tqdm
from google.cloud import storage
from google.oauth2 import service_account

def download_flat_from_gcs():
    # --- CONFIGURATION ---
    BUCKET_NAME = "hoi-sit"
    # Specific GCS folder path
    PREFIX = "qwen3VL-8B/hoi_v2_sft/" 
    # Your local destination
    DESTINATION_DIR = "/media/shaun/workspace/AdaTooler-V/checkpoints/qwen3VL-8B"
    KEY_PATH = "/media/shaun/workspace/AdaTooler-V/service-account-gcs.json"

    # 1. Authenticate
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(BUCKET_NAME)

    # 2. List files ONLY in this folder (delimiter='/' stops recursion)
    print(f"🔍 Searching for files in gs://{BUCKET_NAME}/{PREFIX} (excluding subfolders)...")
    # Using delimiter='/' ensures we don't get files inside any subdirectories
    blobs = list(client.list_blobs(bucket, prefix=PREFIX, delimiter='/'))
    
    # Filter out directory markers and the prefix itself
    files_to_download = [b for b in blobs if not b.name.endswith('/') and b.name != PREFIX]
    
    if not files_to_download:
        print("❌ No files found in the specified path.")
        return

    print(f"📦 Found {len(files_to_download)} files at this level. Starting download...")
    os.makedirs(DESTINATION_DIR, exist_ok=True)

    # 3. Download loop with Progress Bar
    for blob in tqdm(files_to_download, desc="Downloading", unit="file"):
        # Get just the filename (e.g., 'config.json') to save flat into DESTINATION_DIR
        filename = os.path.basename(blob.name)
        local_path = os.path.join(DESTINATION_DIR, filename)
        
        # Skip if file already exists and size matches
        if os.path.exists(local_path) and os.path.getsize(local_path) == blob.size:
            continue

        try:
            # For larger weights on DGX/Workspace, GCS uses resumable download by default
            blob.download_to_filename(local_path)
        except Exception as e:
            print(f"\n❌ Error downloading {blob.name}: {e}")

    print(f"\n✅ Done! Files are now in: {DESTINATION_DIR}")

if __name__ == "__main__":
    download_flat_from_gcs()