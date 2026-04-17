import os
from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    """
    Download complete model weights from Hugging Face mirror to local directory.
    """
    # Check if the directory exists and contains files to avoid redundant downloads
    if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 5:
        print(f"✅ Model {repo_id} already exists in {local_dir}. Skipping download.")
        return
    
    print(f"⬇️ Downloading model: {repo_id} from mirror...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            endpoint="https://hf-mirror.com",  # Use mirror for improved accessibility
            resume_download=True,
            # Exclude unnecessary large files to save disk space
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"] 
        )
        print(f"✅ {repo_id} downloaded successfully!")
    except Exception as e:
        print(f"❌ Download failed for {repo_id}: {e}")