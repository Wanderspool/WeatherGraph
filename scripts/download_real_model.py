import os
import requests
from tqdm import tqdm

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists at {dest_path}")
        return
    
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'wb') as f:
        for data in response.iter_content(1024*1024):
            f.write(data)
    print("Download complete.")

if __name__ == "__main__":
    # Keisler 2022 Reference Model
    model_url = "https://huggingface.co/Wanderspool/Keisler_2022/resolve/main/keisler_2022.onnx"
    download_file(model_url, "models/keisler_2022.onnx")
    
    # Normalization stats
    means_url = "https://huggingface.co/Wanderspool/Keisler_2022/resolve/main/means.npy"
    stds_url = "https://huggingface.co/Wanderspool/Keisler_2022/resolve/main/stds.npy"
    download_file(means_url, "data/weights/means.npy")
    download_file(stds_url, "data/weights/stds.npy")
