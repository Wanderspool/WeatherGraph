import os
import platform
import urllib.request
import getpass
from pathlib import Path

def get_default_cache_dir() -> Path:
    """Returns the OS-specific default cache directory for WeatherGraph."""
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif system == "Darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    
    cache_dir = base / "weathergraph"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def download_file(url: str, dest: Path) -> Path:
    """Downloads a file from a URL to the given destination path."""
    print(f"Downloading {url} to {dest}...")
    req = urllib.request.Request(url, headers={"User-Agent": "weathergraph/1.0"})
    with urllib.request.urlopen(req) as response:
        with open(dest, 'wb') as out_file:
            out_file.write(response.read())
    print("Download complete.")
    return dest

def load_env_file(env_path: Path):
    """Simple parser for .env files that injects values into os.environ."""
    if not env_path.exists():
        return
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, val = line.partition("=")
            if sep:
                os.environ[key.strip()] = val.strip().strip("'\"")

def save_env_vars(env_path: Path, vars_dict: dict):
    """Appends dict variables to the .env file."""
    with open(env_path, "a") as f:
        for k, v in vars_dict.items():
            f.write(f"{k}={v}\n")

def prompt_for_credentials(work_dir: Path) -> bool:
    """Interactively prompts the user for credentials, applies them to env, and optionally saves them.
    Returns True if credentials were provided, False otherwise."""
    print("\n[!] Authentication failure detected.")
    print("Would you like to provide credentials interactively?")
    print("1) AWS (S3)")
    print("2) Google Cloud (GCS)")
    print("3) Copernicus CDS")
    print("4) Skip")
    choice = input("Select an option (1-4): ").strip()
    
    new_vars = {}
    if choice == "1":
        access_key = input("AWS_ACCESS_KEY_ID: ").strip()
        secret_key = getpass.getpass("AWS_SECRET_ACCESS_KEY: ").strip()
        if access_key and secret_key:
            new_vars["AWS_ACCESS_KEY_ID"] = access_key
            new_vars["AWS_SECRET_ACCESS_KEY"] = secret_key
    elif choice == "2":
        print("For GCP, provide the path to your service account JSON file, or paste its contents.")
        gcp_input = input("JSON Path or Paste: ").strip()
        if gcp_input.startswith("{"):
            # It's a JSON block, save it to a temporary file in work_dir
            key_path = work_dir / "gcp_credentials.json"
            with open(key_path, "w") as f:
                f.write(gcp_input)
            new_vars["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)
            print(f"Saved GCP key to {key_path}")
        elif gcp_input:
            new_vars["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_input
    elif choice == "3":
        cds_url = input("CDSAPI_URL (default: https://cds.climate.copernicus.eu/api/v2): ").strip()
        cds_key = getpass.getpass("CDSAPI_KEY: ").strip()
        if cds_key:
            new_vars["CDSAPI_URL"] = cds_url if cds_url else "https://cds.climate.copernicus.eu/api/v2"
            new_vars["CDSAPI_KEY"] = cds_key
    else:
        return False
        
    if not new_vars:
        print("No valid credentials provided.")
        return False
        
    # Apply to current environment
    os.environ.update(new_vars)
    
    save_choice = input(f"\nSave these credentials to {work_dir / '.env'} for future runs? [y/N]: ").strip().lower()
    if save_choice in ['y', 'yes']:
        save_env_vars(work_dir / ".env", new_vars)
        print("Credentials saved.")
    else:
        print("Credentials applied for this session only.")
        
    return True

