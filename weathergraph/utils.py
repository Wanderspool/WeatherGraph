import os
import platform
import urllib.request
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
