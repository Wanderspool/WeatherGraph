# weathergraph.utils

The `weathergraph.utils` module contains helper utilities and system functions for directory caching, file downloading, environment configurations, and interactive user credentials management.

---

## Technical Overview

The utilities module exposes helpers to support cloud pipelines and interactive workflows:
1.  **Cache Directory**: Resolves standard system-specific cache locations (e.g. `~/.cache/weathergraph` on Linux) to store downloaded models, normalization weights, and temporary tiling swap spaces.
2.  **Robust Downloading**: Downloads large assets with block-wise transfer, monitoring progress and validating file sizes.
3.  **Dynamic Environment Management**: Loads and writes `.env` configurations.
4.  **Credential Prompts**: In interactive sessions, prompts users for missing API keys (e.g. CDS tokens) when remote downloads fail and updates environmental stores automatically.

---

## Technical Example

The following script loads local environment variables, downloads an ONNX model file to the system cache, and checks access credentials:

```python
import weathergraph.utils as utils
from pathlib import Path

# 1. Resolve target cache directory
cache_dir = utils.get_default_cache_dir()
print(f"System cache is located at: {cache_dir}")

# 2. Download weights file into cache
weights_url = "https://example.com/weathergraph/means.npy"
dest_path = cache_dir / "means.npy"
utils.download_file(weights_url, dest_path)

# 3. Load environment credentials from file
env_file = Path("config/.env")
if env_file.exists():
    utils.load_env_file(env_file)
else:
    # Prompt user interactively if running in a terminal
    utils.prompt_for_credentials(work_dir=Path("config"))
```

---

## Directory & File Utilities

::: weathergraph.utils.get_default_cache_dir
::: weathergraph.utils.download_file
    options:
      show_source: true

---

## Environment & Credential Managers

::: weathergraph.utils.load_env_file
::: weathergraph.utils.save_env_vars
::: weathergraph.utils.prompt_for_credentials
    options:
      show_source: true
