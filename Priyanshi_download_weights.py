from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "microsoft/deberta-v3-small"
SAVE_DIR = Path("/kaggle/working") / MODEL_ID.split("/")[-1]

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=SAVE_DIR,
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt"],
)

print(sorted(path.name for path in SAVE_DIR.iterdir()))
