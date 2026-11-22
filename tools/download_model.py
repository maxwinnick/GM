"""Download the pinned sentence-embedding model used by BERTopic and verify its checksum.

    python tools/download_model.py            # -> models/all-MiniLM-L6-v2/

The pipeline then loads the model from that folder with Hugging Face network access switched
off, so ECR text never leaves the machine. Only safetensors weights are downloaded (no pickle
files), and the weights' SHA256 must match the pinned value below.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = "sentence-transformers/all-MiniLM-L6-v2"           # Apache-2.0
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SAFETENSORS_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
FILES = ["config.json", "config_sentence_transformers.json", "model.safetensors", "modules.json",
         "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
         "vocab.txt", "1_Pooling/config.json"]
DEST = Path(__file__).resolve().parents[1] / "models" / "all-MiniLM-L6-v2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    weights = DEST / "model.safetensors"
    if weights.exists() and sha256(weights) == SAFETENSORS_SHA256:
        print(f"Model already present and verified: {DEST}")
        return 0
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=REPO, revision=REVISION, local_dir=DEST, allow_patterns=FILES)
    (DEST / "2_Normalize").mkdir(exist_ok=True)
    got = sha256(weights)
    if got != SAFETENSORS_SHA256:
        weights.unlink()
        print(f"Checksum mismatch for {weights}: {got} (expected {SAFETENSORS_SHA256}). Deleted.", file=sys.stderr)
        return 1
    print(f"Downloaded and verified {REPO}@{REVISION[:7]} -> {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
