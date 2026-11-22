# models/

Local copy of the sentence-embedding model used by BERTopic (stage `topics`, approach 2). Not committed.

`make models` (or `python tools/download_model.py`) downloads it and checks its SHA256:

- Model: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache-2.0)
- Pinned revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- `model.safetensors` SHA256: `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`

Only safetensors weights are used (no pickle files), and the pipeline sets `HF_HUB_OFFLINE=1`, so no
ECR text leaves the machine. If the folder is missing, the pipeline skips BERTopic with a warning and
runs everything else.
