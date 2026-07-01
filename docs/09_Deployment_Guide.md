# PRISM Deployment Guide (Hugging Face Spaces)

This guide details how to deploy the PRISM Pediatric Respiratory Intelligence System to Hugging Face Spaces using the Streamlit SDK and automated GitHub Actions.

---

## 1. Prerequisites
- A Hugging Face account with an active access token (`HF_TOKEN`)
- The PRISM source code repository
- `requirements.txt` generated from Poetry

---

## 2. Preparing the Repository
Ensure that your repository has the following files at the root:
- `requirements.txt`: Contains all dependencies required to run the Streamlit app.
- `frontend/app.py`: The main entrypoint for the Streamlit application.
- `README.md`: Must contain the Hugging Face YAML metadata block at the top.

### Generating requirements.txt
If you update dependencies using Poetry, regenerate `requirements.txt`:
```bash
poetry self add poetry-plugin-export
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

### README Metadata
The top of the `README.md` must include:
```yaml
---
title: PRISM - Pediatric Respiratory Intelligence
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: frontend/app.py
pinned: false
---
```

---

## 3. Git LFS (Large File Storage) Configuration
Because AI model files and vector metadata are too large for standard git, PRISM uses Git LFS. **Failure to configure this correctly will result in a deployment rejection.**

Tracked files:
- Checkpoints: `models/checkpoints/*.pt`
- Vector Metadata: `models/embeddings/*.csv`

Ensure these are tracked in `.gitattributes`:
```text
models/checkpoints/*.pt filter=lfs diff=lfs merge=lfs -text
models/embeddings/*.csv filter=lfs diff=lfs merge=lfs -text
```
Ensure they are **NOT** ignored in `.gitignore`:
```text
!models/checkpoints/*.pt
!models/embeddings/embeddings_metadata.csv
```

---

## 4. Automated Deployment (GitHub Actions)

The recommended deployment strategy uses GitHub Actions to automatically sync the `main` branch to Hugging Face Spaces.

### 4.1 Set GitHub Secrets
In your GitHub repository, go to **Settings > Secrets and variables > Actions** and add:
- `HF_TOKEN`: Your Hugging Face access token (with `write` permissions).
- `HF_USERNAME`: Your Hugging Face username.
- `HF_SPACE_NAME`: The name of the target Space (e.g., `PRISM`).

### 4.2 The Workflow File
Ensure `.github/workflows/huggingface_sync.yml` exists with the following configuration. Note the crucial `git lfs fetch --all origin` step, which is required to prevent "missing local objects" errors when pushing to HF.

```yaml
name: Sync to Hugging Face hub
on:
  push:
    branches: [main]

jobs:
  sync-to-hub:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
          lfs: true

      - name: Fetch all LFS objects
        run: git lfs fetch --all origin

      - name: Push to Hugging Face Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_USERNAME: ${{ secrets.HF_USERNAME }}
          HF_SPACE_NAME: ${{ secrets.HF_SPACE_NAME }}
        run: |
          git remote add space https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME
          git push --force https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME main
```

---

## 5. Verification
Once code is pushed to `main`, monitor the GitHub Actions logs. If successful, Hugging Face will automatically begin building the Docker container for the Streamlit app.

If the build succeeds, the App will become **Running**.

### Troubleshooting
- **No checkpoint found / Using untrained weights:** Ensure you did not ignore the `.pt` files in `.gitignore`.
- **Remote rejected (missing LFS objects):** Ensure the GitHub Actions workflow contains `git lfs fetch --all origin`.
- **Missing Module Error:** Check `requirements.txt` to ensure all dependencies are listed.
- **Theme Not Applying:** Ensure `.streamlit/config.toml` is committed to the repository so the dark theme is applied on Hugging Face.
