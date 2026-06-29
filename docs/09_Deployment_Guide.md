# PRISM Deployment Guide (Hugging Face Spaces)

This guide details how to deploy the PRISM Pediatric Respiratory Intelligence System to Hugging Face Spaces using the Streamlit SDK.

## 1. Prerequisites
- A Hugging Face account
- The PRISM source code repository (cleaned and prepped)
- `requirements.txt` generated from Poetry

## 2. Preparing the Repository
Ensure that your repository has the following files at the root:
- `requirements.txt`: Contains all dependencies required to run the Streamlit app.
- `frontend/app.py`: The main entrypoint for the Streamlit application.
- `README.md`: Must contain the Hugging Face YAML metadata block at the top.

### Generating requirements.txt
If you update dependencies using Poetry, you must regenerate `requirements.txt`:
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

## 3. Creating the Space
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Name your Space (e.g., `PRISM`).
3. Select **Streamlit** as the Space SDK.
4. Choose the hardware (the free CPU basic tier is sufficient for PRISM, though inference may be slightly slower).
5. Set visibility to **Public** or **Private** based on your preference.
6. Click **Create Space**.

## 4. Deploying the Code
You can deploy your code in one of two ways:

### Option A: Using the Hugging Face UI
1. Clone the Hugging Face Space repository locally:
   `git clone https://huggingface.co/spaces/<your-username>/PRISM`
2. Copy all PRISM files (excluding `.git`, `datasets/`, `venv/`, etc.) into the cloned folder.
3. Commit and push:
   ```bash
   git add .
   git commit -m "Initial PRISM deployment"
   git push
   ```

### Option B: Connecting a GitHub Repository
If your PRISM code is hosted on GitHub, you can use Hugging Face Spaces' **GitHub integration** or setup **GitHub Actions** to automatically sync your GitHub repository to the Space.

## 5. Model Checkpoints
Hugging Face Spaces have a storage limit. Ensure your `models/checkpoints/` folder contains only the best `.pt` files:
- `cough_detector_best.pt`
- `disease_classifier_v1.pt`
- `temporal_transformer_best.pt`

**Note on LFS:** Since `.pt` files are large, they should be tracked using Git LFS (Large File Storage) when pushing to Hugging Face:
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

## 6. Verification
Once the code is pushed, Hugging Face will automatically begin building the Docker container for the Streamlit app. You can monitor the build logs in the **Logs** tab.

If the build succeeds, the App will become **Running**.

### Troubleshooting
- **Missing Module Error:** Check `requirements.txt` to ensure all dependencies are listed.
- **File Not Found Error:** Ensure that relative paths in the code (e.g., `models/checkpoints/...`) correctly point to the files uploaded to the Space.
- **Theme Not Applying:** Ensure `.streamlit/config.toml` is committed to the repository so the dark theme is applied on Hugging Face.
