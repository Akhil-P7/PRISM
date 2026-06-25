"""Quick inspection of the cough detector manifest."""

import os

import numpy as np
import pandas as pd

df = pd.read_csv("datasets/features/manifest.csv")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print("\nis_cough distribution:")
print(df["is_cough"].value_counts())
subjects = df["subject_id"].nunique()
print(f"\nUnique subjects: {subjects}")
print(f"Sample mel_path: {df['mel_path'].iloc[0]}")
silent = df["is_silent"].sum()
print(f"Silent segments: {silent}")

# Check if mel files actually exist
sample_mel = df["mel_path"].iloc[0]
full_path = os.path.join("datasets/features", sample_mel)
print(f"\nSample mel file exists: {os.path.exists(full_path)}")

# Check a few mel file shapes
if os.path.exists(full_path):
    mel = np.load(full_path)
    print(f"Mel shape: {mel.shape}")
