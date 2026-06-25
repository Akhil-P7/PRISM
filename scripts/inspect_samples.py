import zipfile

import pandas as pd

# === COUGHVID ===
print("=" * 60)
print("COUGHVID - metadata_compiled.csv")
print("=" * 60)
z = zipfile.ZipFile(r"c:\Users\Dell\Desktop\prism\datasets\raw\coughvid.zip", "r")
with z.open("public_dataset_v3/coughvid_20211012/metadata_compiled.csv") as f:
    df = pd.read_csv(f, nrows=5)
    print(f"Total columns: {len(df.columns)}")
    print("\nKey columns sample:")
    key_cols = [
        "uuid",
        "age",
        "gender",
        "respiratory_condition",
        "fever_muscle_pain",
        "status",
        "cough_detected",
    ]
    for c in key_cols:
        if c in df.columns:
            print(f"  {c}: {df[c].tolist()}")

# Count total rows
with z.open("public_dataset_v3/coughvid_20211012/metadata_compiled.csv") as f:
    total = sum(1 for _ in f) - 1
    print(f"\nTotal rows: {total}")
z.close()

# === COSWARA ===
print("\n" + "=" * 60)
print("COSWARA - combined_data.csv")
print("=" * 60)
z = zipfile.ZipFile(r"c:\Users\Dell\Desktop\prism\datasets\raw\coswara.zip", "r")
with z.open("combined_data.csv") as f:
    df = pd.read_csv(f, nrows=5)
    print(f"Total columns: {len(df.columns)}")
    print("\nKey columns sample:")
    key_cols = ["id", "a", "g", "covid_status", "smoker", "asthma", "fever", "cough"]
    for c in key_cols:
        if c in df.columns:
            print(f"  {c}: {df[c].tolist()}")

with z.open("combined_data.csv") as f:
    total = sum(1 for _ in f) - 1
    print(f"\nTotal rows: {total}")

# Check what audio files look like
audio_files = [f for f in z.namelist() if f.endswith(".wav")][:5]
print(f"\nSample audio paths: {audio_files}")
print(f"Total audio files: {len([f for f in z.namelist() if f.endswith('.wav')])}")
z.close()
