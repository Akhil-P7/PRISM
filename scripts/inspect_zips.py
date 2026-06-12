import os
import zipfile

import pandas as pd

raw_dir = r"c:\Users\Dell\Desktop\prism\datasets\raw"
datasets = ["coughvid.zip", "coswara.zip", "icbhi.zip"]

for ds in datasets:
    path = os.path.join(raw_dir, ds)
    print(f"\n================= {ds} =================")
    if not os.path.exists(path):
        print("NOT FOUND")
        continue

    with zipfile.ZipFile(path, "r") as z:
        files = z.namelist()
        metadata_files = [f for f in files if f.endswith(".csv") or f.endswith(".txt")]
        print("Metadata files found:")
        for f in metadata_files:
            print(f" - {f}")
            if f.endswith(".csv"):
                try:
                    with z.open(f) as csv_file:
                        df = pd.read_csv(csv_file, nrows=0)
                        print(f"   Columns: {list(df.columns)}")
                except Exception as e:
                    print(f"   Could not read columns: {e}")
