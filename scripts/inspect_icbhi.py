import os
import zipfile

z = zipfile.ZipFile(r"c:\Users\Dell\Desktop\prism\datasets\raw\icbhi.zip", "r")
files = z.namelist()

# Extensions
exts = set()
for f in files:
    _, ext = os.path.splitext(f)
    if ext:
        exts.add(ext)
print("Extensions:", exts)

# Special metadata files
special = [
    f
    for f in files
    if f.endswith(".csv")
    or "patient" in f.lower()
    or "diagnosis" in f.lower()
    or f.endswith("format.txt")
    or f.endswith("differences.txt")
]
print("\nSpecial files:")
for f in special:
    print(f"  {f}")

# Read filename_format.txt
for f in files:
    if f.endswith("filename_format.txt"):
        print(f"\n--- {f} ---")
        print(z.read(f).decode("utf-8", errors="replace"))
        break

# Read filename_differences.txt
for f in files:
    if f.endswith("filename_differences.txt"):
        print(f"\n--- {f} ---")
        content = z.read(f).decode("utf-8", errors="replace")
        print(content[:1500])
        break

# Sample annotation files
sample = [f for f in files if f.endswith(".txt") and "filename" not in f][:3]
for f in sample:
    print(f"\n--- SAMPLE: {f} ---")
    print(z.read(f).decode("utf-8", errors="replace")[:300])

# Counts
wav_count = len([f for f in files if f.endswith(".wav")])
txt_count = len([f for f in files if f.endswith(".txt") and "filename" not in f])
pids = set()
for f in files:
    b = os.path.basename(f)
    if b.endswith(".wav"):
        pids.add(b.split("_")[0])
print(
    f"\nWAVs: {wav_count}, Annotation TXTs: {txt_count}, Unique Patients: {len(pids)}"
)
print(f"Sample Patient IDs: {sorted(list(pids))[:10]}")
z.close()
