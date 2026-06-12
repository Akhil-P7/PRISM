import os
import zipfile

z = zipfile.ZipFile(r"c:\Users\Dell\Desktop\prism\datasets\raw\coswara.zip", "r")
files = z.namelist()

exts = set()
for f in files:
    _, ext = os.path.splitext(f)
    if ext:
        exts.add(ext)
print("Coswara extensions:", exts)

audio_exts = (".wav", ".mp3", ".ogg", ".flac", ".webm")
audio = [f for f in files if f.endswith(audio_exts)][:5]
print(f"Sample audio: {audio}")
print(f"Total audio: {len([f for f in files if f.endswith(audio_exts)])}")

# Check folder structure
dirs = set()
for f in files:
    parts = f.split("/")
    if len(parts) >= 3:
        dirs.add("/".join(parts[:3]))
sample_dirs = sorted(list(dirs))[:10]
print(f"Sample dirs: {sample_dirs}")
z.close()
