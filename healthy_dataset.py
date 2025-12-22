import os
import numpy as np
import pandas as pd
from app import extract_simple_features

rows = []

# List of folders containing healthy recordings
BASE_DIRS = [
    r"C:\Users\Somya\Downloads\parkinsons_project\backend\combined_healthy",
    r"C:\Users\Somya\Downloads\parkinsons_project\backend\child_healthy_combined"
]

print("Extracting features from healthy recordings...\n")

# Loop through each folder
for BASE_DIR in BASE_DIRS:
    print(f"📁 Processing folder: {BASE_DIR}")
    
    if not os.path.exists(BASE_DIR):
        print(f"⚠️  Folder not found, skipping: {BASE_DIR}\n")
        continue
    
    for fname in os.listdir(BASE_DIR):
        if not fname.lower().endswith((".wav", ".mp3", ".m4a", ".webm")):
            continue
        
        fpath = os.path.join(BASE_DIR, fname)
        try:
            feats = extract_simple_features(fpath)[0]  # (16,)
            rows.append({
                "filename": fname,
                "label": 0,  # healthy
                "fo": feats[0],
                "fhi": feats[1],
                "flo": feats[2],
                "jitter_percent": feats[3],
                "jitter_abs": feats[4],
                "rap": feats[5],
                "ppq": feats[6],
                "jitter_ddp": feats[7],
                "shimmer": feats[8],
                "shimmer_db": feats[9],
                "apq3": feats[10],
                "apq5": feats[11],
                "apq": feats[12],
                "shimmer_dda": feats[13],
                "nhr": feats[14],
                "hnr": feats[15],
            })
            print(f"✅ {fname}")
        except Exception as e:
            print(f"❌ {fname} -> {e}")
    
    print()  # blank line between folders

df_my = pd.DataFrame(rows)
df_my.to_csv("my_healthy_16features.csv", index=False)
print(f"✅ Saved {len(df_my)} healthy samples to my_healthy_16features.csv")
