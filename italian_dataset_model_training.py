# retrain_with_real_audio.py (UPDATED - add NaN handling)
import os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from app import extract_simple_features

print("="*70)
print("🤖 RETRAINING: Real Italian PD Audio + Your Healthy Audio")
print("="*70)

# 1. Extract features from Italian PD audio files
print("\n📁 Extracting features from Italian PD recordings...")

PD_FOLDER = r"C:\Users\Somya\Downloads\parkinsons_project\backend\28 People with Parkinson's disease"

pd_rows = []
file_count = 0

for root, dirs, files in os.walk(PD_FOLDER):
    for fname in files:
        if not fname.lower().endswith((".wav", ".mp3", ".m4a", ".webm", ".ogg")):
            continue
        
        fpath = os.path.join(root, fname)
        try:
            feats = extract_simple_features(fpath)[0]
            pd_rows.append(feats)
            file_count += 1
            if file_count % 10 == 0:
                print(f"  Processed {file_count} files...")
        except Exception as e:
            print(f"❌ {fname} -> {e}")

X_pd = np.array(pd_rows)
y_pd = np.ones(len(X_pd), dtype=int)

print(f"\n✅ Italian PD audio samples extracted: {len(X_pd)}")

# 2. Load your healthy samples
df_my = pd.read_csv("my_healthy_16features.csv")
X_h = df_my[[
    "fo","fhi","flo",
    "jitter_percent","jitter_abs","rap","ppq","jitter_ddp",
    "shimmer","shimmer_db","apq3","apq5","apq","shimmer_dda",
    "nhr","hnr"
]].values
y_h = np.zeros(len(X_h), dtype=int)

print(f"Your healthy samples: {len(X_h)}")

# 3. Combine
X = np.vstack([X_h, X_pd])
y = np.concatenate([y_h, y_pd])

print(f"\nCombined (before cleaning): {X.shape[0]} samples, {X.shape[1]} features")

# ✅ NEW: Remove rows with NaN or inf values
valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
X_clean = X[valid_mask]
y_clean = y[valid_mask]

removed_count = len(X) - len(X_clean)
print(f"⚠️  Removed {removed_count} samples with invalid features (NaN/inf)")
print(f"✅ Clean dataset: {X_clean.shape[0]} samples")
print(f"  Healthy: {(y_clean==0).sum()}")
print(f"  PD: {(y_clean==1).sum()}")

if len(X_clean) < 50:
    print("❌ ERROR: Too few valid samples after cleaning!")
    exit()

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, stratify=y_clean, random_state=42
)

# 5. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train SVM
print("\n" + "="*70)
print("🎯 TRAINING SVM")
print("="*70)

svc = SVC(
    kernel="rbf",
    C=10,
    gamma=0.01,
    probability=True,
    class_weight="balanced",
    random_state=42,
)
svc.fit(X_train_scaled, y_train)

# 7. Calibrate
calibrated = CalibratedClassifierCV(svc, method="sigmoid", cv=5)
calibrated.fit(X_train_scaled, y_train)

# 8. Evaluate
y_pred = calibrated.predict(X_test_scaled)
print("\n" + "="*70)
print("TEST RESULTS")
print("="*70)
print(classification_report(y_test, y_pred, target_names=["Healthy","PD"]))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

probs_test = calibrated.predict_proba(X_test_scaled)[:,1]
print(f"\nAUC: {roc_auc_score(y_test, probs_test):.4f}")

# 9. Save
dump(calibrated, "model_audio.pkl")
dump(scaler, "scaler_audio.pkl")

print("\n" + "="*70)
print("✅ SAVED NEW MODEL")
print("="*70)
print("  model_audio.pkl")
print("  scaler_audio.pkl")
print(f"  Features: {X_clean.shape[1]}")
print("\n🔄 Restart Flask and test Italian PD recording!")
