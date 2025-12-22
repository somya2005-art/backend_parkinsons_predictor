import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("="*70)
print("🔥 COMPLETE FRESH RETRAIN (FIXED)")
print("="*70)

# Load CSV
df = pd.read_csv('parkinsons.csv')

# Extract features and labels - CONVERT TO NUMPY IMMEDIATELY
X = df.drop(columns=['name', 'status'], axis=1).values  # ← .values is critical
y = df['status'].values  # ← .values is critical

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Healthy: {sum(y==0)}, Parkinson's: {sum(y==1)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2, stratify=y
)

# Create fresh scaler - fit on NUMPY array (no feature names!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nScaler fitted:")
print(f"  Mean (first 3): {scaler.mean_[:3]}")
print(f"  Scale (first 3): {scaler.scale_[:3]}")

# Train model
print("\nTraining SVM...")
model = SVC(
    kernel='rbf',
    C=10,
    gamma=0.1,
    probability=True,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)

print(f"\n{'='*70}")
print("RESULTS:")
print(f"{'='*70}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"\n{classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinsons'])}")

# CRITICAL TEST: Verify model works with known data
print(f"\n{'='*70}")
print("VERIFICATION TEST:")
print(f"{'='*70}")

# Use actual test samples (first healthy and first Parkinson's from test set)
healthy_idx = np.where(y_test == 0)[0][0]
parkinsons_idx = np.where(y_test == 1)[0][0]

test_samples = [
    ("Test Set Healthy", X_test_scaled[healthy_idx:healthy_idx+1], 0),
    ("Test Set Parkinson's", X_test_scaled[parkinsons_idx:parkinsons_idx+1], 1)
]

for name, features, expected in test_samples:
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    result = "✅" if pred == expected else "❌"
    
    print(f"\n{name}:")
    print(f"  Expected: {expected}, Got: {pred} {result}")
    print(f"  Confidence: {proba[pred]*100:.1f}%")

# Test with diverse synthetic inputs
print(f"\n{'='*70}")
print("DIVERSITY TEST:")
print(f"{'='*70}")

synthetic_tests = [
    ("All high", np.ones(22) * 200),
    ("All medium", np.ones(22) * 100),
    ("All low", np.ones(22) * 50)
]

predictions = []
for name, raw in synthetic_tests:
    scaled = scaler.transform(raw.reshape(1, -1))
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    predictions.append(pred)
    print(f"{name}: Pred={pred}, Proba=[{proba[0]*100:.1f}%, {proba[1]*100:.1f}%]")

print(f"\nUnique predictions: {len(set(predictions))}")

# Save files
print(f"\n{'='*70}")
print("SAVING FILES:")
print(f"{'='*70}")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ model.pkl (SVC object)")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ scaler.pkl (StandardScaler object)")

model_info = {
    'model_name': 'SVM (RBF)',
    'test_accuracy': test_acc,
    'num_features': 22,
    'has_probability': True,
    'sklearn_version': '1.6.1'
}

with open('model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("✅ model_info.pkl")

print(f"\n{'='*70}")
print("✅ DONE!")
print(f"{'='*70}")
print("If VERIFICATION TEST shows ✅✅, download files!")
print("If it shows ❌, the dataset itself has issues.")
print(f"{'='*70}")
