from feature_extraction import extract_features_from_audio
import pickle
import numpy as np

# Extract features from your test audio
features = extract_features_from_audio('my_voice.wav')

print(f"Extracted {len(features)} features:")
print(f"First 5: {features[:5]}")
print(f"Last 5: {features[-5:]}")

# Load model and test
with open('parkinsons_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Test with extracted features
scaled = scaler.transform(features.reshape(1, -1))
pred = model.predict(scaled)[0]
proba = model.predict_proba(scaled)[0]

print(f"\nPrediction: {pred} ({'Parkinson' if pred==1 else 'Healthy'})")
print(f"Probability: [{proba[0]*100:.1f}%, {proba[1]*100:.1f}%]")
