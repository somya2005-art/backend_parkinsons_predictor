import numpy as np
import pickle
from app import extract_features_from_audio

print("="*70)
print("🎤 TESTING YOUR RECORDED AUDIO")
print("="*70)

# Your audio file path
audio_path = r"C:\Users\Somya\Downloads\Recording.wav"  # Update if different

print(f"\nAudio file: {audio_path}")
print(f"Extracting features...")

try:
    # Extract features
    features = extract_features_from_audio(audio_path)
    print(f"✅ Features extracted: {features.shape}")
    print(f"\nFeature values:")
    print(features)
    
    # Load model
    with open('model_audio.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler_audio.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"\n✅ Model loaded")
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = float(probabilities[prediction] * 100)
    
    print(f"\n{'='*70}")
    print("🎯 PREDICTION RESULT")
    print(f"{'='*70}")
    print(f"Prediction: {prediction} ({'Parkinson\'s' if prediction==1 else 'Healthy'})")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Probabilities:")
    print(f"  Healthy: {probabilities[0]*100:.2f}%")
    print(f"  Parkinson's: {probabilities[1]*100:.2f}%")
    print(f"{'='*70}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
