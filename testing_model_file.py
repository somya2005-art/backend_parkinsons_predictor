from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load YOUR working model for manual input
try:
    with open('model_manual.pkl', 'rb') as f:
        model_manual = pickle.load(f)
    with open('scaler_manual.pkl', 'rb') as f:
        scaler_manual = pickle.load(f)
    with open('model_info_manual.pkl', 'rb') as f:
        model_info = pickle.load(f)
    print("✅ Manual input model loaded successfully")
    print(f"   Model: {model_info['model_name']}")
    print(f"   Test Accuracy: {model_info['test_accuracy']*100:.2f}%")
except Exception as e:
    model_manual = None
    scaler_manual = None
    model_info = None
    print(f"⚠️ Could not load manual model: {e}")

@app.route('/predict/manual', methods=['POST'])
def predict_manual():
    try:
        data = request.json
        features_dict = data.get('features', {})
        
        # Feature names in correct order
        feature_names = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
            'spread1', 'spread2', 'D2', 'PPE'
        ]
        
        # Helper function to handle comma/period decimal separators
        def parse_float(value):
            """Convert string to float, handling both comma and period as decimal separator"""
            if isinstance(value, (int, float)):
                return float(value)
            # Replace comma with period
            value_str = str(value).replace(',', '.')
            return float(value_str)
        
        # Convert to numpy array in correct order
        features = np.array([parse_float(features_dict[name]) for name in feature_names]).reshape(1, -1)
        
        if model_manual is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Scale and predict using YOUR model
        features_scaled = scaler_manual.transform(features)
        prediction = int(model_manual.predict(features_scaled)[0])
        probabilities = model_manual.predict_proba(features_scaled)[0]
        confidence = float(probabilities[prediction] * 100)
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'healthy': float(probabilities[0] * 100),
                'parkinsons': float(probabilities[1] * 100)
            },
            'model_info': {
                'name': model_info['model_name'] if model_info else 'SVM (RBF)',
                'test_accuracy': model_info['test_accuracy'] * 100 if model_info else 92.31
            }
        })
        
    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_manual is not None,
        'model_info': model_info if model_manual else None
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    # Placeholder for audio analysis (will implement with Italian dataset)
    return jsonify({'error': 'Audio analysis not yet implemented. Use manual input for now.'}), 501

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 FLASK SERVER STARTING")
    print("="*70)
    app.run(debug=True, port=5000)
