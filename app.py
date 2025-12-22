from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import tempfile
import joblib
import parselmouth
from parselmouth.praat import call
from collections import defaultdict
import noisereduce as nr
from scipy.io import wavfile
import tempfile

import os

# Force pydub to use these binaries
os.environ["FFMPEG_BINARY"]  = r"C:\Users\Somya\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
os.environ["FFPROBE_BINARY"] = r"C:\Users\Somya\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe"

from pydub import AudioSegment, utils
from pydub.utils import which


FFMPEG_BIN   = r"C:\Users\Somya\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
FFPROBE_BIN  = r"C:\Users\Somya\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe"

AudioSegment.converter = FFMPEG_BIN
AudioSegment.ffprobe   = FFPROBE_BIN

def get_prober_name():
    return FFPROBE_BIN

utils.get_prober_name = get_prober_name  # <-- crucial on Windows


app = Flask(__name__)
CORS(app)

model = None
scaler = None

# MOVE THIS OUTSIDE THE FUNCTION so it persists across requests
person_stats = defaultdict(lambda: {"n": 0, "sum_pd": 0.0})

def load_models():
    global model, scaler
    if model is None or scaler is None:
        from joblib import load
        model = load('model_audio.pkl')
        scaler = load('scaler_audio.pkl')
        print(f"✅ Audio model loaded: expects {model.n_features_in_} features")

# Load manual model
try:
    with open('model_manual.pkl', 'rb') as f:
        model_manual = pickle.load(f)
    with open('scaler_manual.pkl', 'rb') as f:
        scaler_manual = pickle.load(f)
    print("✅ Manual model loaded")
except Exception:
    model_manual = None
    scaler_manual = None


def convert_to_valid_wav(audio_path):
    """
    Convert any audio format to valid WAV that Parselmouth can read
    """
    try:
        file_ext = os.path.splitext(audio_path)[1].lower()
        
        if file_ext == '.wav':
            try:
                test_sound = parselmouth.Sound(audio_path)
                return audio_path
            except:
                print("   ⚠️ WAV file has invalid header, converting...")
        
        print(f"   🔄 Converting {file_ext} to valid WAV...")
        
        if file_ext == '.webm':
            audio = AudioSegment.from_file(audio_path, format='webm')
        elif file_ext == '.m4a':
            audio = AudioSegment.from_file(audio_path, format='mp4')
        elif file_ext == '.mp3':
            audio = AudioSegment.from_file(audio_path, format='mp3')
        else:
            print("DEBUG converter:", AudioSegment.converter)
            print("DEBUG ffprobe:", AudioSegment.ffprobe)
            audio = AudioSegment.from_file(audio_path)
        
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Single, correct temp file block
        temp_wav = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='_converted.wav',
            dir=os.path.dirname(audio_path)
        )
        temp_wav_name = temp_wav.name
        temp_wav.close()  # ensure closed before export

        audio.export(temp_wav_name, format='wav')
        
        print("   ✅ Converted successfully")
        return temp_wav_name

    except Exception as e:
        print(f"   ❌ Conversion failed: {e}")
        raise Exception(f"Could not convert audio file: {e}")


def extract_simple_features(audio_path):
    """Extract features matching Italian Parkinson's dataset"""
    try:
        print(f"   Loading: {audio_path}")

        # Load with Parselmouth (Praat in Python)
        sound = parselmouth.Sound(audio_path)

        # Get pitch object
        pitch = sound.to_pitch()

        # Get point process for jitter/shimmer
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)

        # 1-3: Pitch features
        fo = call(pitch, "Get mean", 0, 0, "Hertz")
        fhi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        flo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

        # Handle invalid pitch
        if fo == 0 or np.isnan(fo):
            fo = 150.0
        if fhi == 0 or np.isnan(fhi):
            fhi = fo * 1.2
        if flo == 0 or np.isnan(flo):
            flo = fo * 0.8

        # 4-8: Jitter features
        jitter_percent = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ddp = rap * 3

        # 9-14: Shimmer features
        shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_dda = apq3 * 3

        # 15-16: Harmonicity features
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        nhr = 1 / (hnr + 0.00001)

        # BUILD FEATURE ARRAY
        features = [
            fo, fhi, flo,
            jitter_percent, jitter_abs, rap, ppq, jitter_ddp,
            shimmer, shimmer_db, apq3, apq5, apq, shimmer_dda,
            nhr, hnr
        ]

        print(f"   ✅ F0={fo:.1f}Hz, Jitter={jitter_percent*100:.3f}%, Shimmer={shimmer*100:.3f}%, HNR={hnr:.2f}")

        return np.array(features, dtype=float).reshape(1, -1)

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.route('/predict/manual', methods=['POST'])
def predict_manual():
    try:
        data = request.json
        features_dict = data.get('features', {})

        feature_names = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
            'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
            'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
            'Shimmer:APQ3', 'Shimmer:APQ5',
            'MDVP:APQ', 'Shimmer:DDA',
            'NHR', 'HNR',
            'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ]


        features = np.array([
            float(str(features_dict[name]).replace(',', '.')) for name in feature_names
        ]).reshape(1, -1)

        features_scaled = scaler_manual.transform(features)
        prediction = int(model_manual.predict(features_scaled)[0])
        probabilities = model_manual.predict_proba(features_scaled)[0]

        return jsonify({
            'prediction': prediction,
            'confidence': float(probabilities[prediction] * 100)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main audio prediction endpoint"""
    try:
        global person_stats  # Access the global person_stats
        
        load_models()

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        person_id = request.form.get('person_id', 'anonymous')

        # Save uploaded file with original extension
        file_ext = os.path.splitext(audio_file.filename)[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        print(f"\n🎤 Analyzing audio for person: {person_id}")
        
        # Convert to valid WAV if needed
        converted_path = None
        try:
            wav_path = convert_to_valid_wav(tmp_path)
            if wav_path != tmp_path:
                converted_path = wav_path  # just track path

            # Extract features from converted WAV
            features = extract_simple_features(wav_path)

        finally:
            # Clean up original uploaded file ONLY
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except PermissionError:
                    # File is still locked by ffmpeg/subprocess; ignore
                    pass
            # Do NOT remove converted_path here



        print(f"\n📊 RAW FEATURES (first clip for debug):")
        print(f"   F0={features[0][0]:.2f}, Jitter={features[0][3]:.6f}, Shimmer={features[0][8]:.6f}")
        print(f"   NHR={features[0][14]:.6f}, HNR={features[0][15]:.2f}")

        # SCALE with scaler
        features_scaled = scaler.transform(features)

        # Probabilities for this clip
        probabilities = model.predict_proba(features_scaled)[0]
        p_healthy_clip = float(probabilities[0])
        p_pd_clip = float(probabilities[1])

        # ---- PER-PERSON AGGREGATION ----
        stats = person_stats[person_id]
        stats["n"] += 1
        stats["sum_pd"] += p_pd_clip
        mean_pd = stats["sum_pd"] / stats["n"]

        # Convert to percentages
        healthy_prob_clip = p_healthy_clip * 100.0
        parkinsons_prob_clip = p_pd_clip * 100.0
        mean_pd_percent = mean_pd * 100.0

        # ---- DECISION BASED ON PERSON-LEVEL MEAN PD ----
        if mean_pd >= 0.75:
            stage = "Parkinson's Detected"
            prediction = 1
            confidence = (mean_pd - 0.5) * 200
        elif 0.55 <= mean_pd < 0.75:
            stage = "Borderline (Possible Parkinson's)"
            prediction = 1
            confidence = (mean_pd - 0.5) * 200
        elif 0.35 <= mean_pd < 0.55:
            stage = "Borderline (Inconclusive)"
            prediction = 0
            confidence = (0.5 - abs(mean_pd - 0.5)) * 200
        else:
            stage = "Healthy"
            prediction = 0
            confidence = (0.5 - mean_pd) * 200

        confidence = max(0.0, min(confidence, 100.0))

        print(f"\n🎯 PREDICTION (clip {stats['n']} for {person_id}):")
        print(f"  Clip Probabilities: [Healthy={healthy_prob_clip:.2f}%, Parkinsons={parkinsons_prob_clip:.2f}%]")
        print(f"  Mean PD probability for person: {mean_pd_percent:.2f}%")
        print(f"  Stage (person-level): {stage}")
        print(f"  Confidence: {confidence:.1f}%")

        return jsonify({
            "prediction": int(prediction),
            "stage": stage,
            "confidence": float(confidence),
            "probabilities": {
                "Healthy_clip": round(healthy_prob_clip, 2),
                "Parkinsons_clip": round(parkinsons_prob_clip, 2),
                "Mean_PD_person": round(mean_pd_percent, 2),
                "Clips_count": stats["n"],
            },
            "voice_features": {
    "F0_Hz": {
        "value": round(float(features[0][0]), 2),
        "label": "Average pitch (F0)",
        "description": "Average speaking pitch in Hertz; higher for higher voices."  # [web:192]
    },
    "Fhi_Hz": {
        "value": round(float(features[0][1]), 2),
        "label": "Max pitch (Fhi)",
        "description": "Highest pitch reached in the sample."  # [web:192]
    },
    "Flo_Hz": {
        "value": round(float(features[0][2]), 2),
        "label": "Min pitch (Flo)",
        "description": "Lowest pitch reached in the sample."  # [web:192]
    },
    "Jitter_percent": {
        "value": round(float(features[0][3] * 100), 4),
        "label": "Jitter (%)",
        "description": "Cycle‑to‑cycle pitch variation; higher values mean less stable vocal fold vibration."  # [web:192][web:195]
    },
    "Jitter_abs": {
        "value": round(float(features[0][4]), 6),
        "label": "Jitter (Abs)",
        "description": "Absolute pitch period variation in seconds."  # [web:192]
    },
    "Shimmer": {
        "value": round(float(features[0][8]), 6),
        "label": "Shimmer",
        "description": "Cycle‑to‑cycle loudness variation; higher values mean less stable voice amplitude."  # [web:192][web:195]
    },
    "Shimmer_dB": {
        "value": round(float(features[0][9]), 4),
        "label": "Shimmer (dB)",
        "description": "Shimmer expressed in decibels."  # [web:192]
    },
    "NHR": {
        "value": round(float(features[0][14]), 6),
        "label": "Noise-to-Harmonics Ratio (NHR)",
        "description": "Amount of noise relative to harmonic (voiced) energy; higher suggests noisier, rougher voice."  # [web:192][web:178]
    },
    "HNR": {
        "value": round(float(features[0][15]), 2),
        "label": "Harmonics-to-Noise Ratio (HNR)",
        "description": "How strongly the voice is dominated by clear harmonic components; lower values indicate more noise/hoarseness."  # [web:192][web:178]
    }
}

        })

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ADD THIS: Reset person stats endpoint (useful for testing)
@app.route('/api/reset_person/<person_id>', methods=['POST'])
def reset_person(person_id):
    global person_stats
    if person_id in person_stats:
        del person_stats[person_id]
        return jsonify({'message': f'Reset stats for {person_id}'})
    return jsonify({'message': f'No stats found for {person_id}'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("\n🚀 Parkinson's Detection Server")
    print("=" * 50)
    app.run(debug=True, port=5000)
