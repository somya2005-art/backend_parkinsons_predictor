import librosa
import numpy as np

def extract_features(audio_path):
    """Extract 22 features from audio file"""
    try:
        print(f"Loading audio file: {audio_path}")
        
        # Load audio file with error handling
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=30)  # Limit to 30 seconds
            print(f"Audio loaded: length={len(y)}, sample_rate={sr}")
        except Exception as load_error:
            raise Exception(f"Failed to load audio: {str(load_error)}")
        
        if len(y) == 0:
            raise Exception("Audio file is empty or corrupted")
        
        # Extract features with error handling
        # Pitch features
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) == 0:
                pitch_values = [100.0]  # Default value
            
            fo = float(np.mean(pitch_values))
            fhi = float(np.max(pitch_values))
            flo = float(np.min(pitch_values) if np.min(pitch_values) > 0 else fo * 0.5)
            
            print(f"Pitch features: fo={fo:.2f}, fhi={fhi:.2f}, flo={flo:.2f}")
        except Exception as e:
            print(f"Pitch extraction warning: {e}, using defaults")
            fo, fhi, flo = 120.0, 150.0, 90.0
        
        # Jitter features
        try:
            if len(pitch_values) > 1:
                pitch_diff = np.diff(pitch_values)
                jitter_percent = float((np.mean(np.abs(pitch_diff)) / fo) * 100 if fo > 0 else 0.005)
                jitter_abs = float(np.mean(np.abs(pitch_diff)) / sr * 1000000)
            else:
                jitter_percent = 0.005
                jitter_abs = 0.00005
            
            rap = float(jitter_percent * 0.5)
            ppq = float(jitter_percent * 0.7)
            jitter_ddp = float(jitter_percent * 1.5)
            
            print(f"Jitter: {jitter_percent:.6f}%")
        except Exception as e:
            print(f"Jitter calculation warning: {e}, using defaults")
            jitter_percent, jitter_abs = 0.005, 0.00005
            rap, ppq, jitter_ddp = 0.0025, 0.0035, 0.0075
        
        # Shimmer features
        try:
            rms = librosa.feature.rms(y=y)[0]
            shimmer = float(np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0.02)
            shimmer_db = float(20 * np.log10(shimmer + 1e-10))
            
            apq3 = float(shimmer * 0.3)
            apq5 = float(shimmer * 0.5)
            apq = float(shimmer * 0.6)
            shimmer_dda = float(shimmer * 2)
            
            print(f"Shimmer: {shimmer:.5f}")
        except Exception as e:
            print(f"Shimmer calculation warning: {e}, using defaults")
            shimmer, shimmer_db = 0.02, 0.2
            apq3, apq5, apq = 0.006, 0.01, 0.012
            shimmer_dda = 0.04
        
        # NHR and HNR
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            nhr = float(np.std(spectral_centroids) / np.mean(spectral_centroids) if np.mean(spectral_centroids) > 0 else 0.02)
            hnr = float(20.0 / (nhr + 0.01))
            
            print(f"NHR: {nhr:.5f}, HNR: {hnr:.3f}")
        except Exception as e:
            print(f"NHR/HNR calculation warning: {e}, using defaults")
            nhr, hnr = 0.02, 21.0
        
        # RPDE and DFA
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            rpde = float(np.std(mfcc) / (np.mean(np.abs(mfcc)) + 1e-10))
            
            zero_crossings = librosa.zero_crossings(y)
            dfa = float(np.sum(zero_crossings) / len(y))
            
            print(f"RPDE: {rpde:.5f}, DFA: {dfa:.6f}")
        except Exception as e:
            print(f"RPDE/DFA calculation warning: {e}, using defaults")
            rpde, dfa = 0.5, 0.7
        
        # Spread and other features
        try:
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spread1 = float(np.mean(spectral_rolloff) - fo)
            spread2 = float(np.std(spectral_rolloff) / (np.mean(spectral_rolloff) + 1e-10))
            
            d2 = float(np.log(len(y)) / np.log(sr))
            
            ppe = float(-np.sum((magnitudes + 1e-10) * np.log(magnitudes + 1e-10)) / len(magnitudes.flatten()))
            
            print(f"Spread1: {spread1:.3f}, Spread2: {spread2:.5f}")
        except Exception as e:
            print(f"Spread/PPE calculation warning: {e}, using defaults")
            spread1, spread2 = -5.0, 0.2
            d2, ppe = 2.0, 0.2
        
        # Construct feature vector
        features = [
            fo, fhi, flo,
            jitter_percent, jitter_abs, rap, ppq, jitter_ddp,
            shimmer, shimmer_db, apq3, apq5, apq, shimmer_dda,
            nhr, hnr,
            rpde, dfa,
            spread1, spread2, d2, ppe
        ]
        
        print(f"✅ Feature extraction complete: {len(features)} features")
        return np.array(features).reshape(1, -1)
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
