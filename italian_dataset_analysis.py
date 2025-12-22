import os
import librosa
import numpy as np
import pandas as pd

print("="*70)
print("📊 ITALIAN PARKINSON'S DATASET ANALYSIS")
print("="*70)

# Dataset path
DATASET_PATH = r"C:\Users\Somya\Downloads\Italian Parkinson's Voice and speech"

folders = {
    "Young Healthy": os.path.join(DATASET_PATH, "15 Young Healthy Control"),
    "Elderly Healthy": os.path.join(DATASET_PATH, "22 Elderly Healthy Control"),
    "Parkinsons": os.path.join(DATASET_PATH, "28 People with Parkinson's disease")
}

# Analyze each folder
analysis_data = []

for label, folder_path in folders.items():
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        continue
    
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.m4a', '.ogg'))]
    
    print(f"\n{label}:")
    print(f"  Number of files: {len(audio_files)}")
    
    if len(audio_files) > 0:
        # Sample first file
        sample_file = os.path.join(folder_path, audio_files[0])
        try:
            y, sr = librosa.load(sample_file, sr=None, duration=5)
            duration = len(y) / sr
            
            print(f"  Sample file: {audio_files[0]}")
            print(f"  Sample rate: {sr} Hz")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Audio length: {len(y)} samples")
            
            # Check multiple files
            durations = []
            sample_rates = []
            for audio_file in audio_files[:10]:
                try:
                    y_temp, sr_temp = librosa.load(os.path.join(folder_path, audio_file), sr=None, duration=5)
                    durations.append(len(y_temp) / sr_temp)
                    sample_rates.append(sr_temp)
                except Exception as e:
                    print(f"    ⚠️ Error loading {audio_file}: {e}")
            
            if durations:
                print(f"  Avg duration (first 10): {np.mean(durations):.2f}s")
                print(f"  Duration range: {np.min(durations):.2f}s - {np.max(durations):.2f}s")
                print(f"  Sample rates: {set(sample_rates)}")
            
            analysis_data.append({
                'Group': label,
                'Num_Files': len(audio_files),
                'Sample_Rate': sr,
                'Avg_Duration': np.mean(durations) if durations else 0,
                'File_Types': list(set([f.split('.')[-1] for f in audio_files]))
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")

# Summary
print("\n" + "="*70)
print("📈 DATASET SUMMARY")
print("="*70)

if analysis_data:
    df_summary = pd.DataFrame(analysis_data)
    print(df_summary.to_string(index=False))
    
    total_healthy = df_summary[df_summary['Group'] != 'Parkinsons']['Num_Files'].sum()
    total_parkinsons = df_summary[df_summary['Group'] == 'Parkinsons']['Num_Files'].sum()
    
    print(f"\n✅ Total Healthy files: {total_healthy}")
    print(f"❌ Total Parkinson's files: {total_parkinsons}")
    print(f"📊 Total files: {total_healthy + total_parkinsons}")
    print(f"📊 Class ratio: {total_parkinsons/total_healthy:.2f}:1 (PD:Healthy)")
    
    if abs(total_healthy - total_parkinsons) < total_healthy * 0.3:
        print("✅ Dataset is reasonably BALANCED")
    else:
        print("⚠️  Dataset is IMBALANCED - we'll handle this in training")

print("\n" + "="*70)
print("✅ Analysis complete! Next step: Extract features from all audio files")
print("="*70)
