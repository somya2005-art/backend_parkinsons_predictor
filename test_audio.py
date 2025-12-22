import numpy as np
import soundfile as sf

# Create a proper voice-like recording (5 seconds)
duration = 5
sample_rate = 22050
frequency = 150  # Voice frequency

# Generate sine wave
t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.3 * np.sin(2 * np.pi * frequency * t)

# Add slight variation (like natural voice)
audio += 0.05 * np.sin(2 * np.pi * frequency * 2 * t)

# Save it properly
sf.write('test_audio.wav', audio, sample_rate)
print("✓ test_audio.wav created successfully!")

# Verify it works
data, sr = sf.read('test_audio.wav')
print(f"✓ File verified: {len(data)} samples at {sr} Hz")
