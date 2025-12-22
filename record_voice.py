import sounddevice as sd
import scipy.io.wavfile as wav
import time

print("="*70)
print("🎙️  VOICE RECORDER")
print("="*70)
print("\n📋 Instructions:")
print("  1. Get ready to say 'Ahhhhh'")
print("  2. Take a deep breath")
print("  3. Say 'Ahhhhh' steadily for 5 seconds")
print("\nRecording starts in:")

for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

print("\n🎙️ RECORDING NOW! Say 'Ahhhhh'...\n")

fs = 22050
duration = 5

recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')

# Show countdown
for i in range(5, 0, -1):
    print(f"  {i} seconds remaining...")
    time.sleep(1)

sd.wait()

print("\n✅ Recording complete!")
print("💾 Saving...")

wav.write('my_voice.wav', fs, recording)

print("✅ Saved as my_voice.wav")
print(f"   Duration: {duration}s")
print(f"   Sample rate: {fs}Hz")
print(f"   Channels: 1 (mono)")
print(f"\n{'='*70}")
print("Now run: python test_features.py my_voice.wav")
print(f"{'='*70}")
