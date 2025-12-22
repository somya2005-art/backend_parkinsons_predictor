from pydub import AudioSegment

print("Converting M4A to proper WAV...")

try:
    # Load M4A
    audio = AudioSegment.from_file('Recording.m4a', format='m4a')
    
    # Convert to mono, 22050 Hz
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(22050)
    
    # Export as proper WAV
    audio.export('my_voice.wav', format='wav')
    
    print("✅ Converted successfully to my_voice.wav")
    print(f"   Duration: {len(audio)/1000:.2f}s")
    print(f"   Sample rate: 22050 Hz")
    print(f"   Channels: 1 (mono)")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTry installing ffmpeg:")
    print("  choco install ffmpeg  (Windows)")
