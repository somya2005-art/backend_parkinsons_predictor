import os
from pydub import AudioSegment

FFMPEG_BIN = r"C:\Users\Somya\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
FFPROBE_BIN = r"C:\Users\Somya\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe"

print(f"ffmpeg.exe exists: {os.path.exists(FFMPEG_BIN)}")
print(f"ffprobe.exe exists: {os.path.exists(FFPROBE_BIN)}")

AudioSegment.converter = FFMPEG_BIN
AudioSegment.ffprobe = FFPROBE_BIN

# Try to use it
try:
    audio = AudioSegment.silent(duration=1000)
    audio.export("test.wav", format="wav")
    print("✅ FFmpeg working!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
