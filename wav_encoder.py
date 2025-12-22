import struct
import numpy as np

def encode_wav(audio_data, sample_rate=22050):
    """Encode numpy array to WAV format bytes"""
    
    # Normalize audio
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Convert to 16-bit PCM
    audio_data = np.int16(audio_data * 32767)
    
    # WAV file parameters
    num_channels = 1
    bytes_per_sample = 2
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    
    # WAV file header
    wav_bytes = b'RIFF'
    wav_bytes += struct.pack('<I', 36 + len(audio_data) * bytes_per_sample)
    wav_bytes += b'WAVE'
    
    # Subchunk1 (fmt)
    wav_bytes += b'fmt '
    wav_bytes += struct.pack('<I', 16)  # Subchunk1Size
    wav_bytes += struct.pack('<H', 1)   # AudioFormat (PCM)
    wav_bytes += struct.pack('<H', num_channels)
    wav_bytes += struct.pack('<I', sample_rate)
    wav_bytes += struct.pack('<I', byte_rate)
    wav_bytes += struct.pack('<H', block_align)
    wav_bytes += struct.pack('<H', 16)  # BitsPerSample
    
    # Subchunk2 (data)
    wav_bytes += b'data'
    wav_bytes += struct.pack('<I', len(audio_data) * bytes_per_sample)
    wav_bytes += audio_data.tobytes()
    
    return wav_bytes
