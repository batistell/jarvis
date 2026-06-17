import ctypes
import os
import numpy as np

# Find the DLL in .venv
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dll_path = os.path.join(project_root, ".venv", "Lib", "site-packages", "pyrnnoise", "rnnoise.dll")

try:
    lib = ctypes.CDLL(dll_path)
    lib.rnnoise_create.argtypes = [ctypes.c_void_p]
    lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
    lib.rnnoise_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.rnnoise_create.restype = ctypes.c_void_p
    lib.rnnoise_get_frame_size.restype = ctypes.c_int
    lib.rnnoise_process_frame.restype = ctypes.c_float
    
    FRAME_SIZE = lib.rnnoise_get_frame_size()
    state = lib.rnnoise_create(None)
    
    # Let's mock a 30ms frame at 48000Hz (1440 samples)
    # Containing a sine wave at 440Hz with some random noise
    t = np.linspace(0, 0.03, 1440, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.1 * np.random.normal(size=1440)
    mock_audio_48k = (sine + noise).astype(np.float32)
    
    # Scale to [-32768, 32767]
    scaled_audio = mock_audio_48k * 32767.0
    
    # Process in chunks of FRAME_SIZE (480)
    cleaned_scaled = np.zeros_like(scaled_audio)
    speech_probs = []
    
    for i in range(0, len(scaled_audio), FRAME_SIZE):
        chunk = scaled_audio[i:i+FRAME_SIZE].copy()
        ptr = chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # In-place processing
        prob = lib.rnnoise_process_frame(state, ptr, ptr)
        cleaned_scaled[i:i+FRAME_SIZE] = chunk
        speech_probs.append(prob)
        
    # Scale back to [-1.0, 1.0]
    cleaned_audio_48k = cleaned_scaled / 32767.0
    
    # Downsample to 16kHz (take every 3rd sample)
    cleaned_audio_16k = cleaned_audio_48k[::3]
    
    print("Denoising pipeline ran successfully!")
    print("Input shape (48kHz):", mock_audio_48k.shape)
    print("Cleaned shape (48kHz):", cleaned_audio_48k.shape)
    print("Downsampled shape (16kHz):", cleaned_audio_16k.shape)
    print("Speech probabilities for the three 10ms chunks:", speech_probs)
    
    lib.rnnoise_destroy(state)
    
except Exception as e:
    import traceback
    traceback.print_exc()
