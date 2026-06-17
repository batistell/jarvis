import ctypes
import os
import numpy as np

# Find the DLL in .venv
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dll_path = os.path.join(project_root, ".venv", "Lib", "site-packages", "pyrnnoise", "rnnoise.dll")

print("Checking DLL path:", dll_path)
print("Exists:", os.path.exists(dll_path))

try:
    lib = ctypes.CDLL(dll_path)
    print("DLL loaded successfully via ctypes!")
    
    # Configure argtypes and restypes
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
    print("FRAME_SIZE:", FRAME_SIZE)
    
    # Try creating state
    state = lib.rnnoise_create(None)
    print("State created:", state)
    
    # Test mono frame processing
    # RNNoise expects 480 samples at 48kHz
    dummy_frame = np.zeros(FRAME_SIZE, dtype=np.float32)
    ptr = dummy_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    speech_prob = lib.rnnoise_process_frame(state, ptr, ptr)
    print("Processed dummy float32 frame! Speech prob:", speech_prob)
    
    # Destroy state
    lib.rnnoise_destroy(state)
    print("State destroyed successfully.")
    
except Exception as e:
    import traceback
    traceback.print_exc()
