import pytest
import numpy as np
from jarvis.stt.noise_reducer import NoiseReducer


def test_noise_reducer_initialization():
    """Valida que o NoiseReducer carrega a DLL e inicializa o estado do Ctypes."""
    reducer = NoiseReducer()
    assert reducer.state is not None
    assert reducer.frame_size == 480
    reducer.close()
    assert reducer.state is None


def test_noise_reducer_processing():
    """Valida que o processamento e downsampling de 48kHz para 16kHz funciona de ponta a ponta."""
    reducer = NoiseReducer()
    
    # Cria uma entrada simulada de 1440 samples (30ms a 48kHz)
    dummy_input = np.zeros(1440, dtype=np.float32)
    cleaned_output = reducer.process_chunk(dummy_input)
    
    # Downsampled para 16kHz deve resultar em exatamente 480 samples (30ms a 16kHz)
    assert len(cleaned_output) == 480
    assert cleaned_output.dtype == np.float32
    assert not np.isnan(cleaned_output).any()
    
    reducer.close()
