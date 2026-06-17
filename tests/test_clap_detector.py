import time
import pytest
import asyncio
from unittest.mock import MagicMock, patch

mock_loop = MagicMock()
mock_loop.call_soon_threadsafe = lambda f, *a: f(*a)

@pytest.fixture(autouse=True)
def patch_running_loop():
    with patch("asyncio.get_running_loop", return_value=mock_loop):
        yield

from jarvis.stt.mic_capture import MicCapture


def test_clap_detector_no_claps():
    """Valida que silêncio ou áudio contínuo baixo não ativa o detector de palmas."""
    mic = MicCapture()
    mic._loop = mock_loop
    
    called = False
    def on_clap():
        nonlocal called
        called = True
        
    mic.on_double_clap = on_clap
    
    # Envia um frame com silêncio (tudo zero)
    dummy_silent = np_zeros = np = __import__('numpy').zeros((mic.chunk_size, 1), dtype=__import__('numpy').float32)
    mic._audio_callback(dummy_silent, mic.chunk_size, {}, None)
    
    assert not called
    assert mic.last_clap_time == 0.0


def test_clap_detector_single_clap():
    """Valida que uma única palma (um único pico alto) não ativa o callback."""
    mic = MicCapture()
    mic._loop = mock_loop
    
    called = False
    def on_clap():
        nonlocal called
        called = True
        
    mic.on_double_clap = on_clap
    
    # Frame com um pico alto (palma)
    np = __import__('numpy')
    clap_frame = np.zeros((mic.chunk_size, 1), dtype=np.float32)
    clap_frame[10] = 0.8
    
    mic._audio_callback(clap_frame, mic.chunk_size, {}, None)
    
    assert not called
    assert mic.last_clap_time > 0.0  # Primeira palma registrada


def test_clap_detector_double_clap_success():
    """Valida que duas palmas com intervalo válido (ex: 400ms) disparam o callback."""
    mic = MicCapture()
    mic._loop = mock_loop
    
    called = False
    def on_clap():
        nonlocal called
        called = True
        
    mic.on_double_clap = on_clap
    
    np = __import__('numpy')
    # 1ª palma
    clap1 = np.zeros((mic.chunk_size, 1), dtype=np.float32)
    clap1[10] = 0.8
    mic._audio_callback(clap1, mic.chunk_size, {}, None)
    assert not called
    t1 = mic.last_clap_time
    assert t1 > 0.0
    
    # Simula passagem de tempo de 400ms
    mic.last_clap_time = t1 - 0.400
    mic.cooldown_ends = 0.0  # Reseta cooldown para permitir 2ª detecção
    
    # 2ª palma
    clap2 = np.zeros((mic.chunk_size, 1), dtype=np.float32)
    clap2[20] = 0.9
    mic._audio_callback(clap2, mic.chunk_size, {}, None)
    
    assert called
    assert mic.last_clap_time == 0.0  # Estado resetado com sucesso


def test_clap_detector_double_clap_cooldown():
    """Valida que duas palmas ocorrendo rápido demais (<150ms) são ignoradas pelo cooldown."""
    mic = MicCapture()
    mic._loop = mock_loop
    
    called = False
    def on_clap():
        nonlocal called
        called = True
        
    mic.on_double_clap = on_clap
    
    np = __import__('numpy')
    # 1ª palma
    clap1 = np.zeros((mic.chunk_size, 1), dtype=np.float32)
    clap1[10] = 0.8
    mic._audio_callback(clap1, mic.chunk_size, {}, None)
    
    # 2ª palma enviada logo em seguida (cooldown ainda ativo)
    clap2 = np.zeros((mic.chunk_size, 1), dtype=np.float32)
    clap2[20] = 0.9
    mic._audio_callback(clap2, mic.chunk_size, {}, None)
    
    assert not called


def test_clap_detector_double_clap_timeout():
    """Valida que duas palmas com intervalo longo demais (>800ms) não ativam o callback."""
    mic = MicCapture()
    mic._loop = mock_loop
    
    called = False
    def on_clap():
        nonlocal called
        called = True
        
    mic.on_double_clap = on_clap
    
    np = __import__('numpy')
    # 1ª palma
    clap1 = np.zeros((mic.chunk_size, 1), dtype=np.float32)
    clap1[10] = 0.8
    mic._audio_callback(clap1, mic.chunk_size, {}, None)
    t1 = mic.last_clap_time
    
    # Simula passagem de tempo de 1.2 segundos (excedendo threshold de 800ms)
    mic.last_clap_time = t1 - 1.2
    mic.cooldown_ends = 0.0
    
    # 2ª palma
    clap2 = np.zeros((mic.chunk_size, 1), dtype=np.float32)
    clap2[20] = 0.9
    mic._audio_callback(clap2, mic.chunk_size, {}, None)
    
    # Como passou o tempo limite, a 2ª palma deve apenas ser tratada como uma nova "1ª palma"
    assert not called
    assert mic.last_clap_time > 0.0
