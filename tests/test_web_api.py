"""Jarvis — Unit tests for FastAPI Web Interface."""

from __future__ import annotations

import pytest
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from fastapi.testclient import TestClient
from jarvis.ui.web import app


def run_in_thread(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(func, *args, **kwargs).result()
    return wrapper


@run_in_thread
def test_web_index_route() -> None:
    """Verifica se a rota raiz '/' retorna o HTML do dashboard."""
    import sys, asyncio, threading
    import anyio._core._eventloop as ev
    print(f"\n[DEBUG MAIN THREAD] Thread: {threading.current_thread().name}", file=sys.stderr)
    try:
        loop = asyncio.get_running_loop()
        print(f"[DEBUG MAIN THREAD] Running loop in main thread: {loop}", file=sys.stderr)
    except RuntimeError as e:
        print(f"[DEBUG MAIN THREAD] No running loop in main thread: {e}", file=sys.stderr)
    print(f"[DEBUG MAIN THREAD] ev.sniffio: {ev.sniffio}", file=sys.stderr)
    print(f"[DEBUG MAIN THREAD] ev.current_async_library(): {ev.current_async_library()}", file=sys.stderr)

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "JARVIS" in response.text


@run_in_thread
def test_web_stats_route() -> None:
    """Verifica se o endpoint '/api/stats' retorna as métricas da GPU."""
    client = TestClient(app)
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    
    assert "vram_total" in data
    assert "vram_used" in data
    assert "temperature" in data
    assert "cuda_load" in data
    assert "success" in data


@run_in_thread
def test_web_ha_devices_route() -> None:
    """Verifica se o endpoint '/api/ha/devices' responde corretamente."""
    client = TestClient(app)
    response = client.get("/api/ha/devices")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@run_in_thread
def test_web_chat_websocket() -> None:
    """Simula uma conexão WebSocket no canal de chat '/ws/chat'."""
    client = TestClient(app)
    with client.websocket_connect("/ws/chat") as websocket:
        # No momento da conexão, o uvicorn manda dados de telemetria
        data = websocket.receive_json()
        assert data["type"] == "telemetry"
        
        # Envia uma pergunta mockada por texto
        websocket.send_json({
            "type": "message",
            "text": "test prompt"
        })
        
        # Verifica se o broadcast re-transmite a mensagem
        # (O broadcast re-transmite a mensagem do usuário com sender 'user' e origin 'Navegador')
        broadcast_data = websocket.receive_json()
        assert broadcast_data["type"] == "message"
        assert broadcast_data["sender"] == "user"
        assert broadcast_data["text"] == "test prompt"


def test_tts_audio_header_concatenation() -> None:
    """Verifica se a concatenação de cabeçalho do sample rate funciona perfeitamente."""
    import numpy as np
    sample_rate = 22050
    audio_float_array = np.array([0.1, -0.2, 0.35], dtype=np.float32)
    
    header = np.array([float(sample_rate)], dtype=np.float32)
    combined = np.concatenate((header, audio_float_array))
    
    assert combined.dtype == np.float32
    assert len(combined) == 4
    assert combined[0] == 22050.0
    assert np.allclose(combined[1:], audio_float_array)
