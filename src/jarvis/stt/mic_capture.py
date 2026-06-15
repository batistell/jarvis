"""Jarvis — Captura de áudio do microfone em tempo real.

Usa a biblioteca sounddevice para capturar blocos de áudio do microfone
e disponibilizá-los em um gerador assíncrono.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import numpy as np
import sounddevice as sd

from jarvis.config.settings import get_settings
from jarvis.core.logging import get_logger

log = get_logger(__name__)


class MicCapture:
    """Captura de áudio do microfone em tempo real."""

    def __init__(self) -> None:
        settings = get_settings()
        self.sample_rate = settings.audio.sample_rate
        self.channels = settings.audio.channels
        self.chunk_duration_ms = settings.audio.chunk_duration_ms
        # Número de samples por chunk (ex: 16000 * 0.030 = 480 samples)
        self.chunk_size = int(self.sample_rate * (self.chunk_duration_ms / 1000))

        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self._stream: sd.InputStream | None = None
        self._running = False

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags
    ) -> None:
        """Callback chamada pelo sounddevice em uma thread separada."""
        if status:
            log.warning("Status do stream de áudio: {}", status)

        # Copia o áudio para colocar na fila assíncrona com segurança
        self._loop.call_soon_threadsafe(self._queue.put_nowait, indata.copy())

    def start(self) -> None:
        """Inicia a captura do microfone."""
        if self._running:
            return

        log.info(
            "Iniciando captura do microfone: rate={}Hz, channels={}, chunk_size={} samples ({}ms)",
            self.sample_rate,
            self.channels,
            self.chunk_size,
            self.chunk_duration_ms,
        )

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",  # Whisper espera float32 normalizado [-1, 1]
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._running = True

    def stop(self) -> None:
        """Para a captura do microfone."""
        if not self._running:
            return

        log.info("Parando captura do microfone")
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._running = False
        # Limpa a fila
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def stream(self) -> AsyncGenerator[np.ndarray, None]:
        """Gerador assíncrono que entrega chunks de áudio conforme chegam."""
        if not self._running:
            self.start()

        try:
            while self._running:
                chunk = await self._queue.get()
                yield chunk
        finally:
            self.stop()
