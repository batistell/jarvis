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
from jarvis.stt.noise_reducer import NoiseReducer

log = get_logger(__name__)


class MicCapture:
    """Captura de áudio do microfone em tempo real com redução de ruído ativa."""

    def __init__(self) -> None:
        settings = get_settings()
        self.noise_reduction_enabled = settings.audio.noise_reduction_enabled
        self.channels = settings.audio.channels
        self.chunk_duration_ms = settings.audio.chunk_duration_ms
        
        if self.noise_reduction_enabled:
            # Captura fisicamente em 48kHz para alimentar o RNNoise
            self.sample_rate = 48000
            self.chunk_size = int(self.sample_rate * (self.chunk_duration_ms / 1000))  # 1440 samples
            
            try:
                self.noise_reducer = NoiseReducer()
                log.info("Redução de ruído ativa ativa no microfone (48kHz -> 16kHz).")
            except Exception as e:
                log.error("Falha ao inicializar o NoiseReducer. Desativando redução de ruído: {}", e)
                self.noise_reduction_enabled = False
                # Reverte para as configurações padrão de 16kHz
                self.sample_rate = settings.audio.sample_rate
                self.chunk_size = int(self.sample_rate * (self.chunk_duration_ms / 1000))
                self.noise_reducer = None
        else:
            self.sample_rate = settings.audio.sample_rate
            self.chunk_size = int(self.sample_rate * (self.chunk_duration_ms / 1000))  # 480 samples
            self.noise_reducer = None

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

        audio_data = indata.copy()
        
        if self.noise_reduction_enabled and self.noise_reducer:
            try:
                # Processa o áudio de 48kHz, reduzindo para 16kHz limpo
                audio_data = self.noise_reducer.process_chunk(audio_data)
            except Exception as e:
                log.error("Falha durante processamento do filtro de ruído: {}", e)
                # Fallback de decimação simples se o processador falhar
                audio_data = audio_data[::3]

        # Copia o áudio para colocar na fila assíncrona com segurança
        self._loop.call_soon_threadsafe(self._queue.put_nowait, audio_data)

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
