"""Jarvis — Transcrição de áudio com Faster Whisper.

Carrega o modelo Whisper localmente e realiza a transcrição rápida de buffers
de áudio de forma assíncrona.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from jarvis.config.settings import get_settings
from jarvis.core.logging import get_logger

log = get_logger(__name__)


class Transcriber:
    """Wrapper para o Faster Whisper de alta performance."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model_size = settings.stt.model_size
        self.compute_type = settings.stt.compute_type
        self.device = settings.stt.device
        self.language = settings.stt.language

        self._model: WhisperModel | None = None
        # Executor para rodar a inferência pesada fora do loop de eventos principal
        self._executor = ThreadPoolExecutor(max_workers=1)

    def load_model(self) -> None:
        """Carrega o modelo na memória (GPU ou CPU)."""
        if self._model is not None:
            return

        log.info(
            "Carregando modelo Faster Whisper: size={}, device={}, compute={}",
            self.model_size,
            self.device,
            self.compute_type,
        )

        try:
            # Tenta carregar na GPU/CUDA conforme configurado
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            log.info("Modelo Faster Whisper carregado com sucesso na GPU!")
        except Exception as e:
            log.warning(
                "Falha ao carregar na GPU ({}) com compute={}. Tentando fallback para CPU (float32)...",
                e,
                self.compute_type,
            )
            try:
                # Fallback para CPU
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="float32",
                )
                log.info("Modelo Faster Whisper carregado com sucesso na CPU.")
            except Exception as e_cpu:
                log.critical("Falha crítica ao carregar modelo Whisper na CPU: {}", e_cpu)
                raise e_cpu

    def _run_transcription(self, audio: np.ndarray) -> str:
        """Executa a transcrição síncrona (interna)."""
        if self._model is None:
            self.load_model()

        assert self._model is not None

        # Roda a transcrição do buffer numpy directamente
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=False,  # Já estamos filtrando áudio com nosso VAD local
        )

        # Junta os segmentos de texto
        text_segments = []
        for segment in segments:
            text_segments.append(segment.text)

        return "".join(text_segments).strip()

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcreve um buffer de áudio numpy de forma assíncrona.

        Args:
            audio: Array 1D do numpy contendo áudio em 16kHz float32.

        Returns:
            Texto transcrito.
        """
        loop = asyncio.get_running_loop()
        # Executa no thread pool para não bloquear o loop do asyncio
        return await loop.run_in_executor(self._executor, self._run_transcription, audio)
