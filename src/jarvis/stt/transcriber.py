"""Jarvis — Transcrição de áudio com Faster Whisper.

Carrega o modelo Whisper localmente e realiza a transcrição rápida de buffers
de áudio de forma assíncrona.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# Otimização Windows: Adiciona os DLLs dos pacotes pip da NVIDIA ao search path
import os
import sys
# Lista global para manter as referências de DLL vivas na memória
_dll_handles = []

if sys.platform == "win32":
    # Procura em site-packages/nvidia
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia_dir = site_packages / "nvidia"
    print(f"[debug] Checking site-packages/nvidia path: {nvidia_dir} (exists={nvidia_dir.exists()})")
    if nvidia_dir.exists():
        for bin_dir in nvidia_dir.glob("**/bin"):
            try:
                print(f"[debug] Registering and keeping DLL path: {bin_dir.resolve()}")
                handle = os.add_dll_directory(str(bin_dir.resolve()))
                _dll_handles.append(handle)
            except Exception as e:
                print(f"[debug] Error adding DLL path {bin_dir}: {e}")

from faster_whisper import WhisperModel

from jarvis.config.settings import get_settings
from jarvis.core.logging import get_logger
from jarvis.core.executor import get_gpu_executor

log = get_logger(__name__)


class Transcriber:
    """Wrapper para o Faster Whisper de alta performance."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model_size = settings.stt.model_size
        self.compute_type = settings.stt.compute_type
        self.device = settings.stt.device
        self.language = settings.stt.language
        self.initial_prompt = settings.stt.initial_prompt

        self._model: WhisperModel | None = None
        # Usa o executor compartilhado de thread única para GPU
        self._executor = get_gpu_executor()

    def load_model(self) -> None:
        """Carrega o modelo na memória (GPU ou CPU) com dry-run para verificar DLLs."""
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
            model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            
            # DRY-RUN: Faster Whisper carrega cuBLAS/cuDNN preguiçosamente.
            # Transcrevemos 1 segundo de silêncio para forçar o carregamento das DLLs de GPU
            # e garantir que não haverá erros em tempo de execução.
            dummy_audio = np.zeros(16000, dtype=np.float32)
            list(model.transcribe(dummy_audio, beam_size=1)[0])
            
            self._model = model
            log.info("Modelo Faster Whisper carregado com sucesso ({})!", self.device.upper())
            
        except Exception as e:
            log.warning(
                "Falha ao inicializar GPU/CUDA ({}) com compute={}. "
                "Realizando fallback automático para CPU (float32)...",
                e,
                self.compute_type,
            )
            try:
                # Fallback para CPU
                model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="float32",
                )
                # Dry-run na CPU para garantir que funciona
                dummy_audio = np.zeros(16000, dtype=np.float32)
                list(model.transcribe(dummy_audio, beam_size=1)[0])
                
                self._model = model
                log.info("Modelo Faster Whisper carregado com sucesso na CPU.")
            except Exception as e_cpu:
                log.critical("Falha crítica ao carregar modelo Whisper na CPU: {}", e_cpu)
                raise e_cpu


    def _filter_hallucinations(self, text: str) -> str:
        """Filtra alucinações ou preenchimentos comuns do Whisper causados por ruído ou silêncio."""
        cleaned = text.strip().lower()
        # Remove pontuação comum para normalizar a comparação
        for p in [".", ",", "!", "?", "-", '"', "'"]:
            cleaned = cleaned.replace(p, "")
        cleaned = cleaned.strip()

        # Alucinações típicas do Whisper sob ruído ou silêncio (inglês e português)
        hallucinations = {
            "thank you",
            "thank you very much",
            "thank you for watching",
            "thanks for watching",
            "subtitles by amaraorg",
            "subtitles",
            "amaraorg",
            "you",
            "ha",
            "yeah",
            "bye",
            "please",
            "ok",
            "right",
            "obrigado",
            "obrigada",
            "obrigado por assistir",
            "muito obrigado",
            "tchau",
        }

        if cleaned in hallucinations:
            log.warning("STT: Transcrição de ruído/silêncio ignorada (alucinação filtrada): '{}'", text)
            return ""

        # Ignora ruídos que resultam em strings extremamente curtas
        if len(cleaned) < 2:
            return ""

        return text

    def _run_transcription(self, audio: np.ndarray) -> tuple[str, str]:
        """Executa a transcrição síncrona (interna)."""
        if self._model is None:
            self.load_model()

        assert self._model is not None

        # Se language for auto, passa None para o Faster Whisper realizar auto-detecção
        lang = self.language if self.language and self.language != "auto" else None

        # Otimizado para o idioma configurado e máxima velocidade (beam_size=1)
        segments, info = self._model.transcribe(
            audio,
            language=lang,
            beam_size=1,
            best_of=1,
            vad_filter=False,  # VAD já é feito externamente
            initial_prompt=self.initial_prompt,
        )

        # Junta os segmentos de texto
        text_segments = []
        for segment in segments:
            text_segments.append(segment.text)

        raw_text = "".join(text_segments).strip()
        cleaned_text = self._filter_hallucinations(raw_text)
        return cleaned_text, info.language

    async def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """Transcreve um buffer de áudio numpy de forma assíncrona.

        Args:
            audio: Array 1D do numpy contendo áudio em 16kHz float32.

        Returns:
            Tupla contendo (texto transcrito, código do idioma detectado).
        """
        loop = asyncio.get_running_loop()
        # Executa no thread pool para não bloquear o loop do asyncio
        return await loop.run_in_executor(self._executor, self._run_transcription, audio)
