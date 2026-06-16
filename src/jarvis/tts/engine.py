"""Jarvis — Text-to-Speech Engine using local Piper."""

from __future__ import annotations

import queue
import threading
import urllib.request
from pathlib import Path
from loguru import logger
import sounddevice as sd
from piper import PiperVoice

from jarvis.config.settings import get_settings


class TTSEngine:
    """Wrapper para síntese de voz local usando Piper TTS com fila de reprodução assíncrona."""

    def __init__(self) -> None:
        self.settings = get_settings().tts
        self._voices: dict[str, PiperVoice] = {}
        self._is_playing = False
        self._queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _download_model(self, lang: str) -> tuple[Path, Path]:
        """Baixa o modelo e configuração do Piper caso não existam localmente."""
        if lang == "pt":
            model_name = "pt_BR-faber-medium.onnx"
            base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium"
        else:
            model_name = "en_US-lessac-medium.onnx"
            base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

        model_path = Path("models") / model_name
        config_path = model_path.with_suffix(".onnx.json")

        if model_path.exists() and config_path.exists():
            return model_path, config_path

        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_url = f"{base_url}/{model_name}"
        config_url = f"{base_url}/{model_name}.json"

        logger.info(f"TTS: Modelo de voz ({lang}) não encontrado localmente. Iniciando download...")
        
        try:
            logger.info(f"Baixando configuração de voz para {lang}...")
            urllib.request.urlretrieve(config_url, str(config_path))
            
            logger.info(f"Baixando modelo ONNX (~15-18MB) para {lang}...")
            urllib.request.urlretrieve(model_url, str(model_path))
            
            logger.info(f"TTS: Modelo e configuração para {lang} baixados com sucesso!")
        except Exception as e:
            logger.error(f"Erro ao baixar modelo de voz {lang}: {e}")
            if model_path.exists():
                model_path.unlink()
            if config_path.exists():
                config_path.unlink()
            raise e

        return model_path, config_path

    def load_model(self, lang: str = "pt") -> PiperVoice:
        """Carrega o modelo Piper para o idioma especificado na memória."""
        if lang in self._voices:
            return self._voices[lang]

        model_path, config_path = self._download_model(lang)

        logger.info(f"TTS: Carregando modelo Piper ONNX para {lang}...")
        try:
            # PiperVoice espera o arquivo ONNX e a configuração correspondente
            voice = PiperVoice.load(str(model_path))
            self._voices[lang] = voice
            logger.info(f"TTS: Modelo de voz {lang} carregado com sucesso.")
            return voice
        except Exception as e:
            logger.exception(f"Erro ao inicializar o motor Piper TTS para {lang}: {e}")
            raise e

    def start_worker(self) -> None:
        """Inicia a thread de trabalho que consome a fila de reprodução."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Loop da thread que processa a fila de áudio."""
        logger.info("TTS: Thread de reprodução iniciada.")
        while not self._stop_event.is_set():
            try:
                # Espera por um item com timeout para verificar se deve parar
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                self._queue.task_done()
                break

            text, lang = item
            try:
                self._is_playing = True
                voice = self.load_model(lang)
                
                # Gera e reproduz os chunks de áudio (um por frase/sentença)
                for chunk in voice.synthesize(text):
                    if not self._is_playing or self._stop_event.is_set():
                        break
                    sd.play(chunk.audio_float_array, chunk.sample_rate)
                    sd.wait()
            except Exception as e:
                logger.error(f"TTS: Erro na síntese/reprodução: {e}")
            finally:
                self._is_playing = False
                self._queue.task_done()

        logger.info("TTS: Thread de reprodução encerrada.")

    def speak(self, text: str) -> None:
        """Legado: adiciona o texto inteiro na fila padrão."""
        self.speak_stream(text, "pt")

    def speak_stream(self, text: str, lang: str = "pt") -> None:
        """Gera e reproduz a fala do texto fornecido de forma síncrona/fila.

        Deve ser executado fora da thread de loop principal do asyncio.
        """
        if not self.settings.enabled:
            return

        self.start_worker()
        self._queue.put((text, lang))

    def stop(self) -> None:
        """Interrompe qualquer reprodução ou geração de áudio atual e limpa a fila."""
        self._is_playing = False
        sd.stop()
        
        # Esvazia a fila de reprodução
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        logger.info("TTS: Reprodução de voz interrompida e fila limpa.")
