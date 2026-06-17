"""Jarvis — Voice Activity Detection (VAD).

Detecta se há fala ou silêncio em um bloco de áudio.
Usa a extensão 'webrtcvad' se disponível, com fallback automático
para um detector baseado em energia (RMS) caso a extensão não esteja instalada.
"""

from __future__ import annotations

import numpy as np

from jarvis.config.settings import get_settings
from jarvis.core.logging import get_logger

log = get_logger(__name__)

# Tenta importar webrtcvad
try:
    import webrtcvad
    _HAS_WEBRTCVAD = True
except ImportError:
    _HAS_WEBRTCVAD = False
    log.warning("webrtcvad não instalado. Usando fallback baseado em energia (RMS).")


class VADDetector:
    """Detector de atividade de voz (VAD)."""

    def __init__(self) -> None:
        settings = get_settings()
        self.sample_rate = settings.audio.sample_rate
        self.aggressiveness = settings.audio.vad_aggressiveness

        # Configurações do webrtcvad
        self._vad = None
        if _HAS_WEBRTCVAD:
            try:
                self._vad = webrtcvad.Vad(self.aggressiveness)
            except Exception as e:
                log.error("Erro ao inicializar webrtcvad: {}. Usando fallback RMS.", e)
                self._vad = None

        # Configurações do fallback de energia (RMS)
        # Threshold de energia padrão (pode ser calibrado automaticamente depois)
        self.rms_threshold = 0.015  # Sensibilidade para áudio float32 normalizado
        self.calibration_frames = 30  # Número de frames para calibração inicial
        self._noise_floor = 0.005
        self._calibrated = False
        self._calibration_data: list[float] = []

    def calibrate(self, rms: float) -> None:
        """Calibra o nível de ruído de fundo (noise floor) dinamicamente."""
        if self._calibrated:
            return

        self._calibration_data.append(rms)
        if len(self._calibration_data) >= self.calibration_frames:
            # Define o threshold de ruído com base na média + desvio padrão
            mean_noise = np.mean(self._calibration_data)
            std_noise = np.std(self._calibration_data)
            self._noise_floor = float(mean_noise)
            # Threshold = ruído médio + 3 * desvio padrão (mínimo de 0.01, máximo de 0.035)
            self.rms_threshold = min(max(float(mean_noise + 3 * std_noise), 0.01), 0.035)
            self._calibrated = True
            log.info(
                "VAD Calibrado: Noise Floor={:.5f}, RMS Threshold={:.5f}",
                self._noise_floor,
                self.rms_threshold,
            )

    def is_speech(self, chunk: np.ndarray, is_jarvis_busy: bool = False) -> bool:
        """Verifica se o bloco de áudio contém fala.

        Args:
            chunk: Array numpy contendo o áudio em float32.
            is_jarvis_busy: Indica se Jarvis está ativamente falando ou gerando resposta.

        Returns:
            True se contiver fala, False caso contrário.
        """
        # Calcula RMS do bloco (energia geral)
        rms = float(np.sqrt(np.mean(chunk**2)))

        # Se não estiver calibrado, acumula dados
        if not self._calibrated:
            self.calibrate(rms)

        # Ajusta dinamicamente a sensibilidade se Jarvis estiver falando/gerando.
        # Reduzimos o limiar de RMS necessário (multiplicando por 0.4) para aumentar
        # a sensibilidade, facilitando a detecção de stop words em volumes menores.
        threshold = self.rms_threshold * 0.4 if is_jarvis_busy else self.rms_threshold

        # Se tivermos webrtcvad, tentamos usá-lo
        if self._vad is not None:
            try:
                # webrtcvad espera int16 PCM de 16-bit
                # Converte float32 [-1.0, 1.0] para int16 [-32768, 32767]
                int16_chunk = (chunk * 32767).astype(np.int16).tobytes()
                # webrtcvad suporta apenas frames de 10, 20 ou 30ms
                is_speech_webrtc = self._vad.is_speech(int16_chunk, self.sample_rate)
                # Dual-Gate: padrão de voz webrtc AND amplitude de energia acima do ruído calibrado
                return is_speech_webrtc and rms > threshold
            except Exception as e:
                # Em caso de erro (ex: tamanho de frame inválido), faz fallback para RMS
                log.debug("Erro no webrtcvad: {}. Usando fallback RMS.", e)

        # Fallback ou detector principal por energia RMS
        return rms > threshold
