"""Jarvis — Redução de ruído com RNNoise via Ctypes.

Carrega a DLL compilada do RNNoise e processa áudio de 48kHz,
reduzindo a amostragem para 16kHz de forma limpa.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
import numpy as np

from jarvis.config.settings import get_settings
from jarvis.core.logging import get_logger

log = get_logger(__name__)


class NoiseReducer:
    """Wrapper Ctypes de alta performance para o filtro de ruído RNNoise."""

    def __init__(self) -> None:
        settings = get_settings()
        
        # 1. Caminho absoluto do DLL do RNNoise no ambiente virtual ou site-packages
        project_root = settings.project_root
        
        dll_options = [
            project_root / ".venv" / "Lib" / "site-packages" / "pyrnnoise" / "rnnoise.dll",
            Path(sys.prefix) / "Lib" / "site-packages" / "pyrnnoise" / "rnnoise.dll",
        ]
        
        dll_path = None
        for path in dll_options:
            if path.exists():
                dll_path = path
                break
                
        if not dll_path:
            log.error("DLL do RNNoise não localizado nos caminhos padrão. Redução de ruído desativada.")
            raise FileNotFoundError("rnnoise.dll não encontrado.")
            
        log.info("Carregando DLL do RNNoise em: {}", dll_path)
        try:
            self.lib = ctypes.CDLL(str(dll_path.resolve()))
            
            # 2. Configura assinaturas das funções C
            self.lib.rnnoise_create.argtypes = [ctypes.c_void_p]
            self.lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
            self.lib.rnnoise_process_frame.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            self.lib.rnnoise_create.restype = ctypes.c_void_p
            self.lib.rnnoise_get_frame_size.restype = ctypes.c_int
            self.lib.rnnoise_process_frame.restype = ctypes.c_float
            
            self.frame_size = self.lib.rnnoise_get_frame_size()  # Normalmente 480 (10ms a 48kHz)
            self.state = self.lib.rnnoise_create(None)
            log.info("Filtro RNNoise inicializado. Tamanho de frame interno: {} samples.", self.frame_size)
            
        except Exception as e:
            log.error("Erro crítico ao inicializar o motor Ctypes do RNNoise: {}", e)
            raise e

    def process_chunk(self, chunk_48k: np.ndarray) -> np.ndarray:
        """Processa um bloco de áudio float32 a 48kHz e retorna áudio limpo a 16kHz.

        Args:
            chunk_48k: Array 1D contendo samples de áudio a 48.000 Hz em float32 [-1, 1].

        Returns:
            Array 1D contendo samples limpos a 16.000 Hz em float32 [-1, 1].
        """
        # Garante que o input está normalizado como float32 e plano
        audio = chunk_48k.flatten().astype(np.float32)
        
        # O RNNoise espera que a amplitude da onda esteja no range de 16 bits [-32768, 32767]
        scaled_audio = audio * 32767.0
        
        cleaned_scaled = np.zeros_like(scaled_audio)
        
        # Processa em blocos de 480 (frame_size do RNNoise)
        for i in range(0, len(scaled_audio), self.frame_size):
            # Fatia o bloco
            block = scaled_audio[i:i + self.frame_size].copy()
            
            # Se for menor que o esperado (fim do buffer), faz padding com zeros
            if len(block) < self.frame_size:
                block = np.pad(block, (0, self.frame_size - len(block)))
                
            # Ponteiro C
            ptr = block.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # Processamento in-place (resultado escrito de volta em block)
            self.lib.rnnoise_process_frame(self.state, ptr, ptr)
            
            # Grava no buffer de saída
            end_idx = min(i + self.frame_size, len(scaled_audio))
            cleaned_scaled[i:end_idx] = block[:end_idx - i]
            
        # Reduz escala de volta para [-1.0, 1.0]
        cleaned_48k = cleaned_scaled / 32767.0
        
        # Downsample de 48kHz para 16kHz (pega 1 de cada 3 samples)
        cleaned_16k = cleaned_48k[::3]
        
        return cleaned_16k

    def close(self) -> None:
        """Libera os recursos C alocados."""
        if hasattr(self, "state") and self.state:
            try:
                self.lib.rnnoise_destroy(self.state)
                self.state = None
                log.info("Instância C do RNNoise liberada com sucesso.")
            except Exception as e:
                log.error("Erro ao liberar recursos do RNNoise: {}", e)

    def __del__(self) -> None:
        self.close()
