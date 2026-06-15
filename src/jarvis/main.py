"""Jarvis — Entry point principal com transcrição em tempo real.

Inicializa as configurações, logging, carrega o modelo Faster Whisper
e inicia a captura de áudio com transcrição em milissegundos e
detecção automática de silêncio (fim de frase).
"""

from __future__ import annotations

# Otimização Windows: Garante o registro do DLL Path de CUDA antes de qualquer import do projeto
import os
import sys
import ctypes
from pathlib import Path
_main_dll_handles = []
if sys.platform == "win32":
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia_dir = site_packages / "nvidia"
    if nvidia_dir.exists():
        # 1. Registra os caminhos no add_dll_directory
        for bin_dir in nvidia_dir.glob("**/bin"):
            try:
                _main_dll_handles.append(os.add_dll_directory(str(bin_dir.resolve())))
            except Exception:
                pass
        
        # 2. Pré-carrega as DLLs na memória do processo via ctypes
        try:
            # CUDA Runtime
            cudart_path = nvidia_dir / "cuda_runtime" / "bin" / "cudart64_12.dll"
            if cudart_path.exists():
                ctypes.CDLL(str(cudart_path.resolve()))
            
            # cuBLAS Lt
            cublaslt_path = nvidia_dir / "cublas" / "bin" / "cublasLt64_12.dll"
            if cublaslt_path.exists():
                ctypes.CDLL(str(cublaslt_path.resolve()))
                
            # cuBLAS
            cublas_path = nvidia_dir / "cublas" / "bin" / "cublas64_12.dll"
            if cublas_path.exists():
                ctypes.CDLL(str(cublas_path.resolve()))
                
            # cuDNN
            cudnn_path = nvidia_dir / "cudnn" / "bin" / "cudnn64_9.dll"
            if cudnn_path.exists():
                ctypes.CDLL(str(cudnn_path.resolve()))
        except Exception:
            pass

import asyncio
import time

import numpy as np
from rich.console import Console
from rich.panel import Panel

from jarvis.config.settings import get_settings
from jarvis.core.logging import get_logger, setup_logging
from jarvis.stt.mic_capture import MicCapture
from jarvis.stt.transcriber import Transcriber
from jarvis.stt.vad import VADDetector

console = Console()
log = get_logger("jarvis.main")


def _print_banner() -> None:
    """Exibe o banner do Jarvis."""
    banner = """
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
    """
    console.print(Panel(banner, title="🤖 Jarvis STT Test", border_style="cyan"))


async def run_stt_loop() -> None:
    """Loop principal de captura e transcrição em tempo real."""
    settings = get_settings()

    console.print("[yellow]⏳ Carregando componentes do STT...[/yellow]")
    mic = MicCapture()
    vad = VADDetector()
    transcriber = Transcriber()

    # Força o carregamento do modelo Whisper no startup
    transcriber.load_model()
    console.print("[green]✅ Modelo Whisper carregado e pronto![/green]\n")

    # Parâmetros de controle
    # Silêncio necessário para fechar uma frase (ms)
    silence_threshold_ms = settings.audio.silence_threshold_ms
    # Duração de cada chunk (ms)
    chunk_duration_ms = settings.audio.chunk_duration_ms
    # Chunks de silêncio tolerados antes de finalizar a frase
    max_silent_chunks = silence_threshold_ms // chunk_duration_ms

    # Buffers de estado
    audio_buffer: list[np.ndarray] = []
    is_speaking = False
    silent_chunks = 0

    # Controle de transcrição parcial (para não sobrecarregar)
    last_partial_time = 0.0
    partial_interval_s = 0.35  # Transcreve parcial a cada 350ms

    console.print("[bold cyan]🎙️  Microfone Ativo. Pode começar a falar![/bold cyan]")
    console.print("[dim]Pressione Ctrl+C para encerrar.[/dim]\n")

    mic.start()
    try:
        async for chunk in mic.stream():
            # 1. Verifica se há fala no chunk de 30ms
            speech_detected = vad.is_speech(chunk)

            if speech_detected:
                if not is_speaking:
                    is_speaking = True
                    log.debug("Fala detectada - Iniciando frase")
                # Reseta contador de silêncio e acumula áudio
                silent_chunks = 0
                audio_buffer.append(chunk)
            else:
                if is_speaking:
                    # Acumula o silêncio também para não cortar palavras
                    audio_buffer.append(chunk)
                    silent_chunks += 1

                    # Se atingir o threshold de silêncio, finaliza a frase
                    if silent_chunks >= max_silent_chunks:
                        is_speaking = False
                        # Concatena todo o áudio acumulado
                        full_audio = np.concatenate(audio_buffer, axis=0).flatten()

                        # Se o áudio for longo o suficiente, transcreve e comete
                        if len(full_audio) > settings.audio.sample_rate * 0.5:
                            final_text = await transcriber.transcribe(full_audio)
                            if final_text:
                                # Apaga a linha parcial e imprime a final com destaque
                                sys.stdout.write("\r\033[K")  # Limpa linha
                                console.print(f"[bold green]🗣️  Você:[/bold green] {final_text}")
                            else:
                                sys.stdout.write("\r\033[K")
                        else:
                            sys.stdout.write("\r\033[K")

                        # Reseta buffers
                        audio_buffer.clear()
                        silent_chunks = 0

            # 2. Transcrição Parcial (Real-time Feedback)
            if is_speaking and len(audio_buffer) > 0:
                now = time.time()
                if now - last_partial_time >= partial_interval_s:
                    last_partial_time = now

                    # Concatena o áudio acumulado até o momento
                    partial_audio = np.concatenate(audio_buffer, axis=0).flatten()

                    # Transcreve em segundo plano
                    # Criamos uma task para não bloquear o loop de captura de áudio
                    async def transcribe_partial(audio_data: np.ndarray) -> None:
                        text = await transcriber.transcribe(audio_data)
                        if text and is_speaking:
                            # Imprime em cinza no terminal sobrescrevendo a linha atual
                            sys.stdout.write(f"\r\033[K[dim]🗣️  Escrevendo: {text}...[/dim]")
                            sys.stdout.flush()

                    asyncio.create_task(transcribe_partial(partial_audio))

    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        console.print("\n[yellow]🎙️  Microfone desativado.[/yellow]")


async def _async_main() -> None:
    """Main principal assíncrono."""
    _print_banner()
    await run_stt_loop()


def main() -> None:
    """Entry point do script console."""
    setup_logging()
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        console.print("\n[dim]Jarvis encerrado.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
