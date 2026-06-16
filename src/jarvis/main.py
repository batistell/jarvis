"""Jarvis — Entry point principal com transcrição em tempo real.

Inicializa as configurações, logging, carrega o modelo Faster Whisper
e inicia a captura de áudio com transcrição em milissegundos e
detecção automática de silêncio (fim de frase).
"""

from __future__ import annotations

# Otimização Windows: Garante o registro do DLL Path de CUDA antes de qualquer import do projeto
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import ctypes
from pathlib import Path
_main_dll_handles = []
if sys.platform == "win32":
    # Garante suporte a UTF-8 no terminal Windows para evitar UnicodeEncodeError ao exibir emojis
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
from jarvis.llm.engine import LLMEngine
from jarvis.tts.engine import TTSEngine

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


def extract_sentences(buffer: str) -> tuple[list[str], str]:
    """Extrai sentenças completas de um buffer de texto.

    Retorna uma lista de sentenças completas e o restante do buffer.
    """
    sentences = []
    current = []
    i = 0
    n = len(buffer)
    while i < n:
        char = buffer[i]
        current.append(char)
        if char in ('.', '!', '?', '\n'):
            is_boundary = False
            if char == '\n':
                is_boundary = True
            elif i + 1 < n and buffer[i + 1] in (' ', '\n', '\t'):
                is_boundary = True
            
            if is_boundary:
                sentence = "".join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []
        i += 1
        
    return sentences, "".join(current)


async def run_stt_loop() -> None:
    """Loop principal de captura e transcrição em tempo real."""
    settings = get_settings()

    llm = LLMEngine()
    console.print("[yellow]⏳ Carregando modelo LLM (Qwen 14B)...[/yellow]")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(llm._executor, llm.load_model)
    console.print("[green]✅ Modelo LLM carregado e pronto![/green]")

    console.print("[yellow]⏳ Carregando banco de dados e motor de busca (RAG)...[/yellow]")
    await loop.run_in_executor(None, llm.pre_load_rag)
    console.print("[green]✅ Motor de busca (RAG) carregado e pronto![/green]")

    console.print("[yellow]⏳ Carregando componentes do STT...[/yellow]")
    mic = MicCapture()
    vad = VADDetector()
    transcriber = Transcriber()

    # Força o carregamento do modelo Whisper no startup (no executor de GPU)
    await loop.run_in_executor(transcriber._executor, transcriber.load_model)
    console.print("[green]✅ Modelo Whisper carregado e pronto![/green]")

    tts = TTSEngine()
    console.print("[yellow]⏳ Carregando sintetizador de voz (TTS)...[/yellow]")
    await loop.run_in_executor(None, tts.load_model)
    console.print("[green]✅ Sintetizador de voz (TTS) carregado e pronto![/green]\n")

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

    # Controle de tarefas assíncronas para o LLM
    llm_task: asyncio.Task | None = None
    llm_generating = False

    async def generate_response(prompt_text: str, lang: str) -> None:
        nonlocal llm_generating
        llm_generating = True
        console.print("🤖 [bold cyan]Jarvis:[/bold cyan] ", end="")
        full_response_parts = []
        sentence_buffer = ""
        try:
            async for token in llm.generate_stream(prompt_text, language=lang):
                sys.stdout.write(token)
                sys.stdout.flush()
                full_response_parts.append(token)

                sentence_buffer += token
                sentences, sentence_buffer = extract_sentences(sentence_buffer)
                for sentence in sentences:
                    # Envia a frase para a fila do TTS em segundo plano
                    tts.speak_stream(sentence, lang)

            # Envia a última parte restante do buffer ao finalizar a geração
            remaining_sentence = sentence_buffer.strip()
            if remaining_sentence:
                tts.speak_stream(remaining_sentence, lang)
        except asyncio.CancelledError:
            # Geração foi cancelada por interrupção de fala
            tts.stop()
            console.print(" [bold red][Interrompido][/bold red]")
        except Exception as e:
            console.print(f"\n[red]Erro ao gerar resposta do Jarvis: {e}[/red]")
        finally:
            sys.stdout.write("\n\n")
            sys.stdout.flush()
            llm_generating = False

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

            # Se o LLM está gerando e detectamos nova fala do usuário, interrompemos!
            if speech_detected and llm_generating:
                if llm_task and not llm_task.done():
                    llm.interrupt()  # Sinaliza interrupção do loop C++ do llama.cpp
                    tts.stop()  # Para o áudio imediatamente e limpa a fila
                    llm_task.cancel()  # Cancela a task assíncrona da resposta
                    log.info("Conversa: Jarvis foi interrompido pela fala do usuário.")

                # Reseta buffers e estados para começar a capturar a fala do usuário imediatamente
                audio_buffer.clear()
                is_speaking = True
                silent_chunks = 0
                audio_buffer.append(chunk)
                continue

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
                            res = await transcriber.transcribe(full_audio)
                            if res:
                                final_text, detected_lang = res
                                if final_text:
                                    # Apaga a linha parcial e imprime a final com destaque
                                    sys.stdout.write("\r\033[K")  # Limpa linha
                                    console.print(f"[bold green]🗣️  Você ({detected_lang}):[/bold green] {final_text}")
                                    
                                    # Dispara a geração da resposta em uma task assíncrona separada
                                    llm_task = asyncio.create_task(generate_response(final_text, detected_lang))
                                else:
                                    sys.stdout.write("\r\033[K")
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
                        res = await transcriber.transcribe(audio_data)
                        if res:
                            text, _ = res
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
