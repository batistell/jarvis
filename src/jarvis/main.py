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
from jarvis.core.homeassistant import HomeAssistantClient

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


def match_local_ha_command(text: str, entities: list[dict]) -> tuple[str, str, str, dict] | None:
    """Realiza o casamento semântico/sintático rápido de comandos de voz locais.

    Identifica intenções básicas de ligar/desligar e mapeia para a melhor entidade correspondente.
    """
    # Limpa o texto
    cmd = text.lower().replace("jarvis", "").strip()
    for p in [".", ",", "!", "?", "-"]:
        cmd = cmd.replace(p, "")
    cmd = cmd.strip()

    # Se contiver termos de encadeamento/tempo, pula para deixar o LLM processar via tool calling
    if " e " in cmd or " and " in cmd or "depois" in cmd or "then" in cmd:
        return None

    # Mapeamento de intenções
    intents = {
        "turn_on": ["liga", "ligue", "ligar", "turn on", "switch on", "ativar", "ative", "abrir", "open"],
        "turn_off": ["desliga", "desligue", "desligar", "turn off", "switch off", "desativar", "desative", "apagar", "apague", "fechar", "close"],
        "toggle": ["alternar", "alterne", "toggle", "inverter"]
    }

    import re
    matched_intent = None
    query = cmd

    for intent, triggers in intents.items():
        for trigger in triggers:
            # Verifica se o gatilho ocorre como uma palavra inteira (limite de palavra \b)
            pattern = rf"\b{re.escape(trigger)}\b"
            match = re.search(pattern, cmd)
            if match:
                idx = match.start()
                query = cmd[idx + len(trigger):].strip()
                matched_intent = intent
                break
        if matched_intent:
            break

    if not matched_intent or not query:
        return None

    best_entity = None
    best_score = 0.0
    query_clean = query.lower().strip()

    for e in entities:
        name_clean = e["name"].lower().strip()
        entity_id_clean = e["entity_id"].lower().strip()

        # 1. Match exato
        if query_clean == name_clean or query_clean == entity_id_clean:
            best_entity = e
            best_score = 1.0
            break

        # 2. Match por contensão de substring
        if query_clean in name_clean or name_clean in query_clean:
            score = len(query_clean) / len(name_clean) if len(name_clean) > len(query_clean) else len(name_clean) / len(query_clean)
            score = 0.8 + score * 0.19
            if score > best_score:
                best_score = score
                best_entity = e

        # 3. Match por interseção de palavras (com suporte a sinônimos bilíngues)
        synonyms = {
            "escritorio": ["office"],
            "escritório": ["office"],
            "office": ["escritorio", "escritório"],
            "sala": ["living", "room"],
            "living": ["sala"],
            "room": ["sala", "quarto"],
            "quarto": ["bedroom", "bed"],
            "bedroom": ["quarto"],
            "cozinha": ["kitchen"],
            "kitchen": ["cozinha"],
            "banheiro": ["bathroom", "bath"],
            "bathroom": ["banheiro"],
            "luz": ["light", "lâmpada"],
            "light": ["luz", "lâmpada"],
            "lâmpada": ["light", "luz"],
            "lampada": ["light", "luz"],
            "interruptor": ["switch"],
            "switch": ["interruptor", "tomada"],
            "tomada": ["switch"],
        }
        q_words = set(query_clean.split())
        expanded_q_words = set(q_words)
        for w in q_words:
            if w in synonyms:
                expanded_q_words.update(synonyms[w])

        n_words = set(name_clean.replace(".", " ").replace("_", " ").replace("-", " ").split())
        if expanded_q_words and n_words:
            overlap = expanded_q_words.intersection(n_words)
            score = len(overlap) / len(n_words)
            if score > best_score:
                best_score = score
                best_entity = e

    # Se o match for confiante (score >= 0.6)
    if best_entity and best_score >= 0.6:
        domain = best_entity["entity_id"].split(".")[0]
        return matched_intent, domain, best_entity["entity_id"], best_entity

    return None


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

    # Inicializa o cliente do Home Assistant
    ha_client = HomeAssistantClient()
    if ha_client.is_configured:
        console.print("[yellow]⏳ Conectando e sincronizando dispositivos do Home Assistant...[/yellow]")
        await ha_client.get_entities()
        if ha_client.entities:
            console.print(f"[green]✅ Sincronizados {len(ha_client.entities)} dispositivos do Home Assistant![/green]\n")
        else:
            console.print("[yellow]⚠️ Nenhum dispositivo encontrado ou falha na conexão com o Home Assistant.[/yellow]\n")

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
    partial_newline_printed = False
    llm_interrupted_by_voice = False
    partial_in_progress = False

    async def generate_response(prompt_text: str, lang: str, original_query: str | None = None) -> None:
        nonlocal llm_generating
        llm_generating = True
        log.info("PERF: Iniciando geração do LLM para o prompt: '{}'", prompt_text)
        console.print("🤖 [bold cyan]Jarvis:[/bold cyan] ", end="")
        full_response_parts = []
        sentence_buffer = ""
        first_token_received = False
        try:
            async for token in llm.generate_stream(prompt_text, language=lang, ha_client=ha_client):
                if not first_token_received:
                    log.info("PERF: Primeiro token verbal recebido do LLM.")
                    first_token_received = True
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
            
            log.info("PERF: Geração do LLM concluída com sucesso.")
            
            # Salva o turno da conversa no banco de dados temporário
            full_response = "".join(full_response_parts)
            user_msg = original_query if original_query is not None else prompt_text
            cleaned_user_msg = user_msg
            # Remove "jarvis" do início se houver para deixar o contexto de histórico mais limpo
            if cleaned_user_msg.lower().strip().startswith("jarvis"):
                cleaned_user_msg = cleaned_user_msg.strip()[6:].strip(", ").strip()
            asyncio.create_task(llm.save_conversation_turn(cleaned_user_msg, full_response))
        except asyncio.CancelledError:
            # Geração foi cancelada por interrupção de fala
            log.info("PERF: Geração do LLM cancelada por interrupção de voz.")
            tts.stop()
            console.print(" [bold red][Interrompido][/bold red]")
        except Exception as e:
            log.error("PERF: Erro na geração de resposta do LLM: {}", e)
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

            if speech_detected:
                if not is_speaking:
                    is_speaking = True
                    partial_newline_printed = False
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

                        # Se o áudio for longo o suficiente, transcreve e comete (mínimo de 300ms para comandos rápidos)
                        if len(full_audio) > settings.audio.sample_rate * 0.3:
                            log.info("PERF: Fim de fala detectado (VAD finalizado). Iniciando transcrição com Whisper STT...")
                            res = await transcriber.transcribe(full_audio)
                            if res:
                                final_text, detected_lang = res
                                log.info("PERF: Transcrição Whisper concluída. Texto: '{}' (Idioma: {})", final_text, detected_lang)
                                if final_text:
                                    # Define se Jarvis estava ativo (gerando, falando ou recém-interrompido)
                                    was_jarvis_active = llm_generating or tts._is_playing or not tts._queue.empty() or llm_interrupted_by_voice

                                    # Definição das palavras de parada e ativação suportadas
                                    stop_words = ("jarvis", "para", "pare", "parar", "cala a boca", "silêncio", "quieto", "stop", "shut up", "be quiet", "silence", "pera", "espera", "calma", "chega", "shh", "shush")

                                    # Se Jarvis estava ativo, interrompe apenas com palavras de parada suportadas.
                                    # Se Jarvis estava ocioso, exige a palavra de ativação "jarvis".
                                    if was_jarvis_active:
                                        should_process = any(word in final_text.lower() for word in stop_words)
                                    else:
                                        should_process = "jarvis" in final_text.lower()

                                    if should_process:
                                        # Limpa o comando para analisar
                                        cleaned_cmd = final_text.lower().replace("jarvis", "").strip()
                                        for p in [".", ",", "!", "?", "-"]:
                                            cleaned_cmd = cleaned_cmd.replace(p, "")
                                        cleaned_cmd = cleaned_cmd.strip()

                                        is_stop_term = any(term in cleaned_cmd for term in ("pare", "parar", "cala a boca", "silêncio", "quieto", "stop", "shut up", "be quiet", "silence", "pera", "espera", "calma", "chega", "shh", "shush"))
                                        is_just_name = cleaned_cmd == "" or final_text.lower().strip() in ("jarvis", "jarvis.", "jarvis!", "jarvis?")
                                        is_stop_command = is_stop_term or (is_just_name and was_jarvis_active)

                                        if is_stop_command:
                                            # Interrompe tudo e fica em silêncio
                                            if llm_task and not llm_task.done():
                                                llm.interrupt()
                                                llm_task.cancel()
                                            tts.stop()
                                            log.info("Conversa: Jarvis foi silenciado por comando de voz.")

                                            if not partial_newline_printed:
                                                sys.stdout.write("\n")
                                                sys.stdout.flush()
                                            sys.stdout.write("\r\033[K")
                                            console.print(f"[bold green]🗣️  Você ({detected_lang}):[/bold green] {final_text} [bold red][Interrompido][/bold red]")
                                        else:
                                            # Se o LLM está gerando ou o TTS está reproduzindo, interrompe a geração anterior antes de responder
                                            if llm_generating or tts._is_playing or (llm_task and not llm_task.done()):
                                                if llm_task and not llm_task.done():
                                                    llm.interrupt()
                                                    llm_task.cancel()
                                                tts.stop()
                                                log.info("Conversa: Jarvis foi interrompido pela fala do usuário (nova pergunta).")

                                            # Apaga a linha parcial e imprime a final com destaque
                                            if not partial_newline_printed:
                                                sys.stdout.write("\n")
                                                sys.stdout.flush()
                                            sys.stdout.write("\r\033[K")  # Limpa linha
                                            console.print(f"[bold green]🗣️  Você ({detected_lang}):[/bold green] {final_text}")
                                            
                                            # Tenta fazer o matching semântico local rápido para o Home Assistant
                                            matched = match_local_ha_command(final_text, ha_client.entities) if ha_client and ha_client.entities else None
                                            
                                            if matched:
                                                intent, domain, entity_id, entity_obj = matched
                                                log.info("PERF: [Local Match] Casamento semântico local detectado! Entidade: {} ({}) -> Intent: {}", entity_obj['name'], entity_id, intent)
                                                
                                                # Dispara a requisição REST de controle imediatamente em paralelo
                                                asyncio.create_task(
                                                    ha_client.control_entity(
                                                        domain=domain,
                                                        service=intent,
                                                        entity_id=entity_id
                                                    )
                                                )
                                                
                                                # Cria um prompt instrutivo para o LLM apenas confirmar o sucesso verbalmente
                                                service_desc = "ligar" if intent == "turn_on" else ("desligar" if intent == "turn_off" else "alternar")
                                                spoken_prompt = (
                                                    f"O usuário solicitou {service_desc} o dispositivo '{entity_obj['name']}' (ID: {entity_id}). "
                                                    "Eu já executei a ação no Home Assistant com sucesso em segundo plano. "
                                                    "Por favor, confirme verbalmente a conclusão dessa ação com uma frase curta, educada e elegante."
                                                )
                                                llm_task = asyncio.create_task(
                                                    generate_response(
                                                        prompt_text=spoken_prompt,
                                                        lang=detected_lang,
                                                        original_query=final_text
                                                    )
                                                )
                                            else:
                                                # Fallback padrão: Envia para o LLM resolver via tool calling
                                                llm_task = asyncio.create_task(
                                                    generate_response(
                                                        prompt_text=final_text,
                                                        lang=detected_lang,
                                                        original_query=final_text
                                                    )
                                                )
                                    else:
                                        if not partial_newline_printed:
                                            sys.stdout.write("\n")
                                            sys.stdout.flush()
                                        sys.stdout.write("\r\033[K")  # Limpa linha
                                        log.info("Conversa: Frase ignorada (não contém palavra de parada/ativação necessária).")
                                else:
                                    if partial_newline_printed:
                                        sys.stdout.write("\r\033[K")
                                        sys.stdout.flush()
                            else:
                                if partial_newline_printed:
                                    sys.stdout.write("\r\033[K")
                                    sys.stdout.flush()
                        else:
                            if partial_newline_printed:
                                sys.stdout.write("\r\033[K")
                                sys.stdout.flush()

                        # Reseta buffers
                        audio_buffer.clear()
                        silent_chunks = 0
                        llm_interrupted_by_voice = False

            # 2. Transcrição Parcial (Real-time Feedback)
            if is_speaking and len(audio_buffer) > 0 and not partial_in_progress:
                now = time.time()
                if now - last_partial_time >= partial_interval_s:
                    last_partial_time = now

                    # Concatena o áudio acumulado até o momento
                    partial_audio = np.concatenate(audio_buffer, axis=0).flatten()

                    # Transcreve em segundo plano
                    # Criamos uma task para não bloquear o loop de captura de áudio
                    async def transcribe_partial(audio_data: np.ndarray) -> None:
                        nonlocal partial_newline_printed, partial_in_progress
                        partial_in_progress = True
                        try:
                            res = await transcriber.transcribe(audio_data, is_partial=True)
                            if res:
                                text, _ = res
                                if text:
                                    # Se o usuário disser uma palavra de parada específica durante a geração ou reprodução, interrompe IMEDIATAMENTE!
                                    # Verificado mesmo se o VAD já mudou is_speaking para False
                                    if llm_generating or tts._is_playing or not tts._queue.empty():
                                        cleaned_text = text.lower().strip()
                                        for p in [".", ",", "!", "?", "-"]:
                                            cleaned_text = cleaned_text.replace(p, "")
                                        cleaned_text = cleaned_text.strip()

                                        stop_words = ("jarvis", "para", "pare", "parar", "cala a boca", "silêncio", "quieto", "stop", "shut up", "be quiet", "silence", "pera", "espera", "calma", "chega", "shh", "shush")
                                        if any(word in cleaned_text for word in stop_words):
                                            if llm_task and not llm_task.done():
                                                nonlocal llm_interrupted_by_voice
                                                llm.interrupt()
                                                llm_task.cancel()
                                                llm_interrupted_by_voice = True
                                            tts.stop()
                                            log.info(f"Conversa: Jarvis foi interrompido imediatamente ao detectar a palavra de parada '{cleaned_text}'.")

                                    # Exibição do feedback de digitação em tempo real (apenas se o usuário ainda estiver falando)
                                    if is_speaking:
                                        if not partial_newline_printed:
                                            sys.stdout.write("\n")
                                            sys.stdout.flush()
                                            partial_newline_printed = True
                                        sys.stdout.write(f"\r\033[K[dim]🗣️  Escrevendo: {text}...[/dim]")
                                        sys.stdout.flush()
                        except Exception as e:
                            log.error(f"Erro na transcrição parcial: {e}")
                        finally:
                            partial_in_progress = False

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
