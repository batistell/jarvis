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
import jarvis.ui.web as web
from jarvis.core.ssl_gen import generate_self_signed_cert

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
        if char in ('.', '!', '?', '\n', ',', ';', ':'):
            is_boundary = False
            if char == '\n':
                is_boundary = True
            elif i + 1 < n and buffer[i + 1] in (' ', '\n', '\t'):
                # Para pontuações fracas como vírgula, dois pontos e ponto e vírgula, só divide se houver conteúdo mínimo
                if char in (',', ';', ':'):
                    if len(current) >= 30:
                        is_boundary = True
                else:
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
    mic = MicCapture()
    vad = VADDetector()
    transcriber = Transcriber()
    tts = TTSEngine()

    loop = asyncio.get_running_loop()
    console.print("[yellow]⏳ Carregando todos os modelos em paralelo...[/yellow]\n")

    async def _load_gpu_models() -> None:
        """LLM e STT sequencialmente no executor de GPU compartilhado."""
        await loop.run_in_executor(llm._executor, llm.load_model)
        console.print("[green]  ✅ LLM (Qwen 14B) carregado![/green]")
        await loop.run_in_executor(transcriber._executor, transcriber.load_model)
        console.print("[green]  ✅ Whisper STT carregado![/green]")

    async def _load_rag() -> None:
        """Embedding engine na CPU — em paralelo com os modelos de GPU."""
        await loop.run_in_executor(None, llm.pre_load_rag)
        console.print("[green]  ✅ RAG / Embedding (bge-m3) carregado![/green]")

    async def _load_tts() -> None:
        """Piper TTS na CPU — em paralelo com os modelos de GPU."""
        await loop.run_in_executor(None, tts.load_model)
        console.print("[green]  ✅ TTS (Piper) carregado![/green]")

    # GPU: LLM → STT (sequencial no mesmo executor)
    # CPU: Embedding + TTS (paralelo na thread pool padrão)
    await asyncio.gather(
        _load_gpu_models(),
        _load_rag(),
        _load_tts(),
    )
    console.print("\n[bold green]✅ Todos os modelos prontos![/bold green]\n")


    # Inicializa o cliente do Home Assistant
    ha_client = HomeAssistantClient()
    if ha_client.is_configured:
        console.print("[yellow]⏳ Conectando e sincronizando dispositivos do Home Assistant...[/yellow]")
        await ha_client.get_entities()
        if ha_client.entities:
            console.print(f"[green]✅ Sincronizados {len(ha_client.entities)} dispositivos do Home Assistant![/green]\n")
        else:
            console.print("[yellow]⚠️ Nenhum dispositivo encontrado ou falha na conexão com o Home Assistant.[/yellow]\n")

    # 1. Gera certificados autoassinados se HTTPS estiver ativo
    ssl_cert = Path(settings.web.ssl_cert_path)
    ssl_key = Path(settings.web.ssl_key_path)
    if settings.web.ssl_enabled:
        generate_self_signed_cert(ssl_cert, ssl_key)

    # 2. Injeta as instâncias dos motores carregados no módulo web
    web.llm_engine = llm
    web.tts_engine = tts
    web.ha_client = ha_client
    web.transcriber_engine = transcriber

    # 3. Inicia o servidor Uvicorn em background (no mesmo event loop)
    import uvicorn
    web_config = uvicorn.Config(
        "jarvis.ui.web:app",
        host=settings.web.host,
        port=settings.web.port,
        ssl_keyfile=str(ssl_key) if settings.web.ssl_enabled else None,
        ssl_certfile=str(ssl_cert) if settings.web.ssl_enabled else None,
        log_level="warning",
    )
    web_server = uvicorn.Server(web_config)
    asyncio.create_task(web_server.serve())
    
    # 4. Inicia o túnel Cloudflare
    cf_url = None
    try:
        from jarvis.core.cloudflare import start_cloudflare_tunnel
        cf_url, _ = await loop.run_in_executor(
            None,
            start_cloudflare_tunnel,
            settings.web.port,
            settings.web.ssl_enabled,
            settings.project_root
        )
    except Exception as e:
        log.error(f"Erro ao iniciar o túnel Cloudflare: {e}")

    # Exibe URLs amigáveis para o usuário
    protocol = "https" if settings.web.ssl_enabled else "http"
    local_ip = "127.0.0.1"
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    console.print(f"[green]✅ Servidor Web {protocol.upper()} ativo![/green]")
    if settings.web.host == "0.0.0.0":
        console.print(f"👉 Local: [bold cyan]{protocol}://localhost:{settings.web.port}[/bold cyan]")
        if local_ip != "127.0.0.1":
            console.print(f"👉 Rede Local (WLAN/LAN): [bold cyan]{protocol}://{local_ip}:{settings.web.port}[/bold cyan]")
    else:
        console.print(f"👉 Endereço: [bold cyan]{protocol}://{settings.web.host}:{settings.web.port}[/bold cyan]")
        
    if cf_url:
        console.print(f"👉 Cloudflare (Acesso Público): [bold cyan]{cf_url}[/bold cyan]")
    console.print("")

    # Define a ação executada ao detectar duas palmas
    async def handle_double_clap() -> None:
        log.info("Conversa: Callback de duas palmas disparado!")
        console.print("\n[bold yellow]👏 Duas Palmas Detectadas! Executando ação...[/bold yellow]")
        
        # Procura a primeira lâmpada configurada no Home Assistant, priorizando 'office' / 'escritorio'
        light_entity = None
        fallback_light = None
        if ha_client and ha_client.entities:
            for e in ha_client.entities:
                if e["entity_id"].startswith("light."):
                    name_lower = e["name"].lower()
                    id_lower = e["entity_id"].lower()
                    
                    # Prioriza qualquer luz que contenha "office", "escritorio" ou "escritório"
                    if "office" in name_lower or "office" in id_lower or \
                       "escritorio" in name_lower or "escritorio" in id_lower or \
                       "escritório" in name_lower:
                        light_entity = e
                        break
                    
                    # Salva a primeira luz genérica caso não encontre nenhuma do escritório
                    if fallback_light is None:
                        fallback_light = e
            
            # Se não encontrou nenhuma luz de escritório, usa o fallback genérico
            if not light_entity:
                light_entity = fallback_light
        
        if light_entity:
            entity_id = light_entity["entity_id"]
            name = light_entity["name"]
            log.info("Palmas: Alternando estado da luz: {} ({})", name, entity_id)
            
            # Executa a alternância assíncrona
            asyncio.create_task(
                ha_client.control_entity(
                    domain="light",
                    service="toggle",
                    entity_id=entity_id
                )
            )
            tts.speak_stream("Luz alternada, Senhor.", "pt")
        else:
            log.info("Palmas: Nenhuma luz encontrada no Home Assistant. Emitindo confirmação verbal.")
            tts.speak_stream("Sim, Senhor. Detectei duas palmas.", "pt")

    mic.on_double_clap = lambda: asyncio.create_task(handle_double_clap())


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
    conversa_fluida_active = False

    # Controle de tarefas assíncronas e estados compartilhados com web.py
    web.tts_last_active_time = 0.0
    web.llm_task = None
    web.llm_generating = False
    web.llm_interrupted_by_voice = False

    partial_newline_printed = False
    partial_in_progress = False

    async def generate_response(
        prompt_text: str,
        lang: str,
        original_query: str | None = None,
        on_audio_chunk: callable | None = None,
        origin: str = "Terminal"
    ) -> None:
        web.llm_generating = True
        web.llm_task = asyncio.current_task()
        web.llm_interrupted_by_voice = False
        log.info("PERF: Iniciando geração do LLM para o prompt: '{}' (Origem: {})", prompt_text, origin)
        console.print(f"🤖 [bold cyan]Jarvis [{origin}]:[/bold cyan] ", end="")
        
        # Avisa a interface web que o Jarvis começou a falar/responder
        await web.broadcast_chat_status("speaking")
        
        full_response_parts = []
        sentence_buffer = ""
        first_token_received = False
        try:
            async for token in llm.generate_stream(prompt_text, language=lang, ha_client=ha_client):
                if llm._is_cancelled or web.llm_interrupted_by_voice:
                    raise asyncio.CancelledError()
                if not first_token_received:
                    log.info("PERF: Primeiro token verbal recebido do LLM.")
                    first_token_received = True
                sys.stdout.write(token)
                sys.stdout.flush()
                full_response_parts.append(token)

                # Transmite o texto gerado em tempo real para a interface do navegador
                current_text = "".join(full_response_parts)
                asyncio.create_task(
                    web.broadcast_chat_message({
                        "type": "token",
                        "sender": "jarvis",
                        "origin": origin,
                        "text": current_text
                    })
                )

                sentence_buffer += token
                sentences, sentence_buffer = extract_sentences(sentence_buffer)
                for sentence in sentences:
                    # Envia a frase para a fila do TTS em segundo plano (com o callback do navegador se disponível)
                    tts.speak_stream(sentence, lang, on_audio_chunk=on_audio_chunk)
                    if origin == "Navegador":
                        # Registra nos textos recentemente falados do navegador para AEC
                        clean_s = sentence.lower().strip()
                        for p in [".", ",", "!", "?", "-", '"', "'"]:
                            clean_s = clean_s.replace(p, "")
                        clean_s = clean_s.strip()
                        if clean_s:
                            web.browser_recent_texts.append((time.time(), clean_s))

            if llm._is_cancelled or web.llm_interrupted_by_voice:
                raise asyncio.CancelledError()

            # Envia a última parte restante do buffer ao finalizar a geração
            remaining_sentence = sentence_buffer.strip()
            if remaining_sentence:
                tts.speak_stream(remaining_sentence, lang, on_audio_chunk=on_audio_chunk)
                if origin == "Navegador":
                    clean_s = remaining_sentence.lower().strip()
                    for p in [".", ",", "!", "?", "-", '"', "'"]:
                        clean_s = clean_s.replace(p, "")
                    clean_s = clean_s.strip()
                    if clean_s:
                        web.browser_recent_texts.append((time.time(), clean_s))
            
            log.info("PERF: Geração do LLM concluída com sucesso.")
            
            # Salva o turno da conversa no banco de dados temporário
            full_response = "".join(full_response_parts)
            user_msg = original_query if original_query is not None else prompt_text
            cleaned_user_msg = user_msg
            # Remove "jarvis" do início se houver para deixar o contexto de histórico mais limpo
            if cleaned_user_msg.lower().strip().startswith("jarvis"):
                cleaned_user_msg = cleaned_user_msg.strip()[6:].strip(", ").strip()
            asyncio.create_task(llm.save_conversation_turn(cleaned_user_msg, full_response))
            
            # Notifica os chats web com o texto final da resposta
            await web.broadcast_chat_message({
                "type": "message",
                "sender": "jarvis",
                "origin": origin,
                "text": full_response
            })
        except asyncio.CancelledError:
            # Geração foi cancelada por interrupção de fala
            log.info("PERF: Geração do LLM cancelada por interrupção de voz.")
            tts.stop()
            console.print(" [bold red][Interrompido][/bold red]")
            await web.broadcast_chat_message({
                "type": "message",
                "sender": "jarvis",
                "origin": origin,
                "text": "".join(full_response_parts) + " [Interrompido]"
            })
        except Exception as e:
            log.error("PERF: Erro na geração de resposta do LLM: {}", e)
            console.print(f"\n[red]Erro ao gerar resposta do Jarvis: {e}[/red]")
        finally:
            sys.stdout.write("\n\n")
            sys.stdout.flush()
            web.llm_generating = False
            web.llm_interrupted_by_voice = False
            await web.broadcast_chat_status("idle")

    # Vincula o callback para que web.py possa disparar respostas no mesmo fluxo
    web.generate_response_callback = generate_response

    # Loop em background para capturar instruções via texto diretamente no terminal
    async def terminal_input_loop() -> None:
        import sys
        loop = asyncio.get_running_loop()
        while True:
            try:
                # Como sys.stdin.readline é blocante, executamos no executor padrão
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                text = line.strip()
                if not text:
                    continue

                # Trata comando de parada textual
                text_clean = text.lower().strip()
                for p in [".", ",", "!", "?", "-"]:
                    text_clean = text_clean.replace(p, "")
                text_clean = text_clean.strip()

                is_stop_command = text_clean in ("pare", "parar", "stop", "silêncio", "quieto", "espera")
                if is_stop_command:
                    if web.llm_task and not web.llm_task.done():
                        llm.interrupt()
                        web.llm_task.cancel()
                    tts.stop()
                    log.info("Conversa: Jarvis foi silenciado por comando de parada textual no Terminal.")
                    # Notifica a interface web
                    await web.broadcast_chat_message({
                        "type": "message",
                        "sender": "user",
                        "origin": "Terminal",
                        "text": text + " [Interrompido]"
                    })
                    await web.broadcast_chat_message({
                        "type": "message",
                        "sender": "jarvis",
                        "origin": "Terminal",
                        "text": "[Interrompido]"
                    })
                    continue

                # Interrompe qualquer geração/reprodução anterior ativa
                if web.llm_generating or tts._is_playing or (web.llm_task and not web.llm_task.done()):
                    if web.llm_task and not web.llm_task.done():
                        llm.interrupt()
                        web.llm_task.cancel()
                    tts.stop()
                    log.info("Conversa: Jarvis interrompido por entrada de texto no Terminal.")

                # Imprime no console terminal local indicando que recebeu o texto
                console.print(f"[bold green]🗣️  Você [Terminal] (texto):[/bold green] {text}")

                # Sincroniza o chat web
                await web.broadcast_chat_message({
                    "type": "message",
                    "sender": "user",
                    "origin": "Terminal",
                    "text": text
                })

                # Dispara a geração de resposta
                web.llm_task = asyncio.create_task(
                    generate_response(
                        prompt_text=text,
                        lang="pt",
                        original_query=text,
                        origin="Terminal"
                    )
                )
            except Exception as e:
                log.error(f"Erro no loop de entrada de texto do terminal: {e}")

    # Controle de transcrição parcial (para não sobrecarregar)
    last_partial_time = 0.0
    partial_interval_s = 0.35  # Transcreve parcial a cada 350ms

    console.print("[bold cyan]🎙️  Microfone Ativo. Pode começar a falar![/bold cyan]")
    console.print("[dim]Pressione Ctrl+C para encerrar ou digite texto para enviar instruções.[/dim]\n")

    mic.start()
    input_task = asyncio.create_task(terminal_input_loop())
    try:
        async for chunk in mic.stream():
            # Mantém tts_last_active_time atualizado enquanto o Jarvis estiver ocupado gerando ou falando
            is_jarvis_busy = web.llm_generating or tts._is_playing or not tts._queue.empty()
            if is_jarvis_busy:
                web.tts_last_active_time = time.time()

            # 1. Verifica se há fala no chunk de 30ms
            speech_detected = vad.is_speech(chunk, is_jarvis_busy=is_jarvis_busy)

            # Suspende a escuta de palmas enquanto o Jarvis ou o usuário estão falando
            mic.is_listening_for_claps = not (is_jarvis_busy or is_speaking)

            # Atualização dinâmica do indicador visual no console para Conversa Fluida
            now = time.time()
            if web.tts_last_active_time > 0.0:
                is_conversa_fluida = (now - web.tts_last_active_time) < (settings.audio.full_duplex_cooldown_ms / 1000)
                if is_conversa_fluida:
                    # Mostra o indicador somente quando Jarvis terminar de falar/processar para não poluir
                    if not is_jarvis_busy and not conversa_fluida_active:
                        conversa_fluida_active = True
                        console.print("\n[bold yellow]⚡ Escuta Direta Ativa (Pode falar sem chamar 'Jarvis')...[/bold yellow]")
                else:
                    if conversa_fluida_active:
                        conversa_fluida_active = False
                        console.print("\n[dim]💤 Escuta Ociosa (Diga 'Jarvis' para reativar).[/dim]")


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

                        # Se o áudio for longo o suficiente, transcreve e comete (mínimo de 800ms para evitar RTF ruim em clips curtos)
                        if len(full_audio) > settings.audio.sample_rate * 0.8:
                            log.info("PERF: Fim de fala detectado (VAD finalizado). Iniciando transcrição com Whisper STT...")
                            res = await transcriber.transcribe(full_audio)
                            if res:
                                final_text, detected_lang = res
                                log.info("PERF: Transcrição Whisper concluída. Texto: '{}' (Idioma: {})", final_text, detected_lang)
                                if final_text:
                                    # Define se Jarvis estava ativo (gerando, falando ou recém-interrompido)
                                    was_jarvis_active = web.llm_generating or tts._is_playing or not tts._queue.empty() or web.llm_interrupted_by_voice

                                    # Limpa a transcrição para verificar se é eco da própria voz do Jarvis
                                    final_text_clean = final_text.lower().strip()
                                    for p in [".", ",", "!", "?", "-", '"', "'"]:
                                        final_text_clean = final_text_clean.replace(p, "")
                                    final_text_clean = final_text_clean.strip()

                                    # Rastreia e cancela se for eco do TTS
                                    is_echo = False
                                    current_spoken = tts.current_spoken_text
                                    if current_spoken:
                                        # Bypass AEC if the user is clearly speaking a stop command (to handle barge-in while speaking)
                                        has_stop_word = any(word in final_text_clean for word in ("pare", "parar", "cala a boca", "silêncio", "quieto", "stop", "shut up", "be quiet", "silence", "pera", "espera", "calma", "chega", "shh", "shush"))
                                        
                                        if not has_stop_word:
                                            # 1. Correspondência direta de substring
                                            if final_text_clean in current_spoken or current_spoken in final_text_clean:
                                                is_echo = True
                                            # 2. Correspondência de interseção de palavras
                                            else:
                                                words_trans = set(final_text_clean.split())
                                                words_spok = set(current_spoken.split())
                                                if words_trans and words_spok:
                                                    intersection = words_trans.intersection(words_spok)
                                                    if len(intersection) / len(words_trans) > 0.6:
                                                        is_echo = True

                                    if is_echo:
                                        log.info("AEC: Transcrição de eco ignorada: '{}'", final_text)
                                        audio_buffer.clear()
                                        silent_chunks = 0
                                        web.llm_interrupted_by_voice = False
                                        continue

                                    # Definição das palavras de parada e ativação suportadas
                                    stop_words = ("jarvis", "para", "pare", "parar", "cala a boca", "silêncio", "quieto", "stop", "shut up", "be quiet", "silence", "pera", "espera", "calma", "chega", "shh", "shush")

                                    # Define se estamos no período de conversa continuada
                                    is_conversa_fluida = False
                                    if web.tts_last_active_time > 0.0:
                                        is_conversa_fluida = (time.time() - web.tts_last_active_time) < (settings.audio.full_duplex_cooldown_ms / 1000)

                                    # Se Jarvis estava ativo, interrompe apenas com palavras de parada suportadas.
                                    # Se estiver na janela de conversa fluida, aceita qualquer comando diretamente.
                                    # Se estiver ocioso, exige a palavra de ativação "jarvis".
                                    if was_jarvis_active:
                                        should_process = any(word in final_text.lower() for word in stop_words)
                                    elif is_conversa_fluida:
                                        should_process = True
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
                                            if web.llm_task and not web.llm_task.done():
                                                llm.interrupt()
                                                web.llm_task.cancel()
                                            tts.stop()
                                            log.info("Conversa: Jarvis foi silenciado por comando de voz.")

                                            if not partial_newline_printed:
                                                sys.stdout.write("\n")
                                                sys.stdout.flush()
                                            sys.stdout.write("\r\033[K")
                                            console.print(f"[bold green]🗣️  Você [Terminal] ({detected_lang}):[/bold green] {final_text} [bold red][Interrompido][/bold red]")
                                            # Sincroniza o chat web
                                            asyncio.create_task(
                                                web.broadcast_chat_message({
                                                    "type": "message",
                                                    "sender": "user",
                                                    "origin": "Terminal",
                                                    "text": final_text + " [Interrompido]",
                                                    "lang": detected_lang
                                                })
                                            )
                                        else:
                                            # Se o LLM está gerando ou o TTS está reproduzindo, interrompe a geração anterior antes de responder
                                            if web.llm_generating or tts._is_playing or (web.llm_task and not web.llm_task.done()):
                                                if web.llm_task and not web.llm_task.done():
                                                    llm.interrupt()
                                                    web.llm_task.cancel()
                                                tts.stop()
                                                log.info("Conversa: Jarvis foi interrompido pela fala do usuário (nova pergunta).")

                                            # Apaga a linha parcial e imprime a final com destaque
                                            if not partial_newline_printed:
                                                sys.stdout.write("\n")
                                                sys.stdout.flush()
                                            sys.stdout.write("\r\033[K")  # Limpa linha
                                            console.print(f"[bold green]🗣️  Você [Terminal] ({detected_lang}):[/bold green] {final_text}")
                                            
                                            # Sincroniza o chat web
                                            asyncio.create_task(
                                                web.broadcast_chat_message({
                                                    "type": "message",
                                                    "sender": "user",
                                                    "origin": "Terminal",
                                                    "text": final_text,
                                                    "lang": detected_lang
                                                })
                                            )
                                            
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
                                                
                                                # Resposta canned imediata — pula LLM inteiramente (~1.1s economizados)
                                                _action_map = {
                                                    "turn_on":  ("Ligando",    "turned on"),
                                                    "turn_off": ("Desligando", "turned off"),
                                                    "toggle":   ("Alternando", "toggled"),
                                                }
                                                _verb_pt, _state_en = _action_map.get(intent, ("Executando", "done"))
                                                if detected_lang == "en":
                                                    canned_response = f"{entity_obj['name']} {_state_en}, Sir."
                                                else:
                                                    canned_response = f"{_verb_pt} {entity_obj['name']}, Senhor."

                                                # Fala imediatamente via TTS sem passar pelo LLM
                                                await asyncio.to_thread(tts.speak_stream, canned_response, detected_lang)
                                                asyncio.create_task(
                                                    web.broadcast_chat_message({
                                                        "type": "message", "sender": "jarvis",
                                                        "origin": "Terminal", "text": canned_response
                                                    })
                                                )
                                                asyncio.create_task(
                                                    llm.save_conversation_turn(final_text, canned_response)
                                                )
                                            else:
                                                # Fallback padrão: Envia para o LLM resolver via tool calling
                                                web.llm_task = asyncio.create_task(
                                                    generate_response(
                                                        prompt_text=final_text,
                                                        lang=detected_lang,
                                                        original_query=final_text,
                                                        origin="Terminal"
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
                        web.llm_interrupted_by_voice = False

            # 2. Transcrição Parcial (Real-time Feedback)
            if is_speaking and len(audio_buffer) > 0 and not partial_in_progress:
                now = time.time()
                is_jarvis_busy = web.llm_generating or tts._is_playing or not tts._queue.empty()
                current_interval = 0.20 if is_jarvis_busy else partial_interval_s
                if now - last_partial_time >= current_interval:
                    last_partial_time = now

                    # Concatena o áudio acumulado até o momento
                    partial_audio = np.concatenate(audio_buffer, axis=0).flatten()

                    # Slice the last 1.5 seconds of audio to avoid Whisper queue lag and keep transcriptions instant
                    max_samples = int(settings.audio.sample_rate * 1.5)
                    if len(partial_audio) > max_samples:
                        partial_audio = partial_audio[-max_samples:]

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
                                    if web.llm_generating or tts._is_playing or not tts._queue.empty():
                                        cleaned_text = text.lower().strip()
                                        for p in [".", ",", "!", "?", "-"]:
                                            cleaned_text = cleaned_text.replace(p, "")
                                        cleaned_text = cleaned_text.strip()

                                        stop_words = ("pare", "parar", "cala a boca", "silêncio", "quieto", "stop", "shut up", "be quiet", "silence", "pera", "espera", "calma", "chega", "shh", "shush")
                                        if any(word in cleaned_text for word in stop_words):
                                            # AEC check for terminal partial transcription
                                            is_echo = False
                                            current_spoken = tts.current_spoken_text
                                            if current_spoken:
                                                if cleaned_text in current_spoken or current_spoken in cleaned_text:
                                                    is_echo = True
                                                else:
                                                    words_trans = set(cleaned_text.split())
                                                    words_spok = set(current_spoken.split())
                                                    if words_trans and words_spok:
                                                        intersection = words_trans.intersection(words_spok)
                                                        if len(intersection) / len(words_trans) > 0.6:
                                                            is_echo = True
                                            
                                            if is_echo:
                                                log.debug("AEC Parcial: Eco ignorado no terminal: '{}'", text)
                                                return
                                            
                                            if web.llm_task and not web.llm_task.done():
                                                llm.interrupt()
                                                web.llm_task.cancel()
                                                web.llm_interrupted_by_voice = True
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
        input_task.cancel()
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
