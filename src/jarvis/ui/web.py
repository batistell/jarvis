"""Jarvis — Servidor Backend FastAPI & WebSockets.

Gerencia as conexões do navegador, o streaming de áudio bidirecional e a
sincronização de mensagens e telemetria da GPU.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import FileResponse
from loguru import logger

# Configurações do Jarvis
from jarvis.config.settings import get_settings

app = FastAPI(title="Jarvis Web Dashboard")

# Referências globais dos motores pre-carregados (injetadas por main.py)
llm_engine = None
tts_engine = None
ha_client = None
transcriber_engine = None
generate_response_callback: Callable | None = None

# Lista de conexões WebSocket de chat ativas
active_chat_connections: list[WebSocket] = []

# Estado da conversa compartilhado entre terminal e navegador
llm_generating = False
llm_task: asyncio.Task | None = None
tts_last_active_time = 0.0
llm_interrupted_by_voice = False

# Rastreamento específico do áudio no navegador
browser_tts_end_time = 0.0
browser_recent_texts: list[tuple[float, str]] = []

# Mapeia caminhos de arquivos estáticos do dashboard
UI_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
UI_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


def get_gpu_stats() -> dict:
    """Lê as estatísticas de telemetria da GPU RTX 3060 via NVML ou nvidia-smi."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "vram_total": round(info.total / (1024 ** 2), 1),
            "vram_used": round(info.used / (1024 ** 2), 1),
            "vram_free": round(info.free / (1024 ** 2), 1),
            "vram_percent": round((info.used / info.total) * 100, 1),
            "temperature": temp,
            "cuda_load": util.gpu,
            "success": True
        }
    except Exception as nvml_err:
        logger.debug(f"Falha ao carregar NVML (pynvml): {nvml_err}. Tentando fallback nvidia-smi...")
        # Fallback de leitura via comando nvidia-smi (compatível com Windows)
        try:
            import subprocess
            cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.used,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            parts = res.stdout.strip().split(",")
            v_total = float(parts[0].strip())
            v_used = float(parts[1].strip())
            temp = int(parts[2].strip())
            util = int(parts[3].strip())
            return {
                "vram_total": v_total,
                "vram_used": v_used,
                "vram_free": v_total - v_used,
                "vram_percent": round((v_used / v_total) * 100, 1),
                "temperature": temp,
                "cuda_load": util,
                "success": True
            }
        except Exception as smi_err:
            logger.debug(f"nvidia-smi indisponível: {smi_err}")
            # Retorna estatísticas mockadas seguras para fins de desenvolvimento/testes
            return {
                "vram_total": 12288.0,
                "vram_used": 5500.0,
                "vram_free": 6788.0,
                "vram_percent": 44.8,
                "temperature": 55,
                "cuda_load": 12,
                "success": False
            }


async def broadcast_chat_message(message_data: dict) -> None:
    """Envia uma mensagem de chat em broadcast para todos os navegadores abertos."""
    for ws in list(active_chat_connections):
        try:
            await ws.send_json(message_data)
        except Exception:
            if ws in active_chat_connections:
                active_chat_connections.remove(ws)


async def broadcast_chat_status(status_str: str) -> None:
    """Informa o status do Jarvis (idle, recording, thinking, speaking) aos clientes."""
    await broadcast_chat_message({
        "type": "status",
        "status": status_str
    })


# --- Rotas REST ---

@app.get("/")
async def get_index() -> FileResponse:
    """Retorna a interface visual principal."""
    index_file = UI_TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html não encontrado no diretório templates.")
    return FileResponse(index_file)


@app.get("/manifest.json")
async def get_manifest() -> dict:
    """Retorna o manifesto PWA do aplicativo."""
    return {
        "short_name": "Jarvis",
        "name": "Jarvis Web Dashboard",
        "description": "Painel de controle por voz do assistente Jarvis",
        "icons": [
            {
                "src": "/icon.svg",
                "sizes": "192x192 512x512",
                "type": "image/svg+xml",
                "purpose": "any maskable"
            }
        ],
        "start_url": "/",
        "background_color": "#080300",
        "theme_color": "#ff5500",
        "display": "standalone",
        "orientation": "portrait"
    }


@app.get("/icon.svg")
async def get_icon() -> Response:
    """Retorna o ícone SVG futurista do Jarvis."""
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
        <!-- Background -->
        <rect width="512" height="512" rx="128" fill="#080300"/>
        
        <!-- Faint Outer Ring -->
        <circle cx="256" cy="256" r="200" fill="none" stroke="rgba(255, 85, 0, 0.05)" stroke-width="4"/>
        
        <!-- Outer dashed HUD circle (Glowing Orange) -->
        <!-- Outer Glow -->
        <circle cx="256" cy="256" r="130" fill="none" stroke="#ff5500" stroke-width="28" stroke-dasharray="24, 24" opacity="0.25"/>
        <!-- Core line -->
        <circle cx="256" cy="256" r="130" fill="none" stroke="#ff5500" stroke-width="12" stroke-dasharray="28, 20"/>
        
        <!-- Inner solid HUD circle (Glowing Gold/Yellow) -->
        <!-- Inner Glow -->
        <circle cx="256" cy="256" r="70" fill="none" stroke="#ffaa00" stroke-width="20" stroke-dasharray="none" opacity="0.3"/>
        <!-- Core line -->
        <circle cx="256" cy="256" r="70" fill="none" stroke="#ffaa00" stroke-width="8"/>
        
        <!-- Central Core (Glowing White/Orange Dot) -->
        <circle cx="256" cy="256" r="20" fill="#ff5500" opacity="0.3"/>
        <circle cx="256" cy="256" r="10" fill="#ffffff"/>
    </svg>"""
    return Response(content=svg_content, media_type="image/svg+xml")


@app.get("/sw.js")
async def get_service_worker() -> Response:
    """Retorna um Service Worker básico para habilitar instalação de PWA no navegador."""
    sw_content = """
    self.addEventListener('install', (e) => {
        self.skipWaiting();
    });
    self.addEventListener('fetch', (e) => {
        // Pass-through
    });
    """
    return Response(content=sw_content, media_type="application/javascript")


@app.get("/api/stats")
async def get_stats_api() -> dict:
    """API REST para telemetria de GPU."""
    return get_gpu_stats()


@app.get("/api/ha/devices")
async def get_ha_devices() -> list[dict]:
    """Retorna os dispositivos integrados do Home Assistant."""
    if ha_client and ha_client.entities:
        return ha_client.entities
    return []


@app.post("/api/ha/control")
async def post_ha_control(data: dict) -> dict:
    """Executa ações rápidas de automação do Home Assistant."""
    if not ha_client:
        raise HTTPException(status_code=503, detail="Home Assistant não configurado.")
    domain = data.get("domain")
    service = data.get("service")
    entity_id = data.get("entity_id")
    extra_data = data.get("data")

    if not service or not entity_id:
        raise HTTPException(status_code=400, detail="Parâmetros inválidos.")

    asyncio.create_task(
        ha_client.control_entity(
            domain=domain or entity_id.split(".")[0],
            service=service,
            entity_id=entity_id,
            data=extra_data
        )
    )
    return {"status": "triggered"}


# --- WebSockets ---

@app.websocket("/ws/chat")
async def ws_chat_endpoint(websocket: WebSocket) -> None:
    """Gerencia conexões WebSocket de chat e atualizações de telemetria."""
    await websocket.accept()
    active_chat_connections.append(websocket)
    logger.info("WS Chat: Cliente conectado.")

    # Loop para enviar atualizações de telemetria periódicas e ler mensagens textuais
    async def telemetry_loop():
        try:
            import time
            settings = get_settings()
            while websocket in active_chat_connections:
                # Calcula se a escuta direta (conversa fluida) está ativa no momento
                is_fluida = False
                if tts_last_active_time > 0.0:
                    is_fluida = (time.time() - tts_last_active_time) < (settings.audio.full_duplex_cooldown_ms / 1000)

                is_browser_playing = time.time() < (browser_tts_end_time + 0.5)
                is_jarvis_busy = llm_generating or is_browser_playing or (tts_engine and tts_engine._is_playing) or (tts_engine and not tts_engine._queue.empty())

                conversa_fluida_active = is_fluida and not is_jarvis_busy

                stats = get_gpu_stats()
                await websocket.send_json({
                    "type": "telemetry",
                    "data": stats,
                    "conversa_fluida": conversa_fluida_active
                })
                await asyncio.sleep(1.5)
        except Exception:
            pass

    telemetry_task = asyncio.create_task(telemetry_loop())

    try:
        while True:
            data_str = await websocket.receive_text()
            data = json.loads(data_str)

            # Trata mensagens textuais enviadas pela interface web
            if data.get("type") == "message" and data.get("text"):
                text = data["text"]
                logger.info(f"WS Chat: Recebida pergunta textual do Navegador: '{text}'")

                # Trata comando de parada textual
                text_clean = text.lower().strip()
                for p in [".", ",", "!", "?", "-"]:
                    text_clean = text_clean.replace(p, "")
                text_clean = text_clean.strip()

                is_stop_command = text_clean in ("pare", "parar", "stop", "silêncio", "quieto", "espera")
                if is_stop_command:
                    if llm_task and not llm_task.done():
                        llm_engine.interrupt()
                        llm_task.cancel()
                    if tts_engine:
                        tts_engine.stop()
                    llm_interrupted_by_voice = True
                    browser_tts_end_time = 0.0
                    logger.info("WS Chat: Jarvis silenciado por comando de parada textual do Navegador.")

                    await broadcast_chat_message({
                        "type": "message",
                        "sender": "user",
                        "origin": "Navegador",
                        "text": text + " [Interrompido]"
                    })
                    await broadcast_chat_message({
                        "type": "message",
                        "sender": "jarvis",
                        "origin": "Navegador",
                        "text": "[Interrompido]"
                    })
                    await broadcast_chat_status("idle")
                    continue

                # Imprime no terminal local indicando origem
                print(f"\n🗣️  Você [Navegador] (texto): {text}")

                # Re-transmite a pergunta em broadcast para atualizar todos os chats
                await broadcast_chat_message({
                    "type": "message",
                    "sender": "user",
                    "origin": "Navegador",
                    "text": text
                })

                # Dispara a resposta do LLM em segundo plano
                if generate_response_callback:
                    # Envia o áudio gerado pelo TTS de volta pelo WebSocket de áudio associado se ativo
                    # Ou cria um broadcast genérico de áudio se preferível.
                    # Como o canal de texto e áudio são separados, localizamos conexões de áudio.
                    # Para simplificar, mensagens de texto respondem apenas via texto e fala local,
                    # ou mandam a resposta no chat visual (broadcast de tokens já acontece).
                    asyncio.create_task(
                        generate_response_callback(
                            prompt_text=text,
                            lang="pt",  # assume português por padrão em interações textuais locais
                            original_query=text,
                            origin="Navegador"
                        )
                    )
    except WebSocketDisconnect:
        logger.info("WS Chat: Cliente desconectado.")
    except Exception as e:
        logger.error(f"WS Chat: Erro na conexão: {e}")
    finally:
        telemetry_task.cancel()
        if websocket in active_chat_connections:
            active_chat_connections.remove(websocket)


@app.websocket("/ws/audio")
async def ws_audio_endpoint(websocket: WebSocket) -> None:
    """Canal de baixa latência para streaming de áudio bidirecional."""
    global llm_generating, llm_task, tts_last_active_time, llm_interrupted_by_voice, browser_tts_end_time, browser_recent_texts
    await websocket.accept()
    logger.info("WS Audio: Conexão de áudio estabelecida.")

    # Detector VAD e Transcriber exclusivos para este fluxo de microfone móvel
    from jarvis.stt.vad import VADDetector
    vad = VADDetector()

    sample_buffer = np.array([], dtype=np.float32)
    audio_buffer = []
    is_speaking = False
    silent_chunks = 0

    settings = get_settings()
    max_silent_chunks = settings.audio.silence_threshold_ms // settings.audio.chunk_duration_ms

    # Parâmetros de transcrição parcial
    last_partial_time = 0.0
    partial_interval_s = 0.35
    partial_in_progress = False

    # Callback para capturar e enviar os pacotes do TTS de volta a esta conexão via loop asyncio
    loop = asyncio.get_running_loop()
    def send_tts_chunk(audio_float_array: np.ndarray, sample_rate: int = 22050) -> None:
        import time
        duration = len(audio_float_array) / sample_rate
        now = time.time()
        global browser_tts_end_time
        if browser_tts_end_time < now:
            browser_tts_end_time = now + duration
        else:
            browser_tts_end_time += duration

        async def send():
            try:
                # Transmite os dados binários float32 prepended com o sample_rate (como float32 de 4 bytes)
                # para que o navegador possa ler dinamicamente a frequência correta.
                header = np.array([float(sample_rate)], dtype=np.float32)
                combined = np.concatenate((header, audio_float_array))
                await websocket.send_bytes(combined.tobytes())
            except Exception:
                pass
        asyncio.run_coroutine_threadsafe(send(), loop)

    try:
        while True:
            # Recebe o buffer binário de áudio (resampled a 16kHz Float32 no frontend)
            data = await websocket.receive_bytes()
            if not data:
                break

            chunk = np.frombuffer(data, dtype=np.float32)
            if len(chunk) == 0:
                continue

            sample_buffer = np.concatenate((sample_buffer, chunk))

            # Alimenta o VAD com fatias de 30ms (480 samples)
            while len(sample_buffer) >= 480:
                vad_chunk = sample_buffer[:480]
                sample_buffer = sample_buffer[480:]

                # Mantém tts_last_active_time atualizado se Jarvis estiver gerando ou tocando no navegador
                import time
                is_browser_playing = time.time() < (browser_tts_end_time + 0.5)
                is_jarvis_busy = llm_generating or is_browser_playing or (tts_engine and tts_engine._is_playing) or (tts_engine and not tts_engine._queue.empty())
                if is_jarvis_busy:
                    tts_last_active_time = time.time()

                speech_detected = vad.is_speech(vad_chunk, is_jarvis_busy=is_jarvis_busy)

                if speech_detected:
                    if not is_speaking:
                        is_speaking = True
                        logger.info("WS Audio: Captação de voz activa iniciada.")
                        await broadcast_chat_status("recording")
                        llm_interrupted_by_voice = False
                    silent_chunks = 0
                    audio_buffer.append(vad_chunk)
                else:
                    if is_speaking:
                        audio_buffer.append(vad_chunk)
                        silent_chunks += 1

                        if silent_chunks >= max_silent_chunks:
                            is_speaking = False
                            logger.info("WS Audio: Silêncio de fim de frase detectado.")
                            await broadcast_chat_status("thinking")

                            full_audio = np.concatenate(audio_buffer, axis=0)
                            audio_buffer = []
                            silent_chunks = 0

                            # Executa a transcrição apenas se houver tamanho mínimo de voz (800ms — abaixo disso o RTF do Whisper é > 1×)
                            if len(full_audio) > 16000 * 0.8 and transcriber_engine:
                                res = await transcriber_engine.transcribe(full_audio)
                                if res:
                                    text, lang = res
                                    if text:
                                        import time
                                        text_clean = text.lower().strip()
                                        for p in [".", ",", "!", "?", "-", '"', "'"]:
                                            text_clean = text_clean.replace(p, "")
                                        text_clean = text_clean.strip()

                                        # Define se Jarvis estava ativo (gerando, falando ou recém-interrompido)
                                        is_browser_playing = time.time() < (browser_tts_end_time + 0.5)
                                        was_jarvis_active = llm_generating or is_browser_playing or (tts_engine and tts_engine._is_playing) or (tts_engine and not tts_engine._queue.empty()) or llm_interrupted_by_voice

                                        # AEC Check (Echo Cancellation)
                                        is_echo = False
                                        now = time.time()
                                        browser_recent_texts = [entry for entry in browser_recent_texts if now - entry[0] < 20.0]

                                        # Se o usuário falou uma palavra de parada, faz bypass do AEC para garantir que a interrupção ocorra
                                        has_stop_word = any(word in text_clean for word in ("pare", "parar", "cala a boca", "silêncio", "quieto", "stop", "shut up", "be quiet", "silence", "pera", "espera", "calma", "chega", "shh", "shush"))

                                        if not has_stop_word:
                                            for _, spoken_text in browser_recent_texts:
                                                if text_clean in spoken_text or spoken_text in text_clean:
                                                    is_echo = True
                                                    break
                                                else:
                                                    words_trans = set(text_clean.split())
                                                    words_spok = set(spoken_text.split())
                                                    if words_trans and words_spok:
                                                        intersection = words_trans.intersection(words_spok)
                                                        if len(intersection) / len(words_trans) > 0.6:
                                                            is_echo = True
                                                            break

                                        if is_echo:
                                            logger.info(f"WS Audio AEC: Eco do TTS ignorado no Navegador: '{text}'")
                                            await broadcast_chat_status("idle")
                                            llm_interrupted_by_voice = False
                                            continue

                                        # Regras de processamento (Wake Word / Stop Words / Conversa Fluida)
                                        stop_words = ("jarvis", "para", "pare", "parar", "cala a boca", "silêncio", "quieto", "stop", "shut up", "be quiet", "silence", "pera", "espera", "calma", "chega", "shh", "shush")

                                        is_conversa_fluida = False
                                        if tts_last_active_time > 0.0:
                                            is_conversa_fluida = (time.time() - tts_last_active_time) < (settings.audio.full_duplex_cooldown_ms / 1000)

                                        if was_jarvis_active:
                                            should_process = any(word in text.lower() for word in stop_words)
                                        elif is_conversa_fluida:
                                            should_process = True
                                        else:
                                            should_process = "jarvis" in text.lower()

                                        if not should_process:
                                            logger.info(f"WS Audio: Frase ignorada (sem palavra de ativação/parada): '{text}'")
                                            await broadcast_chat_status("idle")
                                            llm_interrupted_by_voice = False
                                            continue

                                        # Limpa comando de "jarvis"
                                        cleaned_cmd = text.lower().replace("jarvis", "").strip()
                                        for p in [".", ",", "!", "?", "-"]:
                                            cleaned_cmd = cleaned_cmd.replace(p, "")
                                        cleaned_cmd = cleaned_cmd.strip()

                                        is_stop_term = any(term in cleaned_cmd for term in ("pare", "parar", "cala a boca", "silêncio", "quieto", "stop", "shut up", "be quiet", "silence", "pera", "espera", "calma", "chega", "shh", "shush"))
                                        is_just_name = cleaned_cmd == "" or text.lower().strip() in ("jarvis", "jarvis.", "jarvis!", "jarvis?")
                                        is_stop_command = is_stop_term or (is_just_name and was_jarvis_active)

                                        if is_stop_command:
                                            # Interrompe geração e áudio
                                            if llm_task and not llm_task.done():
                                                llm_engine.interrupt()
                                                llm_task.cancel()
                                            if tts_engine:
                                                tts_engine.stop()
                                            llm_interrupted_by_voice = True
                                            browser_tts_end_time = 0.0
                                            logger.info("WS Audio: Jarvis silenciado por comando de voz do Navegador.")

                                            await broadcast_chat_message({
                                                "type": "message",
                                                "sender": "user",
                                                "origin": "Navegador",
                                                "text": text + " [Interrompido]",
                                                "lang": lang
                                            })
                                            await broadcast_chat_message({
                                                "type": "message",
                                                "sender": "jarvis",
                                                "origin": "Navegador",
                                                "text": "[Interrompido]"
                                            })
                                            await broadcast_chat_status("idle")
                                            continue
                                        else:
                                            # Interrompe geração anterior se ativa antes de responder nova pergunta
                                            if llm_generating or is_browser_playing or (tts_engine and tts_engine._is_playing) or (llm_task and not llm_task.done()):
                                                if llm_task and not llm_task.done():
                                                    llm_engine.interrupt()
                                                    llm_task.cancel()
                                                if tts_engine:
                                                    tts_engine.stop()
                                                browser_tts_end_time = 0.0
                                                logger.info("WS Audio: Jarvis interrompido por nova pergunta do Navegador.")

                                            # Imprime no terminal local
                                            print(f"\n🗣️  Você [Navegador] ({lang}): {text}")

                                            # Notifica todas as janelas do chat
                                            await broadcast_chat_message({
                                                "type": "message",
                                                "sender": "user",
                                                "origin": "Navegador",
                                                "text": text,
                                                "lang": lang
                                            })

                                            # Dispara inferência
                                            if generate_response_callback:
                                                asyncio.create_task(
                                                    generate_response_callback(
                                                        prompt_text=text,
                                                        lang=lang,
                                                        original_query=text,
                                                        on_audio_chunk=send_tts_chunk,
                                                        origin="Navegador"
                                                    )
                                                )
                            else:
                                await broadcast_chat_status("idle")

                            # Safety reset of voice interruption flag at the end of final processing
                            llm_interrupted_by_voice = False

            # 2. Transcrição Parcial (Real-time Feedback e Interrupção Imediata)
            if is_speaking and len(audio_buffer) > 0 and not partial_in_progress:
                import time
                now = time.time()
                is_browser_playing = time.time() < (browser_tts_end_time + 0.5)
                is_jarvis_busy = llm_generating or is_browser_playing or (tts_engine and tts_engine._is_playing) or (tts_engine and not tts_engine._queue.empty())
                current_interval = 0.20 if is_jarvis_busy else partial_interval_s
                if now - last_partial_time >= current_interval:
                    last_partial_time = now
                    partial_audio = np.concatenate(audio_buffer, axis=0).flatten()

                    # Slice the last 1.5 seconds of audio to avoid Whisper queue lag and keep transcriptions instant
                    max_samples = int(16000 * 1.5)
                    if len(partial_audio) > max_samples:
                        partial_audio = partial_audio[-max_samples:]

                    async def transcribe_partial(audio_data: np.ndarray) -> None:
                        nonlocal partial_in_progress, audio_buffer, is_speaking, silent_chunks
                        global llm_interrupted_by_voice, browser_tts_end_time, browser_recent_texts
                        partial_in_progress = True
                        try:
                            if transcriber_engine:
                                res = await transcriber_engine.transcribe(audio_data, is_partial=True)
                                if res:
                                    text, _ = res
                                    if text:
                                        # Verifica se Jarvis está ativo
                                        import time
                                        is_browser_playing = time.time() < (browser_tts_end_time + 0.5)
                                        was_jarvis_active = llm_generating or is_browser_playing or (tts_engine and tts_engine._is_playing) or (tts_engine and not tts_engine._queue.empty()) or llm_interrupted_by_voice

                                        if was_jarvis_active:
                                            text_clean = text.lower().strip()
                                            for p in [".", ",", "!", "?", "-"]:
                                                text_clean = text_clean.replace(p, "")
                                            text_clean = text_clean.strip()

                                            stop_words = ("pare", "parar", "cala a boca", "silêncio", "quieto", "stop", "shut up", "be quiet", "silence", "pera", "espera", "calma", "chega", "shh", "shush")
                                            if any(word in text_clean for word in stop_words):
                                                # Para transcrição parcial com palavras de parada, fazemos bypass completo do AEC
                                                # para garantir interrupção de voz ultra-responsiva
                                                is_echo = False

                                                # Interrompe geração e áudio imediatamente!
                                                if llm_task and not llm_task.done():
                                                    llm_engine.interrupt()
                                                    llm_task.cancel()
                                                if tts_engine:
                                                    tts_engine.stop()

                                                llm_interrupted_by_voice = True
                                                browser_tts_end_time = 0.0

                                                logger.info(f"WS Audio: Jarvis foi interrompido imediatamente ao detectar a palavra de parada '{text_clean}' via transcrição parcial.")

                                                # Limpa buffers de fala atuais
                                                audio_buffer = []
                                                is_speaking = False
                                                silent_chunks = 0

                                                # Notifica todos os clientes
                                                await broadcast_chat_status("idle")
                                                await broadcast_chat_message({
                                                    "type": "message",
                                                    "sender": "user",
                                                    "origin": "Navegador",
                                                    "text": text + " [Interrompido]"
                                                })
                                                await broadcast_chat_message({
                                                    "type": "message",
                                                    "sender": "jarvis",
                                                    "origin": "Navegador",
                                                    "text": "[Interrompido]"
                                                })
                        except Exception as pe:
                            logger.error(f"WS Audio: Erro na transcrição parcial: {pe}")
                        finally:
                            partial_in_progress = False

                    asyncio.create_task(transcribe_partial(partial_audio))

    except WebSocketDisconnect:
        logger.info("WS Audio: Conexão de áudio desconectada pelo cliente.")
    except Exception as e:
        logger.error(f"WS Audio: Falha no loop de recebimento: {e}")
