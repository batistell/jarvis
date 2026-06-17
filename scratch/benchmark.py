"""
Jarvis — Comprehensive Pipeline Benchmark
==========================================
Mede a latência e uso de VRAM de cada estágio da pipeline:
  1. Embedding Engine (BAAI/bge-m3 na CPU)
  2. ChromaDB Vector Store Query
  3. LLM Engine — Load + TTFT + Tokens/s
  4. Faster Whisper STT — Load + Latência de transcrição
  5. Piper TTS — Load + Latência do 1º chunk de áudio
  6. Simulação End-to-End (voz → LLM → TTS)

Uso:
    .venv\\Scripts\\python scratch\\benchmark.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Windows CUDA DLL setup (espelhado do main.py)
import ctypes
_dll_handles = []
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia_dir = site_packages / "nvidia"
    if nvidia_dir.exists():
        for bin_dir in nvidia_dir.glob("**/bin"):
            try:
                _dll_handles.append(os.add_dll_directory(str(bin_dir.resolve())))
            except Exception:
                pass
        for dll_name, rel_path in [
            ("cudart64_12.dll", "cuda_runtime/bin"),
            ("cublasLt64_12.dll", "cublas/bin"),
            ("cublas64_12.dll", "cublas/bin"),
            ("cudnn64_9.dll", "cudnn/bin"),
        ]:
            dll_path = nvidia_dir / rel_path / dll_name
            if dll_path.exists():
                try:
                    ctypes.CDLL(str(dll_path.resolve()))
                except Exception:
                    pass

# ── VRAM helper ──────────────────────────────────────────────────────────────
def get_vram_mb() -> dict[str, float]:
    """Retorna uso de VRAM via pynvml com fallback para nvidia-smi."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {
            "used_mb": round(info.used / 1024**2, 1),
            "free_mb": round(info.free / 1024**2, 1),
            "total_mb": round(info.total / 1024**2, 1),
        }
    except Exception:
        pass
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        parts = r.stdout.strip().split(",")
        return {
            "used_mb": float(parts[0].strip()),
            "free_mb": float(parts[1].strip()),
            "total_mb": float(parts[2].strip()),
        }
    except Exception:
        return {"used_mb": -1, "free_mb": -1, "total_mb": -1}


# ── Pretty print helpers ─────────────────────────────────────────────────────
SEP = "─" * 68

def section(title: str) -> None:
    print(f"\n{'═' * 68}")
    print(f"  {title}")
    print('═' * 68)

def row(label: str, value: Any, unit: str = "", warn_above: float | None = None) -> None:
    val_str = f"{value:.3f}" if isinstance(value, float) else str(value)
    line = f"  {label:<42} {val_str:>10} {unit}"
    if warn_above is not None and isinstance(value, (int, float)) and value > warn_above:
        print(f"⚠️  {line}  ← GARGALO")
    else:
        print(f"   {line}")

def vram_row(label: str, before: dict, after: dict) -> None:
    delta = after["used_mb"] - before["used_mb"]
    sign = "+" if delta >= 0 else ""
    print(f"   {'VRAM ' + label:<42} {after['used_mb']:>8.1f} MB  ({sign}{delta:.1f} MB)")


# ── Benchmark stages ─────────────────────────────────────────────────────────
results: dict[str, Any] = {}


async def bench_embedding() -> None:
    section("1 · EMBEDDING ENGINE  (BAAI/bge-m3 — CPU)")
    from jarvis.vectorstore.embeddings import EmbeddingEngine

    v0 = get_vram_mb()
    t0 = time.perf_counter()
    engine = EmbeddingEngine(device="cpu")
    load_time = time.perf_counter() - t0
    v1 = get_vram_mb()

    row("Load time", load_time, "s", warn_above=15.0)
    vram_row("after load", v0, v1)

    prompt = "Qual a latência do modelo de linguagem local?"
    times = []
    for _ in range(3):
        t = time.perf_counter()
        emb = engine.get_query_embedding(prompt)
        times.append(time.perf_counter() - t)

    avg = sum(times) / len(times)
    row("Embedding latency avg (3 runs)", avg * 1000, "ms", warn_above=300)
    row("Embedding dimensions", len(emb), "dims")

    results["embedding_load_s"] = load_time
    results["embedding_avg_ms"] = avg * 1000
    results["_emb_engine"] = engine
    results["_embedding"] = emb


async def bench_vectorstore() -> None:
    section("2 · VECTOR STORE  (ChromaDB)")
    from jarvis.vectorstore.store import VectorStore

    emb = results["_embedding"]
    store = VectorStore()

    times = []
    for _ in range(3):
        t = time.perf_counter()
        docs = await store.query_collection(
            collection_name="documents",
            query_embeddings=[emb],
            limit=3,
        )
        times.append(time.perf_counter() - t)

    avg = sum(times) / len(times)
    row("Query latency avg (3 runs)", avg * 1000, "ms", warn_above=100)
    row("Documents returned", len(docs), "chunks")

    results["chroma_avg_ms"] = avg * 1000


async def bench_llm() -> None:
    section("3 · LLM ENGINE  (llama.cpp — Qwen 14B GGUF)")
    from jarvis.llm.engine import LLMEngine

    llm = LLMEngine()

    v0 = get_vram_mb()
    t0 = time.perf_counter()
    llm.load_model()
    load_time = time.perf_counter() - t0
    v1 = get_vram_mb()

    row("Model load time", load_time, "s", warn_above=30.0)
    vram_row("after LLM load", v0, v1)

    # Pré-aquece o RAG (necessário para generate_stream funcionar)
    llm.pre_load_rag()

    # Warm-up run (descartada)
    warmup_prompt = "Diga apenas: ok"
    async for _ in llm.generate_stream(warmup_prompt, language="pt"):
        break

    # Benchmark run
    bench_prompt = "Qual a capital da França? Responda em uma frase."
    t_start = time.perf_counter()
    ttft = None
    tokens: list[str] = []

    async for tok in llm.generate_stream(bench_prompt, language="pt"):
        if ttft is None:
            ttft = time.perf_counter() - t_start
        tokens.append(tok)

    total_time = time.perf_counter() - t_start
    decode_time = total_time - (ttft or 0)
    tps = len(tokens) / decode_time if decode_time > 0 else 0

    row("TTFT (Time to First Token)", (ttft or 0), "s", warn_above=5.0)
    row("Total generation time", total_time, "s")
    row("Tokens generated", len(tokens), "tokens")
    row("Throughput (decode phase)", tps, "tok/s", warn_above=1)
    print(f"\n   Response preview: {''.join(tokens)[:120].strip()!r}")

    results["llm_load_s"] = load_time
    results["llm_ttft_s"] = ttft or 0
    results["llm_total_s"] = total_time
    results["llm_tokens"] = len(tokens)
    results["llm_tps"] = tps
    results["_llm"] = llm


async def bench_stt() -> None:
    section("4 · STT  (Faster Whisper — large-v3-turbo)")
    from jarvis.stt.transcriber import Transcriber

    transcriber = Transcriber()

    v0 = get_vram_mb()
    t0 = time.perf_counter()
    transcriber.load_model()
    load_time = time.perf_counter() - t0
    v1 = get_vram_mb()

    row("Model load time", load_time, "s", warn_above=20.0)
    vram_row("after Whisper load", v0, v1)

    def _make_speech_audio(duration_s: float, sr: int = 16000) -> np.ndarray:
        """Gera áudio sintético com forma de onda similar a fala (ruído rosa + modulação)."""
        n = int(sr * duration_s)
        noise = np.random.randn(n).astype(np.float32)
        # Modula para simular envoltória de voz
        t = np.linspace(0, duration_s, n)
        envelope = np.abs(np.sin(2 * np.pi * 2.5 * t)) * 0.08
        return noise * envelope

    for dur_s in (1.0, 2.0, 4.0):
        audio = _make_speech_audio(dur_s)
        times = []
        for _ in range(2):
            t0 = time.perf_counter()
            text, lang = await transcriber.transcribe(audio)
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)
        rtf = avg / dur_s  # Real-Time Factor (< 1.0 é mais rápido que tempo real)
        warn = avg > dur_s  # Mais lento que tempo real é problemático
        row(
            f"Transcription {dur_s:.0f}s audio → RTF",
            rtf,
            f"  ({avg*1000:.0f}ms)" + (" ⚠️ SLOW" if warn else ""),
        )
        if dur_s == 2.0:
            results["stt_rtf_2s"] = rtf
            results["stt_lat_2s_ms"] = avg * 1000

    results["stt_load_s"] = load_time
    results["_transcriber"] = transcriber


async def bench_tts() -> None:
    section("5 · TTS  (Piper — CPU ONNX)")
    from jarvis.tts.engine import TTSEngine

    tts = TTSEngine()

    t0 = time.perf_counter()
    tts.load_model("pt")
    load_time = time.perf_counter() - t0
    row("Model load time (pt)", load_time, "s", warn_above=5.0)

    phrases = [
        "Ok.",
        "Prontamente, Senhor.",
        "Entendido, vou executar a ação solicitada imediatamente.",
        (
            "Certamente, Senhor. A capital da França é Paris, uma cidade com rica história"
            " cultural e arquitetônica que atrai milhões de visitantes todos os anos."
        ),
    ]

    for phrase in phrases:
        first_chunk_t: list[float] = []
        t_start = time.perf_counter()

        def _cb(audio, sr=22050, *_):
            if not first_chunk_t:
                first_chunk_t.append(time.perf_counter() - t_start)

        tts.speak_stream(phrase, lang="pt", on_audio_chunk=_cb)
        # Drena a fila de forma síncrona
        while tts._is_playing or not tts._queue.empty():
            await asyncio.sleep(0.02)

        lat = first_chunk_t[0] if first_chunk_t else -1.0
        words = len(phrase.split())
        row(
            f"First-chunk latency ({words}w)",
            lat,
            "s",
            warn_above=0.8,
        )

    tts.stop()
    results["tts_load_s"] = load_time
    results["_tts"] = tts


async def bench_e2e() -> None:
    section("6 · END-TO-END SIMULATION  (STT → LLM → TTS)")
    transcriber = results.get("_transcriber")
    llm = results.get("_llm")
    tts = results.get("_tts")

    if not (transcriber and llm and tts):
        print("   ⚠️  Módulos não disponíveis — pulando E2E.")
        return

    def _make_speech_audio(duration_s: float, sr: int = 16000) -> np.ndarray:
        n = int(sr * duration_s)
        noise = np.random.randn(n).astype(np.float32)
        t = np.linspace(0, duration_s, n)
        envelope = np.abs(np.sin(2 * np.pi * 2.5 * t)) * 0.08
        return noise * envelope

    # Simula 2s de "fala" do usuário → transcrição → LLM → primeiro chunk TTS
    audio = _make_speech_audio(2.0)

    t_e2e_start = time.perf_counter()

    # 1. STT
    t0 = time.perf_counter()
    text, lang = await transcriber.transcribe(audio)
    stt_lat = time.perf_counter() - t0

    # Usa prompt fixo para o LLM (para garantir consistência)
    prompt = "Qual a capital do Brasil? Responda em uma frase curta."

    # 2. LLM (TTFT)
    t0 = time.perf_counter()
    ttft_e2e = None
    tts_first_chunk_t: list[float] = []
    all_tokens: list[str] = []
    sentence_buffer = ""
    tts_started = False

    async for tok in llm.generate_stream(prompt, language="pt"):
        if ttft_e2e is None:
            ttft_e2e = time.perf_counter() - t0
        all_tokens.append(tok)
        sentence_buffer += tok

        # Envia para TTS ao encontrar fronteira de frase
        if any(c in sentence_buffer for c in ".!?,\n") and len(sentence_buffer) >= 20:
            if not tts_started:
                t_tts = time.perf_counter()

                def _cb(audio_data, sr=22050, *_):
                    if not tts_first_chunk_t:
                        tts_first_chunk_t.append(time.perf_counter() - t_tts)

                tts.speak_stream(sentence_buffer.strip(), lang="pt", on_audio_chunk=_cb)
                tts_started = True
            else:
                tts.speak_stream(sentence_buffer.strip(), lang="pt")
            sentence_buffer = ""

    llm_total = time.perf_counter() - t0

    # Aguarda o 1º chunk TTS chegar (até 8s)
    waited = 0.0
    while not tts_first_chunk_t and waited < 8.0:
        await asyncio.sleep(0.05)
        waited += 0.05

    tts_first = tts_first_chunk_t[0] if tts_first_chunk_t else -1.0
    total_e2e = time.perf_counter() - t_e2e_start

    tts.stop()

    row("STT (2s audio) latency", stt_lat, "s", warn_above=2.0)
    row("LLM TTFT", ttft_e2e or 0, "s", warn_above=5.0)
    row("LLM total generation", llm_total, "s")
    row("TTS first-chunk latency", tts_first, "s", warn_above=0.8)
    row("Total E2E (STT+LLM+TTS start)", total_e2e, "s", warn_above=10.0)
    print(f"\n   Tokens gerados: {len(all_tokens)}  |  Lang: {lang}")

    results["e2e_stt_s"] = stt_lat
    results["e2e_ttft_s"] = ttft_e2e or 0
    results["e2e_llm_s"] = llm_total
    results["e2e_tts_first_s"] = tts_first
    results["e2e_total_s"] = total_e2e


# ── Final report ─────────────────────────────────────────────────────────────
def print_report() -> None:
    section("📊 RESUMO EXECUTIVO — GARGALOS IDENTIFICADOS")

    v_final = get_vram_mb()
    print(f"\n   VRAM total usada ao final: {v_final['used_mb']:.1f} MB / {v_final['total_mb']:.1f} MB")
    print(f"   VRAM livre restante:        {v_final['free_mb']:.1f} MB\n")
    print(SEP)

    items = [
        ("Embedding load", results.get("embedding_load_s", 0) * 1000, "ms"),
        ("Embedding query (avg)", results.get("embedding_avg_ms", 0), "ms"),
        ("ChromaDB query (avg)", results.get("chroma_avg_ms", 0), "ms"),
        ("LLM load", results.get("llm_load_s", 0), "s"),
        ("LLM TTFT", results.get("llm_ttft_s", 0), "s"),
        ("LLM throughput", results.get("llm_tps", 0), "tok/s"),
        ("STT load", results.get("stt_load_s", 0), "s"),
        ("STT RTF (2s audio)", results.get("stt_rtf_2s", 0), "×RT"),
        ("STT latency (2s audio)", results.get("stt_lat_2s_ms", 0), "ms"),
        ("TTS load", results.get("tts_load_s", 0), "s"),
        ("E2E STT", results.get("e2e_stt_s", 0) * 1000, "ms"),
        ("E2E LLM TTFT", results.get("e2e_ttft_s", 0), "s"),
        ("E2E TTS first chunk", results.get("e2e_tts_first_s", 0) * 1000, "ms"),
        ("E2E total", results.get("e2e_total_s", 0), "s"),
    ]

    for label, val, unit in items:
        print(f"   {label:<36} {val:>10.3f}  {unit}")

    print(f"\n{SEP}")
    print("\n   🔍 ANÁLISE DE GARGALOS\n")

    bottlenecks = []

    llm_tps = results.get("llm_tps", 99)
    if llm_tps < 5:
        bottlenecks.append(
            f"   🔴 LLM throughput crítico ({llm_tps:.1f} tok/s) — "
            "Reduza n_gpu_layers ou use um modelo menor (7B)"
        )
    elif llm_tps < 10:
        bottlenecks.append(
            f"   🟠 LLM throughput baixo ({llm_tps:.1f} tok/s) — "
            "Considere reduzir n_ctx de 2048 para 1024"
        )

    llm_ttft = results.get("llm_ttft_s", 0)
    if llm_ttft > 5:
        bottlenecks.append(
            f"   🔴 LLM TTFT alto ({llm_ttft:.1f}s) — "
            "Prefill lento, muitas camadas na CPU (aumente n_gpu_layers)"
        )
    elif llm_ttft > 2:
        bottlenecks.append(
            f"   🟠 LLM TTFT moderado ({llm_ttft:.1f}s) — "
            "Algumas camadas no CPU. Verifique VRAM disponível"
        )

    stt_rtf = results.get("stt_rtf_2s", 0)
    if stt_rtf > 1.0:
        bottlenecks.append(
            f"   🔴 STT mais lento que tempo real (RTF={stt_rtf:.2f}×) — "
            "Use 'medium' ou 'small' em vez de large-v3-turbo, ou reduza n_gpu_layers do LLM"
        )
    elif stt_rtf > 0.5:
        bottlenecks.append(
            f"   🟡 STT RTF moderado ({stt_rtf:.2f}×) — "
            "Funcional mas há margem de melhora"
        )

    emb_ms = results.get("embedding_avg_ms", 0)
    if emb_ms > 500:
        bottlenecks.append(
            f"   🟠 Embedding lento ({emb_ms:.0f}ms) — "
            "Considere usar device='cuda' ou modelo menor"
        )

    chroma_ms = results.get("chroma_avg_ms", 0)
    if chroma_ms > 200:
        bottlenecks.append(
            f"   🟡 ChromaDB lento ({chroma_ms:.0f}ms) — "
            "Verifique tamanho da coleção e índice HNSW"
        )

    vram = get_vram_mb()
    if vram["free_mb"] > 0 and vram["free_mb"] < 500:
        bottlenecks.append(
            f"   🔴 VRAM crítica! Apenas {vram['free_mb']:.0f}MB livre — "
            "Alto risco de OOM. Reduza n_gpu_layers do LLM"
        )
    elif vram["free_mb"] > 0 and vram["free_mb"] < 1500:
        bottlenecks.append(
            f"   🟠 VRAM baixa ({vram['free_mb']:.0f}MB livre) — "
            "Pode causar OOM esporádico durante STT+LLM simultâneos"
        )

    e2e = results.get("e2e_total_s", 0)
    if e2e > 10:
        bottlenecks.append(
            f"   🔴 E2E muito lento ({e2e:.1f}s) — "
            "Experiência degradada. Pipeline em gargalo severo"
        )
    elif e2e > 6:
        bottlenecks.append(
            f"   🟠 E2E moderado ({e2e:.1f}s) — "
            "Aceitável, mas o ideal é < 4s para conversação fluida"
        )

    if bottlenecks:
        for b in bottlenecks:
            print(b)
    else:
        print("   ✅ Nenhum gargalo crítico detectado. Pipeline dentro dos limites ideais.")

    print(f"\n{SEP}\n")


# ── Main ─────────────────────────────────────────────────────────────────────
async def main() -> None:
    print("\n" + "═" * 68)
    print("  JARVIS PIPELINE BENCHMARK")
    print("═" * 68)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")

    v = get_vram_mb()
    if v["total_mb"] > 0:
        print(f"  GPU VRAM: {v['used_mb']:.0f} / {v['total_mb']:.0f} MB used at start")
    else:
        print("  GPU VRAM: nvidia-smi não disponível")

    try:
        await bench_embedding()
    except Exception as e:
        print(f"   ❌ Embedding falhou: {e}")

    try:
        await bench_vectorstore()
    except Exception as e:
        print(f"   ❌ VectorStore falhou: {e}")

    try:
        await bench_llm()
    except Exception as e:
        print(f"   ❌ LLM falhou: {e}")
        import traceback; traceback.print_exc()

    try:
        await bench_stt()
    except Exception as e:
        print(f"   ❌ STT falhou: {e}")
        import traceback; traceback.print_exc()

    try:
        await bench_tts()
    except Exception as e:
        print(f"   ❌ TTS falhou: {e}")

    try:
        await bench_e2e()
    except Exception as e:
        print(f"   ❌ E2E falhou: {e}")
        import traceback; traceback.print_exc()

    print_report()


if __name__ == "__main__":
    asyncio.run(main())
