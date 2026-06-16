"""Jarvis — Script para verificar RAG com o LLM."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from loguru import logger

# Otimização Windows: Garante o registro do DLL Path de CUDA antes de qualquer import do projeto
import os
import sys
import ctypes
from pathlib import Path
_main_dll_handles = []
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
                _main_dll_handles.append(os.add_dll_directory(str(bin_dir.resolve())))
            except Exception:
                pass
        try:
            cudart_path = nvidia_dir / "cuda_runtime" / "bin" / "cudart64_12.dll"
            if cudart_path.exists():
                ctypes.CDLL(str(cudart_path.resolve()))
            cublaslt_path = nvidia_dir / "cublas" / "bin" / "cublasLt64_12.dll"
            if cublaslt_path.exists():
                ctypes.CDLL(str(cublaslt_path.resolve()))
            cublas_path = nvidia_dir / "cublas" / "bin" / "cublas64_12.dll"
            if cublas_path.exists():
                ctypes.CDLL(str(cublas_path.resolve()))
            cudnn_path = nvidia_dir / "cudnn" / "bin" / "cudnn64_9.dll"
            if cudnn_path.exists():
                ctypes.CDLL(str(cudnn_path.resolve()))
        except Exception:
            pass

# Adiciona o diretório src ao path do Python para encontrar o pacote jarvis
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jarvis.llm.engine import LLMEngine


async def main() -> None:
    logger.info("Carregando o LLM Engine...")
    llm = LLMEngine()
    
    # Carrega o modelo de forma idêntica ao main.py
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(llm._executor, llm.load_model)
    logger.info("Modelo LLM carregado com sucesso!")

    queries = [
        "Qual o meu nome completo?",
        "Onde eu moro atualmente?"
    ]

    for q in queries:
        print(f"\nPergunta: {q}")
        print("Resposta: ", end="")
        sys.stdout.flush()
        
        # Windows console might fail with non-cp1252 characters
        encoding = sys.stdout.encoding or "utf-8"
        
        async for token in llm.generate_stream(q):
            try:
                sys.stdout.write(token)
            except UnicodeEncodeError:
                sys.stdout.write(token.encode(encoding, errors="replace").decode(encoding))
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    asyncio.run(main())
