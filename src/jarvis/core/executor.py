"""Jarvis — Shared GPU Executor.

Define um ThreadPoolExecutor compartilhado e de thread única para garantir que todas
as operações de GPU/CUDA executem sequencialmente na mesma thread do SO,
evitando conflitos de contexto CUDA no Windows.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

# Executor compartilhado com uma única thread de trabalho dedicada para GPU
_gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu_worker")


def get_gpu_executor() -> ThreadPoolExecutor:
    """Retorna a instância do executor de thread única para tarefas de GPU."""
    return _gpu_executor
