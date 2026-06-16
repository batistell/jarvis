"""Jarvis — Logging estruturado com Loguru.

Configura o loguru com:
- Formatação rica com cores e ícones por módulo
- Rotação de arquivos de log
- Filtragem por nível configurável
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from jarvis.config.settings import get_settings

# Ícones por módulo para fácil identificação visual
_MODULE_ICONS: dict[str, str] = {
    "jarvis.stt": "🎤",
    "jarvis.llm": "🧠",
    "jarvis.vectorstore": "🔍",
    "jarvis.knowledge": "📚",
    "jarvis.orchestrator": "⚡",
    "jarvis.ui": "🖥️",
    "jarvis.core": "⚙️",
    "jarvis.config": "📋",
}

_LOG_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "{extra[icon]} "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

_configured = False


def setup_logging() -> None:
    """Configura o sistema de logging do Jarvis.

    Deve ser chamado uma única vez no startup (``main.py``).
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return

    import logging

    settings = get_settings()

    # Silencia bibliotecas de terceiros verbosas
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("chromadb").setLevel(logging.ERROR)

    # Configura valor default para extra['icon'] para evitar KeyError
    logger.configure(extra={"icon": "🤖"})

    # Remove handlers padrão do loguru
    logger.remove()

    # Handler: stderr (terminal) - limita a ERROR se estiver em modo terminal para chat limpo
    console_level = "ERROR" if settings.ui.mode == "terminal" else settings.log_level.upper()
    logger.add(
        sys.stderr,
        format=_LOG_FORMAT,
        level=console_level,
        colorize=True,
        backtrace=True,
        diagnose=settings.env.value == "development",
    )

    # Handler: arquivo de log com rotação
    log_dir = settings.project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.add(
        str(log_dir / "jarvis_{time:YYYY-MM-DD}.log"),
        format=_LOG_FORMAT,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
    )

    _configured = True
    logger.info("Logging configurado — nível={}", settings.log_level)


def get_logger(name: str) -> logger:
    """Retorna um logger contextualizado com ícone do módulo.

    Args:
        name: Nome do módulo (ex: ``jarvis.stt``).

    Returns:
        Logger do loguru com ``extra['icon']`` preenchido.
    """
    # Encontra o ícone mais específico
    icon = "🤖"
    for prefix, mod_icon in _MODULE_ICONS.items():
        if name.startswith(prefix):
            icon = mod_icon
            break

    return logger.bind(icon=icon, module=name)
