"""Jarvis — Conexão com o banco de dados.

Gerencia a engine SQLAlchemy assíncrona e sessões.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from jarvis.config.settings import get_settings
from jarvis.core.logging import get_logger

log = get_logger(__name__)

_engine = None
_session_factory = None


def get_engine():
    """Retorna a engine assíncrona (singleton)."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.db.async_url,
            echo=settings.env.value == "development",
            pool_size=5,
            max_overflow=10,
        )
        log.info("Engine PostgreSQL criada — host={}", settings.db.host)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Retorna a factory de sessões assíncronas."""
    global _session_factory  # noqa: PLW0603
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def close_engine() -> None:
    """Fecha a engine e libera conexões."""
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        log.info("Engine PostgreSQL encerrada")
