"""Jarvis — Entry point.

Inicializa configurações, logging e exibe status do sistema.
Será expandido nas fases seguintes com STT, LLM e orchestrator.
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from jarvis.config.settings import get_settings
from jarvis.core.logging import get_logger, setup_logging

console = Console()
log = get_logger("jarvis.main")


def _print_banner() -> None:
    """Exibe banner do Jarvis no terminal."""
    banner = """
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
    """
    console.print(Panel(banner, title="🤖 Assistente Pessoal Local", border_style="cyan"))


def _print_status(settings) -> None:
    """Exibe tabela de status de configuração."""
    table = Table(title="⚙️  Configuração do Sistema", border_style="dim")
    table.add_column("Componente", style="cyan", no_wrap=True)
    table.add_column("Configuração", style="green")
    table.add_column("Status", style="yellow")

    # Ambiente
    table.add_row("Ambiente", settings.env.value, "✅")
    table.add_row("Log Level", settings.log_level, "✅")

    # PostgreSQL
    db_status = "⏳ Não conectado"
    table.add_row(
        "PostgreSQL",
        f"{settings.db.host}:{settings.db.port}/{settings.db.name}",
        db_status,
    )

    # LLM
    model_path = settings.llm.resolved_model_path
    llm_status = "✅ Encontrado" if model_path.exists() else "❌ Não encontrado"
    table.add_row("LLM Model", str(model_path.name), llm_status)

    # STT
    table.add_row(
        "STT",
        f"{settings.stt.model_size} ({settings.stt.compute_type})",
        "⏳ Não carregado",
    )

    # ChromaDB
    chroma_path = settings.chroma.resolved_persist_dir
    chroma_status = "✅ Existe" if chroma_path.exists() else "📁 Será criado"
    table.add_row("ChromaDB", str(chroma_path), chroma_status)

    # Embeddings
    table.add_row(
        "Embeddings",
        f"{settings.embedding.model} ({settings.embedding.device})",
        "⏳ Não carregado",
    )

    console.print(table)


async def _check_db_connection(settings) -> bool:
    """Testa conexão com o PostgreSQL."""
    try:
        from jarvis.knowledge.database import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
            result.scalar()
        log.info("Conexão com PostgreSQL estabelecida com sucesso")
        return True
    except Exception as e:
        log.warning("Falha ao conectar ao PostgreSQL: {}", str(e))
        return False


async def _async_main() -> None:
    """Entry point assíncrono."""
    settings = get_settings()

    _print_banner()
    _print_status(settings)

    console.print()

    # Testar conexão com banco
    console.print("[dim]Testando conexão com PostgreSQL...[/dim]")
    db_ok = await _check_db_connection(settings)
    if db_ok:
        console.print("[green]✅ PostgreSQL conectado com sucesso![/green]")
    else:
        console.print(
            "[yellow]⚠️  PostgreSQL não disponível. "
            "Execute 'python scripts/setup_db.py' para configurar.[/yellow]"
        )

    console.print()
    console.print(
        Panel(
            "[dim]Fase 1 concluída — Fundação do projeto.\n"
            "Próximos passos: Fase 2 (STT) e Fase 3 (LLM).[/dim]",
            title="📋 Status",
            border_style="dim",
        )
    )


def main() -> None:
    """Entry point síncrono (chamado pelo console_scripts)."""
    setup_logging()
    log.info("Jarvis inicializando...")

    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        console.print("\n[dim]Jarvis encerrado pelo usuário.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
