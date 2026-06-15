"""Jarvis — Script de inicialização do banco de dados PostgreSQL.

Cria o database, instala extensões (pgvector, pg_trgm) e cria todas as tabelas
definidas nos modelos SQLAlchemy.

Uso:
    python scripts/setup_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Adiciona src/ ao path para importar jarvis
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sqlalchemy
from rich.console import Console
from sqlalchemy import create_engine, text

from jarvis.config.settings import get_settings
from jarvis.knowledge.models import Base

console = Console()


def create_database_if_not_exists(settings) -> None:
    """Cria o database jarvis_kb se não existir."""
    # Conecta ao database padrão 'postgres' para criar o novo DB
    pwd = settings.db.password.get_secret_value()
    admin_url = (
        f"postgresql://{settings.db.user}:{pwd}"
        f"@{settings.db.host}:{settings.db.port}/postgres"
    )

    engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")

    with engine.connect() as conn:
        # Verifica se o database já existe
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": settings.db.name},
        )
        exists = result.scalar() is not None

        if not exists:
            conn.execute(text(f'CREATE DATABASE "{settings.db.name}"'))
            console.print(f"[green]✅ Database '{settings.db.name}' criado.[/green]")
        else:
            console.print(f"[dim]Database '{settings.db.name}' já existe.[/dim]")

    engine.dispose()


def install_extensions(settings) -> None:
    """Instala extensões necessárias no database."""
    engine = create_engine(settings.db.sync_url)

    extensions = ["vector", "pg_trgm"]

    with engine.connect() as conn:
        for ext in extensions:
            try:
                conn.execute(text(f'CREATE EXTENSION IF NOT EXISTS "{ext}"'))
                conn.commit()
                console.print(f"[green]✅ Extensão '{ext}' instalada/verificada.[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Extensão '{ext}': {e}[/yellow]")

    engine.dispose()


def create_tables(settings) -> None:
    """Cria todas as tabelas definidas nos modelos SQLAlchemy."""
    engine = create_engine(settings.db.sync_url)

    Base.metadata.create_all(engine)
    console.print("[green]✅ Tabelas criadas/verificadas com sucesso.[/green]")

    # Lista as tabelas criadas
    inspector = sqlalchemy.inspect(engine)
    tables = inspector.get_table_names()
    for table in sorted(tables):
        console.print(f"   📋 {table}")

    engine.dispose()


def main() -> None:
    """Executa setup completo do banco de dados."""
    console.print("[bold cyan]🗄️  Jarvis — Setup do Banco de Dados[/bold cyan]\n")

    settings = get_settings()

    console.print(
        f"[dim]Conectando a PostgreSQL em "
        f"{settings.db.host}:{settings.db.port}...[/dim]\n"
    )

    try:
        # Passo 1: Criar database
        console.print("[bold]1. Criando database...[/bold]")
        create_database_if_not_exists(settings)

        # Passo 2: Instalar extensões
        console.print("\n[bold]2. Instalando extensões...[/bold]")
        install_extensions(settings)

        # Passo 3: Criar tabelas
        console.print("\n[bold]3. Criando tabelas...[/bold]")
        create_tables(settings)

        console.print("\n[bold green]✅ Setup concluído com sucesso![/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]❌ Erro no setup: {e}[/bold red]")
        console.print(
            "\n[dim]Verifique se o PostgreSQL está rodando e as credenciais "
            "no .env estão corretas.[/dim]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
