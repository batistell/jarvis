"""Jarvis — Script para download de modelos.

Baixa os modelos necessários do Hugging Face:
- Modelo LLM (GGUF)
- Modelo de embeddings (sentence-transformers)
- Modelo STT (Faster Whisper — baixado automaticamente na primeira execução)

Uso:
    python scripts/download_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.panel import Panel

console = Console()

# Modelos GGUF recomendados para RTX 3060 12GB
RECOMMENDED_MODELS = {
    "mistral-7b": {
        "repo": "TheBloke/Mistral-7B-Instruct-v0.3-GGUF",
        "file": "mistral-7b-instruct-v0.3.Q4_K_M.gguf",
        "size": "~4.4 GB",
        "description": "Mistral 7B Instruct v0.3 — Boa qualidade geral, rápido",
    },
    "qwen2.5-7b": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "file": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size": "~4.7 GB",
        "description": "Qwen2.5 7B — Excelente em código e raciocínio",
    },
}


def print_instructions() -> None:
    """Exibe instruções para download dos modelos."""
    console.print("[bold cyan]📥 Jarvis — Download de Modelos[/bold cyan]\n")

    # LLM Models
    console.print("[bold]1. Modelo LLM (escolha um):[/bold]\n")
    for name, info in RECOMMENDED_MODELS.items():
        console.print(
            Panel(
                f"[bold]{name}[/bold]\n"
                f"Repositório: {info['repo']}\n"
                f"Arquivo: {info['file']}\n"
                f"Tamanho: {info['size']}\n"
                f"Descrição: {info['description']}\n\n"
                f"[dim]Comando:[/dim]\n"
                f"  huggingface-cli download {info['repo']} {info['file']} "
                f"--local-dir models/",
                border_style="dim",
            )
        )

    # Embeddings
    console.print("\n[bold]2. Modelo de Embeddings:[/bold]")
    console.print(
        "  O modelo [cyan]all-MiniLM-L6-v2[/cyan] será baixado automaticamente\n"
        "  pelo sentence-transformers na primeira execução (~80 MB).\n"
    )

    # STT
    console.print("[bold]3. Modelo STT (Faster Whisper):[/bold]")
    console.print(
        "  O modelo [cyan]large-v3[/cyan] será baixado automaticamente\n"
        "  pelo faster-whisper na primeira execução (~3 GB).\n"
        "  Alternativa mais leve: [cyan]medium[/cyan] (~1.5 GB).\n"
    )

    # Requisitos de VRAM
    console.print(
        Panel(
            "[bold]Estimativa de uso de VRAM (RTX 3060 12GB):[/bold]\n\n"
            "  LLM Q4_K_M 7B:        ~4.5 GB\n"
            "  Faster Whisper large:  ~3.0 GB\n"
            "  Overhead CUDA:         ~0.5 GB\n"
            "  ─────────────────────────────\n"
            "  Total estimado:        ~8.0 GB  ✅ Cabe na 3060!\n\n"
            "  Embeddings (CPU):      ~0.1 GB RAM\n"
            "  ChromaDB:              ~0.1 GB RAM",
            title="💾 VRAM Budget",
            border_style="green",
        )
    )


def try_auto_download() -> None:
    """Tenta baixar o modelo padrão automaticamente via huggingface_hub."""
    try:
        from huggingface_hub import hf_hub_download

        model_info = RECOMMENDED_MODELS["mistral-7b"]
        models_dir = Path(__file__).resolve().parents[1] / "models"
        models_dir.mkdir(exist_ok=True)

        target = models_dir / model_info["file"]
        if target.exists():
            console.print(f"[green]✅ Modelo já existe: {target.name}[/green]")
            return

        console.print(f"[yellow]⏳ Baixando {model_info['file']}...[/yellow]")
        hf_hub_download(
            repo_id=model_info["repo"],
            filename=model_info["file"],
            local_dir=str(models_dir),
        )
        console.print(f"[green]✅ Modelo baixado: {target.name}[/green]")

    except ImportError:
        console.print(
            "[dim]huggingface_hub não instalado. "
            "Instale com: pip install huggingface_hub[/dim]"
        )
        console.print("[dim]Ou baixe manualmente seguindo as instruções acima.[/dim]")
    except Exception as e:
        console.print(f"[red]❌ Erro no download: {e}[/red]")


def main() -> None:
    """Entry point do script."""
    print_instructions()

    console.print("\n[bold]Deseja tentar o download automático do Mistral 7B? (s/n)[/bold]")
    try:
        resp = input("> ").strip().lower()
        if resp in ("s", "sim", "y", "yes"):
            try_auto_download()
        else:
            console.print("[dim]Ok, baixe manualmente quando estiver pronto.[/dim]")
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Cancelado.[/dim]")


if __name__ == "__main__":
    main()
