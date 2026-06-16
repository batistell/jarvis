"""Jarvis — Script de ingestão em batch."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger

# Adiciona o diretório src ao path do Python para encontrar o pacote jarvis
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jarvis.vectorstore.embeddings import EmbeddingEngine
from jarvis.vectorstore.store import VectorStore
from jarvis.vectorstore.ingest import DocumentIngester


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script de ingestão em batch de conhecimento para o Jarvis."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Caminho do arquivo ou diretório contendo os documentos (.md, .txt, .pdf).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="documents",
        choices=["documents", "conversations", "code_snippets"],
        help="Nome da coleção do ChromaDB onde os dados serão inseridos (default: documents).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Dispositivo para a geração de embeddings ('cuda' ou 'cpu', default: cuda).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Desativa a busca recursiva em subdiretórios se a origem for uma pasta.",
    )

    args = parser.parse_args()
    source_path = Path(args.source)
    recursive = not args.no_recursive

    if not source_path.exists():
        logger.error(f"O caminho de origem fornecido não existe: {source_path}")
        sys.exit(1)

    logger.info("Iniciando o processo de ingestão do VectorStore...")

    # Inicializa as classes de motor e armazenamento
    try:
        embedding_engine = EmbeddingEngine(device=args.device)
        vector_store = VectorStore()
        ingester = DocumentIngester(
            embedding_engine=embedding_engine, vector_store=vector_store
        )

        if source_path.is_file():
            total_chunks = await ingester.ingest_file(
                source_path, collection_name=args.collection
            )
        else:
            total_chunks = await ingester.ingest_directory(
                source_path, collection_name=args.collection, recursive=recursive
            )

        logger.info(f"Ingestão concluída! Total de chunks indexados: {total_chunks}")

    except Exception as e:
        logger.exception(f"Erro inesperado durante a execução da ingestão: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Configura o logger do loguru para o terminal
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Executa a função main assíncrona
    asyncio.run(main())
