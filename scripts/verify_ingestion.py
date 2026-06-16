"""Jarvis — Script para verificar os embeddings inseridos."""

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


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script de verificação de busca semântica para o Jarvis."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="documents",
        choices=["documents", "conversations", "code_snippets"],
        help="Nome da coleção do ChromaDB onde os dados serão consultados (default: documents).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Dispositivo para a geração de embeddings de consulta ('cuda' ou 'cpu', default: cuda).",
    )

    args = parser.parse_args()

    logger.info("Inicializando o motor de embeddings e a conexão com o VectorStore...")

    try:
        embedding_engine = EmbeddingEngine(device=args.device)
        vector_store = VectorStore()

        # Consultas de teste baseadas no resume.md
        test_queries = [
            "Gabriel Batistell",
            "Java and Go experience",
            "AKS Kubernetes infrastructure cost",
            "Florianópolis, Santa Catarina",
        ]

        logger.info(f"Iniciando busca semântica na coleção '{args.collection}'...")

        for query in test_queries:
            logger.info(f"\nConsulta: '{query}'")
            query_emb = embedding_engine.get_query_embedding(query)
            
            results = await vector_store.query_collection(
                collection_name=args.collection,
                query_embeddings=[query_emb],
                limit=2,
            )

            if not results:
                logger.warning("Nenhum resultado retornado para esta consulta.")
                continue

            for idx, res in enumerate(results):
                def safe_print(msg: str) -> None:
                    try:
                        print(msg)
                    except UnicodeEncodeError:
                        encoding = sys.stdout.encoding or "utf-8"
                        print(msg.encode(encoding, errors="replace").decode(encoding))

                safe_print(f"  [{idx + 1}] Distância: {res['distance']:.4f}")
                safe_print(f"      Origem: {res['metadata'].get('source')}")
                safe_print(f"      Chunk: {res['metadata'].get('chunk_index')} / {res['metadata'].get('total_chunks')}")
                safe_print(f"      Hash: {res['metadata'].get('file_hash')}")
                content = res['document'].replace('\n', ' ')
                snippet = content[:150] + "..." if len(content) > 150 else content
                safe_print(f"      Conteúdo: {snippet}")

    except Exception as e:
        logger.exception(f"Erro inesperado durante a verificação: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    asyncio.run(main())
