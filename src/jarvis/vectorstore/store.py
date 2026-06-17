"""Jarvis — ChromaDB operations."""

from __future__ import annotations

import asyncio
from loguru import logger
import chromadb
from jarvis.config.settings import get_settings


class VectorStore:
    """Wrapper assíncrono para operações do ChromaDB persistente."""

    def __init__(self) -> None:
        settings = get_settings()
        persist_dir = settings.chroma.resolved_persist_dir

        # Garante que a pasta de destino exista
        persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Conectando ao ChromaDB persistente em: {persist_dir}")
        self.client = chromadb.PersistentClient(path=str(persist_dir))

        # Inicializa coleções padrão
        self._init_collections()

    def _init_collections(self) -> None:
        """Inicializa as coleções principais em modo síncrono."""
        self.collections = {
            "documents": self.client.get_or_create_collection(
                name="documents", metadata={"hnsw:space": "cosine"}
            ),
            "conversations": self.client.get_or_create_collection(
                name="conversations", metadata={"hnsw:space": "cosine"}
            ),
            "code_snippets": self.client.get_or_create_collection(
                name="code_snippets", metadata={"hnsw:space": "cosine"}
            ),
        }
        logger.info("Coleções do ChromaDB inicializadas com sucesso.")

    def _get_collection(self, name: str) -> chromadb.Collection:
        if name not in self.collections:
            raise ValueError(
                f"Coleção '{name}' desconhecida. Coleções válidas: {list(self.collections.keys())}"
            )
        return self.collections[name]

    async def add_documents(
        self,
        collection_name: str,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """Adiciona documentos com seus embeddings pré-calculados de forma assíncrona."""
        collection = self._get_collection(collection_name)

        def _sync_add() -> None:
            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

        logger.debug(f"Adicionando {len(texts)} documentos na coleção '{collection_name}'...")
        await asyncio.to_thread(_sync_add)
        logger.info(f"{len(texts)} documentos adicionados à coleção '{collection_name}'.")

    async def query_collection(
        self,
        collection_name: str,
        query_embeddings: list[list[float]],
        limit: int = 5,
    ) -> list[dict]:
        """Consulta a coleção usando embeddings de consulta de forma assíncrona.

        Retorna uma lista de dicionários contendo id, document, metadata e distance.
        """
        collection = self._get_collection(collection_name)

        def _sync_query() -> dict:
            return collection.query(
                query_embeddings=query_embeddings,
                n_results=limit,
            )

        logger.debug(f"Consultando coleção '{collection_name}'...")
        raw_results = await asyncio.to_thread(_sync_query)

        # Processa a estrutura de retorno do ChromaDB para um formato amigável
        results = []
        if not raw_results or "ids" not in raw_results or not raw_results["ids"]:
            return results

        # O retorno do ChromaDB é mapeado por listas aninhadas
        ids = raw_results["ids"][0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        for idx in range(len(ids)):
            results.append({
                "id": ids[idx],
                "document": documents[idx] if documents else "",
                "metadata": metadatas[idx] if metadatas else {},
                "distance": distances[idx] if distances else 0.0,
            })

        return results

    async def delete_by_ids(self, collection_name: str, ids: list[str]) -> None:
        """Deleta documentos específicos por ID de forma assíncrona."""
        collection = self._get_collection(collection_name)

        def _sync_delete() -> None:
            collection.delete(ids=ids)

        logger.debug(f"Removendo {len(ids)} documentos da coleção '{collection_name}'...")
        await asyncio.to_thread(_sync_delete)
        logger.info(f"{len(ids)} documentos removidos da coleção '{collection_name}'.")

    def clear_collection(self, name: str) -> None:
        """Deleta e recria uma coleção para limpá-la de forma síncrona."""
        if name not in self.collections:
            raise ValueError(
                f"Coleção '{name}' desconhecida. Coleções válidas: {list(self.collections.keys())}"
            )
        try:
            self.client.delete_collection(name)
        except Exception as e:
            logger.warning(f"Erro ao deletar coleção '{name}': {e}")
        self.collections[name] = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Coleção '{name}' limpa e recriada com sucesso.")

    async def get_chronological_conversations(self, limit: int = 5) -> list[str]:
        """Recupera os turnos de conversa da sessão atual em ordem cronológica (mais antigos primeiro)."""
        collection = self._get_collection("conversations")

        def _sync_get() -> dict:
            return collection.get()

        try:
            raw_results = await asyncio.to_thread(_sync_get)
            if not raw_results or "ids" not in raw_results or not raw_results["ids"]:
                return []

            documents = raw_results.get("documents") or []
            metadatas = raw_results.get("metadatas") or []

            # Associa cada documento com seu timestamp correspondente
            turns = []
            for doc, meta in zip(documents, metadatas):
                ts = meta.get("timestamp", 0.0) if meta else 0.0
                turns.append((ts, doc))

            # Ordena por timestamp crescente (mais antigo primeiro)
            turns.sort(key=lambda x: x[0])

            # Pega os últimos 'limit' turnos
            recent_turns = turns[-limit:]

            return [doc for ts, doc in recent_turns]
        except Exception as e:
            logger.error(f"Erro ao recuperar conversas cronológicas: {e}")
            return []
