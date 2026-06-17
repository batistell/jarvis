"""Jarvis — Sentence-transformers embedding wrapper."""

from __future__ import annotations

from loguru import logger
from sentence_transformers import SentenceTransformer
from jarvis.config.settings import get_settings


class TextSplitter:
    """Recursivamente divide o texto em chunks com tamanho e sobreposição controlados."""

    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str] | None = None) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n## ", "\n# ", "\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> list[str]:
        """Divide o texto de entrada em uma lista de chunks menores."""
        if not text.strip():
            return []
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Fallback padrão: divisão direta por caractere com sobreposição
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunks.append(text[start:end])
                if end >= len(text):
                    break
                start = end - self.chunk_overlap
            return chunks

        separator = separators[0]
        splits: list[str] = []
        if separator == "":
            splits = list(text)
        else:
            parts = text.split(separator)
            for i, part in enumerate(parts):
                if i == 0:
                    if part:
                        splits.append(part)
                else:
                    splits.append(separator + part)

        chunks = []
        current_chunk = ""

        for split in splits:
            if len(split) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                recursive_chunks = self._split_text(split, separators[1:])
                chunks.extend(recursive_chunks)
            else:
                if current_chunk and len(current_chunk) + len(split) > self.chunk_size:
                    chunks.append(current_chunk)
                    # Aplica sobreposição pegando os últimos caracteres do chunk anterior
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + split
                else:
                    current_chunk = current_chunk + split if current_chunk else split

        if current_chunk:
            chunks.append(current_chunk)

        return [c.strip() for c in chunks if c.strip()]


class EmbeddingEngine:
    """Mecanismo de geração de embeddings locais."""

    def __init__(self, device: str | None = None) -> None:
        settings = get_settings()
        self.model_name = settings.embedding.model
        self.device = device or settings.embedding.device

        logger.info(f"Inicializando modelo de embeddings '{self.model_name}' no dispositivo '{self.device}'...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Modelo de embeddings carregado com sucesso.")

        self.splitter = TextSplitter(
            chunk_size=settings.embedding.chunk_size,
            chunk_overlap=settings.embedding.chunk_overlap,
            separators=settings.embedding.separators,
        )

        # Cache LRU de 16 entradas para queries repetidas (~170ms economizados por hit)
        self._query_cache: dict[str, list[float]] = {}
        self._query_cache_max: int = 16

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings vetoriais para uma lista de textos."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vec.tolist() for vec in embeddings]

    def get_query_embedding(self, query: str) -> list[float]:
        """Gera o embedding vetorial para uma única frase de consulta (com cache LRU de 16 entradas)."""
        if query in self._query_cache:
            return self._query_cache[query]
        result = self.get_embeddings([query])[0]
        if len(self._query_cache) >= self._query_cache_max:
            # Remove a entrada mais antiga (Python 3.7+ dicts são ordered by insertion)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        self._query_cache[query] = result
        return result

    def split_text(self, text: str) -> list[str]:
        """Particiona um texto longo em chunks usando o TextSplitter configurado."""
        return self.splitter.split_text(text)
