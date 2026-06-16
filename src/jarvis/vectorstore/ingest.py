"""Jarvis — Ingestion pipeline for documents."""

from __future__ import annotations

import hashlib
from pathlib import Path
from loguru import logger
from jarvis.config.settings import get_settings
from jarvis.vectorstore.embeddings import EmbeddingEngine
from jarvis.vectorstore.store import VectorStore


class DocumentIngester:
    """Orquestrador do pipeline de ingestão de documentos para o VectorStore."""

    def __init__(self, embedding_engine: EmbeddingEngine, vector_store: VectorStore) -> None:
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.settings = get_settings()

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcula o hash MD5 do arquivo para versionamento/deduplicação."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extrai todo o conteúdo de texto de um arquivo PDF."""
        logger.debug(f"Extraindo texto do PDF: {file_path}")
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.error("O pacote 'pypdf' é necessário para ler arquivos PDF. Instale-o com 'pip install pypdf'.")
            raise ImportError("Pacote 'pypdf' não encontrado. Por favor, instale-o para processar PDFs.")
        reader = PdfReader(str(file_path))
        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)

    def _extract_text(self, file_path: Path) -> str:
        """Extrai o texto com base na extensão do arquivo."""
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_text_from_pdf(file_path)
        elif suffix in (".md", ".txt"):
            return file_path.read_text(encoding="utf-8", errors="ignore")
        else:
            raise ValueError(f"Formato de arquivo não suportado: {suffix}")

    async def ingest_file(self, file_path: Path, collection_name: str = "documents") -> int:
        """Processa um arquivo individual, gera embeddings e insere no ChromaDB."""
        if not file_path.exists() or not file_path.is_file():
            logger.error(f"Arquivo não encontrado: {file_path}")
            return 0

        suffix = file_path.suffix.lower()
        if suffix not in (".md", ".txt", ".pdf"):
            logger.debug(f"Pulando arquivo não suportado: {file_path}")
            return 0

        logger.info(f"Iniciando ingestão do arquivo: {file_path.name}")
        try:
            text = self._extract_text(file_path)
            if not text.strip():
                logger.warning(f"O arquivo {file_path.name} está vazio ou sem texto extraível.")
                return 0

            # Divide o texto em chunks controlados
            chunks = self.embedding_engine.split_text(text)
            if not chunks:
                logger.warning(f"Nenhum chunk gerado para o arquivo {file_path.name}.")
                return 0

            logger.info(f"Arquivo particionado em {len(chunks)} chunks.")

            # Gera metadados e IDs determinísticos baseados em hash
            file_hash = self._calculate_file_hash(file_path)
            # Simplifica o caminho para salvar como origem relativa à raiz do projeto se possível
            try:
                rel_path = file_path.relative_to(self.settings.project_root)
            except ValueError:
                rel_path = file_path

            metadatas = []
            ids = []
            for idx, _ in enumerate(chunks):
                chunk_id = f"{file_hash}_chunk_{idx}"
                ids.append(chunk_id)
                metadatas.append({
                    "source": str(rel_path.as_posix()),
                    "type": suffix[1:],  # 'pdf', 'md', 'txt'
                    "file_hash": file_hash,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                })

            # Gera embeddings em lote (pode rodar na GPU dependendo da inicialização do motor)
            logger.info("Gerando embeddings para os chunks...")
            embeddings = self.embedding_engine.get_embeddings(chunks)

            # Insere no ChromaDB de forma assíncrona
            await self.vector_store.add_documents(
                collection_name=collection_name,
                texts=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"Ingestão concluída com sucesso para: {file_path.name}")
            return len(chunks)

        except Exception as e:
            logger.exception(f"Erro ao processar o arquivo {file_path.name}: {e}")
            return 0

    async def ingest_directory(
        self, dir_path: Path, collection_name: str = "documents", recursive: bool = True
    ) -> int:
        """Processa todos os arquivos de um diretório recursivamente ou não."""
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Diretório não encontrado: {dir_path}")
            return 0

        logger.info(f"Escaneando diretório: {dir_path}")
        pattern = "**/*" if recursive else "*"
        files = [p for p in dir_path.glob(pattern) if p.is_file() and p.suffix.lower() in (".md", ".txt", ".pdf")]

        if not files:
            logger.warning("Nenhum arquivo compatível (.md, .txt, .pdf) encontrado.")
            return 0

        logger.info(f"Encontrados {len(files)} arquivos para processamento.")
        total_chunks = 0
        for file_path in files:
            chunks_count = await self.ingest_file(file_path, collection_name)
            total_chunks += chunks_count

        logger.info(f"Ingestão do diretório concluída. Total de chunks inseridos: {total_chunks}")
        return total_chunks
