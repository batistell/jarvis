"""Integration tests for the document ingestion pipeline."""

import shutil
from pathlib import Path
import pytest
import torch

from jarvis.config.settings import Settings, ChromaSettings
from jarvis.vectorstore.embeddings import EmbeddingEngine
from jarvis.vectorstore.store import VectorStore
from jarvis.vectorstore.ingest import DocumentIngester


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Fixture that overrides settings to use a temporary Chroma DB path."""
    settings = Settings()
    settings.chroma = ChromaSettings(persist_dir=str(tmp_path / "chroma_test"))
    return settings


@pytest.mark.asyncio
async def test_resume_ingestion_and_query(test_settings: Settings) -> None:
    """Integration test to verify full ingestion and query flow on resume.md."""
    # Resolve absolute path to the resume.md in the project
    project_root = test_settings.project_root
    resume_path = project_root / "documents" / "resume.md"
    
    assert resume_path.exists(), f"resume.md not found at {resume_path}"

    # Determine device dynamically to support environments with or without CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Instantiate the actual engine and store
    embedding_engine = EmbeddingEngine(device=device)
    
    # Overwrite the settings singleton
    from jarvis.config import settings as config_settings
    
    original_settings = config_settings._settings
    config_settings._settings = test_settings
    
    try:
        vector_store = VectorStore()
        ingester = DocumentIngester(embedding_engine, vector_store)
        
        # 1. Ingest file
        chunks_count = await ingester.ingest_file(resume_path, collection_name="documents")
        assert chunks_count > 0, "Ingestion should successfully split resume.md into chunks"
        
        # 2. Query search
        query = "Gabriel Batistell"
        query_emb = embedding_engine.get_query_embedding(query)
        
        results = await vector_store.query_collection(
            collection_name="documents",
            query_embeddings=[query_emb],
            limit=2
        )
        
        assert len(results) > 0, "Query should return results"
        
        # 3. Assert metadata and content
        first_result = results[0]
        assert first_result["id"].startswith(first_result["metadata"]["file_hash"])
        assert "documents/resume.md" in first_result["metadata"]["source"]
        assert first_result["metadata"]["type"] == "md"
        assert "Gabriel Batistell" in first_result["document"] or "Software Engineer" in first_result["document"]
        
    finally:
        # Restore original settings singleton
        config_settings._settings = original_settings
