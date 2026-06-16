"""Tests for vectorstore module."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from jarvis.config.settings import Settings, ChromaSettings, EmbeddingSettings
from jarvis.vectorstore.embeddings import TextSplitter, EmbeddingEngine
from jarvis.vectorstore.store import VectorStore
from jarvis.vectorstore.ingest import DocumentIngester


def test_text_splitter() -> None:
    splitter = TextSplitter(chunk_size=20, chunk_overlap=5, separators=["\n", " "])
    text = "Hello world\nThis is a test of splitting\nAnother line"
    chunks = splitter.split_text(text)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) <= 20
        assert chunk.strip() != ""


@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    """Fixture to provide test settings pointing to a temporary directory."""
    settings = Settings()
    settings.chroma = ChromaSettings(persist_dir=str(tmp_path / "chroma"))
    settings.embedding = EmbeddingSettings(
        model="BAAI/bge-m3",
        device="cpu",
        chunk_size=100,
        chunk_overlap=10,
        separators=["\n\n", "\n", " ", ""],
    )
    return settings


@patch("jarvis.vectorstore.embeddings.SentenceTransformer")
@patch("jarvis.vectorstore.embeddings.get_settings")
def test_embedding_engine(
    mock_get_settings: MagicMock, mock_st_class: MagicMock, mock_settings: Settings
) -> None:
    mock_get_settings.return_value = mock_settings

    # Mock SentenceTransformer encode
    mock_st_instance = MagicMock()
    mock_st_class.return_value = mock_st_instance
    mock_st_instance.encode.return_value = [[0.1, 0.2, 0.3]]

    engine = EmbeddingEngine(device="cpu")

    # Verify instantiation
    mock_st_class.assert_called_once_with("BAAI/bge-m3", device="cpu")

    # Verify embedding generation
    emb = engine.get_embeddings(["test text"])
    assert emb == [[0.1, 0.2, 0.3]]
    mock_st_instance.encode.assert_called_once_with(
        ["test text"], show_progress_bar=False, convert_to_numpy=True
    )

    # Verify query embedding
    q_emb = engine.get_query_embedding("query")
    assert q_emb == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
@patch("jarvis.vectorstore.store.chromadb.PersistentClient")
@patch("jarvis.vectorstore.store.get_settings")
async def test_vector_store(
    mock_get_settings: MagicMock, mock_client_class: MagicMock, mock_settings: Settings
) -> None:
    mock_get_settings.return_value = mock_settings

    # Mock Chroma Client and Collection
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    store = VectorStore()

    # Verify collection creation
    assert mock_client.get_or_create_collection.call_count == 3

    # Test add_documents
    await store.add_documents(
        collection_name="documents",
        texts=["doc1"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"source": "test"}],
        ids=["id1"],
    )
    mock_collection.add.assert_called_once_with(
        documents=["doc1"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"source": "test"}],
        ids=["id1"],
    )

    # Test query_collection
    mock_collection.query.return_value = {
        "ids": [["id1"]],
        "documents": [["doc1"]],
        "metadatas": [[{"source": "test"}]],
        "distances": [[0.1]],
    }

    results = await store.query_collection(
        collection_name="documents", query_embeddings=[[0.1, 0.2]], limit=1
    )
    assert len(results) == 1
    assert results[0]["id"] == "id1"
    assert results[0]["document"] == "doc1"
    assert results[0]["metadata"] == {"source": "test"}
    assert results[0]["distance"] == 0.1

    # Test delete_by_ids
    await store.delete_by_ids("documents", ["id1"])
    mock_collection.delete.assert_called_once_with(ids=["id1"])


@pytest.mark.asyncio
@patch("jarvis.vectorstore.embeddings.SentenceTransformer")
@patch("jarvis.vectorstore.embeddings.get_settings")
@patch("jarvis.vectorstore.store.chromadb.PersistentClient")
@patch("jarvis.vectorstore.store.get_settings")
async def test_document_ingester(
    mock_store_settings_get: MagicMock,
    mock_client_class: MagicMock,
    mock_emb_settings_get: MagicMock,
    mock_st_class: MagicMock,
    mock_settings: Settings,
    tmp_path: Path,
) -> None:
    mock_store_settings_get.return_value = mock_settings
    mock_emb_settings_get.return_value = mock_settings

    # Create a dummy file to ingest
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is test file content.\nLine 2 of test.", encoding="utf-8")

    # Mock embeddings and client
    mock_st_instance = MagicMock()
    mock_st_class.return_value = mock_st_instance
    mock_st_instance.encode.return_value = [[0.1, 0.2]]

    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    engine = EmbeddingEngine(device="cpu")
    store = VectorStore()
    ingester = DocumentIngester(engine, store)

    # Ingest the file
    chunks_count = await ingester.ingest_file(test_file, collection_name="documents")

    # Assertions
    assert chunks_count > 0
    mock_collection.add.assert_called_once()
    added_kwargs = mock_collection.add.call_args[1]
    assert len(added_kwargs["documents"]) == chunks_count
    # Relative path check (will resolve based on project_root)
    expected_rel = str(test_file.relative_to(mock_settings.project_root).as_posix())
    assert added_kwargs["metadatas"][0]["source"] == expected_rel
    assert added_kwargs["metadatas"][0]["type"] == "txt"
