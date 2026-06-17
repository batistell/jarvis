"""Jarvis — Unit tests for Codebase RAG."""

import sys
from pathlib import Path
import pytest

# Adiciona a raiz do projeto ao PATH do Python para importar de scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.vectorstore.store import VectorStore
from jarvis.vectorstore.embeddings import EmbeddingEngine
from scripts.index_codebase import CodebaseIndexer


@pytest.mark.asyncio
async def test_codebase_indexer_parsing() -> None:
    """Verifica se o parser do indexador AST extrai classes e funções de um arquivo teste."""
    # Cria arquivo Python temporário para teste de parsing
    test_code = (
        "class TestClass:\n"
        "    \"\"\"Docstring da classe.\"\"\"\n"
        "    def test_method(self):\n"
        "        \"\"\"Docstring do método.\"\"\"\n"
        "        return True\n\n"
        "def test_function():\n"
        "    \"\"\"Docstring da função.\"\"\"\n"
        "    return False\n"
    )
    
    test_file = Path("tests/temp_test_parse.py")
    test_file.write_text(test_code, encoding="utf-8")

    try:
        # Inicializa o indexador sem motores carregados para testar apenas o parsing estático
        indexer = CodebaseIndexer(None, None)  # type: ignore
        indexer.project_root = Path("tests")
        
        entities = indexer.parse_python_file(test_file)
        
        assert len(entities) == 3
        
        # Valida classe
        cls_ent = [e for e in entities if e["type"] == "class"][0]
        assert cls_ent["name"] == "TestClass"
        assert "Docstring da classe." in cls_ent["docstring"]
        
        # Valida método
        method_ent = [e for e in entities if e["type"] == "method"][0]
        assert method_ent["name"] == "TestClass.test_method"
        assert "Docstring do método." in method_ent["docstring"]
        
        # Valida função
        func_ent = [e for e in entities if e["type"] == "function"][0]
        assert func_ent["name"] == "test_function"
        assert "Docstring da função." in func_ent["docstring"]

    finally:
        if test_file.exists():
            test_file.unlink()


@pytest.mark.asyncio
async def test_query_code_snippets() -> None:
    """Verifica se a consulta na coleção 'code_snippets' responde corretamente."""
    vector_store = VectorStore()
    embedding_engine = EmbeddingEngine(device="cpu")
    
    # Adiciona snippet mock
    test_text = "Arquivo: tests/mock.py\nTipo: Função\nNome: mock_run\nLinhas: 1-5\nCódigo:\ndef mock_run(): pass"
    emb = embedding_engine.get_query_embedding(test_text)
    
    await vector_store.add_documents(
        collection_name="code_snippets",
        texts=[test_text],
        embeddings=[emb],
        metadatas=[{"source": "tests/mock.py", "type": "function", "name": "mock_run"}],
        ids=["code_mock_test_123"]
    )
    
    try:
        # Consulta
        query_emb = embedding_engine.get_query_embedding("mock_run function tests")
        results = await vector_store.query_collection(
            collection_name="code_snippets",
            query_embeddings=[query_emb],
            limit=1
        )
        
        assert len(results) == 1
        assert results[0]["metadata"]["name"] == "mock_run"
        assert results[0]["metadata"]["source"] == "tests/mock.py"
    finally:
        # Limpa
        await vector_store.delete_by_ids("code_snippets", ["code_mock_test_123"])
