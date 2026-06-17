import asyncio
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Ensure src/ is in sys.path
sys.path.insert(0, "src")

from jarvis.llm.engine import LLMEngine
from jarvis.vectorstore.store import VectorStore
from jarvis.vectorstore.embeddings import EmbeddingEngine

async def main():
    print("=== Testando Memória de Conversa com ChromaDB ===")
    
    # Instancia e limpa coleções no startup (pre-load RAG)
    engine = LLMEngine()
    
    # 1. Simular startup (limpeza das conversas)
    print("Pre-carregando RAG...")
    engine.pre_load_rag()
    
    store = engine._vector_store
    emb_engine = engine._embedding_engine
    
    # Verifica que a coleção de conversas está vazia
    col = store._get_collection("conversations")
    print(f"Número inicial de documentos em 'conversations': {col.count()}")
    assert col.count() == 0, "A coleção deveria estar limpa após o pre-load."
    
    # 2. Salva um turno de conversa
    user_prompt = "Meu nome é Gabriel Batista e eu sou engenheiro de software."
    response = "Prazer em conhecê-lo, Gabriel! Vou me lembrar que você é engenheiro de software."
    print(f"\nSalvando turno:\n - U: {user_prompt}\n - J: {response}")
    
    print("Chamando save_conversation_turn...")
    try:
        await engine.save_conversation_turn(user_prompt, response)
        print("save_conversation_turn concluído com sucesso!")
    except Exception as e:
        print(f"Erro em save_conversation_turn: {e}")
    
    print(f"Número de documentos em 'conversations' após salvar: {col.count()}")
    assert col.count() == 1, "Deveria ter 1 documento salvo."
    
    # 3. Faz uma consulta cronológica
    print("\nBuscando turnos em ordem cronológica...")
    results = await store.get_chronological_conversations(limit=5)
    
    print("Resultados recuperados:")
    for doc in results:
        print(f" - Doc: {doc}")
        
    assert len(results) > 0, "Deveria recuperar o histórico."
    assert "Gabriel Batista" in results[0], "O histórico deveria conter a menção ao nome."
    
    print("\n✅ Todos os testes de memória local e RAG temporário passaram!")

if __name__ == "__main__":
    asyncio.run(main())
