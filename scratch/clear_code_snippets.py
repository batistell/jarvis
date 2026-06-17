import sys
from pathlib import Path

# Adiciona o diretório 'src' ao sys.path para carregar o módulo jarvis
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chromadb
from jarvis.config.settings import get_settings

def clear_collection():
    settings = get_settings()
    persist_dir = settings.chroma.resolved_persist_dir
    print(f"Directory: Conectando ao banco ChromaDB em: {persist_dir.resolve()}")
    
    if not persist_dir.exists():
        print("Error: Diretorio do banco nao existe ou esta vazio.")
        return
        
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        print("Limpando colecao 'code_snippets'...")
        client.delete_collection("code_snippets")
        # Recria vazia
        client.get_or_create_collection(
            name="code_snippets", metadata={"hnsw:space": "cosine"}
        )
        print("Colecao 'code_snippets' deletada e recriada com sucesso (vazia).")
    except Exception as e:
        print(f"Error ao limpar colecao: {e}")

if __name__ == "__main__":
    clear_collection()
