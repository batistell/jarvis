import sys
from pathlib import Path

# Garante suporte a UTF-8 no terminal Windows para evitar UnicodeEncodeError ao exibir emojis/acentos
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Adiciona o diretório 'src' ao sys.path para carregar o módulo jarvis
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chromadb
from jarvis.config.settings import get_settings

def inspect_chromadb():
    settings = get_settings()
    persist_dir = settings.chroma.resolved_persist_dir
    print(f"Directory: Conectando ao banco ChromaDB em: {persist_dir.resolve()}")
    
    if not persist_dir.exists():
        print("Error: Diretorio do banco nao existe ou esta vazio.")
        return
        
    client = chromadb.PersistentClient(path=str(persist_dir))
    collections = client.list_collections()
    
    if not collections:
        print("Error: Nenhuma colecao encontrada no ChromaDB.")
        return
        
    print(f"Collections: Colecoes encontradas: {len(collections)}")
    for col in collections:
        col_name = col.name
        count = col.count()
        print(f"\n==========================================")
        print(f"Collection Name: {col_name}")
        print(f"Item Count: {count}")
        print(f"==========================================")
        
        if count > 0:
            # Pega as primeiras 5 amostras da coleção
            data = col.get(limit=5)
            ids = data.get("ids", [])
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])
            
            for i in range(len(ids)):
                doc_snippet = documents[i] if documents else "N/A"
                if len(doc_snippet) > 200:
                    doc_snippet = doc_snippet[:200] + "... [truncado]"
                print(f"\n   ID: {ids[i]}")
                print(f"   Metadata: {metadatas[i] if metadatas else 'N/A'}")
                print(f"   Content: {doc_snippet}")
        else:
            print("   (Colecao vazia)")

if __name__ == "__main__":
    inspect_chromadb()
