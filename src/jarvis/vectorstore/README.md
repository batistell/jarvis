# 🔍 Vector Store & Ingestion Pipeline

Este módulo encapsula o **ChromaDB** persistente e o motor de geração de embeddings local baseados no modelo **BGE-M3** (`BAAI/bge-m3`). Ele é responsável por indexar documentos estruturados, conversas passadas e snippets de código para busca semântica em tempo real pelo Jarvis.

---

## 🚀 Como Executar a Ingestão em Batch

A ingestão de novos conhecimentos é feita a partir de um script CLI independente. Recomenda-se rodar este script **apenas com o assistente Jarvis desligado**, permitindo que o processamento pesado de arquivos utilize 100% da aceleração por GPU.

### Uso Básico:
```bash
# Ative o ambiente virtual
.venv\Scripts\activate

# Indexar uma pasta com arquivos .md, .txt e/ou .pdf na GPU
python scripts/ingest.py --source ./data/knowledge/ --collection documents --device cuda
```

### Argumentos da CLI:
| Argumento | Descrição | Padrão |
|---|---|---|
| `--source` | Caminho do arquivo ou diretório contendo os documentos a serem indexados. *(Obrigatório)* | - |
| `--collection` | Coleção alvo no ChromaDB (`documents`, `conversations`, `code_snippets`). | `documents` |
| `--device` | Dispositivo de hardware a ser utilizado (`cuda` para GPU ou `cpu`). | `cuda` |
| `--no-recursive` | Se passado, desativa a busca recursiva por subpastas (apenas se a origem for uma pasta). | Busca recursiva ativa |

---

## 🛠️ Configurações e Ajuste Fino de Chunks

Você pode customizar o comportamento de particionamento e indexação no seu arquivo `.env` local. As variáveis mapeadas são:

```env
# Configurações do Motor de Embeddings
JARVIS_EMBEDDING_MODEL="BAAI/bge-m3"
JARVIS_EMBEDDING_DEVICE="cpu"                # Padrão no Jarvis (sobrescrito via CLI na ingestão)
JARVIS_EMBEDDING_CHUNK_SIZE=1000             # Tamanho máximo do chunk em caracteres
JARVIS_EMBEDDING_CHUNK_OVERLAP=150           # Sobreposição de caracteres entre chunks adjacentes
```

---

## 💡 Guia de Eficiência e Boas Práticas

Para obter os melhores resultados de busca semântica e evitar estouro de VRAM, siga as diretrizes abaixo:

### 1. Estruture bem seus Documentos (Markdown)
O divisor de texto recursivo (`TextSplitter`) tenta manter parágrafos e tópicos inteiros juntos, dividindo o texto nos seguintes caracteres prioritários: `\n## `, `\n# `, `\n\n`, `\n`, ` `.
* **Use títulos (`#` e `##`) para delimitar tópicos:** Isso garante que o splitter quebre o documento exatamente nas seções lógicas, fornecendo ao Jarvis um bloco com contexto completo e não frases cortadas ao meio.
* **Mantenha parágrafos objetivos:** Evite blocos gigantescos de texto contínuo sem quebras de linha.

### 2. Aproveite o Hashing (Evite Retrabalho)
O pipeline calcula um hash MD5 exclusivo para cada arquivo indexado. Os IDs gerados no ChromaDB seguem o formato:
`{hash_do_arquivo}_chunk_{indice_do_chunk}`
* **Por que isso é eficiente?** Se você rodar o script de ingestão na mesma pasta duas vezes, os documentos não serão duplicados. O ChromaDB apenas sobrescreverá (atualizará) os registros existentes se houver modificação no arquivo, ou manterá intocado se o conteúdo for idêntico.

### 3. Híbrido GPU/CPU
* **Ingestão:** Sempre use `--device cuda` para documentos grandes (especialmente PDFs com centenas de páginas). O modelo `BGE-M3` rodará em paralelo na GPU RTX 3060, processando dezenas de páginas por segundo.
* **Consulta:** Deixe o Jarvis configurado com `device = "cpu"` no arquivo `.env` para execução normal. Gerar embeddings para uma pergunta do usuário de 15 palavras na CPU leva menos de 50 milissegundos e economiza cerca de 1.1 GB de VRAM da GPU, que fica livre para o modelo LLM e o Faster Whisper.
