# 🤖 Jarvis — Assistente Pessoal Local

> Agente de IA de alta disponibilidade que roda **100 % local** em uma RTX 3060 (12 GB VRAM).
> Combina **Speech-to-Text em tempo real**, **LLM local**, **busca vetorial** e **base de conhecimento PostgreSQL** para entregar respostas rápidas e contextualizadas em paralelo.

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura](#arquitetura)
3. [Stack Tecnológica](#stack-tecnológica)
4. [Pré-requisitos](#pré-requisitos)
5. [Roadmap de Desenvolvimento](#roadmap-de-desenvolvimento)
6. [Estrutura do Projeto](#estrutura-do-projeto)
7. [Como Executar](#como-executar)
8. [Licença](#licença)

---

## Visão Geral

O **Jarvis** é um assistente pessoal projetado para rodar inteiramente na máquina local. A ideia central é:

1. **Ouvir** — capturar áudio do microfone e converter em texto com alta precisão e baixa latência (Speech-to-Text).
2. **Pensar em paralelo** — disparar simultaneamente três fontes de resposta:
   - 🧠 **LLM local** — modelo de linguagem otimizado para GPU consumer (RTX 3060 12 GB).
   - 🔍 **Base de Vetores** — busca semântica em embeddings de documentos, snippets e conversas anteriores.
   - 📚 **Base de Conhecimento (PostgreSQL)** — consulta estruturada em conteúdos detalhados de programação, documentação e desenvolvimento.
3. **Sintetizar** — combinar os resultados das três fontes, opcionalmente passando por um modelo de síntese, ou apresentar cada resultado assim que estiver disponível (streaming progressivo).
4. **Aprender** — armazenar histórico de conversas, correções do usuário e novos conhecimentos para melhorar respostas futuras.

### Objetivos Principais

| Objetivo | Descrição |
|---|---|
| **Alta Disponibilidade** | Agente sempre pronto para escutar, sem depender de cloud |
| **Respostas Rápidas** | STT de baixa latência + busca vetorial retorna resultados parciais em < 1s |
| **Respostas Detalhadas** | Base de conhecimento PostgreSQL fornece conteúdo profundo e estruturado |
| **Execução em Paralelo** | LLM, vetores e PostgreSQL consultados simultaneamente via `asyncio` |
| **Customização Contínua** | Histórico de conversas, correções e RAG alimentam o contexto do modelo |
| **Hardware Acessível** | Tudo roda em uma RTX 3060 12 GB — sem cloud, sem custos recorrentes |

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────────┐
│                          JARVIS CORE                                │
│                                                                     │
│  ┌──────────┐    ┌──────────────────────────────────────────────┐   │
│  │          │    │           ORCHESTRATOR (asyncio)             │   │
│  │   MIC    │───▶│                                              │   │
│  │  INPUT   │    │  ┌─────────┐  ┌───────────┐  ┌───────────┐  │   │
│  │          │    │  │   LLM   │  │  VECTOR   │  │ POSTGRES  │  │   │
│  └──────────┘    │  │  LOCAL  │  │   STORE   │  │    KB     │  │   │
│       │          │  │         │  │           │  │           │  │   │
│       ▼          │  │ (llama  │  │ (Chroma/  │  │ (pgvector │  │   │
│  ┌──────────┐    │  │  .cpp)  │  │  FAISS)   │  │  + full   │  │   │
│  │  STT     │    │  │         │  │           │  │  text)    │  │   │
│  │ (Faster  │───▶│  └────┬────┘  └─────┬─────┘  └─────┬─────┘  │   │
│  │ Whisper) │    │       │             │               │        │   │
│  └──────────┘    │       ▼             ▼               ▼        │   │
│                  │  ┌──────────────────────────────────────────┐ │   │
│                  │  │         RESPONSE AGGREGATOR              │ │   │
│                  │  │  (merge / rank / re-rank via LLM)        │ │   │
│                  │  └──────────────────┬───────────────────────┘ │   │
│                  └────────────────────┼─────────────────────────┘   │
│                                       ▼                             │
│                          ┌─────────────────────┐                    │
│                          │   OUTPUT (TTS / UI)  │                   │
│                          └─────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Fluxo de Dados

```
Áudio ──▶ STT (Faster Whisper) ──▶ Texto
                                      │
                        ┌─────────────┼─────────────┐
                        ▼             ▼             ▼
                   LLM Local    Vector Store    PostgreSQL
                   (Geração)    (Semântica)     (Estruturada)
                        │             │             │
                        ▼             ▼             ▼
                   ┌──────────────────────────────────┐
                   │   Aggregator / Synthesis Model    │
                   │   (combina ou apresenta em        │
                   │    streaming por ordem de chegada) │
                   └──────────────┬───────────────────┘
                                  ▼
                          Resposta Final
                          + Salva no Histórico
```

---

## Stack Tecnológica

| Componente | Tecnologia | Justificativa |
|---|---|---|
| **Linguagem** | Python 3.11+ | Ecossistema rico para IA/ML, asyncio nativo |
| **Speech-to-Text** | [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2-based, ~4x mais rápido que Whisper original, roda em GPU |
| **Captura de Áudio** | `sounddevice` + `webrtcvad` | Captura em tempo real com detecção de atividade de voz (VAD) |
| **LLM Local** | [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | Binding Python do llama.cpp, suporta GGUF quantizados (Q4/Q5) na 3060 |
| **Modelo LLM** | Mistral 7B / Qwen2.5 7B (GGUF Q4_K_M) | Modelos 7B quantizados cabem na 3060 com sobra para STT |
| **Embeddings** | `sentence-transformers` (all-MiniLM-L6-v2) | Modelo leve de embeddings, roda na CPU sem impactar a GPU |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | Embutido, sem servidor, persistente, integração Python nativa |
| **Base de Conhecimento** | PostgreSQL + pgvector | Busca estruturada (SQL) + busca vetorial híbrida no mesmo banco |
| **ORM / Queries** | SQLAlchemy + asyncpg | Acesso assíncrono ao PostgreSQL |
| **Orquestração Async** | `asyncio` + `asyncio.TaskGroup` | Disparo paralelo das 3 fontes de resposta |
| **Configuração** | Pydantic Settings + `.env` | Validação e tipagem de configurações |
| **Interface (futuro)** | Textual (TUI) ou FastAPI + WebSocket | Interface terminal rica ou web local |

---

## Pré-requisitos

- **Sistema Operacional:** Windows 10/11
- **GPU:** NVIDIA RTX 3060 (12 GB VRAM) com CUDA 12.x instalado
- **Python:** 3.11 ou superior
- **PostgreSQL:** 15+ com extensão `pgvector` instalada
- **RAM:** 16 GB mínimo (32 GB recomendado)
- **Microfone:** Qualquer microfone USB ou integrado

---

## Roadmap de Desenvolvimento

O desenvolvimento é organizado em **fases incrementais**. Cada fase entrega valor funcional e pode ser testada independentemente.

---

### 📌 Fase 1 — Fundação do Projeto

> **Objetivo:** Estruturar o projeto, configurar ambiente e garantir que as dependências funcionam.

**Passo a passo:**

1. **Criar estrutura de diretórios do projeto**
   - Definir os módulos: `core/`, `stt/`, `llm/`, `vectorstore/`, `knowledge/`, `orchestrator/`, `config/`
   - Criar `pyproject.toml` com todas as dependências
   - Configurar virtual environment com `uv` ou `venv`

2. **Configurar sistema de configuração**
   - Criar `config/settings.py` com Pydantic Settings
   - Definir variáveis: caminhos de modelos, conexão PostgreSQL, parâmetros de áudio
   - Suportar `.env` para overrides locais

3. **Configurar logging estruturado**
   - Usar `loguru` ou `logging` com formatação rica
   - Níveis por módulo (STT, LLM, DB, etc.)

4. **Configurar PostgreSQL + pgvector**
   - Criar database `jarvis_kb`
   - Instalar extensão `pgvector`
   - Criar schemas iniciais: `knowledge`, `conversations`, `embeddings`

**Entregável:** Projeto inicializa sem erros, conecta ao PostgreSQL, carrega configurações.

---

### 📌 Fase 2 — Speech-to-Text (STT)

> **Objetivo:** Capturar áudio do microfone e transcrever em tempo real com Faster Whisper.

**Passo a passo:**

1. **Implementar captura de áudio**
   - Usar `sounddevice` para capturar stream do microfone
   - Implementar buffer circular para acumular chunks de áudio
   - Configurar sample rate (16kHz), canais (mono), dtype (float32)

2. **Implementar Voice Activity Detection (VAD)**
   - Integrar `webrtcvad` ou `silero-vad` para detectar início/fim de fala
   - Definir thresholds de silêncio para segmentar frases
   - Evitar enviar silêncio para o modelo STT

3. **Integrar Faster Whisper**
   - Carregar modelo `large-v3` ou `medium` (conforme VRAM disponível)
   - Configurar `compute_type="float16"` para GPU
   - Implementar transcrição com timestamps para feedback em tempo real

4. **Criar pipeline STT completo**
   - `MicCapture → VAD → AudioBuffer → FasterWhisper → Texto`
   - Emitir eventos assíncronos quando uma frase completa é detectada
   - Implementar callback pattern para desacoplar do restante do sistema

**Entregável:** Falar no microfone → texto aparece no terminal em tempo real.

---

### 📌 Fase 3 — LLM Local

> **Objetivo:** Carregar e servir um modelo de linguagem local via llama.cpp.

**Passo a passo:**

1. **Baixar modelo GGUF**
   - Escolher modelo: Mistral 7B Instruct Q4_K_M ou Qwen2.5 7B Q4_K_M
   - Armazenar em `models/` (pasta ignorada pelo git)
   - Documentar como baixar via `huggingface-cli`

2. **Configurar llama-cpp-python**
   - Compilar com suporte CUDA (`CMAKE_ARGS="-DGGML_CUDA=on"`)
   - Instanciar `Llama()` com `n_gpu_layers=-1` (offload total para GPU)
   - Configurar `n_ctx=4096` (ou 8192 se couber na VRAM)

3. **Implementar módulo de geração**
   - Criar classe `LLMEngine` com interface assíncrona
   - Suportar geração com streaming (token a token)
   - Implementar template de prompt com system prompt customizado
   - Suportar injeção de contexto (resultados do vector store e KB)

4. **Gerenciamento de VRAM**
   - Monitorar uso de VRAM para coexistir com Faster Whisper
   - Implementar loading/unloading de modelos se necessário
   - Testar configurações de quantização vs. qualidade

**Entregável:** Enviar prompt de texto → receber resposta gerada localmente, com streaming.

---

### 📌 Fase 4 — Vector Store (Busca Semântica)

> **Objetivo:** Indexar documentos e conversas em embeddings para busca semântica rápida.

**Passo a passo:**

1. **Configurar ChromaDB**
   - Inicializar ChromaDB com persistência em disco (`persist_directory`)
   - Criar coleções: `documents`, `conversations`, `code_snippets`

2. **Implementar pipeline de embeddings**
   - Usar `sentence-transformers` com modelo `all-MiniLM-L6-v2` (CPU)
   - Criar classe `EmbeddingEngine` para gerar embeddings de texto
   - Implementar chunking inteligente para documentos longos (overlap de 10-20%)

3. **Implementar busca semântica**
   - Criar classe `VectorSearcher` com interface assíncrona
   - Busca por similaridade coseno com top-k configurável
   - Retornar documentos + scores + metadados

4. **Ingestão de conteúdo**
   - Script para indexar arquivos `.md`, `.py`, `.txt` de diretórios locais
   - Indexar automaticamente conversas do histórico
   - Suportar re-indexação incremental

**Entregável:** Indexar documentos → buscar por similaridade semântica → retornar resultados rankeados.

---

### 📌 Fase 5 — Base de Conhecimento PostgreSQL

> **Objetivo:** Armazenar e consultar conteúdo estruturado de programação e desenvolvimento.

**Passo a passo:**

1. **Modelagem do banco de dados**
   ```sql
   -- Tabelas principais
   knowledge_articles    -- Artigos/documentação com conteúdo completo
   code_examples         -- Snippets de código categorizados
   conversations         -- Histórico completo de conversas
   conversation_corrections  -- Correções feitas pelo usuário sobre respostas
   tags                  -- Sistema de tags para categorização
   ```

2. **Implementar camada de acesso a dados**
   - Modelos SQLAlchemy com tipagem completa
   - Repository pattern para cada entidade
   - Queries assíncronas com `asyncpg`

3. **Implementar busca híbrida**
   - Full-text search do PostgreSQL (`tsvector` + `tsquery`) para busca textual
   - `pgvector` para busca vetorial dentro do próprio PostgreSQL
   - Combinar scores de ambas as buscas (Reciprocal Rank Fusion)

4. **Sistema de ingestão**
   - CLI para adicionar artigos, snippets e documentação
   - Parser de markdown para extrair seções e metadados
   - Importar de fontes externas (docs locais, READMEs de projetos)

**Entregável:** Base populada → buscar por texto e/ou semântica → retornar conteúdo detalhado e estruturado.

---

### 📌 Fase 6 — Orquestrador Paralelo

> **Objetivo:** Disparar as 3 fontes de resposta em paralelo e agregar os resultados.

**Passo a passo:**

1. **Implementar Orchestrator**
   - Classe central `Orchestrator` que recebe texto transcrito do STT
   - Disparar `asyncio.TaskGroup` com 3 tasks simultâneas:
     - `task_llm`: gerar resposta via LLM local
     - `task_vector`: buscar no vector store
     - `task_knowledge`: buscar na base PostgreSQL

2. **Implementar Response Aggregator**
   - Coletar resultados conforme cada task finaliza (streaming por ordem de chegada)
   - Duas estratégias (configuráveis):
     - **Streaming progressivo:** apresentar cada resultado assim que chega
     - **Síntese final:** passar todos os resultados por um segundo prompt no LLM para gerar resposta unificada

3. **Implementar sistema de contexto**
   - Antes de disparar as tasks, buscar contexto relevante:
     - Últimas N conversas do histórico
     - Correções anteriores do usuário sobre temas similares
   - Injetar contexto no prompt do LLM

4. **Implementar feedback loop**
   - Após resposta, permitir que o usuário corrija ou avalie
   - Armazenar correções na tabela `conversation_corrections`
   - Usar correções como contexto em futuras perguntas similares

**Entregável:** Falar → STT transcreve → 3 fontes respondem em paralelo → resultados agregados exibidos.

---

### 📌 Fase 7 — Persistência e Aprendizado

> **Objetivo:** Salvar conversas, aprender com correções e melhorar respostas ao longo do tempo.

**Passo a passo:**

1. **Persistir conversas completas**
   - Salvar no PostgreSQL: pergunta, respostas de cada fonte, resposta final
   - Gerar embeddings da conversa e indexar no vector store
   - Manter metadados: timestamp, tópico inferido, tags

2. **Sistema de correções**
   - Comando para o usuário corrigir uma resposta anterior
   - Associar correção à conversa original
   - Na próxima pergunta similar, incluir a correção no contexto do prompt

3. **Re-ranking baseado em histórico**
   - Ponderar resultados do vector store com base em preferências aprendidas
   - Priorizar fontes que o usuário mais validou

**Entregável:** Conversas são salvas → correções influenciam respostas futuras → qualidade melhora com o uso.

---

### 📌 Fase 8 — Interface e Experiência do Usuário

> **Objetivo:** Criar uma interface para visualizar resultados em tempo real.

**Passo a passo:**

1. **Interface Terminal Rica (TUI) — primeira versão**
   - Usar `Textual` ou `Rich` para interface no terminal
   - Painéis separados: transcrição STT, resposta LLM, resultados vetoriais, resultados KB
   - Streaming em tempo real em cada painel

2. **Interface Web Local (evolução futura)**
   - FastAPI + WebSocket para backend
   - Frontend minimalista (HTML/JS) ou React
   - Exibir resultados em cards por fonte, na ordem de chegada
   - Indicadores visuais de loading por fonte

3. **Text-to-Speech (opcional)**
   - Integrar `piper-tts` ou `edge-tts` para resposta por voz
   - Configurável: apenas voz, apenas texto, ou ambos

**Entregável:** Interface funcional que exibe resultados de múltiplas fontes em tempo real.

---

### 📌 Fase 9 — Otimização e Refinamento

> **Objetivo:** Polir performance, qualidade de respostas e experiência geral.

1. **Otimizar uso de VRAM** — profiling com `nvidia-smi`, ajustar batch sizes e quantização
2. **Tuning de prompts** — refinar system prompts e templates de síntese
3. **Melhorar chunking** — chunking semântico (por parágrafos/seções) ao invés de tamanho fixo
4. **Cache de embeddings** — evitar recalcular embeddings de conteúdo já indexado
5. **Testes de latência** — medir e otimizar tempo de cada etapa do pipeline
6. **Hot-reload de configurações** — alterar parâmetros sem reiniciar o sistema

---

## Estrutura do Projeto

```
jarvis/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── src/
│   └── jarvis/
│       ├── __init__.py
│       ├── main.py                  # Entry point
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py          # Pydantic Settings
│       │
│       ├── stt/
│       │   ├── __init__.py
│       │   ├── mic_capture.py       # Captura de áudio do microfone
│       │   ├── vad.py               # Voice Activity Detection
│       │   └── transcriber.py       # Faster Whisper integration
│       │
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── engine.py            # llama-cpp-python wrapper
│       │   ├── prompts.py           # Templates de prompt
│       │   └── context_builder.py   # Construtor de contexto com RAG
│       │
│       ├── vectorstore/
│       │   ├── __init__.py
│       │   ├── embeddings.py        # Sentence-transformers wrapper
│       │   ├── store.py             # ChromaDB operations
│       │   └── ingest.py            # Pipeline de ingestão
│       │
│       ├── knowledge/
│       │   ├── __init__.py
│       │   ├── models.py            # SQLAlchemy models
│       │   ├── repository.py        # Data access layer
│       │   └── search.py            # Full-text + vector search
│       │
│       ├── orchestrator/
│       │   ├── __init__.py
│       │   ├── orchestrator.py      # Disparo paralelo das fontes
│       │   ├── aggregator.py        # Combina/sintetiza respostas
│       │   └── history.py           # Gerenciamento de histórico
│       │
│       └── ui/
│           ├── __init__.py
│           ├── terminal.py          # Interface TUI (Textual/Rich)
│           └── web.py               # Interface Web (FastAPI + WS)
│
├── models/                          # Modelos GGUF (gitignored)
│   └── .gitkeep
│
├── data/
│   ├── chroma/                      # ChromaDB persistence
│   └── knowledge/                   # Arquivos para ingestão
│
├── scripts/
│   ├── setup_db.py                  # Inicialização do PostgreSQL
│   ├── download_models.py           # Download de modelos
│   └── ingest.py                    # Ingestão em batch
│
├── tests/
│   ├── test_stt.py
│   ├── test_llm.py
│   ├── test_vectorstore.py
│   ├── test_knowledge.py
│   └── test_orchestrator.py
│
└── docs/
    ├── architecture.md
    └── setup.md
```

---

## Como Executar

```bash
# 1. Clonar o repositório
git clone https://github.com/seu-usuario/jarvis.git
cd jarvis

# 2. Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate       # Windows

# 3. Instalar dependências
pip install -e ".[dev]"

# 4. Configurar variáveis de ambiente
copy .env.example .env
# Editar .env com suas configurações

# 5. Configurar banco de dados
python scripts/setup_db.py

# 6. Baixar modelos
python scripts/download_models.py

# 7. Executar
python -m jarvis
```

---

## Licença

Este projeto está licenciado sob os termos da licença presente no arquivo [LICENSE](LICENSE).