# 🤖 Jarvis — Assistente Pessoal Local

O **Jarvis** é um assistente pessoal por voz de alto desempenho projetado para ser executado **100% localmente**, otimizado para GPUs de consumo (como a NVIDIA RTX 3060 de 12 GB). O sistema integra captação de áudio em tempo real, detecção de atividade de voz inteligente, transcrição com inteligência artificial, inferência com um Large Language Model (LLM) local acoplado a um sistema RAG (Retrieval-Augmented Generation), controle inteligente de automação residencial e síntese de voz (TTS) fluida.

Este documento fornece uma descrição técnica detalhada da arquitetura e do funcionamento de todos os componentes implementados no projeto.

---

## 📋 Sumário
1. [Arquitetura Geral e Fluxo de Dados](#1-arquitetura-geral-e-fluxo-de-dados)
2. [Otimizações de Baixa Latência e Windows/CUDA](#2-otimizações-de-baixa-latência-e-windowscuda)
3. [Descrição Detalhada dos Módulos](#3-descrição-detalhada-dos-módulos)
   - [Configuração (`jarvis.config.settings`)](#configuração-jarvisconfigsettings)
   - [Logging (`jarvis.core.logging`)](#logging-jarviscorelogging)
   - [Captura de Áudio (`jarvis.stt.mic_capture`)](#captura-de-áudio-jarvissttmic_capture)
   - [Voice Activity Detection (`jarvis.stt.vad`)](#voice-activity-detection-jarvissttvad)
   - [Transcrição STT (`jarvis.stt.transcriber`)](#transcrição-stt-jarvisstttranscriber)
   - [Motor LLM Local (`jarvis.llm.engine`)](#motor-llm-local-jarvisllmengine)
   - [Integração Home Assistant (`jarvis.core.homeassistant`)](#integração-home-assistant-jarviscorehomeassistant)
   - [Sintetizador de Voz TTS (`jarvis.tts.engine`)](#sintetizador-de-voz-tts-jarvisttsengine)
   - [Inicialização e Tratamento de Exceções (`run_jarvis.py`)](#inicialização-e-tratamento-de-exceções-run_jarvispy)
4. [Armazenamento e RAG (ChromaDB & PostgreSQL)](#4-armazenamento-e-rag-chromadb--postgresql)
5. [Scripts e Utilitários de Teste](#5-scripts-e-utilitários-de-teste)
6. [Guia de Configuração e Execução](#6-guia-de-configuração-e-execução)

---

## 1. Arquitetura Geral e Fluxo de Dados

O Jarvis utiliza uma arquitetura baseada em **Loops de Eventos Assíncronos (`asyncio`)** e execução multithreaded inteligente para coordenar as etapas do pipeline de voz sem bloquear a interface de usuário.

```
                   ┌──────────────────────────────────────────────┐
                   │               MICROFONE CLIENTE              │
                   └──────────────────────┬───────────────────────┘
                                          │ (sd.InputStream - 16kHz float32)
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │          CAPTURA & BUFFER DE ÁUDIO           │
                   └──────────────────────┬───────────────────────┘
                                          │
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │    DETECTOR DE VOZ VAD (Dual-Gate / RMS)     │
                   └──────────────────────┬───────────────────────┘
                                          │ (Detecção de Início/Fim de Fala)
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │        TRANSCRITOR STT (Faster Whisper)       │
                   └──────────────────────┬───────────────────────┘
                                          │ (Texto Transcrito)
                                          ▼
                       ┌──────────────────┴──────────────────┐
                       │   Casamento Semântico Local (HA)?   │
                       └──────────┬───────────────────┬──────┘
                                  │ (Sim)             │ (Não)
                                  ▼                   ▼
                     ┌──────────────────────┐   ┌──────────────────────────┐
                     │ HA REST API (Direto) │   │ LLM ENGINE (Local GGUF)  │
                     └────────────┬─────────┘   └─────────────┬────────────┘
                                  │                           │
                                  │ (Prompt Confirmador)      │ (Geração em Stream)
                                  └───────────┬───────────────┘
                                              │
                                              ▼
                   ┌──────────────────────────────────────────────┐
                   │            EXTRATOR DE SENTENÇAS             │
                   └──────────────────────┬───────────────────────┘
                                          │ (Frases completas na hora)
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │         SINTETIZADOR TTS (Piper local)       │
                   └──────────────────────┬───────────────────────┘
                                          │ (sd.play - Fila Thread-Safe)
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │              REPRODUÇÃO DE ÁUDIO             │
                   └──────────────────────────────────────────────┘
```

### O Fluxo Passo a Passo:
1. **Captação**: O áudio é capturado continuamente em tempo real em blocos de 30 ms (480 samples @ 16kHz) e inserido em uma fila assíncrona.
2. **VAD**: Um classificador de atividade de voz processa os blocos. Quando detecta voz, ele inicia o buffer de áudio acumulado. Quando detecta silêncio contínuo superior ao threshold definido (ex: 800 ms), fecha a frase.
3. **Transcrição**: O áudio acumulado é enviado para o Faster Whisper. Há suporte para transcrição parcial em tempo real (feedback de escrita no terminal) e transcrição final consolidada.
4. **Mapeamento de Ações**:
   - O texto passa por uma validação rápida de comandos locais para o Home Assistant (ex: *"ligar luz do escritório"*). Se houver match semântico local de alta confiança (score >= 0.6), a requisição REST é disparada imediatamente em paralelo, e o LLM é provocado com uma diretiva simples para gerar apenas uma confirmação verbal curta e polida.
   - Caso contrário (fallback), a solicitação de voz inteira é enviada ao LLM para processamento convencional (incluindo tool calling nativo para automação se necessário).
5. **Contexto (RAG)**: O LLM recupera informações do ChromaDB:
   - **Histórico cronológico de conversas** (últimos 5 turnos de diálogo na sessão).
   - **Documentos permanentes** cadastrados na base de conhecimento (similaridade de cosseno com threshold < 0.8).
6. **Streaming Verbal (TTS em paralelo)**: À medida que os tokens são gerados pelo LLM, eles passam por um extrator de sentenças (baseado em quebras de pontuação `.`, `!`, `?` ou novas linhas). Cada frase completa é imediatamente enviada para a fila do Piper TTS. Isso reduz significativamente a latência percebida, pois a voz começa a falar muito antes de a geração completa do LLM terminar.
7. **Interrupção Dinâmica**: Se o usuário começar a falar enquanto o Jarvis estiver gerando uma resposta ou reproduzindo áudio, uma detecção de fala imediata ou palavra-chave de interrupção (*"para"*, *"pare"*, *"espera"*, *"silêncio"*, etc.) cancela instantaneamente a tarefa de inferência do LLM e esvazia a fila de reprodução de voz, permitindo uma comunicação natural de duas vias.

---

## 2. Otimizações de Baixa Latência e Windows/CUDA

A execução de modelos de IA locais no Windows impõe desafios relacionados ao gerenciamento de threads e ao contexto de execução do CUDA. O Jarvis implementa soluções específicas para esses gargalos:

*   **Registro Automático de DLLs NVIDIA**: No startup do transcritor (`transcriber.py`) e do ponto de entrada principal (`main.py`), o script localiza a pasta `site-packages/nvidia` do ambiente virtual e adiciona recursivamente os diretórios `bin/` das bibliotecas (CUDA Runtime, cuBLAS Lt, cuBLAS, cuDNN) ao path de pesquisa do Windows utilizando `os.add_dll_directory`. Além disso, pré-carrega as DLLs dinamicamente na memória via `ctypes.CDLL`. Isso evita erros comuns de CUDA não encontrado no ambiente do Windows.
*   **Dry-run de Inicialização (Faster Whisper Warmup)**: O Faster Whisper carrega cuBLAS e cuDNN de forma preguiçosa (*lazy loading*) apenas na primeira inferência. O Jarvis força um *warmup* no startup, transcrevendo 1 segundo de silêncio artificial (array de zeros). Isso garante que o atraso de carregamento ocorra durante a inicialização do programa, e não na primeira pergunta do usuário.
*   **Shared GPU Executor (`gpu_worker`)**: Operações assíncronas do PyTorch e llama.cpp em múltiplas threads de sistema podem causar travamentos ou estouro de contexto de CUDA. Para resolver isso, o Jarvis implementa um `ThreadPoolExecutor` centralizado e **exclusivamente de thread única** (`max_workers=1`). Toda inferência do LLM (`llama-cpp-python`) e transcrição do STT (`faster-whisper`) é enviada para este executor compartilhado, forçando a serialização correta das operações de GPU na mesma thread de sistema.
*   **Geração de Embeddings na CPU**: Enquanto o LLM e o Whisper ocupam a VRAM da GPU, a geração de embeddings de consulta (feita pelo `BAAI/bge-m3` via `SentenceTransformer`) é executada intencionalmente na **CPU**. Isso economiza cerca de 1 a 2 GB de VRAM, assegurando que o sistema opere estavelmente dentro do limite de 12 GB da RTX 3060.
*   **Paralelismo de Áudio TTS**: A síntese de voz (Piper TTS) é rodada em uma thread dedicada separada do loop do `asyncio`. Ela opera consumindo uma fila síncrona thread-safe, permitindo que a geração do áudio e a reprodução (via `sounddevice.play`) ocorram em background enquanto o asyncio cuida da captação de microfone e rede.

---

## 3. Descrição Detalhada dos Módulos

### Configuração (`jarvis.config.settings`)
*   Implementado usando `pydantic-settings`. As configurações são totalmente tipadas e carregam dados de variáveis de ambiente com o prefixo `JARVIS_` (ex: `JARVIS_DB_HOST`) ou diretamente a partir do arquivo `.env`.
*   O arquivo [settings.py](file:///src/jarvis/config/settings.py) expõe a função `get_settings()` que retorna uma instância singleton de `Settings`.
*   Inclui dicionário padrão de termos técnicos de programação (Java, Spring Boot, Kafka, SQL, RAG) como `initial_prompt` no Whisper para orientar a precisão ortográfica.

### Logging (`jarvis.core.logging`)
*   Interface estruturada com o `Loguru`.
*   Adiciona decorações visuais e ícones característicos dependendo do módulo emissor (ex: 🎤 para `stt`, 🧠 para `llm`, 🔍 para `vectorstore`).
*   Supressão automática de logs de depuração verbosos de bibliotecas de terceiros (como `chromadb`, `transformers` e `urllib3`).
*   Os logs detalhados de nível `DEBUG` são persistidos de forma rotacionada em arquivos `.log` dentro da pasta `logs/` com limite de 10MB por arquivo e compressão em formato `.zip`.
*   No console, os logs são limitados a mensagens de erro (`ERROR`) caso o sistema esteja configurado para rodar no terminal, garantindo um chat limpo para o usuário.

### Captura de Áudio (`jarvis.stt.mic_capture`)
*   Implementado na classe `MicCapture` no arquivo [mic_capture.py](file:///src/jarvis/stt/mic_capture.py).
*   Configura um stream de entrada de áudio através da biblioteca `sounddevice` (`sd.InputStream`), capturando dados de microfone em formato de ponto flutuante de 32 bits (`float32`), mono, com taxa de amostragem padrão de 16.000 Hz.
*   O callback de interrupção física do hardware insere os chunks de áudio na fila de eventos assíncronos de forma thread-safe utilizando `_loop.call_soon_threadsafe`.
*   O fluxo de áudio é consumido através de um gerador assíncrono.

### Voice Activity Detection (`jarvis.stt.vad`)
*   Gerenciado pela classe `VADDetector` no arquivo [vad.py](file:///src/jarvis/stt/vad.py).
*   Oferece suporte à biblioteca C `webrtcvad`. Se ela não estiver instalada na máquina, realiza um fallback automático e transparente para um detector baseado em energia RMS da amplitude da onda.
*   **Dual-Gate**: Quando o `webrtcvad` está ativo, o Jarvis aplica uma verificação dupla (Dual-Gate): o frame deve ser classificado como voz pelo WebRTC VAD **E** a energia RMS do sinal de áudio deve superar um limite mínimo de ruído. Isso impede a ativação acidental por ruídos de sopro, digitação ou estalos de ambiente.
*   **Calibração Dinâmica**: O detector analisa os primeiros 30 frames coletados para calcular o *noise floor* (ruído médio de fundo do ambiente) e define automaticamente o threshold de energia com base na fórmula: `threshold = média + 3 * desvio padrão` (garantindo o patamar mínimo de 0.01).

### Transcrição STT (`jarvis.stt.transcriber`)
*   Implementado na classe `Transcriber` no arquivo [transcriber.py](file:///src/jarvis/stt/transcriber.py).
*   Utiliza a biblioteca `faster-whisper` executando localmente.
*   Configurações flexíveis de tamanho de modelo (ex: `large-v3`, `medium`) e tipo de computação (padrão `float16` na GPU CUDA).
*   **Filtro de Alucinações**: O Whisper tende a alucinar sob silêncio ou ruído contínuo, repetindo expressões fixas (como *"Obrigado"*, *"Thank you"*, *"Thank you for watching"*, etc.). O método `_filter_hallucinations` normaliza o texto transcrito e barra essas strings de ruído conhecidas, além de descartar transcrições com menos de 2 caracteres.
*   **Redirecionamento Dinâmico de Idioma**: O Jarvis suporta apenas os idiomas Português (`pt`) e Inglês (`en`). Se o Whisper detectar dinamicamente outro idioma (ex: espanhol ou latim) em uma fala curta, o transcritor verifica os níveis de probabilidade de todas as linguagens calculados pelo Whisper. Ele força uma re-transcrição imediata fixando o idioma (`pt` ou `en`) que obteve a maior probabilidade mútua, eliminando desvios estranhos de dialeto.

### Motor LLM Local (`jarvis.llm.engine`)
*   Gerenciado pela classe `LLMEngine` no arquivo [engine.py](file:///src/jarvis/llm/engine.py).
*   Faz o envelopamento assíncrono das chamadas de chat da biblioteca `llama-cpp-python` de modo a rodar no executor compartilhado dedicado à GPU.
*   Injeta o histórico e o contexto de arquivos pessoais e base de conhecimento (RAG) recuperados do ChromaDB nas mensagens do prompt do sistema (`messages`).
*   **Streaming de Ferramentas (Tool Calling)**: Se dispositivos do Home Assistant estiverem sincronizados, o prompt do sistema é enriquecido com um schema de função descrevendo a ferramenta `control_home_device`.
    *   O motor monitora os tokens do stream em busca de tags customizadas geradas pelo modelo local (`<tool_call>` e `</tool_call>`).
    *   Ao ler uma chamada de ferramenta, intercepta a geração verbal, decodifica o payload JSON contendo os parâmetros do dispositivo (service, entity_id, data, delay), dispara o comando de controle assíncrono para o Home Assistant em segundo plano e, em seguida, chama uma inferência interna de *follow-up* para que o LLM retome o stream com a confirmação falada de sucesso ao usuário.

### Integração Home Assistant (`jarvis.core.homeassistant`)
*   A classe `HomeAssistantClient` em [homeassistant.py](file:///src/jarvis/core/homeassistant.py) faz requisições REST autenticadas via token JWT contra a API do Home Assistant OS local.
*   No startup do Jarvis, faz uma sincronização de dispositivos controláveis, filtrando entidades de domínios específicos de automação (como `light`, `switch`, `climate`, `media_player`, `fan`, `cover` e `lock`).
*   **Match Semântico Local Rápido**: Em vez de sempre esperar que o áudio transcrito vá para o LLM, o arquivo [main.py](file:///src/jarvis/main.py#L122) expõe a função `match_local_ha_command`. Esta função realiza um processamento léxico rápido na string transcrita:
    1. Identifica a intenção (ligar, desligar, toggle) usando buscas de palavras inteiras (expressões regulares).
    2. Compara os nomes amigáveis ou ids dos dispositivos integrados no HA de três formas: busca por correspondência exata, busca por substring proporcional e busca expandida de interseção de palavras utilizando um dicionário de sinônimos bilíngues (ex: mapeando "escritório" para "office", "luz" para "light", etc.).
    3. Se encontrar um dispositivo correspondente com score superior a 0.60, a ação REST de controle é enviada imediatamente (latência de milissegundos). O prompt do LLM é então modificado apenas para emitir uma confirmação de sucesso educada, ignorando o processamento pesado de tool-calling tradicional.

### Sintetizador de Voz TTS (`jarvis.tts.engine`)
*   Gerenciado pela classe `TTSEngine` no arquivo [engine.py](file:///src/jarvis/tts/engine.py).
*   Utiliza a biblioteca local rápida e de alta fidelidade **Piper TTS** baseada em modelos ONNX.
*   Se o modelo e o arquivo de configuração `.json` correspondentes ao idioma não estiverem na pasta `models/`, o motor faz o download automático em lote das vozes hospedadas no Hugging Face:
    *   Português (`pt`): Voz `pt_BR-cadu-medium.onnx` (Voz masculina brasileira cadu).
    *   Inglês (`en`): Voz `en_GB-alan-medium.onnx` (Voz clássica Alan de mordomo britânico).
*   Usa o gerador `voice.synthesize` para processar a string recebida de cada frase e reproduz os frames resultantes em blocos no hardware com auxílio de `sounddevice.play`.

### Inicialização e Tratamento de Exceções (`run_jarvis.py`)
*   O script raiz [run_jarvis.py](file:///run_jarvis.py) inicializa o ciclo de vida da aplicação.
*   Se o programa for executado fora do ambiente virtual no Windows, o script detecta a presença da pasta `.venv` local e auto-delega a execução chamando novamente o processo passando a linha de comando para o interpretador `.venv/Scripts/python.exe`.
*   Registra um hook global de captura de exceções em threads quaisquer (`sys.excepthook`) e uma rotina personalizada de tratamento de erros assíncronos no event loop (`loop.set_exception_handler`). Qualquer travamento ou exceção inesperada é capturada, formata uma stack trace amigável e salva em disco no arquivo `crash_report.log` para depuração.

---

## 4. Armazenamento e RAG (ChromaDB & PostgreSQL)

O Jarvis adota um armazenamento híbrido para viabilizar RAG sem latência e manter a memória a longo prazo.

### 1. Vector Store Local (ChromaDB)
Persistido na pasta `data/chroma/`, é gerenciado pelo wrapper `VectorStore` no arquivo [store.py](file:///src/jarvis/vectorstore/store.py). O banco opera com distância de cosseno (`hnsw:space: cosine`) dividido em três coleções principais:
*   `documents`: Contém pedaços (*chunks*) indexados de documentos, manuais de projeto e currículos úteis para o RAG de conhecimento geral.
*   `conversations`: Coleção temporária contendo as interações realizadas na sessão ativa. É limpa no início de cada execução e salva interações no formato `"Usuário: ... \n Jarvis: ..."`. Ao buscar, os itens são lidos, associados aos timestamps gerados no metadado do ChromaDB e classificados cronologicamente (do mais antigo para o mais recente) para remontar a linha de raciocínio no prompt do LLM.
*   `code_snippets`: Coleção secundária voltada ao armazenamento rápido de exemplos e códigos de programação.

### 2. Base de Conhecimento Estruturada (PostgreSQL + pgvector)
Implementado na pasta `src/jarvis/knowledge/` sob os modelos definidos no arquivo [models.py](file:///file:///o:/Python/jarvis/src/jarvis/knowledge/models.py). O banco é voltado a consultas avançadas estruturadas e busca semântica unificada no banco relacional via ORM:
*   `knowledge_articles`: Tabela para artigos completos em Markdown. Inclui coluna `embedding` do tipo `Vector(384)` (otimizado para o encoder leve `all-MiniLM-L6-v2`) e índice GIN na coluna título para pesquisas textuais (*Full-Text Search*).
*   `code_examples`: Armazena trechos de código categorizados por linguagem (`language`), framework (`framework`), com sua respectiva representação em vetor (`embedding` de 384 dimensões).
*   `conversations`: Registra permanentemente todos os diálogos passados na máquina, salvando perguntas do usuário, respostas providas por cada fonte (LLM puro, vetor, base estruturada), a resposta final consolidada e o score de satisfação (1-5) imputado pelo usuário.
*   `conversation_corrections`: Permite registrar anotações e correções fornecidas pelo usuário de forma explícita sobre interações prévias. O vetor da correção auxilia o RAG a resgatar feedbacks passados no prompt de perguntas similares, criando uma curva de aprendizado contínua.
*   `tags`, `article_tags` & `code_example_tags`: Sistema relacional muitos-para-muitos (many-to-many) para classificação de artigos e códigos.

---

## 5. Scripts e Utilitários de Teste

O projeto disponibiliza cinco scripts utilitários na pasta `scripts/` para automação e testes locais:

1.  **`setup_db.py`**: Conecta ao servidor PostgreSQL, cria o banco de dados `jarvis_kb` (caso não exista), ativa as extensões `vector` (pgvector) e `pg_trgm` (trigramas de texto) e cria a estrutura física de tabelas e relacionamentos mapeados no SQLAlchemy.
2.  **`download_models.py`**: Guia o usuário na instalação local dos modelos. Exibe comandos do `huggingface-cli` para baixar os modelos quantizados GGUF recomendados (Mistral 7B Instruct v0.3 ou Qwen 2.5 7B) e permite o download automatizado do Mistral 7B via script de Python. Apresenta também o planejamento de consumo de VRAM detalhado para garantir a execução estável na placa RTX 3060.
3.  **`ingest.py`**: CLI de ingestão em lote. Lê uma pasta ou arquivo contendo documentos `.md`, `.txt` ou `.pdf` (extraindo texto via biblioteca `pypdf`), divide em chunks com base nas configurações de sobreposição, calcula o hash MD5 dos arquivos para evitar duplicações e indexa os embeddings gerados no banco vetorial ChromaDB.
    *   *Exemplo*: `python scripts/ingest.py --source documents/resume.md --device cpu`
4.  **`verify_ingestion.py`**: Permite testar e validar o ChromaDB realizando buscas semânticas de teste. O script converte perguntas de amostra em embeddings e imprime as duas correspondências de maior proximidade encontradas no banco vetorial (exibindo a distância de cosseno, o índice do chunk, o hash MD5 e um snippet do texto recuperado).
5.  **`verify_rag.py`**: Script de validação integrado. Inicializa o motor local do LLM em linha de comando, simula a recuperação de contexto através do ChromaDB e gera as respostas em tempo real por streaming, permitindo auditar o comportamento lógico do assistente e do banco.

---

## 6. Guia de Configuração e Execução

### 1. Requisitos do Sistema
*   **SO**: Windows 10/11 64-bit.
*   **Hardware**: NVIDIA GeForce RTX 3060 (12 GB de VRAM) ou superior.
*   **Software**: Python 3.11 instalado, PostgreSQL 15+ com a extensão `pgvector` habilitada no sistema operacional.

### 2. Configurando o Ambiente
Abra o terminal na raiz do projeto e crie o ambiente virtual:
```bash
# Cria o virtual env
python -m venv .venv

# Ativa o virtual env no Windows
.venv\Scripts\activate

# Instala o pacote local em modo editável com dependências de desenvolvimento
pip install -e ".[dev]"
```

### 3. Copiando as Configurações
Faça uma cópia do arquivo `.env.example` nomeando-a como `.env` e configure suas variáveis de conexão do PostgreSQL e credenciais do Home Assistant (se aplicável):
```bash
copy .env.example .env
```

### 4. Preparando a Base e os Modelos
Execute os scripts de setup para preparar o ecossistema:
```bash
# 1. Configurar banco de dados PostgreSQL
python scripts/setup_db.py

# 2. Baixar o modelo LLM local (siga as instruções do painel de VRAM)
python scripts/download_models.py

# 3. Ingerir arquivos de exemplo para o RAG
python scripts/ingest.py --source documents/
```

### 5. Execução
Para iniciar o Jarvis no modo escuta ativo pelo microfone:
```bash
python run_jarvis.py
```
*(O script irá carregar o motor de embeddings, o ChromaDB, o LLM Mistral, o STT Faster Whisper e o sintetizador Piper, abrindo a captura de microfone logo em seguida).*

### 6. Interação por Voz
*   **Ativação**: Diga a palavra-chave **"Jarvis"** para iniciar uma solicitação ou comando verbal.
*   **Interrupção**: Se o assistente estiver falando ou pensando e você disser termos de parada (como *"pare"*, *"para"*, *"parar"*, *"cala a boca"*, *"silêncio"*, *"espera"*, *"calma"*), a resposta será interrompida instantaneamente no terminal e no alto-falante.