"""Jarvis — LLM Engine (llama-cpp-python wrapper).

Carrega e executa inferência assíncrona com streaming usando o modelo GGUF local.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncIterator

from jarvis.config.settings import get_settings
from jarvis.core.logging import get_logger
from jarvis.core.executor import get_gpu_executor
from jarvis.llm.prompts import BUTLER_SYSTEM_PROMPT_PT, BUTLER_SYSTEM_PROMPT_EN
from jarvis.core.homeassistant import HomeAssistantClient

log = get_logger(__name__)


class LLMEngine:
    """Wrapper de alta performance para o modelo LLM GGUF local via llama.cpp."""

    def __init__(self) -> None:
        self.settings = get_settings().llm
        self._model = None
        # Usa o executor compartilhado de thread única para GPU
        self._executor = get_gpu_executor()
        self._is_cancelled = False
        self._lock: asyncio.Lock | None = None

    def load_model(self) -> None:
        """Carrega o modelo na memória de forma síncrona.

        Deve ser executado através do executor para não bloquear a thread principal.
        """
        if self._model is not None:
            return

        # Import local para não penalizar o tempo de startup global do Jarvis
        from llama_cpp import Llama

        model_path = self.settings.resolved_model_path
        if not model_path.exists():
            log.critical("Modelo LLM não encontrado no caminho: {}", model_path)
            raise FileNotFoundError(f"Modelo LLM não encontrado em: {model_path}")

        log.info(
            "Carregando LLM local: {} (n_gpu_layers={}, n_ctx={})",
            model_path.name,
            self.settings.n_gpu_layers,
            self.settings.n_ctx,
        )

        try:
            self._model = Llama(
                model_path=str(model_path.resolve()),
                n_gpu_layers=self.settings.n_gpu_layers,
                n_ctx=self.settings.n_ctx,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                top_p=self.settings.top_p,
                verbose=False,
            )
            log.info("Modelo LLM local carregado com sucesso!")
        except Exception as e:
            log.critical("Falha crítica ao carregar modelo LLM: {}", e)
            raise e

    def pre_load_rag(self) -> None:
        """Pré-carrega o motor de embeddings e a conexão com o banco de dados.

        Deve ser executado no startup para otimizar o tempo de resposta da primeira pergunta.
        """
        from jarvis.vectorstore.embeddings import EmbeddingEngine
        from jarvis.vectorstore.store import VectorStore

        log.info("Pré-carregando o motor de busca vetorial (RAG)...")
        if not hasattr(self, "_embedding_engine"):
            self._embedding_engine = EmbeddingEngine(device="cpu")
        if not hasattr(self, "_vector_store"):
            self._vector_store = VectorStore()
        try:
            self._vector_store.clear_collection("conversations")
            log.info("RAG: Coleção de conversas temporárias limpa para uma nova execução.")
        except Exception as e:
            log.warning("RAG: Falha ao limpar coleção de conversas temporárias: {}", e)
        log.info("RAG pré-carregado com sucesso.")

    async def save_conversation_turn(self, user_prompt: str, assistant_response: str) -> None:
        """Salva a interação atual na coleção temporária de conversas do VectorStore."""
        if not hasattr(self, "_embedding_engine") or not hasattr(self, "_vector_store"):
            return
        
        # Limpa o texto da resposta removendo tags de chamadas de ferramentas se houver
        clean_response = assistant_response
        if "<tool_call>" in clean_response:
            parts = clean_response.split("<tool_call>")
            clean_response = parts[0].strip()
            if len(parts) > 1 and "</tool_call>" in parts[1]:
                subparts = parts[1].split("</tool_call>")
                if len(subparts) > 1 and subparts[1]:
                    clean_response += " " + subparts[1].strip()

        clean_response = clean_response.strip()
        if not clean_response:
            return

        # Trunca mensagens muito longas no histórico para evitar estourar o limite de tokens da janela de contexto
        if len(user_prompt) > 150:
            user_prompt = user_prompt[:150] + "... [pergunta truncada]"
        if len(clean_response) > 350:
            clean_response = clean_response[:350] + "... [resposta truncada]"

        turn_text = f"Usuário: {user_prompt}\nJarvis: {clean_response}"
        
        try:
            # Gera embedding de forma síncrona para evitar problemas de threads com PyTorch no Windows
            emb = self._embedding_engine.get_query_embedding(turn_text)
            import time
            turn_id = f"turn_{time.time_ns()}"
            metadata = {
                "user": user_prompt[:100],
                "assistant": clean_response[:100],
                "timestamp": time.time()
            }
            await self._vector_store.add_documents(
                collection_name="conversations",
                texts=[turn_text],
                embeddings=[emb],
                metadatas=[metadata],
                ids=[turn_id]
            )
            log.info("RAG: Turno de conversa salvo no VectorStore.")
        except Exception as e:
            log.error("RAG: Falha ao salvar turno de conversa: {}", e)

    def interrupt(self) -> None:
        """Interrompe qualquer geração ativa do LLM."""
        self._is_cancelled = True
        log.info("LLM: Geração interrompida pelo usuário.")

    async def generate_stream(
        self,
        prompt: str,
        language: str = "pt",
        ha_client: HomeAssistantClient | None = None,
    ) -> AsyncIterator[str]:
        """Gera resposta do LLM em streaming assíncrono garantindo exclusão mútua de acesso à GPU/Sampler."""
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            async for token in self._generate_stream_inner(prompt, language, ha_client):
                yield token

    async def _generate_stream_inner(
        self,
        prompt: str,
        language: str = "pt",
        ha_client: HomeAssistantClient | None = None,
    ) -> AsyncIterator[str]:
        self._is_cancelled = False
        loop = asyncio.get_running_loop()

        # Garante o carregamento do modelo em segundo plano
        if self._model is None:
            await loop.run_in_executor(self._executor, self.load_model)

        # RAG Context Retrieval — async pipeline com skip para comandos curtos
        _RAG_SKIP_WORDS = 6  # prompts < N palavras pulam embedding (~177ms economizados)
        context = ""
        conv_context = ""
        word_count = len(prompt.split())
        try:
            from jarvis.vectorstore.embeddings import EmbeddingEngine
            from jarvis.vectorstore.store import VectorStore

            # Inicializa o motor de embeddings e a conexão com o VectorStore se não criados
            if not hasattr(self, "_embedding_engine"):
                self._embedding_engine = EmbeddingEngine(device="cpu")
            if not hasattr(self, "_vector_store"):
                self._vector_store = VectorStore()

            if word_count < _RAG_SKIP_WORDS:
                # Comando curto — pula embedding + busca documental (~177ms economizados)
                log.debug("RAG: Pulado para prompt curto ({} palavras < {}).", word_count, _RAG_SKIP_WORDS)
                conv_parts = await self._vector_store.get_chronological_conversations(limit=3)
                if conv_parts:
                    conv_context = "\n---\n".join(conv_parts)
            else:
                # Pipeline assíncrono: embedding e histórico de conversa em paralelo
                query_emb, conv_parts = await asyncio.gather(
                    asyncio.to_thread(self._embedding_engine.get_query_embedding, prompt),
                    self._vector_store.get_chronological_conversations(limit=3),
                )

                if conv_parts:
                    conv_context = "\n---\n".join(conv_parts)
                    log.info("RAG: Recobrados {} turnos de conversa anteriores do VectorStore (ordem cronológica).", len(conv_parts))

                # 1. Recupera documentos permanentes da base de conhecimento (RAG)
                doc_results = await self._vector_store.query_collection(
                    collection_name="documents",
                    query_embeddings=[query_emb],
                    limit=2,
                )
                context_parts = []
                for r in doc_results:
                    if r.get("distance", 1.0) < 0.8:
                        doc_content = r["document"]
                        if len(doc_content) > 600:
                            doc_content = doc_content[:600] + "\n... [Documento truncado]"
                        context_parts.append(doc_content)
                if context_parts:
                    context = "\n\n".join(context_parts)
                    log.info("RAG: Recobrados {} chunks relevantes do VectorStore.", len(context_parts))
        except Exception as e:
            log.warning("RAG: Falha ao recuperar contexto do VectorStore: {}", e)

        # Prepare system prompt
        if language == "en":
            system_prompt = BUTLER_SYSTEM_PROMPT_EN
        else:
            system_prompt = BUTLER_SYSTEM_PROMPT_PT

        # Inject Home Assistant entities if available
        tools = None
        if ha_client and ha_client.entities:
            entities_desc = "\n".join(
                [f"- {e['name']} (entity_id: '{e['entity_id']}')" for e in ha_client.entities]
            )
            if language == "en":
                entity_prompt = (
                    "\n\nYou have control over the following home automation devices via Home Assistant:\n"
                    f"{entities_desc}\n"
                    "If the user asks you to turn a device on, turn a device off, set a value, or control any device listed above, use the 'control_home_device' tool. Do NOT try to simulate control or say you can't; just call the tool. Call it with domain, service, entity_id, and optionally data."
                )
            else:
                entity_prompt = (
                    "\n\nVocê tem controle sobre os seguintes dispositivos de automação residencial através do Home Assistant:\n"
                    f"{entities_desc}\n"
                    "Se o usuário pedir para ligar, desligar, ajustar ou controlar qualquer um dos dispositivos listados acima, chame a ferramenta 'control_home_device' apropriada. Nunca simule o controle nem diga que não é possível; simplesmente chame a ferramenta informando o domain, service, entity_id e opcionalmente o data."
                )
            system_prompt += entity_prompt

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "control_home_device",
                        "description": (
                            "Controla um dispositivo doméstico inteligente no Home Assistant."
                            if language == "pt" else
                            "Control a smart home device in Home Assistant."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "service": {
                                    "type": "string",
                                    "description": "A ação a executar (ex: 'turn_on', 'turn_off', 'toggle', 'set_temperature')."
                                    if language == "pt" else
                                    "The action to perform (e.g., 'turn_on', 'turn_off', 'toggle', 'set_temperature')."
                                },
                                "entity_id": {
                                    "type": "string",
                                    "description": "O ID da entidade no Home Assistant (ex: 'light.living_room_light')."
                                    if language == "pt" else
                                    "The target entity ID in Home Assistant (e.g., 'light.living_room_light')."
                                },
                                "data": {
                                    "type": "object",
                                    "description": "Parâmetros extras opcionais (ex: {'temperature': 22.0})."
                                    if language == "pt" else
                                    "Optional extra parameters (e.g., {'temperature': 22.0})."
                                },
                                "delay": {
                                    "type": "integer",
                                    "description": "Tempo opcional em segundos para aguardar antes de executar o comando."
                                    if language == "pt" else
                                    "Optional delay in seconds to wait before executing this command."
                                }
                            },
                            "required": ["service", "entity_id"]
                        }
                    }
                }
            ]

        # Prepare messages
        if language == "en":
            user_content = prompt
            if conv_context:
                user_content = (
                    "Context from previous turns in this session:\n"
                    f"{conv_context}\n\n"
                    f"{user_content}"
                )
            if context:
                user_content = (
                    "You have access to your Master's personal files and knowledge base. "
                    "Use the following relevant snippets to answer the Master's question. "
                    "Since these files belong to and describe your Master, you can assume that personal pronouns (like 'I', 'my', 'me') refer to the person described in these files.\n\n"
                    f"Relevant Context:\n{context}\n\n"
                    f"Question: {user_content}"
                )
        else:
            user_content = prompt
            if conv_context:
                user_content = (
                    "Contexto de turnos anteriores nesta sessão:\n"
                    f"{conv_context}\n\n"
                    f"{user_content}"
                )
            if context:
                user_content = (
                    "Você tem acesso aos arquivos pessoais e base de conhecimento do seu Mestre. "
                    "Use os seguintes trechos relevantes para responder à pergunta do Mestre. "
                    "Como estes arquivos pertencem e descrevem o seu Mestre, você pode assumir que pronomes pessoais (como 'eu', 'meu', 'me') se referem à pessoa descrita nestes arquivos.\n\n"
                    f"Contexto Relevante:\n{context}\n\n"
                    f"Pergunta: {user_content}"
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        def _create_stream():
            assert self._model is not None
            return self._model.create_chat_completion(
                messages=messages,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                stream=True,
                tools=tools,
                tool_choice="auto" if tools else None,
            )

        try:
            log.info("PERF: Chamando create_chat_completion no modelo (início do pre-fill / processamento do prompt)...")
            response_stream = await loop.run_in_executor(self._executor, _create_stream)
            iterator = iter(response_stream)
            log.info("PERF: Stream do LLM inicializado com sucesso.")
        except Exception as e:
            log.error("Erro ao iniciar stream do LLM: {}", e)
            raise e

        def _get_next_chunk(it):
            try:
                return next(it)
            except StopIteration:
                return None
            except Exception as e:
                return e

        in_tool_call = False
        tool_call_buffer = ""
        collected_tool_calls = []
        first_chunk_received = False

        while not self._is_cancelled:
            # Executa o passo de inferência do próximo token no executor de GPU compartilhado
            chunk = await loop.run_in_executor(self._executor, _get_next_chunk, iterator)
            if not first_chunk_received and chunk is not None:
                log.info("PERF: Primeiro chunk recebido do stream do LLM (pre-fill concluído, iniciando geração).")
                first_chunk_received = True

            if chunk is None:
                if collected_tool_calls:
                    log.info("LLM: Executando {} chamadas de ferramentas coletadas no stream...", len(collected_tool_calls))
                    import json
                    for tool_data in collected_tool_calls:
                        func_name = tool_data.get("name")
                        args = tool_data.get("arguments", {})
                        
                        if func_name == "control_home_device" and ha_client:
                            service = args.get("service")
                            entity_id = args.get("entity_id")
                            domain = args.get("domain") or (entity_id.split(".")[0] if entity_id else "")
                            data = args.get("data")
                            delay = float(args.get("delay", 0.0))
                            
                            # Dispara a chamada REST ao Home Assistant em segundo plano (em paralelo, respeitando o delay)
                            asyncio.create_task(
                                ha_client.control_entity(
                                    domain=domain,
                                    service=service,
                                    entity_id=entity_id,
                                    data=data,
                                    delay=delay
                                )
                            )
                            
                            # Registra no histórico de mensagens
                            assistant_msg = {
                                "role": "assistant",
                                "content": f"<tool_call>\n{json.dumps(tool_data)}\n</tool_call>"
                            }
                            tool_result_str = json.dumps({"status": "success", "message": "Service triggered in parallel"})
                            tool_msg = {
                                "role": "tool",
                                "name": "control_home_device",
                                "content": tool_result_str
                            }
                            messages.append(assistant_msg)
                            messages.append(tool_msg)
                            
                    collected_tool_calls.clear()
                    
                    # Gera resposta verbal de confirmação
                    def _create_follow_up():
                        assert self._model is not None
                        return self._model.create_chat_completion(
                            messages=messages,
                            max_tokens=self.settings.max_tokens,
                            temperature=self.settings.temperature,
                            top_p=self.settings.top_p,
                            stream=True,
                        )
                    
                    log.info("PERF: Enviando resultado das ferramentas em segundo plano e iniciando chamada follow-up...")
                    follow_up_stream = await loop.run_in_executor(self._executor, _create_follow_up)
                    iterator = iter(follow_up_stream)
                    first_chunk_received = False
                    in_tool_call = False
                    tool_call_buffer = ""
                    continue
                else:
                    break
            
            if isinstance(chunk, Exception):
                log.error("Erro durante a inferência do token do LLM: {}", chunk)
                raise chunk
            
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                token = delta["content"]
                
                # Tratamento de tags de chamada de ferramenta
                if "<tool_call>" in token:
                    in_tool_call = True
                    parts = token.split("<tool_call>")
                    if parts[0]:
                        yield parts[0]
                    if len(parts) > 1:
                        tool_call_buffer += parts[1]
                    continue
                
                if "</tool_call>" in token:
                    in_tool_call = False
                    parts = token.split("</tool_call>")
                    tool_call_buffer += parts[0]
                    
                    log.info("LLM: Coletada chamada de ferramenta no stream: {}", tool_call_buffer)
                    try:
                        import json
                        tool_json_str = tool_call_buffer.strip()
                        tool_data = json.loads(tool_json_str)
                        collected_tool_calls.append(tool_data)
                    except Exception as e:
                        log.error("LLM: Falha ao decodificar JSON da chamada de ferramenta: {}", e)
                        if language == "pt":
                            yield "Perdão, Senhor. Ocorreu um erro ao processar o comando."
                        else:
                            yield "I am sorry, Sir. An error occurred while processing the command."
                    
                    if len(parts) > 1 and parts[1]:
                        yield parts[1]
                    tool_call_buffer = ""  # Reseta buffer para próxima chamada
                    continue
                
                if in_tool_call:
                    tool_call_buffer += token
                else:
                    yield token
