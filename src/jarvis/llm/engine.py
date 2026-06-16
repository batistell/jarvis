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

log = get_logger(__name__)


class LLMEngine:
    """Wrapper de alta performance para o modelo LLM GGUF local via llama.cpp."""

    def __init__(self) -> None:
        self.settings = get_settings().llm
        self._model = None
        # Usa o executor compartilhado de thread única para GPU
        self._executor = get_gpu_executor()
        self._is_cancelled = False

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
        log.info("RAG pré-carregado com sucesso.")

    def interrupt(self) -> None:
        """Interrompe qualquer geração ativa do LLM."""
        self._is_cancelled = True
        log.info("LLM: Geração interrompida pelo usuário.")

    async def generate_stream(self, prompt: str, language: str = "pt") -> AsyncIterator[str]:
        """Gera resposta do LLM em streaming assíncrono.

        Args:
            prompt: Pergunta ou instrução do usuário.
            language: Idioma da resposta ("pt" ou "en").

        Yields:
            Fragmentos de texto (tokens) conforme gerados pelo modelo.
        """
        self._is_cancelled = False
        loop = asyncio.get_running_loop()

        # Garante o carregamento do modelo em segundo plano
        if self._model is None:
            await loop.run_in_executor(self._executor, self.load_model)

        # RAG Context Retrieval
        context = ""
        try:
            from jarvis.vectorstore.embeddings import EmbeddingEngine
            from jarvis.vectorstore.store import VectorStore

            # Inicializa o motor de embeddings e a conexão com o VectorStore se não criados
            if not hasattr(self, "_embedding_engine"):
                self._embedding_engine = EmbeddingEngine(device="cpu")
            if not hasattr(self, "_vector_store"):
                self._vector_store = VectorStore()

            query_emb = self._embedding_engine.get_query_embedding(prompt)
            results = await self._vector_store.query_collection(
                collection_name="documents",
                query_embeddings=[query_emb],
                limit=3,
            )

            # Filtra chunks relevantes com distância inferior a 0.8
            context_parts = [r["document"] for r in results if r.get("distance", 1.0) < 0.8]
            if context_parts:
                context = "\n\n".join(context_parts)
                log.info("RAG: Recobrados {} chunks relevantes do VectorStore.", len(context_parts))
        except Exception as e:
            log.warning("RAG: Falha ao recuperar contexto do VectorStore: {}", e)

        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()

        def _run_completion() -> None:
            try:
                assert self._model is not None

                if language == "en":
                    system_prompt = BUTLER_SYSTEM_PROMPT_EN
                    user_content = prompt
                    if context:
                        user_content = (
                            "You have access to your Master's personal files and knowledge base. "
                            "Use the following relevant snippets to answer the Master's question. "
                            "Since these files belong to and describe your Master, you can assume that personal pronouns (like 'I', 'my', 'me') refer to the person described in these files.\n\n"
                            f"Relevant Context:\n{context}\n\n"
                            f"Question: {prompt}"
                        )
                else:
                    system_prompt = BUTLER_SYSTEM_PROMPT_PT
                    user_content = prompt
                    if context:
                        user_content = (
                            "Você tem acesso aos arquivos pessoais e base de conhecimento do seu Mestre. "
                            "Use os seguintes trechos relevantes para responder à pergunta do Mestre. "
                            "Como estes arquivos pertencem e descrevem o seu Mestre, você pode assumir que pronomes pessoais (como 'eu', 'meu', 'me') se referem à pessoa descrita nestes arquivos.\n\n"
                            f"Contexto Relevante:\n{context}\n\n"
                            f"Pergunta: {prompt}"
                        )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
                
                response_stream = self._model.create_chat_completion(
                    messages=messages,
                    max_tokens=self.settings.max_tokens,
                    temperature=self.settings.temperature,
                    top_p=self.settings.top_p,
                    stream=True,
                )
                
                for chunk in response_stream:
                    if self._is_cancelled:
                        log.debug("LLM: Geração interrompida pelo sinal de cancelamento.")
                        break
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        # Envia token para a fila na thread principal
                        loop.call_soon_threadsafe(queue.put_nowait, delta["content"])
            except Exception as e:
                log.error("Erro na inferência do LLM: {}", e)
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                # Envia sentinela de encerramento
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # Dispara execução na thread secundária
        self._executor.submit(_run_completion)

        # Consome da fila e cede controle (yield) de forma assíncrona
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
