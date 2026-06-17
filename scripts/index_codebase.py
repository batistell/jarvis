"""Jarvis — Script para indexação semântica do código fonte do projeto no ChromaDB."""

from __future__ import annotations

import argparse
import asyncio
import ast
import hashlib
import os
import sys
import traceback
from pathlib import Path
from loguru import logger

# Adiciona o diretório src ao path do Python para encontrar o pacote jarvis
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jarvis.vectorstore.embeddings import EmbeddingEngine
from jarvis.vectorstore.store import VectorStore
from jarvis.config.settings import get_settings


class CodebaseIndexer:
    """Analisador estático e indexador de código Python para o VectorStore do Jarvis."""

    def __init__(self, embedding_engine: EmbeddingEngine, vector_store: VectorStore) -> None:
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.settings = get_settings()
        self.project_root = self.settings.project_root

    def parse_python_file(self, file_path: Path) -> list[dict]:
        """Parseia o arquivo Python usando AST para extrair classes, funções e métodos."""
        print(f"[debug] Iniciando parse_python_file para: {file_path}", file=sys.stderr)
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            tree = ast.parse(content, filename=str(file_path))
            
            entities = []
            rel_path = file_path.relative_to(self.project_root).as_posix()

            # Percorre os nós no nível do módulo
            for node in tree.body:
                try:
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name
                        class_doc = ast.get_docstring(node) or ""
                        bases = []
                        for b in node.bases:
                            try:
                                bases.append(ast.unparse(b))
                            except Exception:
                                bases.append(str(b))
                        start_line = getattr(node, "lineno", 1)
                        end_line = getattr(node, "end_lineno", len(lines))
                        class_code = "\n".join(lines[max(0, start_line - 1) : min(len(lines), end_line)])

                        entities.append({
                            "type": "class",
                            "name": class_name,
                            "docstring": class_doc,
                            "bases": bases,
                            "start_line": start_line,
                            "end_line": end_line,
                            "code": class_code,
                            "file_path": rel_path
                        })

                        # Extrai métodos dentro da classe
                        for subnode in node.body:
                            if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                method_name = subnode.name
                                method_doc = ast.get_docstring(subnode) or ""
                                m_start = getattr(subnode, "lineno", 1)
                                m_end = getattr(subnode, "end_lineno", len(lines))
                                method_code = "\n".join(lines[max(0, m_start - 1) : min(len(lines), m_end)])

                                entities.append({
                                    "type": "method",
                                    "name": f"{class_name}.{method_name}",
                                    "docstring": method_doc,
                                    "class_name": class_name,
                                    "start_line": m_start,
                                    "end_line": m_end,
                                    "code": method_code,
                                    "file_path": rel_path
                                })

                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_name = node.name
                        func_doc = ast.get_docstring(node) or ""
                        start_line = getattr(node, "lineno", 1)
                        end_line = getattr(node, "end_lineno", len(lines))
                        func_code = "\n".join(lines[max(0, start_line - 1) : min(len(lines), end_line)])

                        entities.append({
                            "type": "function",
                            "name": func_name,
                            "docstring": func_doc,
                            "start_line": start_line,
                            "end_line": end_line,
                            "code": func_code,
                            "file_path": rel_path
                        })
                except Exception as node_err:
                    print(f"[debug] Erro no nó em {file_path.name}: {node_err}", file=sys.stderr)
            
            print(f"[debug] Fim parse_python_file para: {file_path}. Total entidades: {len(entities)}", file=sys.stderr)
            return entities
        except Exception as e:
            print(f"[debug] Erro crítico parse_python_file {file_path.name}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return []

    async def index_project(self) -> int:
        """Varre o projeto inteiro e indexa os elementos Python no ChromaDB."""
        logger.info(f"Escaneando diretórios do projeto a partir de: {self.project_root}")
        
        py_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Ignora diretórios indesejados durante a caminhada
            dirs[:] = [
                d for d in dirs
                if d not in {".venv", ".git", ".pytest_cache", "build", "dist", "data", "models", "logs", "scratch", ".gemini"}
            ]
            for file in files:
                if file.endswith(".py"):
                    py_files.append(Path(root) / file)

        logger.info(f"Encontrados {len(py_files)} arquivos Python para indexação estruturada.")
        
        # Limpa a coleção para evitar elementos obsoletos
        logger.info("Limpando coleção anterior 'code_snippets' no ChromaDB...")
        self.vector_store.clear_collection("code_snippets")

        total_chunks = 0
        for file_path in py_files:
            rel_file = file_path.relative_to(self.project_root)
            logger.info(f"Processando arquivo: {rel_file}")
            
            try:
                entities = self.parse_python_file(file_path)
                if not entities:
                    print(f"[debug] Nenhuma entidade extraída para {rel_file}", file=sys.stderr)
                    continue

                texts = []
                metadatas = []
                ids = []

                for ent in entities:
                    if ent["type"] == "class":
                        rep = (
                            f"Arquivo: {ent['file_path']}\n"
                            f"Tipo: Classe\n"
                            f"Nome: {ent['name']}\n"
                            f"Herança: {', '.join(ent['bases']) if ent['bases'] else 'Nenhuma'}\n"
                            f"Linhas: {ent['start_line']}-{ent['end_line']}\n"
                            f"Descrição: {ent['docstring']}\n"
                            f"Código:\n{ent['code']}"
                        )
                    elif ent["type"] == "method":
                        rep = (
                            f"Arquivo: {ent['file_path']}\n"
                            f"Tipo: Método\n"
                            f"Classe: {ent['class_name']}\n"
                            f"Nome: {ent['name']}\n"
                            f"Linhas: {ent['start_line']}-{ent['end_line']}\n"
                            f"Descrição: {ent['docstring']}\n"
                            f"Código:\n{ent['code']}"
                        )
                    else:  # function
                        rep = (
                            f"Arquivo: {ent['file_path']}\n"
                            f"Tipo: Função\n"
                            f"Nome: {ent['name']}\n"
                            f"Linhas: {ent['start_line']}-{ent['end_line']}\n"
                            f"Descrição: {ent['docstring']}\n"
                            f"Código:\n{ent['code']}"
                        )

                    texts.append(rep)
                    
                    hash_input = f"{ent['file_path']}_{ent['type']}_{ent['name']}"
                    chunk_id = "code_" + hashlib.md5(hash_input.encode("utf-8")).hexdigest()
                    ids.append(chunk_id)

                    metadatas.append({
                        "source": ent["file_path"],
                        "type": ent["type"],
                        "name": ent["name"],
                        "start_line": ent["start_line"],
                        "end_line": ent["end_line"]
                    })

                if texts:
                    print(f"[debug] Adicionando {len(texts)} chunks de {rel_file} no ChromaDB...", file=sys.stderr)
                    embs = self.embedding_engine.get_embeddings(texts)
                    await self.vector_store.add_documents(
                        collection_name="code_snippets",
                        texts=texts,
                        embeddings=embs,
                        metadatas=metadatas,
                        ids=ids
                    )
                    total_chunks += len(texts)
                    logger.info(f"Indexados {len(texts)} elementos de: {rel_file}")
            except Exception as file_err:
                print(f"[debug] Erro crítico ao processar arquivo {rel_file}: {file_err}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        logger.info(f"Indexação de código concluída! Total de {total_chunks} elementos inseridos na coleção 'code_snippets'.")
        return total_chunks


async def run_indexing(device: str) -> None:
    """Executa o fluxo de indexação."""
    embedding_engine = EmbeddingEngine(device=device)
    vector_store = VectorStore()
    indexer = CodebaseIndexer(embedding_engine, vector_store)
    await indexer.index_project()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indexador AST de Código Fonte do Jarvis para Busca Semântica RAG."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Dispositivo para geração de embeddings ('cuda' ou 'cpu', default: cuda).",
    )
    args = parser.parse_args()

    # Loguru configurado para stdout
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    try:
        asyncio.run(run_indexing(args.device))
    except BaseException as e:
        print(f"\nCRITICAL ERROR IN MAIN: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
