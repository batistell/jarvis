"""Jarvis — Modelos SQLAlchemy para a base de conhecimento.

Define as tabelas principais:
- knowledge_articles: Artigos e documentação com conteúdo completo.
- code_examples: Snippets de código categorizados por linguagem e tópico.
- conversations: Histórico completo de conversas com o usuário.
- conversation_corrections: Correções do usuário sobre respostas anteriores.
- tags + tabelas de associação: Sistema de categorização flexível.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Base declarativa para todos os modelos."""

    pass


# ---------------------------------------------------------------------------
# Tabelas de associação (many-to-many)
# ---------------------------------------------------------------------------


class ArticleTag(Base):
    """Associação entre artigos e tags."""

    __tablename__ = "article_tags"

    article_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledge_articles.id"), primary_key=True
    )
    tag_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tags.id"), primary_key=True
    )


class CodeExampleTag(Base):
    """Associação entre code examples e tags."""

    __tablename__ = "code_example_tags"

    code_example_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("code_examples.id"), primary_key=True
    )
    tag_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tags.id"), primary_key=True
    )


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class Tag(Base):
    """Tag para categorização de conteúdo."""

    __tablename__ = "tags"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    category: Mapped[str | None] = mapped_column(String(50))  # ex: "language", "framework", "topic"

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    articles: Mapped[list[KnowledgeArticle]] = relationship(
        secondary="article_tags", back_populates="tags"
    )
    code_examples: Mapped[list[CodeExample]] = relationship(
        secondary="code_example_tags", back_populates="tags"
    )

    def __repr__(self) -> str:
        return f"<Tag(name={self.name!r}, category={self.category!r})>"


# ---------------------------------------------------------------------------
# Knowledge Articles
# ---------------------------------------------------------------------------


class KnowledgeArticle(Base):
    """Artigo ou documentação armazenado na base de conhecimento.

    Contém conteúdo completo em markdown, com título, resumo e metadados.
    Suporta full-text search via tsvector e busca vetorial via pgvector.
    """

    __tablename__ = "knowledge_articles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(String(500))  # URL, filepath, etc.
    source_type: Mapped[str | None] = mapped_column(String(50))  # "markdown", "url", "manual"

    # Embedding do conteúdo para busca vetorial via pgvector
    embedding = mapped_column(Vector(384), nullable=True)  # all-MiniLM-L6-v2 = 384 dims

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    tags: Mapped[list[Tag]] = relationship(
        secondary="article_tags", back_populates="articles"
    )

    __table_args__ = (
        Index("ix_articles_title_fts", "title", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<KnowledgeArticle(title={self.title!r})>"


# ---------------------------------------------------------------------------
# Code Examples
# ---------------------------------------------------------------------------


class CodeExample(Base):
    """Snippet de código categorizado por linguagem e tópico."""

    __tablename__ = "code_examples"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(50), nullable=False)  # "python", "sql", etc.
    framework: Mapped[str | None] = mapped_column(String(100))  # "fastapi", "sqlalchemy", etc.

    # Embedding para busca semântica
    embedding = mapped_column(Vector(384), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    tags: Mapped[list[Tag]] = relationship(
        secondary="code_example_tags", back_populates="code_examples"
    )

    def __repr__(self) -> str:
        return f"<CodeExample(title={self.title!r}, lang={self.language!r})>"


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


class Conversation(Base):
    """Registro de uma conversa completa com o Jarvis.

    Armazena a pergunta do usuário, respostas de cada fonte, resposta final
    sintetizada e metadados como tópico inferido.
    """

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    response_llm: Mapped[str | None] = mapped_column(Text)
    response_vector: Mapped[str | None] = mapped_column(Text)
    response_knowledge: Mapped[str | None] = mapped_column(Text)
    response_final: Mapped[str | None] = mapped_column(Text)

    topic: Mapped[str | None] = mapped_column(String(200))  # Tópico inferido automaticamente
    satisfaction_score: Mapped[int | None] = mapped_column(Integer)  # 1-5, feedback do usuário

    # Embedding da conversa para busca em histórico
    embedding = mapped_column(Vector(384), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    corrections: Mapped[list[ConversationCorrection]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Conversation(query={self.user_query[:50]!r}...)>"


# ---------------------------------------------------------------------------
# Conversation Corrections
# ---------------------------------------------------------------------------


class ConversationCorrection(Base):
    """Correção feita pelo usuário sobre uma resposta anterior.

    Permite ao sistema aprender com feedback explícito e melhorar
    respostas futuras em perguntas similares.
    """

    __tablename__ = "conversation_corrections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False
    )
    original_response: Mapped[str] = mapped_column(Text, nullable=False)
    corrected_response: Mapped[str] = mapped_column(Text, nullable=False)
    correction_note: Mapped[str | None] = mapped_column(Text)  # Explicação do motivo

    # Embedding da correção para buscar correções similares
    embedding = mapped_column(Vector(384), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    conversation: Mapped[Conversation] = relationship(back_populates="corrections")

    def __repr__(self) -> str:
        return f"<ConversationCorrection(conv={self.conversation_id!r})>"
