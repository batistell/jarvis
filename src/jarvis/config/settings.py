"""Jarvis — Pydantic Settings.

Centraliza todas as configurações do sistema, carregando de variáveis de
ambiente e/ou arquivo `.env`.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Raiz do projeto (3 níveis acima: src/jarvis/config -> src/jarvis -> src -> jarvis/)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Carrega as variáveis de ambiente do .env para os sub-modelos herdarem
load_dotenv(_PROJECT_ROOT / ".env")



class Environment(str, Enum):
    """Ambientes de execução."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class DatabaseSettings(BaseSettings):
    """Configurações do PostgreSQL."""

    model_config = SettingsConfigDict(env_prefix="JARVIS_DB_")

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: SecretStr = SecretStr("postgres")
    name: str = "jarvis_kb"

    @property
    def async_url(self) -> str:
        """URL de conexão assíncrona (asyncpg)."""
        pwd = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{pwd}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        """URL de conexão síncrona (psycopg2 / setup scripts)."""
        pwd = self.password.get_secret_value()
        return f"postgresql://{self.user}:{pwd}@{self.host}:{self.port}/{self.name}"


class LLMSettings(BaseSettings):
    """Configurações do modelo de linguagem local (llama.cpp)."""

    model_config = SettingsConfigDict(env_prefix="JARVIS_LLM_")

    model_path: str = "models/mistral-7b-instruct-v0.3.Q4_K_M.gguf"
    n_gpu_layers: int = -1  # -1 = offload total para GPU
    n_ctx: int = 4096
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9

    @property
    def resolved_model_path(self) -> Path:
        """Caminho absoluto do modelo."""
        p = Path(self.model_path)
        if p.is_absolute():
            return p
        return _PROJECT_ROOT / p


class STTSettings(BaseSettings):
    """Configurações do Speech-to-Text (Faster Whisper)."""

    model_config = SettingsConfigDict(env_prefix="JARVIS_STT_")

    model_size: str = "large-v3"
    compute_type: str = "float16"
    device: str = "cuda"
    language: str = "pt"  # Idioma padrão para transcrição


class AudioSettings(BaseSettings):
    """Configurações de captura de áudio."""

    model_config = SettingsConfigDict(env_prefix="JARVIS_AUDIO_")

    sample_rate: int = 16000
    channels: int = 1
    vad_aggressiveness: int = 2  # 0-3 (mais alto = mais agressivo)
    silence_threshold_ms: int = 800  # ms de silêncio para finalizar frase
    chunk_duration_ms: int = 30  # duração de cada chunk de áudio (ms)


class EmbeddingSettings(BaseSettings):
    """Configurações de embeddings."""

    model_config = SettingsConfigDict(env_prefix="JARVIS_EMBEDDING_")

    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"  # Embeddings rodam na CPU para não competir com LLM
    chunk_size: int = 512
    chunk_overlap: int = 64


class ChromaSettings(BaseSettings):
    """Configurações do ChromaDB."""

    model_config = SettingsConfigDict(env_prefix="JARVIS_CHROMA_")

    persist_dir: str = "data/chroma"

    @property
    def resolved_persist_dir(self) -> Path:
        """Caminho absoluto do diretório de persistência."""
        p = Path(self.persist_dir)
        if p.is_absolute():
            return p
        return _PROJECT_ROOT / p


class UISettings(BaseSettings):
    """Configurações de interface."""

    model_config = SettingsConfigDict(env_prefix="JARVIS_UI_")

    mode: Literal["terminal", "web"] = "terminal"


class WebSettings(BaseSettings):
    """Configurações do servidor web."""

    model_config = SettingsConfigDict(env_prefix="JARVIS_WEB_")

    host: str = "127.0.0.1"
    port: int = 8000


class Settings(BaseSettings):
    """Configurações raízes do Jarvis.

    Agrupa todas as sub-configurações e carrega variáveis de ambiente
    do arquivo ``.env`` na raiz do projeto.
    """

    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Environment = Environment.DEVELOPMENT
    log_level: str = "DEBUG"

    # Sub-configurações (instanciadas com defaults próprios)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    ui: UISettings = Field(default_factory=UISettings)
    web: WebSettings = Field(default_factory=WebSettings)

    @property
    def project_root(self) -> Path:
        """Raiz do projeto."""
        return _PROJECT_ROOT


# Singleton — importar onde necessário
_settings: Settings | None = None


def get_settings() -> Settings:
    """Retorna a instância singleton das configurações."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
    return _settings
