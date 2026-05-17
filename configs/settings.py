"""
configs/settings.py
───────────────────
Centralised, validated configuration using Pydantic-Settings.
All values come from environment variables or .env file.
"""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")  # openai | groq | gemini | ollama
    llm_model: str = Field(default="gpt-4o", alias="LLM_MODEL")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")

    # ── SWE-bench ────────────────────────────────────────────────────────────
    swebench_dataset: str = Field(
        default="princeton-nlp/SWE-bench_Lite", alias="SWEBENCH_DATASET"
    )
    swebench_split: str = Field(default="test", alias="SWEBENCH_SPLIT")
    results_dir: Path = Field(default=Path("./results"), alias="RESULTS_DIR")

    # ── Sandbox ──────────────────────────────────────────────────────────────
    sandbox_image: str = Field(
        default="code-agent-sandbox:latest", alias="SANDBOX_IMAGE"
    )
    sandbox_timeout: int = Field(default=60, alias="SANDBOX_TIMEOUT")
    sandbox_memory_limit: str = Field(default="2g", alias="SANDBOX_MEMORY_LIMIT")
    sandbox_cpu_limit: float = Field(default=2.0, alias="SANDBOX_CPU_LIMIT")
    sandbox_network: str = Field(default="none", alias="SANDBOX_NETWORK")

    # ── Caching ──────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    diskcache_dir: Path = Field(default=Path("./.cache/diskcache"), alias="DISKCACHE_DIR")

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(default="./mlruns", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(
        default="code-agent-baseline", alias="MLFLOW_EXPERIMENT_NAME"
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    bm25_top_k: int = Field(default=20, alias="BM25_TOP_K")
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    rrf_alpha_bm25: float = Field(default=0.4, alias="RRF_ALPHA_BM25")
    rrf_alpha_embed: float = Field(default=0.4, alias="RRF_ALPHA_EMBED")
    rrf_alpha_ppr: float = Field(default=0.2, alias="RRF_ALPHA_PPR")

    # ── Agent Loop ────────────────────────────────────────────────────────────
    max_attempts: int = Field(default=3, alias="MAX_ATTEMPTS")
    max_file_tokens: int = Field(default=2000, alias="MAX_FILE_TOKENS")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1", alias="CELERY_BROKER_URL"
    )

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.diskcache_dir.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere
settings = Settings()
