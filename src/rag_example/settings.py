from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8"
    )

    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    LLM_MODEL: str = "Qwen/Qwen3-0.6B"
    MAX_CHUNK_TOKENS: int = 512


settings = Settings()