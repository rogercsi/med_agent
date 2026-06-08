from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    openai_api_key: str = "sk-placeholder"
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_rag: str = "medical_dense"
    qdrant_collection_mem0: str = "medical_mem0"

    # RAG tuning
    embed_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    bm25_top_k: int = 50
    dense_top_k: int = 20
    rerank_top_k: int = 5
    compress_token_threshold: int = 1500

    # Agent
    summarize_every_n_turns: int = 6

    # Eval
    ragas_testset_path: str = "data/eval/testset.json"
    ragas_baseline_path: str = "data/eval/baseline_results.json"

    # Internal paths
    bm25_index_path: str = "data/bm25_index.pkl"
    checkpoints_db_path: str = "data/checkpoints.db"
    raw_docs_path: str = "data/raw"


@lru_cache
def get_settings() -> Settings:
    return Settings()
