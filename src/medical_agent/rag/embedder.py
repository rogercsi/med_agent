from functools import lru_cache

from FlagEmbedding import BGEM3FlagModel

from medical_agent.config import get_settings


@lru_cache(maxsize=1)
def get_embedder() -> BGEM3FlagModel:
    settings = get_settings()
    return BGEM3FlagModel(settings.embed_model, use_fp16=True)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    output = model.encode(texts, batch_size=12, max_length=512)
    return output["dense_vecs"].tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
