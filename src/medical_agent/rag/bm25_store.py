import pickle
from dataclasses import dataclass, field
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from medical_agent.config import get_settings


@dataclass
class BM25Store:
    corpus_ids: list[str] = field(default_factory=list)
    corpus_texts: list[str] = field(default_factory=list)
    _model: BM25Okapi | None = field(default=None, repr=False)

    @property
    def id_to_text(self) -> dict[str, str]:
        if not hasattr(self, "_id_to_text") or len(self._id_to_text) != len(self.corpus_ids):
            self._id_to_text: dict[str, str] = dict(zip(self.corpus_ids, self.corpus_texts, strict=True))
        return self._id_to_text

    def add(self, chunk_id: str, text: str) -> None:
        self.corpus_ids.append(chunk_id)
        self.corpus_texts.append(text)
        self._model = None  # invalidate cache

    def _build(self) -> None:
        tokenized = [list(jieba.cut(t)) for t in self.corpus_texts]
        self._model = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self.corpus_ids:
            return []
        if self._model is None:
            self._build()
        tokens = list(jieba.cut(query))
        scores = self._model.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.corpus_ids[i], float(s)) for i, s in ranked[:top_k] if s > 0]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Store":
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301


def get_bm25_store(force_new: bool = False) -> BM25Store:
    settings = get_settings()
    path = Path(settings.bm25_index_path)
    if not force_new and path.exists():
        return BM25Store.load(path)
    return BM25Store()
