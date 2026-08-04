from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import RetrievalHit


@dataclass
class KnowledgeChunk:
    source: str
    title: str
    chunk: str


class SimpleRAG:
    def __init__(self, kb_dir: str | Path):
        self.kb_dir = Path(kb_dir)
        self.chunks: List[KnowledgeChunk] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None
        self._load()

    def _load(self) -> None:
        docs = sorted(self.kb_dir.glob("*.md"))
        for doc in docs:
            text = doc.read_text(encoding="utf-8")
            title = doc.stem.replace("_", " ").title()
            parts = [p.strip() for p in text.split("\n\n") if p.strip()]
            for part in parts:
                self.chunks.append(KnowledgeChunk(source=doc.name, title=title, chunk=part))

        corpus = [f"{c.title}. {c.chunk}" for c in self.chunks] or ["empty corpus"]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalHit]:
        if not self.chunks:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [
            RetrievalHit(
                source=self.chunks[i].source,
                title=self.chunks[i].title,
                chunk=self.chunks[i].chunk,
                score=float(scores[i]),
            )
            for i in ranked
            if scores[i] > 0
        ]
