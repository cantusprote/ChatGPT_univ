from __future__ import annotations

import re

from .embedding import Embedder
from .models import SearchHit
from .state import StateStore
from .vector import ChromaStore


class HybridSearch:
    def __init__(self, state: StateStore, vectors: ChromaStore, embedder: Embedder):
        self.state, self.vectors, self.embedder = state, vectors, embedder

    def search(self, query: str, limit: int = 8) -> list[dict]:
        pool = max(20, limit * 4)
        semantic = self.vectors.search(self.embedder.query(query), pool)
        lexical = self.state.lexical_search(query, pool)
        exactish = bool(re.search(r'[A-Z]{2,}|\d|[-+.#]', query))
        semantic_weight, lexical_weight = (0.45, 0.55) if exactish else (0.55, 0.45)
        fused: dict[str, tuple[float, SearchHit]] = {}
        for weight, results in ((semantic_weight, semantic), (lexical_weight, lexical)):
            for rank, hit in enumerate(results, 1):
                score = weight / (60 + rank)
                if hit.chunk_id in fused:
                    old_score, old_hit = fused[hit.chunk_id]
                    old_hit.matched_by = sorted(set(old_hit.matched_by + hit.matched_by))
                    fused[hit.chunk_id] = (old_score + score, old_hit)
                else:
                    fused[hit.chunk_id] = (score, hit)
        ranked = sorted(fused.values(), key=lambda item: item[0], reverse=True)[:limit]
        output = []
        for score, hit in ranked:
            context_rows = self.state.context_for(hit.chunk_id)
            context = "\n".join(f"- {row['content']}" for row in context_rows)
            output.append({
                "path": hit.source_path, "title": hit.title, "heading": hit.heading,
                "score": round(score, 6), "matched_by": hit.matched_by,
                "excerpt": hit.content, "context": context or hit.content,
            })
        return output

