from __future__ import annotations

from pathlib import Path

from .models import Chunk, SearchHit


class ChromaStore:
    def __init__(self, path: Path, collection_name: str):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def replace_file(self, source_path: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.delete_file(source_path)
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "source_path": chunk.source_path,
                    "title": chunk.title,
                    "heading": chunk.heading,
                    "heading_path": chunk.heading_path,
                    "chunk_type": chunk.chunk_type,
                    "chunk_index": chunk.chunk_index,
                    "parent_id": chunk.parent_id or "",
                    "tags_text": " ".join(chunk.tags),
                    "aliases_text": " ".join(chunk.aliases),
                }
                for chunk in chunks
            ],
        )

    def delete_file(self, source_path: str) -> None:
        self.collection.delete(where={"source_path": source_path})

    def search(self, embedding: list[float], limit: int) -> list[SearchHit]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[embedding], n_results=min(limit, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        for chunk_id, document, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0],
            result["distances"][0], strict=True,
        ):
            hits.append(SearchHit(
                chunk_id=chunk_id, source_path=metadata["source_path"],
                title=metadata["title"], heading=metadata["heading"],
                content=document, chunk_type=metadata["chunk_type"],
                score=1.0 - float(distance), matched_by=["semantic"],
            ))
        return hits

