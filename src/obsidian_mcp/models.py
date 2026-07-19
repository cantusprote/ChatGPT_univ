from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    source_path: str
    title: str
    heading: str
    heading_path: str
    content: str
    embedding_text: str
    chunk_type: str
    chunk_index: int
    parent_id: str | None = None
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)


@dataclass
class ImageRef:
    target: str
    heading: str
    heading_path: str
    line_number: int
    alt_text: str = ""


@dataclass
class SearchHit:
    chunk_id: str
    source_path: str
    title: str
    heading: str
    content: str
    chunk_type: str
    score: float
    matched_by: list[str]
