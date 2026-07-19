from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import Chunk, ImageRef

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+)$")
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
WIKI_IMAGE_RE = re.compile(r"!\[\[([^\]|]+\.(?:png|jpe?g|webp|tiff?))(?:\|[^\]]+)?\]\]", re.I)
MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+\.(?:png|jpe?g|webp|tiff?))\)", re.I)


def _list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _id(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class ParsedNote:
    chunks: list[Chunk]
    title: str
    image_refs: list[ImageRef]


def parse_note(path: Path, relative_path: str, text: str) -> ParsedNote:
    metadata: dict = {}
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end >= 0:
            try:
                metadata = yaml.safe_load(text[4:end]) or {}
            except yaml.YAMLError:
                metadata = {}
            body = text[end + 4 :].lstrip("\r\n")

    tags = _list(metadata.get("tags"))
    aliases = _list(metadata.get("aliases") or metadata.get("alias"))
    title = str(metadata.get("title") or "").strip() or path.stem
    lines = body.splitlines()
    for line in lines:
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            title = match.group(2).strip()
            break

    sections: list[tuple[str, str, list[str]]] = []
    image_refs: list[ImageRef] = []
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_heading = ""
    current_path = title

    def flush() -> None:
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_heading, current_path, current_lines[:]))
        current_lines = []

    for line_number, line in enumerate(lines, 1):
        match = HEADING_RE.match(line)
        if not match:
            for image_match in WIKI_IMAGE_RE.finditer(line):
                image_refs.append(ImageRef(image_match.group(1), current_heading, current_path, line_number))
            for image_match in MD_IMAGE_RE.finditer(line):
                target = image_match.group(2)
                if not target.lower().startswith(("http://", "https://")):
                    image_refs.append(
                        ImageRef(target, current_heading, current_path, line_number,
                                 image_match.group(1).strip())
                    )
            current_lines.append(line)
            continue
        flush()
        level, heading = len(match.group(1)), match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        current_heading = heading
        current_path = " > ".join(item[1] for item in stack)
    flush()

    chunks: list[Chunk] = []
    for section_index, (heading, heading_path, section_lines) in enumerate(sections):
        content = "\n".join(section_lines).strip()
        if not content:
            continue
        parent_id = _id(relative_path, heading_path, "section", section_index)
        context = f"문서: {title}\n섹션: {heading_path}\n태그: {', '.join(tags)}".strip()
        links = sorted(set(WIKILINK_RE.findall(content)))
        chunks.append(
            Chunk(
                id=parent_id,
                source_path=relative_path,
                title=title,
                heading=heading,
                heading_path=heading_path,
                content=content,
                embedding_text=f"{context}\n\n{content}",
                chunk_type="section",
                chunk_index=section_index,
                tags=tags,
                aliases=aliases,
                links=links,
            )
        )
        bullet_index = 0
        for line in section_lines:
            bullet = BULLET_RE.match(line)
            if not bullet or not bullet.group(1).strip():
                continue
            bullet_text = bullet.group(1).strip()
            chunks.append(
                Chunk(
                    id=_id(relative_path, heading_path, "bullet", bullet_index, bullet_text),
                    source_path=relative_path,
                    title=title,
                    heading=heading,
                    heading_path=heading_path,
                    content=bullet_text,
                    embedding_text=f"{context}\n\n{bullet_text}",
                    chunk_type="bullet",
                    chunk_index=bullet_index,
                    parent_id=parent_id,
                    tags=tags,
                    aliases=aliases,
                    links=sorted(set(WIKILINK_RE.findall(bullet_text))),
                )
            )
            bullet_index += 1
    return ParsedNote(chunks=chunks, title=title, image_refs=image_refs)
