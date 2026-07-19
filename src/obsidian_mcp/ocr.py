from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote

from .models import Chunk, ImageRef

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


class LocalOCR:
    def __init__(self, languages: str = "eng+kor"):
        executable = shutil.which("tesseract")
        if not executable:
            raise RuntimeError("tesseract is not installed")
        self.executable = executable
        available = subprocess.run(
            [executable, "--list-langs"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[1:]
        selected = [language for language in languages.split("+") if language in available]
        self.languages = "+".join(selected or (["eng"] if "eng" in available else available[:1]))
        if not self.languages:
            raise RuntimeError("tesseract has no usable language data")
        self.version = f"tesseract-v1:{self.languages}"

    def extract(self, image_path: Path) -> str:
        result = subprocess.run(
            [self.executable, str(image_path), "stdout", "-l", self.languages, "--psm", "6"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "OCR failed")
        return "\n".join(line.strip() for line in result.stdout.splitlines() if line.strip()).strip()


class AssetResolver:
    def __init__(self, vault: Path):
        self.vault = vault.resolve()
        self.by_name: dict[str, list[Path]] = {}
        for path in vault.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                self.by_name.setdefault(path.name, []).append(path)

    def resolve(self, note_path: Path, ref: ImageRef) -> Path | None:
        target = Path(unquote(ref.target.split("?", 1)[0]))
        for candidate in (note_path.parent / target, self.vault / target):
            resolved = candidate.resolve()
            if resolved.is_file() and resolved.is_relative_to(self.vault):
                return resolved
        matches = self.by_name.get(target.name, [])
        return matches[0].resolve() if len(matches) == 1 else None


def image_chunk(note_relative: str, title: str, ref: ImageRef, asset_relative: str, text: str) -> Chunk:
    chunk_id = hashlib.sha256(
        f"{note_relative}\x1f{asset_relative}\x1f{ref.line_number}\x1fimage".encode()
    ).hexdigest()
    context = f"문서: {title}\n섹션: {ref.heading_path}\n첨부 이미지: {asset_relative}"
    return Chunk(
        id=chunk_id, source_path=note_relative, title=title, heading=ref.heading,
        heading_path=ref.heading_path, content=text, embedding_text=f"{context}\n\n{text}",
        chunk_type="image", chunk_index=ref.line_number, links=[asset_relative],
    )
