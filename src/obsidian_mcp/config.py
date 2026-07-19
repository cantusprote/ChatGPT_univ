from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    vault_path: Path
    data_path: Path
    embedding_model: str = "intfloat/multilingual-e5-small"
    collection_name: str = "obsidian_notes_v1"
    auto_index: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    index_version: int = 2
    ocr_languages: str = "eng+kor"
    ocr_max_bytes: int = 25 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        vault = Path(os.getenv("OBSIDIAN_VAULT_PATH", ".")).expanduser().resolve()
        data = Path(os.getenv("OBSIDIAN_DATA_PATH", ".obsidian-mcp")).expanduser().resolve()
        return cls(
            vault_path=vault,
            data_path=data,
            embedding_model=os.getenv(
                "OBSIDIAN_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
            ),
            collection_name=os.getenv("OBSIDIAN_COLLECTION", "obsidian_notes_v1"),
            auto_index=_bool("OBSIDIAN_AUTO_INDEX", True),
            host=os.getenv("OBSIDIAN_HOST", "127.0.0.1"),
            port=int(os.getenv("OBSIDIAN_PORT", "8000")),
            ocr_languages=os.getenv("OBSIDIAN_OCR_LANGUAGES", "eng+kor"),
            ocr_max_bytes=int(os.getenv("OBSIDIAN_OCR_MAX_BYTES", str(25 * 1024 * 1024))),
        )

    def validate(self) -> None:
        if not self.vault_path.is_dir():
            raise ValueError(f"Vault directory does not exist: {self.vault_path}")
        if self.data_path == self.vault_path or self.vault_path in self.data_path.parents:
            raise ValueError("Data path must not be inside the Obsidian vault")
        self.data_path.mkdir(parents=True, exist_ok=True)
