from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .models import Chunk, SearchHit


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            self.connection.executescript(
                """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS files (
                source_path TEXT PRIMARY KEY,
                file_size INTEGER NOT NULL,
                modified_ns INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                index_version INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'complete'
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                title TEXT NOT NULL,
                heading TEXT NOT NULL,
                heading_path TEXT NOT NULL,
                content TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                parent_id TEXT,
                tags_json TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                links_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks(source_path);
            CREATE INDEX IF NOT EXISTS chunks_parent_idx ON chunks(parent_id);
            CREATE TABLE IF NOT EXISTS assets (
                asset_path TEXT PRIMARY KEY,
                file_size INTEGER NOT NULL,
                modified_ns INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                ocr_text TEXT NOT NULL,
                ocr_version TEXT NOT NULL,
                indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS note_assets (
                source_path TEXT NOT NULL,
                asset_path TEXT NOT NULL,
                PRIMARY KEY(source_path, asset_path)
            );
            CREATE INDEX IF NOT EXISTS note_assets_asset_idx ON note_assets(asset_path);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED, source_path UNINDEXED, title, heading, content, tags, aliases,
                tokenize='unicode61'
            );
                """
            )

    def files(self) -> dict[str, sqlite3.Row]:
        with self._lock:
            return {
                row["source_path"]: row
                for row in self.connection.execute("SELECT * FROM files")
            }

    def replace_file(
        self,
        source_path: str,
        size: int,
        modified_ns: int,
        content_hash: str,
        chunks: list[Chunk],
        index_version: int,
        model_name: str,
    ) -> None:
        with self._lock, self.connection:
            self._delete_chunks(source_path)
            for chunk in chunks:
                self.connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.id, chunk.source_path, chunk.title, chunk.heading,
                        chunk.heading_path, chunk.content, chunk.chunk_type,
                        chunk.chunk_index, chunk.parent_id, json.dumps(chunk.tags),
                        json.dumps(chunk.aliases), json.dumps(chunk.links),
                    ),
                )
                self.connection.execute(
                    "INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chunk.id, chunk.source_path, chunk.title, chunk.heading, chunk.content,
                     " ".join(chunk.tags), " ".join(chunk.aliases)),
                )
            self.connection.execute(
                """
                INSERT OR REPLACE INTO files
                (source_path, file_size, modified_ns, content_hash, chunk_count,
                 indexed_at, index_version, model_name, status)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, 'complete')
                """,
                (source_path, size, modified_ns, content_hash, len(chunks), index_version, model_name),
            )

    def _delete_chunks(self, source_path: str) -> None:
        ids = [row[0] for row in self.connection.execute(
            "SELECT chunk_id FROM chunks WHERE source_path = ?", (source_path,)
        )]
        for chunk_id in ids:
            self.connection.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
        self.connection.execute("DELETE FROM chunks WHERE source_path = ?", (source_path,))

    def delete_file(self, source_path: str) -> None:
        with self._lock, self.connection:
            self._delete_chunks(source_path)
            self.connection.execute("DELETE FROM note_assets WHERE source_path = ?", (source_path,))
            self.connection.execute("DELETE FROM files WHERE source_path = ?", (source_path,))

    def update_file_stat(self, source_path: str, size: int, modified_ns: int) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "UPDATE files SET file_size = ?, modified_ns = ? WHERE source_path = ?",
                (size, modified_ns, source_path),
            )

    def lexical_search(self, query: str, limit: int) -> list[SearchHit]:
        terms = re_fts_terms(query)
        if not terms:
            return []
        with self._lock:
            rows = list(self.connection.execute(
                """
                SELECT c.*, bm25(chunks_fts, 0, 0, 1.5, 1.2, 1.0, 1.0, 1.0) AS rank
                FROM chunks_fts JOIN chunks c USING(chunk_id)
                WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?
                """,
                (" OR ".join(f'"{term}"' for term in terms), limit),
            ))
            return [
                SearchHit(row["chunk_id"], row["source_path"], row["title"], row["heading"],
                          row["content"], row["chunk_type"], -float(row["rank"]), ["keyword"])
                for row in rows
            ]

    def get_chunk(self, chunk_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self.connection.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()

    def context_for(self, chunk_id: str, neighbors: int = 1) -> list[sqlite3.Row]:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
            if not row:
                return []
            if row["chunk_type"] == "section":
                return [row]
            return list(self.connection.execute(
                """
                SELECT * FROM chunks WHERE source_path = ? AND heading_path = ?
                  AND chunk_type = 'bullet' AND chunk_index BETWEEN ? AND ?
                ORDER BY chunk_index
                """,
                (row["source_path"], row["heading_path"],
                 max(0, row["chunk_index"] - neighbors), row["chunk_index"] + neighbors),
            ))

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "files": self.connection.execute("SELECT count(*) FROM files").fetchone()[0],
                "chunks": self.connection.execute("SELECT count(*) FROM chunks").fetchone()[0],
                "failed": self.connection.execute(
                    "SELECT count(*) FROM files WHERE status != 'complete'"
                ).fetchone()[0],
                "assets": self.connection.execute("SELECT count(*) FROM assets").fetchone()[0],
            }

    def assets(self) -> dict[str, sqlite3.Row]:
        with self._lock:
            return {
                row["asset_path"]: row
                for row in self.connection.execute("SELECT * FROM assets")
            }

    def asset(self, asset_path: str) -> sqlite3.Row | None:
        with self._lock:
            return self.connection.execute(
                "SELECT * FROM assets WHERE asset_path = ?", (asset_path,)
            ).fetchone()

    def save_asset(self, asset_path: str, size: int, modified_ns: int, digest: str,
                   ocr_text: str, ocr_version: str) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO assets
                (asset_path, file_size, modified_ns, content_hash, ocr_text, ocr_version, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (asset_path, size, modified_ns, digest, ocr_text, ocr_version),
            )

    def replace_note_assets(self, source_path: str, asset_paths: list[str]) -> None:
        with self._lock, self.connection:
            self.connection.execute("DELETE FROM note_assets WHERE source_path = ?", (source_path,))
            self.connection.executemany(
                "INSERT OR IGNORE INTO note_assets VALUES (?, ?)",
                [(source_path, asset_path) for asset_path in asset_paths],
            )

    def notes_for_asset(self, asset_path: str) -> list[str]:
        with self._lock:
            return [row[0] for row in self.connection.execute(
                "SELECT source_path FROM note_assets WHERE asset_path = ?", (asset_path,)
            )]

    def delete_asset(self, asset_path: str) -> None:
        with self._lock, self.connection:
            self.connection.execute("DELETE FROM assets WHERE asset_path = ?", (asset_path,))
            self.connection.execute("DELETE FROM note_assets WHERE asset_path = ?", (asset_path,))


def re_fts_terms(query: str) -> list[str]:
    import re
    return [term for term in re.findall(r"[\w.+#-]+", query, re.UNICODE) if term]
