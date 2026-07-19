from __future__ import annotations

import hashlib
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from filelock import FileLock, Timeout

from .config import Settings
from .embedding import Embedder
from .ocr import AssetResolver, LocalOCR, image_chunk
from .parser import parse_note
from .state import StateStore
from .vector import ChromaStore

EXCLUDED_PARTS = {".obsidian", ".trash", ".git", ".obsidian-mcp", ".venv"}


@dataclass
class IndexReport:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    ocr_failed: dict[str, str] = field(default_factory=dict)


class Indexer:
    def __init__(self, settings: Settings, state: StateStore, vectors: ChromaStore,
                 embedder: Embedder, ocr: LocalOCR | None = None,
                 progress: Callable[[str], None] | None = None):
        self.settings, self.state, self.vectors, self.embedder = settings, state, vectors, embedder
        self.ocr = ocr or LocalOCR(settings.ocr_languages)
        self.progress = progress

    def _emit(self, message: str) -> None:
        if self.progress:
            self.progress(message)

    def scan(self, dry_run: bool = False) -> IndexReport:
        lock = FileLock(self.settings.data_path / "index.lock", timeout=0)
        try:
            with lock:
                return self._scan_locked(dry_run)
        except Timeout as exc:
            raise RuntimeError("another indexing process is already running") from exc

    def _scan_locked(self, dry_run: bool = False) -> IndexReport:
        report = IndexReport()
        self._emit("vault 파일 목록을 확인하는 중")
        known = self.state.files()
        known_assets = self.state.assets()
        changed_assets: set[str] = set()
        for asset_relative, old_asset in known_assets.items():
            asset_path = self.settings.vault_path / asset_relative
            if not asset_path.is_file():
                changed_assets.add(asset_relative)
                continue
            stat = asset_path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (
                old_asset["file_size"], old_asset["modified_ns"]
            ) or old_asset["ocr_version"] != self.ocr.version or (
                old_asset["content_hash"] == "__dataless__"
                and not (getattr(stat, "st_flags", 0) & stat_module_dataless())
            ):
                changed_assets.add(asset_relative)
        forced_notes = {
            note for asset in changed_assets for note in self.state.notes_for_asset(asset)
        }
        resolver = AssetResolver(self.settings.vault_path)
        current: dict[str, Path] = {}
        for path in self.settings.vault_path.rglob("*.md"):
            relative = path.relative_to(self.settings.vault_path)
            if any(part in EXCLUDED_PARTS or part.startswith(".") for part in relative.parts[:-1]):
                continue
            current[relative.as_posix()] = path

        # Version 1 already contains a valid text index. Upgrade only legacy notes that
        # actually contain image syntax; notes without images keep their existing vectors.
        for relative, path in current.items():
            old = known.get(relative)
            if old and old["index_version"] < self.settings.index_version:
                try:
                    if b"![" in path.read_bytes():
                        forced_notes.add(relative)
                except OSError:
                    pass

        total = len(current)
        self._emit(f"노트 {total}개 검사 시작")
        for position, (relative, path) in enumerate(sorted(current.items()), 1):
            try:
                stat_before = path.stat()
                old = known.get(relative)
                compatible = old and old["index_version"] <= self.settings.index_version and old[
                    "model_name"
                ] == self.embedder.model_name
                if relative not in forced_notes and compatible and old["file_size"] == stat_before.st_size and old[
                    "modified_ns"
                ] == stat_before.st_mtime_ns:
                    report.unchanged += 1
                    if position % 25 == 0 or position == total:
                        self._emit(f"노트 {position}/{total} 검사 중 (변경 없음)")
                    continue
                data = path.read_bytes()
                stat_after = path.stat()
                if (stat_before.st_size, stat_before.st_mtime_ns) != (
                    stat_after.st_size, stat_after.st_mtime_ns
                ):
                    raise RuntimeError("file changed while being read; retry on next scan")
                digest = hashlib.sha256(data).hexdigest()
                if relative not in forced_notes and compatible and digest == old["content_hash"]:
                    if not dry_run:
                        self.state.update_file_stat(
                            relative, stat_after.st_size, stat_after.st_mtime_ns
                        )
                    report.unchanged += 1
                    if position % 25 == 0 or position == total:
                        self._emit(f"노트 {position}/{total} 검사 중 (내용 동일)")
                    continue
                if dry_run:
                    (report.modified if old else report.added).append(relative)
                    self._emit(f"노트 {position}/{total} 변경 감지: {relative}")
                    continue
                parsed = parse_note(path, relative, data.decode("utf-8"))
                asset_paths: list[str] = []
                image_total = len(parsed.image_refs)
                for image_position, ref in enumerate(parsed.image_refs, 1):
                    asset_path = resolver.resolve(path, ref)
                    if not asset_path:
                        continue
                    asset_relative = asset_path.relative_to(self.settings.vault_path).as_posix()
                    asset_paths.append(asset_relative)
                    self._emit(
                        f"노트 {position}/{total}, 이미지 {image_position}/{image_total} OCR: "
                        f"{asset_relative}"
                    )
                    try:
                        ocr_text = (
                            ref.alt_text if len(ref.alt_text) >= 40
                            else self._ocr_text(asset_path, asset_relative, dry_run)
                        )
                        if ocr_text:
                            parsed.chunks.append(
                                image_chunk(relative, parsed.title, ref, asset_relative, ocr_text)
                            )
                    except Exception as exc:
                        report.ocr_failed[asset_relative] = str(exc)
                embeddings = self.embedder.documents([chunk.embedding_text for chunk in parsed.chunks])
                self.vectors.replace_file(relative, parsed.chunks, embeddings)
                self.state.replace_file(
                    relative, stat_after.st_size, stat_after.st_mtime_ns, digest, parsed.chunks,
                    self.settings.index_version, self.embedder.model_name,
                )
                self.state.replace_note_assets(relative, sorted(set(asset_paths)))
                (report.modified if old else report.added).append(relative)
                action = "수정" if old else "추가"
                self._emit(f"노트 {position}/{total} {action} 완료: {relative}")
            except Exception as exc:
                report.failed[relative] = str(exc)
                self._emit(f"노트 {position}/{total} 실패: {relative}")

        for relative in sorted(set(known) - set(current)):
            report.deleted.append(relative)
            if not dry_run:
                self.vectors.delete_file(relative)
                self.state.delete_file(relative)
        if not dry_run:
            for asset_relative in changed_assets:
                if not (self.settings.vault_path / asset_relative).is_file():
                    self.state.delete_asset(asset_relative)
        self._emit(
            f"완료: 추가 {len(report.added)}, 수정 {len(report.modified)}, "
            f"삭제 {len(report.deleted)}, 변경 없음 {report.unchanged}, "
            f"실패 {len(report.failed)}, OCR 실패 {len(report.ocr_failed)}"
        )
        return report

    def _ocr_text(self, asset_path: Path, asset_relative: str, dry_run: bool) -> str:
        stat_before = asset_path.stat()
        if getattr(stat_before, "st_flags", 0) & stat_module_dataless():
            if not dry_run:
                self.state.save_asset(
                    asset_relative, stat_before.st_size, stat_before.st_mtime_ns,
                    "__dataless__", "", self.ocr.version,
                )
            return ""
        if stat_before.st_size > self.settings.ocr_max_bytes:
            raise RuntimeError(
                f"image exceeds OCR size limit ({stat_before.st_size} bytes)"
            )
        old = self.state.asset(asset_relative)
        if old and old["content_hash"] != "__dataless__" and old[
            "ocr_version"
        ] == self.ocr.version and (
            old["file_size"], old["modified_ns"]
        ) == (stat_before.st_size, stat_before.st_mtime_ns):
            return old["ocr_text"]
        try:
            hash_result = subprocess.run(
                ["shasum", "-a", "256", str(asset_path)], capture_output=True,
                text=True, timeout=5, check=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("timed out reading OneDrive image") from exc
        stat_after = asset_path.stat()
        if (stat_before.st_size, stat_before.st_mtime_ns) != (
            stat_after.st_size, stat_after.st_mtime_ns
        ):
            raise RuntimeError("image changed while being read; retry on next scan")
        digest = hash_result.stdout.split()[0]
        if old and old["content_hash"] == digest and old["ocr_version"] == self.ocr.version:
            if not dry_run:
                self.state.save_asset(asset_relative, stat_after.st_size, stat_after.st_mtime_ns,
                                      digest, old["ocr_text"], self.ocr.version)
            return old["ocr_text"]
        text = self.ocr.extract(asset_path)
        if not dry_run:
            self.state.save_asset(asset_relative, stat_after.st_size, stat_after.st_mtime_ns,
                                  digest, text, self.ocr.version)
        return text


def stat_module_dataless() -> int:
    # Python 3.12 on macOS may omit SF_DATALESS even though st_flags exposes it.
    return getattr(stat, "SF_DATALESS", 0x40000000)
