import os
from concurrent.futures import ThreadPoolExecutor

from obsidian_mcp.config import Settings
from obsidian_mcp.embedding import HashEmbedder
from obsidian_mcp.indexer import Indexer
from obsidian_mcp.models import SearchHit
from obsidian_mcp.state import StateStore


class FakeVectors:
    def __init__(self):
        self.records = {}
        self.replaced = []

    def replace_file(self, source_path, chunks, embeddings):
        self.delete_file(source_path)
        self.replaced.append(source_path)
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.records[chunk.id] = (chunk, embedding)

    def delete_file(self, source_path):
        self.records = {
            key: value for key, value in self.records.items()
            if value[0].source_path != source_path
        }

    def search(self, embedding, limit):
        return [
            SearchHit(chunk.id, chunk.source_path, chunk.title, chunk.heading,
                      chunk.content, chunk.chunk_type, 1.0, ["semantic"])
            for chunk, _ in list(self.records.values())[:limit]
        ]


class FakeOCR:
    version = "fake-ocr-v1"

    def extract(self, image_path):
        return image_path.read_text(encoding="utf-8")


def build(tmp_path):
    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()
    settings = Settings(vault_path=vault, data_path=data)
    state = StateStore(data / "state.sqlite")
    vectors = FakeVectors()
    indexer = Indexer(settings, state, vectors, HashEmbedder(), FakeOCR())
    return vault, state, vectors, indexer


def test_add_modify_unchanged_delete_are_incremental(tmp_path):
    vault, state, vectors, indexer = build(tmp_path)
    note = vault / "paper.md"
    note.write_text("# Paper\n## Result\n- U-Net\n", encoding="utf-8")

    first = indexer.scan()
    assert first.added == ["paper.md"]
    assert state.stats() == {"files": 1, "chunks": 2, "failed": 0, "assets": 0}

    second = indexer.scan()
    assert second.unchanged == 1
    assert vectors.replaced == ["paper.md"]

    note.write_text("# Paper\n## Result\n- U-Net++\n", encoding="utf-8")
    modified = indexer.scan()
    assert modified.modified == ["paper.md"]
    assert "U-Net++" in [value[0].content for value in vectors.records.values()]
    assert "U-Net" not in [value[0].content for value in vectors.records.values()]

    note.unlink()
    deleted = indexer.scan()
    assert deleted.deleted == ["paper.md"]
    assert state.stats()["files"] == 0
    assert not vectors.records


def test_timestamp_only_change_does_not_reembed(tmp_path):
    vault, _, vectors, indexer = build(tmp_path)
    note = vault / "paper.md"
    note.write_text("# Paper\n## Result\n- AUROC 0.91\n", encoding="utf-8")
    indexer.scan()
    before = len(vectors.replaced)
    stat = note.stat()
    os.utime(note, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    report = indexer.scan()
    assert report.unchanged == 1
    assert len(vectors.replaced) == before


def test_state_store_can_be_used_from_mcp_worker_threads(tmp_path):
    vault, state, _, indexer = build(tmp_path)
    (vault / "paper.md").write_text("# Paper\n## Result\n- Dice 0.91\n", encoding="utf-8")
    indexer.scan()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: state.stats(), range(12)))

    assert all(result["files"] == 1 and result["chunks"] == 2 for result in results)


def test_linked_image_ocr_is_cached_and_image_change_forces_note_update(tmp_path):
    vault, state, vectors, indexer = build(tmp_path)
    image = vault / "figure.png"
    image.write_text("Hazard ratio 0.72", encoding="utf-8")
    note = vault / "paper.md"
    note.write_text("# Trial\n## Results\n![[figure.png]]\n", encoding="utf-8")

    first = indexer.scan()
    assert first.added == ["paper.md"]
    assert state.stats()["assets"] == 1
    assert "Hazard ratio 0.72" in [value[0].content for value in vectors.records.values()]

    unchanged = indexer.scan()
    assert unchanged.unchanged == 1

    image.write_text("Hazard ratio 0.61", encoding="utf-8")
    changed = indexer.scan()
    assert changed.modified == ["paper.md"]
    contents = [value[0].content for value in vectors.records.values()]
    assert "Hazard ratio 0.61" in contents
    assert "Hazard ratio 0.72" not in contents
