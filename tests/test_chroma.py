from pathlib import Path

from obsidian_mcp.embedding import HashEmbedder
from obsidian_mcp.models import Chunk
from obsidian_mcp.vector import ChromaStore


def test_chroma_replace_and_delete_file(tmp_path: Path):
    store = ChromaStore(tmp_path / "chroma", "test_notes")
    embedder = HashEmbedder()
    chunk = Chunk(
        id="one", source_path="paper.md", title="Paper", heading="Results",
        heading_path="Paper > Results", content="U-Net Dice 0.89",
        embedding_text="Paper Results U-Net Dice 0.89", chunk_type="bullet", chunk_index=0,
    )
    store.replace_file("paper.md", [chunk], embedder.documents([chunk.embedding_text]))
    assert store.collection.count() == 1
    hits = store.search(embedder.query("U-Net"), 3)
    assert hits[0].source_path == "paper.md"
    store.delete_file("paper.md")
    assert store.collection.count() == 0

