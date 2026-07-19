from obsidian_mcp.embedding import HashEmbedder
from obsidian_mcp.search import HybridSearch

from test_incremental import build


def test_hybrid_search_preserves_neighbor_context(tmp_path):
    vault, state, vectors, indexer = build(tmp_path)
    (vault / "paper.md").write_text(
        "# Brain MRI\n## Results\n- U-Net 사용\n- Dice score 0.89\n- augmentation 개선\n",
        encoding="utf-8",
    )
    indexer.scan()
    search = HybridSearch(state, vectors, HashEmbedder())
    results = search.search("Dice 0.89", limit=3)
    assert results
    dice = next(result for result in results if "Dice" in result["excerpt"])
    assert "U-Net 사용" in dice["context"]
    assert "augmentation 개선" in dice["context"]
    assert "keyword" in dice["matched_by"]
