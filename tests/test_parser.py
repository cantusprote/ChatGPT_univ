from pathlib import Path

from obsidian_mcp.parser import parse_note


def test_short_research_bullets_get_section_and_atomic_chunks():
    note = """---
tags: [medical, segmentation]
aliases: [뇌종양 분할]
---
# Brain Tumor

## Results
- U-Net 사용
- Dice score 0.89
- [[BraTS]] 데이터셋
"""
    parsed = parse_note(Path("paper.md"), "papers/paper.md", note)
    section = [chunk for chunk in parsed.chunks if chunk.chunk_type == "section"]
    bullets = [chunk for chunk in parsed.chunks if chunk.chunk_type == "bullet"]
    assert parsed.title == "Brain Tumor"
    assert len(section) == 1
    assert len(bullets) == 3
    assert bullets[1].content == "Dice score 0.89"
    assert "문서: Brain Tumor" in bullets[1].embedding_text
    assert bullets[1].parent_id == section[0].id
    assert bullets[2].links == ["BraTS"]


def test_local_image_references_keep_heading_context():
    note = "# Paper\n## Figure\n![[plot.png]]\n![table](assets/table.jpg)\n"
    parsed = parse_note(Path("paper.md"), "paper.md", note)
    assert [ref.target for ref in parsed.image_refs] == ["plot.png", "assets/table.jpg"]
    assert all(ref.heading == "Figure" for ref in parsed.image_refs)
