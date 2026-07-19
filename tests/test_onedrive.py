import stat
from pathlib import Path

from obsidian_mcp.config import Settings
from obsidian_mcp.embedding import HashEmbedder
from obsidian_mcp.indexer import Indexer
from obsidian_mcp.indexer import stat_module_dataless
from obsidian_mcp.state import StateStore

from test_incremental import FakeOCR, FakeVectors


def test_macos_dataless_flag_has_python_312_fallback():
    expected = getattr(stat, "SF_DATALESS", 0x40000000)
    assert stat_module_dataless() == expected
    assert stat_module_dataless() != 0


def test_dataless_cache_is_not_reused_after_image_download(tmp_path: Path):
    vault, data = tmp_path / "vault", tmp_path / "data"
    vault.mkdir()
    data.mkdir()
    image = vault / "figure.png"
    image.write_text("downloaded OCR text", encoding="utf-8")
    settings = Settings(vault_path=vault, data_path=data)
    state = StateStore(data / "state.sqlite")
    image_stat = image.stat()
    state.save_asset(
        "figure.png", image_stat.st_size, image_stat.st_mtime_ns,
        "__dataless__", "", "fake-ocr-v1",
    )
    indexer = Indexer(settings, state, FakeVectors(), HashEmbedder(), FakeOCR())
    text = indexer._ocr_text(image, "figure.png", dry_run=False)
    assert text == "downloaded OCR text"
    assert state.asset("figure.png")["content_hash"] != "__dataless__"
