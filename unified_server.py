import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
load_dotenv(BASE_DIR / ".env")

import rag_mcp as rag_backend
from obsidian_mcp.app import Services, create_mcp
from obsidian_mcp.cli import print_index_summary, print_progress
from obsidian_mcp.config import Settings
from zotero_mcp import mcp as zotero_mcp

MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "9000"))
MCP_PATH = os.environ.get("MCP_PATH", "/mcp")

hub = FastMCP(
    name="Universal Research Library",
    instructions=(
        "Use rag_* tools for the indexed PDF corpus. "
        "Use zotero_* tools for Zotero metadata and local attachments. "
        "Use obsidian_* tools for Obsidian note search and reading."
    ),
)
hub.mount(rag_backend.mcp, namespace="rag")
hub.mount(zotero_mcp)

obsidian_settings = Settings.from_env()
obsidian_services = Services(obsidian_settings)
hub.mount(create_mcp(obsidian_services), namespace="obsidian")

def main():
    if obsidian_settings.auto_index:
        obsidian_services.indexer.progress = print_progress
        print_index_summary(obsidian_services.indexer.scan())
    rag_backend.start_rag_api()
    try:
        hub.run(
            transport="http",
            host=MCP_HOST,
            port=MCP_PORT,
            path=MCP_PATH,
            allowed_hosts=["*.ngrok-free.dev", "*.ngrok-free.app", "*.ngrok.app"],
            allowed_origins=["https://chatgpt.com", "https://chat.openai.com", "https://*.openai.com"],
        )
    except KeyboardInterrupt:
        pass
    finally:
        process = rag_backend.rag_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except TimeoutError:
                process.kill()

if __name__ == "__main__":
    main()
