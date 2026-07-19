from __future__ import annotations

from fastmcp import FastMCP
from .config import Settings
from .embedding import SentenceTransformerEmbedder
from .indexer import Indexer
from .search import HybridSearch
from .state import StateStore
from .vector import ChromaStore


class Services:
    def __init__(self, settings: Settings):
        settings.validate()
        self.settings = settings
        self.state = StateStore(settings.data_path / "index-state.sqlite")
        self.embedder = SentenceTransformerEmbedder(
            settings.embedding_model, str(settings.data_path / "model-cache")
        )
        self.vectors = ChromaStore(settings.data_path / "chroma", settings.collection_name)
        self.indexer = Indexer(settings, self.state, self.vectors, self.embedder)
        self.search = HybridSearch(self.state, self.vectors, self.embedder)


def create_mcp(services: Services) -> FastMCP:
    mcp = FastMCP("Obsidian Hybrid Search")

    @mcp.tool(annotations={"readOnlyHint": True})
    def search_notes(query: str, limit: int = 8) -> list[dict]:
        """Search Obsidian notes using semantic and exact keyword retrieval."""
        return services.search.search(query, max(1, min(limit, 25)))

    @mcp.tool(annotations={"readOnlyHint": True})
    def read_note(path: str) -> str:
        """Read a Markdown note by its vault-relative path."""
        requested = (services.settings.vault_path / path).resolve()
        vault = services.settings.vault_path.resolve()
        if requested.suffix.lower() != ".md" or not requested.is_relative_to(vault):
            raise ValueError("Only Markdown files inside the configured vault are allowed")
        return requested.read_text(encoding="utf-8")

    @mcp.tool(annotations={"readOnlyHint": True})
    def index_status() -> dict:
        """Return local index counts and configuration details."""
        return {**services.state.stats(), "vault": str(services.settings.vault_path),
                "model": services.embedder.model_name}

    @mcp.tool(annotations={"readOnlyHint": False})
    def refresh_index() -> dict:
        """Incrementally index only added, modified, and deleted notes."""
        report = services.indexer.scan()
        return report.__dict__

    return mcp


class ProxyHostASGI:
    def __init__(self, app, local_host: str = "127.0.0.1:8000"):
        self.app = app
        self.local_host = local_host.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope = dict(scope)
            normalized_headers = []
            for name, value in scope.get("headers", []):
                if name == b"host":
                    value = self.local_host
                elif name == b"origin":
                    value = b"http://" + self.local_host
                normalized_headers.append((name, value))
            scope["headers"] = normalized_headers
        await self.app(scope, receive, send)


def create_http_app(services: Services):
    services.settings.validate()
    mcp = create_mcp(services)
    return ProxyHostASGI(
        mcp.streamable_http_app(), f"{services.settings.host}:{services.settings.port}",
    )
