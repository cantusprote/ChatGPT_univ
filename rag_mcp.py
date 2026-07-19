import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import requests
from fastmcp import FastMCP

# -------------------------------------------------
# Config
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAG_HOST = "127.0.0.1"
RAG_PORT = int(os.environ.get("RAG_PORT", "8000"))
RAG_URL = f"http://{RAG_HOST}:{RAG_PORT}"

MCP_HOST = "0.0.0.0"
MCP_PORT = 9000
RAG_STARTUP_TIMEOUT_SECONDS = 180
RAG_HEALTHCHECK_INTERVAL_SECONDS = 1

rag_process = None

SERVER_INSTRUCTIONS = """
This MCP server provides search, reading, comparison, and research-gap analysis
over a private PDF RAG system.

Use this server when the user wants to:
- search indexed papers
- read a specific chunk or paper
- gather evidence for a topic
- compare papers or trials
- identify research gaps or hypotheses

Prefer:
- search for broad retrieval
- evidence_topic for writing support
- idea_gaps for literature gap exploration
- compare_papers for structured comparison
- read_chunk or read_paper for detailed inspection
"""

mcp = FastMCP(
    name="PDF RAG",
    instructions=SERVER_INSTRUCTIONS,
)

# -------------------------------------------------
# rag_api auto start
# -------------------------------------------------

def rag_healthcheck() -> bool:
    try:
        r = requests.get(f"{RAG_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def start_rag_api() -> None:
    global rag_process

    if rag_healthcheck():
        return

    if rag_process is None or rag_process.poll() is not None:
        rag_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "rag_api:app",
                "--host",
                RAG_HOST,
                "--port",
                str(RAG_PORT),
            ],
            cwd=BASE_DIR,
        )

    deadline = time.monotonic() + RAG_STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if rag_healthcheck():
            return

        if rag_process.poll() is not None:
            raise RuntimeError(
                f"rag_api exited before becoming ready "
                f"(exit code: {rag_process.returncode})"
            )

        time.sleep(RAG_HEALTHCHECK_INTERVAL_SECONDS)

    raise RuntimeError(
        f"rag_api did not become ready within "
        f"{RAG_STARTUP_TIMEOUT_SECONDS} seconds"
    )


def post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    start_rag_api()
    r = requests.post(f"{RAG_URL}{path}", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


def get_json(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    start_rag_api()
    r = requests.get(f"{RAG_URL}{path}", params=params, timeout=180)
    r.raise_for_status()
    return r.json()

# -------------------------------------------------
# Tools
# -------------------------------------------------

@mcp.tool()
async def search(
    query: str,
    top_k: int = 5,
    retrieve_k: int = 30,
    bm25_k: int = 30,
    section: Optional[str] = None,
    content_type: Optional[str] = None,
    year: Optional[str] = None,
    title_contains: Optional[str] = None,
    biomarker: Optional[str] = None,
    drug: Optional[str] = None,
    trial: Optional[str] = None,
    endpoint: Optional[str] = None,
    subtype: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search the indexed paper library.

    Use this when:
    - the user asks for relevant papers or chunks
    - the user wants papers about a biomarker, drug, trial, endpoint, or subtype
    - you need retrieval before reading, comparing, or synthesizing

    Returns ranked chunk-level results.
    """
    return post_json(
        "/search",
        {
            "query": query,
            "top_k": top_k,
            "retrieve_k": retrieve_k,
            "bm25_k": bm25_k,
            "section": section,
            "content_type": content_type,
            "year": year,
            "title_contains": title_contains,
            "biomarker": biomarker,
            "drug": drug,
            "trial": trial,
            "endpoint": endpoint,
            "subtype": subtype,
        },
    )


@mcp.tool()
async def evidence_topic(
    topic: str,
    top_k: int = 12,
    retrieve_k: int = 40,
    bm25_k: int = 40,
    max_papers: int = 5,
    max_chunks_per_paper: int = 2,
    section: Optional[str] = None,
    content_type: Optional[str] = None,
    year: Optional[str] = None,
    title_contains: Optional[str] = None,
    biomarker: Optional[str] = None,
    drug: Optional[str] = None,
    trial: Optional[str] = None,
    endpoint: Optional[str] = None,
    subtype: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a paper-level evidence pack for a topic.

    Use this when:
    - the user wants introduction/discussion/rebuttal support
    - the user asks for key papers or evidence for a topic
    - you want paper-grouped evidence rather than raw chunks
    """
    return post_json(
        "/evidence/topic",
        {
            "topic": topic,
            "top_k": top_k,
            "retrieve_k": retrieve_k,
            "bm25_k": bm25_k,
            "max_papers": max_papers,
            "max_chunks_per_paper": max_chunks_per_paper,
            "section": section,
            "content_type": content_type,
            "year": year,
            "title_contains": title_contains,
            "biomarker": biomarker,
            "drug": drug,
            "trial": trial,
            "endpoint": endpoint,
            "subtype": subtype,
        },
    )


@mcp.tool()
async def idea_gaps(
    topic: str,
    top_k: int = 15,
    retrieve_k: int = 50,
    bm25_k: int = 50,
    max_papers: int = 8,
    max_chunks_per_paper: int = 2,
    section: Optional[str] = None,
    content_type: Optional[str] = None,
    year: Optional[str] = None,
    title_contains: Optional[str] = None,
    biomarker: Optional[str] = None,
    drug: Optional[str] = None,
    trial: Optional[str] = None,
    endpoint: Optional[str] = None,
    subtype: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Identify possible research gaps and next hypotheses.

    Use this when:
    - the user asks about unanswered questions
    - the user wants literature gaps
    - the user wants hypothesis generation grounded in retrieved papers
    """
    return post_json(
        "/idea/gaps",
        {
            "topic": topic,
            "top_k": top_k,
            "retrieve_k": retrieve_k,
            "bm25_k": bm25_k,
            "max_papers": max_papers,
            "max_chunks_per_paper": max_chunks_per_paper,
            "section": section,
            "content_type": content_type,
            "year": year,
            "title_contains": title_contains,
            "biomarker": biomarker,
            "drug": drug,
            "trial": trial,
            "endpoint": endpoint,
            "subtype": subtype,
        },
    )


@mcp.tool()
async def compare_papers(
    topic: str,
    top_k: int = 18,
    retrieve_k: int = 60,
    bm25_k: int = 60,
    max_papers: int = 6,
    max_chunks_per_paper: int = 3,
    section: Optional[str] = None,
    content_type: Optional[str] = None,
    year: Optional[str] = None,
    title_contains: Optional[str] = None,
    biomarker: Optional[str] = None,
    drug: Optional[str] = None,
    trial: Optional[str] = None,
    endpoint: Optional[str] = None,
    subtype: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare papers in a structured, paper-level format.

    Use this when:
    - the user asks to compare trials, drugs, biomarkers, or papers
    - the user wants a comparison table
    - the user wants the strongest representative evidence per paper
    """
    return post_json(
        "/compare/papers",
        {
            "topic": topic,
            "top_k": top_k,
            "retrieve_k": retrieve_k,
            "bm25_k": bm25_k,
            "max_papers": max_papers,
            "max_chunks_per_paper": max_chunks_per_paper,
            "section": section,
            "content_type": content_type,
            "year": year,
            "title_contains": title_contains,
            "biomarker": biomarker,
            "drug": drug,
            "trial": trial,
            "endpoint": endpoint,
            "subtype": subtype,
        },
    )


@mcp.tool()
async def read_chunk(
    chunk_id: str,
    with_preview: bool = False,
) -> Dict[str, Any]:
    """
    Read one specific chunk by chunk_id.

    Use this when:
    - you already have a chunk_id from search/evidence/compare
    - you need the full text of that chunk
    """
    return get_json(
        "/read/chunk",
        {
            "chunk_id": chunk_id,
            "with_preview": with_preview,
        },
    )


@mcp.tool()
async def read_paper(
    file_name: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Read chunks from a paper by file name.

    Use this when:
    - the user wants to inspect one paper
    - you want chunk previews grouped by paper
    """
    return get_json(
        "/read/by-file",
        {
            "file_name": file_name,
            "limit": limit,
        },
    )




@mcp.tool()
async def generate_hypotheses(
    topic: str,
    top_k: int = 15,
    retrieve_k: int = 50,
    bm25_k: int = 50,
    max_papers: int = 8
):
    """
    Generate research hypothesis seeds from literature evidence.
    """

    return post_json(
        "/idea/hypothesis",
        {
            "topic": topic,
            "top_k": top_k,
            "retrieve_k": retrieve_k,
            "bm25_k": bm25_k,
            "max_papers": max_papers,
        },
    )



# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    start_rag_api()
    mcp.run(
        transport="sse",
        host=MCP_HOST,
        port=MCP_PORT,
    )


if __name__ == "__main__":
    main()
