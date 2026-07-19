import os
import json
import re
import threading
import time
from typing import Optional, Any

import chromadb
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# -------------------------------------------------
# Configuration
# -------------------------------------------------

CHROMA_DB = os.path.expanduser("~/.config/pdf-rag/chroma")
COLLECTION = "pdf_chunks"
BM25_FILE = os.path.expanduser("~/.config/pdf-rag/bm25_corpus.jsonl")
BM25_RELOAD_SETTLE_SECONDS = 5

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

PREVIEW_CHARS = 1200
DEFAULT_TOP_K = 5
DEFAULT_RETRIEVE_K = 30
DEFAULT_BM25_K = 30

# -------------------------------------------------
# FastAPI init
# -------------------------------------------------

app = FastAPI(
    title="PDF RAG API",
    version="1.2.0",
    description="Research-grade hybrid PDF RAG API (Claude MCP + ChatGPT)"
)

embed_model = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

# -------------------------------------------------
# Global in-memory BM25 index
# -------------------------------------------------

BM25_INDEX = None
BM25_DOCS = []
BM25_IDS = []
BM25_METAS = []
BM25_FILE_SIGNATURE = None
BM25_LOCK = threading.RLock()

# -------------------------------------------------
# Request Models
# -------------------------------------------------

class SearchRequest(BaseModel):
    query: str

    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)
    retrieve_k: int = Field(DEFAULT_RETRIEVE_K, ge=5, le=100)
    bm25_k: int = Field(DEFAULT_BM25_K, ge=5, le=100)

    section: Optional[str] = None
    content_type: Optional[str] = None
    year: Optional[str] = None
    title_contains: Optional[str] = None

    biomarker: Optional[str] = None
    drug: Optional[str] = None
    trial: Optional[str] = None
    endpoint: Optional[str] = None
    subtype: Optional[str] = None

    preview_chars: int = Field(PREVIEW_CHARS, ge=200, le=5000)


class EvidenceTopicRequest(BaseModel):
    topic: str

    top_k: int = Field(12, ge=3, le=50)
    retrieve_k: int = Field(40, ge=5, le=100)
    bm25_k: int = Field(40, ge=5, le=100)

    max_papers: int = Field(5, ge=1, le=20)
    max_chunks_per_paper: int = Field(2, ge=1, le=5)

    section: Optional[str] = None
    content_type: Optional[str] = None
    year: Optional[str] = None
    title_contains: Optional[str] = None

    biomarker: Optional[str] = None
    drug: Optional[str] = None
    trial: Optional[str] = None
    endpoint: Optional[str] = None
    subtype: Optional[str] = None

    preview_chars: int = Field(PREVIEW_CHARS, ge=200, le=5000)


class IdeaGapRequest(BaseModel):
    topic: str

    top_k: int = Field(15, ge=5, le=50)
    retrieve_k: int = Field(50, ge=5, le=100)
    bm25_k: int = Field(50, ge=5, le=100)

    max_papers: int = Field(8, ge=1, le=20)
    max_chunks_per_paper: int = Field(2, ge=1, le=5)

    section: Optional[str] = None
    content_type: Optional[str] = None
    year: Optional[str] = None
    title_contains: Optional[str] = None

    biomarker: Optional[str] = None
    drug: Optional[str] = None
    trial: Optional[str] = None
    endpoint: Optional[str] = None
    subtype: Optional[str] = None

    preview_chars: int = Field(PREVIEW_CHARS, ge=200, le=5000)


class ComparePapersRequest(BaseModel):
    topic: str

    top_k: int = Field(18, ge=5, le=60)
    retrieve_k: int = Field(60, ge=5, le=120)
    bm25_k: int = Field(60, ge=5, le=120)

    max_papers: int = Field(6, ge=2, le=20)
    max_chunks_per_paper: int = Field(3, ge=1, le=6)

    section: Optional[str] = None
    content_type: Optional[str] = None
    year: Optional[str] = None
    title_contains: Optional[str] = None

    biomarker: Optional[str] = None
    drug: Optional[str] = None
    trial: Optional[str] = None
    endpoint: Optional[str] = None
    subtype: Optional[str] = None

    preview_chars: int = Field(PREVIEW_CHARS, ge=200, le=5000)

# -------------------------------------------------
# DB
# -------------------------------------------------

def get_collection():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB)
        return client.get_collection(COLLECTION)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to open Chroma collection: {e}"
        )

# -------------------------------------------------
# Utils
# -------------------------------------------------

def make_preview(text: str, max_chars: int = PREVIEW_CHARS):
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def normalize_meta(meta: dict[str, Any]):
    return {
        "title": meta.get("title"),
        "journal": meta.get("journal"),
        "year": meta.get("year"),
        "file_name": meta.get("file_name"),
        "source_path": meta.get("source_path"),
        "page": meta.get("page"),
        "section": meta.get("section"),
        "content_type": meta.get("content_type"),
        "anchor_label": meta.get("anchor_label"),
        "method_score": meta.get("method_score"),
        "biomarkers": meta.get("biomarkers"),
        "drugs": meta.get("drugs"),
        "trials": meta.get("trials"),
        "endpoints": meta.get("endpoints"),
        "subtypes": meta.get("subtypes"),
    }


def is_placeholder(v: Optional[str]):
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in ["", "string", "null"]


def split_pipe_tags(v: Optional[str]) -> list[str]:
    if not v:
        return []
    return [x.strip().lower() for x in str(v).split("|") if x.strip()]


def meta_has_tag(meta: dict[str, Any], field: str, value: Optional[str]) -> bool:
    if is_placeholder(value):
        return True
    tags = split_pipe_tags(meta.get(field))
    return str(value).strip().lower() in tags


def uniq_pipe_join(values: list[str]) -> str:
    seen = []
    for v in values:
        if not v:
            continue
        for x in str(v).split("|"):
            x = x.strip()
            if x and x not in seen:
                seen.append(x)
    return "|".join(seen)


def has_requested_tag(pipe_value: Optional[str], requested: Optional[str]) -> bool:
    if is_placeholder(requested):
        return True
    vals = [x.strip().lower() for x in str(pipe_value or "").split("|") if x.strip()]
    return str(requested).strip().lower() in vals


def passes_post_filters(meta: dict[str, Any], req):
    if not is_placeholder(getattr(req, "title_contains", None)):
        title = (meta.get("title") or "").lower()
        if req.title_contains.lower() not in title:
            return False

    if not meta_has_tag(meta, "biomarkers", getattr(req, "biomarker", None)):
        return False
    if not meta_has_tag(meta, "drugs", getattr(req, "drug", None)):
        return False
    if not meta_has_tag(meta, "trials", getattr(req, "trial", None)):
        return False
    if not meta_has_tag(meta, "endpoints", getattr(req, "endpoint", None)):
        return False
    if not meta_has_tag(meta, "subtypes", getattr(req, "subtype", None)):
        return False

    return True


def paper_matches_filters(paper: dict[str, Any], req) -> bool:
    if not has_requested_tag(paper.get("biomarkers"), getattr(req, "biomarker", None)):
        return False
    if not has_requested_tag(paper.get("drugs"), getattr(req, "drug", None)):
        return False
    if not has_requested_tag(paper.get("trials"), getattr(req, "trial", None)):
        return False
    if not has_requested_tag(paper.get("endpoints"), getattr(req, "endpoint", None)):
        return False
    if not has_requested_tag(paper.get("subtypes"), getattr(req, "subtype", None)):
        return False

    if not is_placeholder(getattr(req, "year", None)):
        if str(paper.get("year") or "") != str(req.year):
            return False

    if not is_placeholder(getattr(req, "title_contains", None)):
        title = (paper.get("title") or "").lower()
        if req.title_contains.lower() not in title:
            return False

    return True


def tokenize_for_bm25(text: str) -> list[str]:
    if not text:
        return []
    text = text.lower()
    return re.findall(r"[a-z0-9\-\._/]+", text)

# -------------------------------------------------
# BM25 build / refresh
# -------------------------------------------------

def _bm25_file_signature():
    stat = os.stat(BM25_FILE)
    return (stat.st_mtime_ns, stat.st_size)

def load_external_bm25_index(force: bool = False):
    """Load the externally managed BM25 corpus without modifying it."""
    global BM25_INDEX, BM25_DOCS, BM25_IDS, BM25_METAS, BM25_FILE_SIGNATURE
    signature = _bm25_file_signature()
    if not force and signature == BM25_FILE_SIGNATURE:
        return False
    if not force and time.time() - os.path.getmtime(BM25_FILE) < BM25_RELOAD_SETTLE_SECONDS:
        return False
    rows = {}
    with open(BM25_FILE, "r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            line=line.strip()
            if not line:
                continue
            try:
                row=json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid BM25 JSONL at line {line_number}: {exc}") from exc
            chunk_id=row.get("chunk_id"); document=row.get("text") or ""
            if chunk_id and document:
                rows[chunk_id]=row
    ids=list(rows)
    docs=[rows[chunk_id]["text"] for chunk_id in ids]
    metas=[rows[chunk_id] for chunk_id in ids]
    new_index=BM25Okapi([tokenize_for_bm25(doc) for doc in docs])
    with BM25_LOCK:
        BM25_INDEX=new_index; BM25_DOCS=docs; BM25_IDS=ids; BM25_METAS=metas
        BM25_FILE_SIGNATURE=signature
    return True

def ensure_bm25_ready():
    if BM25_INDEX is None:
        load_external_bm25_index(force=True)
        return
    try:
        load_external_bm25_index(force=False)
    except FileNotFoundError:
        pass

# -------------------------------------------------
# Retrieval helpers
# -------------------------------------------------

def vector_search(req):
    col = get_collection()

    where = {}
    if not is_placeholder(getattr(req, "section", None)):
        where["section"] = req.section
    if not is_placeholder(getattr(req, "content_type", None)):
        where["content_type"] = req.content_type
    if not is_placeholder(getattr(req, "year", None)):
        where["year"] = req.year

    query_text = getattr(req, "query", None) or getattr(req, "topic", None)

    qvec = embed_model.encode(
        [query_text],
        normalize_embeddings=True
    ).tolist()

    kwargs = dict(
        query_embeddings=qvec,
        n_results=max(req.retrieve_k, req.top_k),
        include=["documents", "metadatas", "distances"]
    )

    if where:
        kwargs["where"] = where

    res = col.query(**kwargs)

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        meta = meta or {}
        if not passes_post_filters(meta, req):
            continue

        hits.append({
            "chunk_id": cid,
            "doc": doc,
            "meta": meta,
            "vector_distance": float(dist),
            "vector_rank": len(hits) + 1,
        })

    return hits


def bm25_search(req):
    ensure_bm25_ready()

    query_text = getattr(req, "query", None) or getattr(req, "topic", None)
    q_tokens = tokenize_for_bm25(query_text)

    if not q_tokens:
        return []

    with BM25_LOCK:
        scores = BM25_INDEX.get_scores(q_tokens)
        ids_snapshot = BM25_IDS
        docs_snapshot = BM25_DOCS
        metas_snapshot = BM25_METAS
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    candidate_idx = ranked_idx[:max(req.bm25_k, req.top_k) * 5]
    candidate_ids = [ids_snapshot[idx] for idx in candidate_idx]
    chroma_meta = {}
    if candidate_ids:
        res = get_collection().get(ids=candidate_ids, include=["metadatas"])
        chroma_meta = dict(zip(res.get("ids") or [], res.get("metadatas") or []))
    hits = []
    for idx in candidate_idx:
        chunk_id = ids_snapshot[idx]
        meta = dict(metas_snapshot[idx] or {})
        meta.update(chroma_meta.get(chunk_id) or {})

        if not is_placeholder(getattr(req, "section", None)):
            if (meta.get("section") or "").lower() != req.section.lower():
                continue

        if not is_placeholder(getattr(req, "content_type", None)):
            if (meta.get("content_type") or "").lower() != req.content_type.lower():
                continue

        if not is_placeholder(getattr(req, "year", None)):
            if str(meta.get("year") or "") != str(req.year):
                continue

        if not passes_post_filters(meta, req):
            continue

        hits.append({
            "chunk_id": chunk_id,
            "doc": docs_snapshot[idx],
            "meta": meta,
            "bm25_score": float(scores[idx]),
            "bm25_rank": len(hits) + 1,
        })
        if len(hits) >= max(req.bm25_k, req.top_k):
            break
    return hits


def merge_hits(vector_hits, bm25_hits):
    merged = {}

    for item in vector_hits:
        cid = item["chunk_id"]
        merged[cid] = {
            "chunk_id": cid,
            "doc": item["doc"],
            "meta": item["meta"],
            "vector_distance": item.get("vector_distance"),
            "vector_rank": item.get("vector_rank"),
            "bm25_score": None,
            "bm25_rank": None,
        }

    for item in bm25_hits:
        cid = item["chunk_id"]
        if cid in merged:
            merged[cid]["bm25_score"] = item.get("bm25_score")
            merged[cid]["bm25_rank"] = item.get("bm25_rank")
        else:
            merged[cid] = {
                "chunk_id": cid,
                "doc": item["doc"],
                "meta": item["meta"],
                "vector_distance": None,
                "vector_rank": None,
                "bm25_score": item.get("bm25_score"),
                "bm25_rank": item.get("bm25_rank"),
            }

    return list(merged.values())


def rerank_hits(query, merged_hits, top_k, preview_chars):
    if not merged_hits:
        return []

    pairs = [[query, h["doc"]] for h in merged_hits]
    scores = reranker.predict(pairs).tolist()

    ranked = sorted(
        zip(scores, merged_hits),
        key=lambda x: x[0],
        reverse=True
    )

    results = []
    for rerank_score, hit in ranked[:top_k]:
        meta = hit["meta"] or {}

        results.append({
            "chunk_id": hit["chunk_id"],
            "rerank_score": float(rerank_score),
            "vector_distance": hit.get("vector_distance"),
            "vector_rank": hit.get("vector_rank"),
            "bm25_score": hit.get("bm25_score"),
            "bm25_rank": hit.get("bm25_rank"),
            "text_preview": make_preview(hit["doc"], preview_chars),
            **normalize_meta(meta),
        })

    return results

# -------------------------------------------------
# Builders
# -------------------------------------------------

def build_search_request_from_topic(req: EvidenceTopicRequest) -> SearchRequest:
    return SearchRequest(
        query=req.topic,
        top_k=req.top_k,
        retrieve_k=req.retrieve_k,
        bm25_k=req.bm25_k,
        section=req.section,
        content_type=req.content_type,
        year=req.year,
        title_contains=req.title_contains,
        biomarker=req.biomarker,
        drug=req.drug,
        trial=req.trial,
        endpoint=req.endpoint,
        subtype=req.subtype,
        preview_chars=req.preview_chars,
    )


def build_search_request_from_idea(req: IdeaGapRequest) -> SearchRequest:
    return SearchRequest(
        query=req.topic,
        top_k=req.top_k,
        retrieve_k=req.retrieve_k,
        bm25_k=req.bm25_k,
        section=req.section,
        content_type=req.content_type,
        year=req.year,
        title_contains=req.title_contains,
        biomarker=req.biomarker,
        drug=req.drug,
        trial=req.trial,
        endpoint=req.endpoint,
        subtype=req.subtype,
        preview_chars=req.preview_chars,
    )


def build_search_request_from_compare(req: ComparePapersRequest) -> SearchRequest:
    return SearchRequest(
        query=req.topic,
        top_k=req.top_k,
        retrieve_k=req.retrieve_k,
        bm25_k=req.bm25_k,
        section=req.section,
        content_type=req.content_type,
        year=req.year,
        title_contains=req.title_contains,
        biomarker=req.biomarker,
        drug=req.drug,
        trial=req.trial,
        endpoint=req.endpoint,
        subtype=req.subtype,
        preview_chars=req.preview_chars,
    )

# -------------------------------------------------
# Grouping / summaries
# -------------------------------------------------

def group_evidence_by_paper(results, max_papers: int, max_chunks_per_paper: int, req=None, strict: bool = False):
    grouped = {}
    order = []

    for item in results:
        file_name = item.get("file_name") or "unknown_file"

        if file_name not in grouped:
            grouped[file_name] = {
                "file_name": file_name,
                "title": item.get("title"),
                "journal": item.get("journal"),
                "year": item.get("year"),
                "source_path": item.get("source_path"),
                "_biomarkers_list": [],
                "_drugs_list": [],
                "_trials_list": [],
                "_endpoints_list": [],
                "_subtypes_list": [],
                "items": []
            }
            order.append(file_name)

        grouped[file_name]["_biomarkers_list"].append(item.get("biomarkers"))
        grouped[file_name]["_drugs_list"].append(item.get("drugs"))
        grouped[file_name]["_trials_list"].append(item.get("trials"))
        grouped[file_name]["_endpoints_list"].append(item.get("endpoints"))
        grouped[file_name]["_subtypes_list"].append(item.get("subtypes"))

        if len(grouped[file_name]["items"]) < max_chunks_per_paper:
            grouped[file_name]["items"].append({
                "chunk_id": item.get("chunk_id"),
                "rerank_score": item.get("rerank_score"),
                "vector_distance": item.get("vector_distance"),
                "vector_rank": item.get("vector_rank"),
                "bm25_score": item.get("bm25_score"),
                "bm25_rank": item.get("bm25_rank"),
                "page": item.get("page"),
                "section": item.get("section"),
                "content_type": item.get("content_type"),
                "anchor_label": item.get("anchor_label"),
                "text_preview": item.get("text_preview"),
            })

    evidence = []
    for file_name in order:
        paper = grouped[file_name]

        paper["biomarkers"] = uniq_pipe_join(paper.pop("_biomarkers_list"))
        paper["drugs"] = uniq_pipe_join(paper.pop("_drugs_list"))
        paper["trials"] = uniq_pipe_join(paper.pop("_trials_list"))
        paper["endpoints"] = uniq_pipe_join(paper.pop("_endpoints_list"))
        paper["subtypes"] = uniq_pipe_join(paper.pop("_subtypes_list"))

        if strict and req is not None:
            if not paper_matches_filters(paper, req):
                continue

        evidence.append(paper)
        if len(evidence) >= max_papers:
            break

    return evidence


def build_gap_summary(topic: str, evidence: list[dict[str, Any]]):
    well_established = []
    possible_gaps = []
    conflicting_areas = []
    next_hypotheses = []

    papers = [e.get("title") for e in evidence if e.get("title")]
    journals = [e.get("journal") for e in evidence if e.get("journal")]
    years = [e.get("year") for e in evidence if e.get("year")]

    if len(evidence) >= 3:
        well_established.append(
            f"Multiple retrieved papers address the topic '{topic}', suggesting it is an established area of investigation."
        )

    if len(set([p for p in papers if p])) >= 3:
        well_established.append(
            "The topic appears across multiple distinct papers rather than a single repeated source."
        )

    if years:
        possible_gaps.append(
            "Check whether the most recent studies materially change prior interpretations, especially if the literature spans multiple years."
        )

    possible_gaps.append(
        "Variant-specific, subgroup-specific, or assay-specific differences may be underexplored even when the broad topic is well described."
    )

    possible_gaps.append(
        "Prospective validation and clinically actionable decision frameworks may be less mature than mechanistic or retrospective evidence."
    )

    conflicting_areas.append(
        "Some papers may frame the topic as mechanistic/biologic, while others emphasize clinical predictive or prognostic relevance; these uses are not always equivalent."
    )

    conflicting_areas.append(
        "Differences in cohort selection, assay methods, endpoint definitions, and treatment context may explain apparently inconsistent findings."
    )

    next_hypotheses.append(
        "A more granular, subtype- or variant-specific analysis may reveal clinically relevant heterogeneity hidden in pooled analyses."
    )

    next_hypotheses.append(
        "Longitudinal monitoring or dynamic biomarker assessment may improve patient stratification beyond baseline-only measurements."
    )

    next_hypotheses.append(
        "Cross-study comparison of treatment context, biomarker definition, and endpoint selection may identify where the strongest translational gap remains."
    )

    return {
        "well_established": well_established,
        "possible_gaps": possible_gaps,
        "conflicting_areas": conflicting_areas,
        "next_hypotheses": next_hypotheses,
        "paper_titles": papers[:10],
        "journals": list(dict.fromkeys(journals))[:10],
        "years": list(dict.fromkeys(years))[:10],
    }


def pick_best_lead_item(items: list[dict[str, Any]], req) -> dict[str, Any]:
    if not items:
        return {}

    def score_item(it: dict[str, Any]) -> tuple:
        score = 0.0

        text = (it.get("text_preview") or "").lower()
        section = (it.get("section") or "").lower()
        content_type = (it.get("content_type") or "").lower()

        for field_name in ["endpoint", "drug", "trial", "biomarker", "subtype"]:
            requested = getattr(req, field_name, None)
            if not is_placeholder(requested) and requested.lower() in text:
                score += 5

        if section == "results":
            score += 4
        elif section == "discussion":
            score += 2
        elif section == "methods":
            score -= 1

        if content_type == "body_text":
            score += 3
        elif content_type == "method_text":
            score += 2
        elif content_type in ["figure_caption", "table_caption", "supplement_caption"]:
            score -= 1

        score += float(it.get("rerank_score") or 0)
        score += 0.1 * float(it.get("bm25_score") or 0)

        vr = it.get("vector_rank")
        if vr is not None:
            score += max(0, 5 - int(vr))

        return (score,)

    ranked = sorted(items, key=score_item, reverse=True)
    return ranked[0]


def build_compare_rows(evidence: list[dict[str, Any]], req=None):
    rows = []

    for paper in evidence:
        items = paper.get("items", [])
        lead = pick_best_lead_item(items, req) if req is not None else (items[0] if items else {})

        rows.append({
            "title": paper.get("title"),
            "journal": paper.get("journal"),
            "year": paper.get("year"),
            "file_name": paper.get("file_name"),
            "biomarkers": paper.get("biomarkers"),
            "drugs": paper.get("drugs"),
            "trials": paper.get("trials"),
            "endpoints": paper.get("endpoints"),
            "subtypes": paper.get("subtypes"),
            "lead_page": lead.get("page"),
            "lead_section": lead.get("section"),
            "lead_content_type": lead.get("content_type"),
            "lead_anchor_label": lead.get("anchor_label"),
            "lead_rerank_score": lead.get("rerank_score"),
            "lead_bm25_score": lead.get("bm25_score"),
            "lead_text_preview": lead.get("text_preview"),
        })

    return rows

# -------------------------------------------------
# Health / refresh
# -------------------------------------------------

@app.get("/health")
def health():
    col = get_collection()
    return {
        "status": "ok",
        "documents": col.count(),
        "db": CHROMA_DB,
        "collection": COLLECTION,
        "bm25_file": BM25_FILE,
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "bm25_ready": BM25_INDEX is not None,
        "bm25_docs": len(BM25_DOCS),
    }


@app.post("/refresh/index")
def refresh_index():
    changed = load_external_bm25_index(force=True)
    return {
        "status": "ok",
        "message": "External BM25 corpus reloaded; source file was not modified",
        "reloaded": changed,
        "bm25_docs": len(BM25_DOCS),
    }

# -------------------------------------------------
# Search
# -------------------------------------------------

@app.post("/search")
def search(req: SearchRequest):
    try:
        v_hits = vector_search(req)
        b_hits = bm25_search(req)
        merged = merge_hits(v_hits, b_hits)
        results = rerank_hits(
            query=req.query,
            merged_hits=merged,
            top_k=req.top_k,
            preview_chars=req.preview_chars,
        )

        return {
            "query": req.query,
            "top_k": req.top_k,
            "retrieve_k": req.retrieve_k,
            "bm25_k": req.bm25_k,
            "filters": {
                "section": req.section,
                "content_type": req.content_type,
                "year": req.year,
                "title_contains": req.title_contains,
                "biomarker": req.biomarker,
                "drug": req.drug,
                "trial": req.trial,
                "endpoint": req.endpoint,
                "subtype": req.subtype,
            },
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/search failed: {e}")

# -------------------------------------------------
# Evidence topic
# -------------------------------------------------

@app.post("/evidence/topic")
def evidence_topic(req: EvidenceTopicRequest):
    try:
        search_req = build_search_request_from_topic(req)

        v_hits = vector_search(search_req)
        b_hits = bm25_search(search_req)
        merged = merge_hits(v_hits, b_hits)

        ranked_results = rerank_hits(
            query=search_req.query,
            merged_hits=merged,
            top_k=search_req.top_k,
            preview_chars=search_req.preview_chars,
        )

        evidence = group_evidence_by_paper(
            ranked_results,
            max_papers=req.max_papers,
            max_chunks_per_paper=req.max_chunks_per_paper,
            req=req,
            strict=False,
        )

        return {
            "topic": req.topic,
            "top_k": req.top_k,
            "max_papers": req.max_papers,
            "max_chunks_per_paper": req.max_chunks_per_paper,
            "filters": {
                "section": req.section,
                "content_type": req.content_type,
                "year": req.year,
                "title_contains": req.title_contains,
                "biomarker": req.biomarker,
                "drug": req.drug,
                "trial": req.trial,
                "endpoint": req.endpoint,
                "subtype": req.subtype,
            },
            "paper_count": len(evidence),
            "evidence": evidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/evidence/topic failed: {e}")

# -------------------------------------------------
# Idea gaps
# -------------------------------------------------

@app.post("/idea/gaps")
def idea_gaps(req: IdeaGapRequest):
    try:
        search_req = build_search_request_from_idea(req)

        v_hits = vector_search(search_req)
        b_hits = bm25_search(search_req)
        merged = merge_hits(v_hits, b_hits)

        ranked_results = rerank_hits(
            query=search_req.query,
            merged_hits=merged,
            top_k=search_req.top_k,
            preview_chars=search_req.preview_chars,
        )

        evidence = group_evidence_by_paper(
            ranked_results,
            max_papers=req.max_papers,
            max_chunks_per_paper=req.max_chunks_per_paper,
            req=req,
            strict=False,
        )

        summary = build_gap_summary(req.topic, evidence)

        return {
            "topic": req.topic,
            "filters": {
                "section": req.section,
                "content_type": req.content_type,
                "year": req.year,
                "title_contains": req.title_contains,
                "biomarker": req.biomarker,
                "drug": req.drug,
                "trial": req.trial,
                "endpoint": req.endpoint,
                "subtype": req.subtype,
            },
            "paper_count": len(evidence),
            "well_established": summary["well_established"],
            "possible_gaps": summary["possible_gaps"],
            "conflicting_areas": summary["conflicting_areas"],
            "next_hypotheses": summary["next_hypotheses"],
            "paper_titles": summary["paper_titles"],
            "journals": summary["journals"],
            "years": summary["years"],
            "evidence": evidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/idea/gaps failed: {e}")

# -------------------------------------------------
# Compare papers
# -------------------------------------------------

@app.post("/compare/papers")
def compare_papers(req: ComparePapersRequest):
    try:
        search_req = build_search_request_from_compare(req)

        v_hits = vector_search(search_req)
        b_hits = bm25_search(search_req)
        merged = merge_hits(v_hits, b_hits)

        ranked_results = rerank_hits(
            query=search_req.query,
            merged_hits=merged,
            top_k=search_req.top_k,
            preview_chars=search_req.preview_chars,
        )

        evidence = group_evidence_by_paper(
            ranked_results,
            max_papers=req.max_papers,
            max_chunks_per_paper=req.max_chunks_per_paper,
            req=req,
            strict=True,
        )

        rows = build_compare_rows(evidence, req=req)

        return {
            "topic": req.topic,
            "filters": {
                "section": req.section,
                "content_type": req.content_type,
                "year": req.year,
                "title_contains": req.title_contains,
                "biomarker": req.biomarker,
                "drug": req.drug,
                "trial": req.trial,
                "endpoint": req.endpoint,
                "subtype": req.subtype,
            },
            "paper_count": len(evidence),
            "rows": rows,
            "evidence": evidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/compare/papers failed: {e}")

# -------------------------------------------------
# Read chunk
# -------------------------------------------------

@app.get("/read/chunk")
def read_chunk(
    chunk_id: str = Query(...),
    with_preview: bool = Query(False)
):
    col = get_collection()

    res = col.get(
        ids=[chunk_id],
        include=["documents", "metadatas"]
    )

    ids = res.get("ids") or []
    if not ids:
        raise HTTPException(status_code=404, detail="Chunk not found")

    doc = (res.get("documents") or [""])[0]
    meta = (res.get("metadatas") or [{}])[0] or {}

    payload = {
        "chunk_id": chunk_id,
        "text": doc,
        **normalize_meta(meta)
    }

    if with_preview:
        payload["text_preview"] = make_preview(doc)

    return payload

# -------------------------------------------------
# Read by file
# -------------------------------------------------

@app.get("/read/by-file")
def read_by_file(
    file_name: str,
    limit: int = 20
):
    col = get_collection()

    res = col.get(
        where={"file_name": file_name},
        include=["documents", "metadatas"],
        limit=limit
    )

    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []

    results = []
    for cid, doc, meta in zip(ids, docs, metas):
        meta = meta or {}
        results.append({
            "chunk_id": cid,
            "text_preview": make_preview(doc),
            **normalize_meta(meta),
        })

    return {
        "file_name": file_name,
        "count": len(results),
        "results": results
    }

# -------------------------------------------------
# Read section
# -------------------------------------------------

@app.get("/read/section")
def read_section(
    file_name: str,
    section: str,
    limit: int = 50,
    include_full_text: bool = False
):
    col = get_collection()

    res = col.get(
        where={"file_name": file_name},
        include=["documents", "metadatas"],
        limit=limit
    )

    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []

    matched = []
    for cid, doc, meta in zip(ids, docs, metas):
        meta = meta or {}
        meta_section = (meta.get("section") or "").lower()

        if meta_section != section.lower():
            continue

        item = {
            "chunk_id": cid,
            **normalize_meta(meta)
        }

        if include_full_text:
            item["text"] = doc
        else:
            item["text_preview"] = make_preview(doc)

        matched.append(item)

    return {
        "file_name": file_name,
        "section": section,
        "count": len(matched),
        "results": matched
    }

# -------------------------------------------------
# Read page
# -------------------------------------------------

@app.get("/read/page")
def read_page(
    file_name: str,
    page: int,
    limit: int = 100,
    include_full_text: bool = False
):
    col = get_collection()

    res = col.get(
        where={"file_name": file_name},
        include=["documents", "metadatas"],
        limit=limit
    )

    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []

    matched = []
    for cid, doc, meta in zip(ids, docs, metas):
        meta = meta or {}
        meta_page = meta.get("page")

        if meta_page != page:
            continue

        item = {
            "chunk_id": cid,
            **normalize_meta(meta)
        }

        if include_full_text:
            item["text"] = doc
        else:
            item["text_preview"] = make_preview(doc)

        matched.append(item)

    return {
        "file_name": file_name,
        "page": page,
        "count": len(matched),
        "results": matched
    }

# -------------------------------------------------
# Read figure
# -------------------------------------------------

@app.get("/read/figure")
def read_figure(
    file_name: str,
    figure_label: Optional[str] = None,
    limit: int = 100,
    include_full_text: bool = True
):
    col = get_collection()

    res = col.get(
        where={"file_name": file_name},
        include=["documents", "metadatas"],
        limit=limit
    )

    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []

    matched = []
    for cid, doc, meta in zip(ids, docs, metas):
        meta = meta or {}

        if (meta.get("content_type") or "").lower() != "figure_caption":
            continue

        if figure_label:
            anchor = (meta.get("anchor_label") or "").lower()
            if figure_label.lower() not in anchor:
                continue

        item = {
            "chunk_id": cid,
            **normalize_meta(meta)
        }

        if include_full_text:
            item["text"] = doc
        else:
            item["text_preview"] = make_preview(doc)

        matched.append(item)

    return {
        "file_name": file_name,
        "figure_label": figure_label,
        "count": len(matched),
        "results": matched
    }

# -------------------------------------------------
# Read table
# -------------------------------------------------

@app.get("/read/table")
def read_table(
    file_name: str,
    table_label: Optional[str] = None,
    limit: int = 100,
    include_full_text: bool = True
):
    col = get_collection()

    res = col.get(
        where={"file_name": file_name},
        include=["documents", "metadatas"],
        limit=limit
    )

    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []

    matched = []
    for cid, doc, meta in zip(ids, docs, metas):
        meta = meta or {}

        if (meta.get("content_type") or "").lower() != "table_caption":
            continue

        if table_label:
            anchor = (meta.get("anchor_label") or "").lower()
            if table_label.lower() not in anchor:
                continue

        item = {
            "chunk_id": cid,
            **normalize_meta(meta)
        }

        if include_full_text:
            item["text"] = doc
        else:
            item["text_preview"] = make_preview(doc)

        matched.append(item)

    return {
        "file_name": file_name,
        "table_label": table_label,
        "count": len(matched),
        "results": matched
    }

# -------------------------------------------------
# hypothesis
# -------------------------------------------------


@app.post("/idea/hypothesis")
def idea_hypothesis(req: IdeaGapRequest):
    try:
        evidence_req = EvidenceTopicRequest(
            topic=req.topic,
            top_k=req.top_k,
            retrieve_k=req.retrieve_k,
            bm25_k=req.bm25_k,
            max_papers=req.max_papers,
            max_chunks_per_paper=req.max_chunks_per_paper,
            section=req.section,
            content_type=req.content_type,
            year=req.year,
            title_contains=req.title_contains,
            biomarker=req.biomarker,
            drug=req.drug,
            trial=req.trial,
            endpoint=req.endpoint,
            subtype=req.subtype,
            preview_chars=req.preview_chars,
        )

        evidence_result = evidence_topic(evidence_req)

        hypotheses = []

        for paper in evidence_result.get("evidence", []):
            title = paper.get("title")
            journal = paper.get("journal")
            year = paper.get("year")

            text_blocks = []
            for item in paper.get("items", []):
                text_blocks.append(item.get("text_preview", ""))

            combined = "\n".join(text_blocks)

            hypotheses.append({
                "paper": title,
                "journal": journal,
                "year": year,
                "hypothesis_seed": combined[:1200]
            })

        return {
            "topic": req.topic,
            "paper_count": len(hypotheses),
            "hypothesis_seeds": hypotheses
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"/idea/hypothesis failed: {e}"
        )


# -------------------------------------------------
# Startup
# -------------------------------------------------

@app.on_event("startup")
def startup_build_bm25():
    try:
        load_external_bm25_index(force=True)
        print(f"[startup] BM25 ready: {len(BM25_DOCS)} docs")
    except Exception as e:
        print(f"[startup] BM25 build failed: {e}")