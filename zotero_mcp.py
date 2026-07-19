import os
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from fastmcp import FastMCP
except ImportError:  # Allows syntax checks before installing local deps.
    FastMCP = None

try:
    from pypdf import PdfReader
except ImportError:  # Allows metadata-only tools before installing local deps.
    PdfReader = None


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()

ZOTERO_DB = os.path.expanduser(os.environ.get("ZOTERO_DB", "~/Zotero/zotero.sqlite"))
ZOTERO_STORAGE = os.path.expanduser(os.environ.get("ZOTERO_STORAGE", "~/Zotero/storage"))
ZOTERO_LINKED_ATTACHMENT_BASE_DIR = os.path.expanduser(
    os.environ.get("ZOTERO_LINKED_ATTACHMENT_BASE_DIR", "")
)
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")
MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))
MCP_PATH = os.environ.get("MCP_PATH", "/mcp")
MCP_ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",")
    if host.strip()
]
MCP_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("MCP_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

mcp = FastMCP("zotero") if FastMCP else None


def _tool(fn):
    if mcp is None:
        return fn
    return mcp.tool()(fn)


@contextmanager
def connect():
    """Open a read-only snapshot of Zotero's DB to avoid app lock conflicts."""
    tmp_path = None
    conn = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="zotero_mcp_", suffix=".sqlite")
        os.close(fd)
        shutil.copy2(ZOTERO_DB, tmp_path)
        conn = sqlite3.connect(f"file:{tmp_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        if conn is not None:
            conn.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _rows(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
    return conn.execute(sql, tuple(params)).fetchall()


def _one(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
    return conn.execute(sql, tuple(params)).fetchone()


def _item_id_for_key(conn: sqlite3.Connection, item_key: str) -> Optional[int]:
    row = _one(
        conn,
        """
        SELECT itemID
        FROM items
        WHERE key = ?
          AND itemID NOT IN (SELECT itemID FROM deletedItems)
        """,
        (item_key,),
    )
    return int(row["itemID"]) if row else None


def _field_map(conn: sqlite3.Connection, item_id: int) -> Dict[str, str]:
    rows = _rows(
        conn,
        """
        SELECT fields.fieldName, itemDataValues.value
        FROM itemData
        JOIN fields ON fields.fieldID = itemData.fieldID
        JOIN itemDataValues ON itemDataValues.valueID = itemData.valueID
        WHERE itemData.itemID = ?
        """,
        (item_id,),
    )
    return {row["fieldName"]: row["value"] for row in rows}


def _creators(conn: sqlite3.Connection, item_id: int) -> List[Dict[str, str]]:
    rows = _rows(
        conn,
        """
        SELECT creatorTypes.creatorType,
               creators.firstName,
               creators.lastName,
               itemCreators.orderIndex
        FROM itemCreators
        JOIN creators ON creators.creatorID = itemCreators.creatorID
        JOIN creatorTypes ON creatorTypes.creatorTypeID = itemCreators.creatorTypeID
        WHERE itemCreators.itemID = ?
        ORDER BY itemCreators.orderIndex
        """,
        (item_id,),
    )
    creators = []
    for row in rows:
        first = row["firstName"] or ""
        last = row["lastName"] or ""
        name = " ".join(part for part in (first, last) if part).strip()
        creators.append({"type": row["creatorType"], "name": name})
    return creators


def _tags(conn: sqlite3.Connection, item_id: int) -> List[str]:
    rows = _rows(
        conn,
        """
        SELECT tags.name
        FROM itemTags
        JOIN tags ON tags.tagID = itemTags.tagID
        WHERE itemTags.itemID = ?
        ORDER BY tags.name
        """,
        (item_id,),
    )
    return [row["name"] for row in rows]


def _collections(conn: sqlite3.Connection, item_id: int) -> List[str]:
    rows = _rows(
        conn,
        """
        SELECT collections.collectionName
        FROM collectionItems
        JOIN collections ON collections.collectionID = collectionItems.collectionID
        WHERE collectionItems.itemID = ?
        ORDER BY collections.collectionName
        """,
        (item_id,),
    )
    return [row["collectionName"] for row in rows]


def _item_type(conn: sqlite3.Connection, item_id: int) -> Optional[str]:
    row = _one(
        conn,
        """
        SELECT itemTypes.typeName
        FROM items
        JOIN itemTypes ON itemTypes.itemTypeID = items.itemTypeID
        WHERE items.itemID = ?
        """,
        (item_id,),
    )
    return row["typeName"] if row else None


def _item_summary(conn: sqlite3.Connection, item_id: int, key: str) -> Dict[str, Any]:
    fields = _field_map(conn, item_id)
    creators = _creators(conn, item_id)
    authors = [creator["name"] for creator in creators if creator["type"] == "author"]
    return {
        "key": key,
        "item_type": _item_type(conn, item_id),
        "title": fields.get("title"),
        "year": fields.get("date"),
        "authors": authors,
        "publication": fields.get("publicationTitle") or fields.get("journalAbbreviation"),
        "doi": fields.get("DOI"),
        "url": fields.get("url"),
        "zotero_uri": f"zotero://select/library/items/{key}",
    }


def _resolve_attachment_path(db_path: str, attachment_key: str) -> Optional[str]:
    if not db_path:
        return None

    if db_path.startswith("storage:"):
        filename = db_path[len("storage:") :]
        return os.path.join(ZOTERO_STORAGE, attachment_key, filename)

    if db_path.startswith("attachments:"):
        rel_path = db_path[len("attachments:") :]
        if not ZOTERO_LINKED_ATTACHMENT_BASE_DIR:
            return None
        return os.path.normpath(os.path.join(ZOTERO_LINKED_ATTACHMENT_BASE_DIR, rel_path))

    return os.path.normpath(os.path.expanduser(db_path))


def _file_uri(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return Path(path).expanduser().resolve().as_uri()
    except ValueError:
        return None


def _parse_pages(pages: Optional[str], page_count: int) -> List[int]:
    if not pages:
        return list(range(page_count))

    selected = set()
    for chunk in pages.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
            if end < start:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))

    return [page - 1 for page in sorted(selected) if 1 <= page <= page_count]


def _resolve_first_existing_pdf(item_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    pdf_info = zotero_get_pdf_path(item_key)
    if "error" in pdf_info:
        return None, pdf_info["error"]

    for pdf in pdf_info.get("pdfs", []):
        if pdf.get("exists"):
            return pdf, None

    return None, f"No existing local PDF found for item key '{item_key}'"


@_tool
def zotero_healthcheck() -> Dict[str, Any]:
    """Check local Zotero paths and DB readability."""
    status: Dict[str, Any] = {
        "zotero_db": ZOTERO_DB,
        "zotero_db_exists": os.path.exists(ZOTERO_DB),
        "zotero_storage": ZOTERO_STORAGE,
        "zotero_storage_exists": os.path.isdir(ZOTERO_STORAGE),
        "linked_attachment_base_dir": ZOTERO_LINKED_ATTACHMENT_BASE_DIR or None,
        "linked_attachment_base_dir_exists": (
            os.path.isdir(ZOTERO_LINKED_ATTACHMENT_BASE_DIR)
            if ZOTERO_LINKED_ATTACHMENT_BASE_DIR
            else None
        ),
    }

    try:
        with connect() as conn:
            row = _one(conn, "SELECT COUNT(*) AS count FROM items")
            status["db_readable"] = True
            status["item_count"] = row["count"] if row else 0
    except Exception as exc:
        status["db_readable"] = False
        status["error"] = str(exc)

    return status


@_tool
def zotero_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search Zotero by metadata, creators, and tags.

    Searches title, abstract, date, DOI, publication title, creators, and tags.
    """
    if limit < 1:
        limit = 1
    if limit > 50:
        limit = 50

    like = f"%{query}%"
    with connect() as conn:
        rows = _rows(
            conn,
            """
            SELECT DISTINCT items.itemID, items.key
            FROM items
            JOIN itemTypes ON itemTypes.itemTypeID = items.itemTypeID
            WHERE itemTypes.typeName != 'attachment'
              AND items.itemID NOT IN (SELECT itemID FROM deletedItems)
              AND (
                EXISTS (
                  SELECT 1
                  FROM itemData
                  JOIN fields ON fields.fieldID = itemData.fieldID
                  JOIN itemDataValues ON itemDataValues.valueID = itemData.valueID
                  WHERE itemData.itemID = items.itemID
                    AND fields.fieldName IN (
                      'title', 'abstractNote', 'date', 'DOI',
                      'publicationTitle', 'journalAbbreviation', 'shortTitle'
                    )
                    AND itemDataValues.value LIKE ?
                )
                OR EXISTS (
                  SELECT 1
                  FROM itemCreators
                  JOIN creators ON creators.creatorID = itemCreators.creatorID
                  WHERE itemCreators.itemID = items.itemID
                    AND (creators.firstName || ' ' || creators.lastName) LIKE ?
                )
                OR EXISTS (
                  SELECT 1
                  FROM itemTags
                  JOIN tags ON tags.tagID = itemTags.tagID
                  WHERE itemTags.itemID = items.itemID
                    AND tags.name LIKE ?
                )
              )
            ORDER BY items.dateModified DESC
            LIMIT ?
            """,
            (like, like, like, limit),
        )
        return [_item_summary(conn, int(row["itemID"]), row["key"]) for row in rows]


@_tool
def zotero_get_item(item_key: str) -> Dict[str, Any]:
    """Return detailed metadata for a Zotero item key."""
    with connect() as conn:
        item_id = _item_id_for_key(conn, item_key)
        if item_id is None:
            return {"error": f"No Zotero item found for key '{item_key}'"}

        fields = _field_map(conn, item_id)
        summary = _item_summary(conn, item_id, item_key)
        summary.update(
            {
                "abstract": fields.get("abstractNote"),
                "creators": _creators(conn, item_id),
                "tags": _tags(conn, item_id),
                "collections": _collections(conn, item_id),
                "all_fields": fields,
            }
        )
        return summary


@_tool
def zotero_get_pdf_path(item_key: str) -> Dict[str, Any]:
    """
    Resolve local filesystem paths for PDF attachments.

    Accepts either a parent item key or an attachment item key.
    """
    with connect() as conn:
        rows = _rows(
            conn,
            """
            SELECT child.key, itemAttachments.path, itemTypes.typeName,
                   itemAttachments.contentType
            FROM itemAttachments
            JOIN items child ON child.itemID = itemAttachments.itemID
            LEFT JOIN itemTypes ON itemTypes.itemTypeID = child.itemTypeID
            WHERE itemAttachments.parentItemID = (
                SELECT itemID FROM items WHERE key = ?
            )
              AND itemAttachments.path IS NOT NULL
            """,
            (item_key,),
        )

        if not rows:
            rows = _rows(
                conn,
                """
                SELECT child.key, itemAttachments.path, itemTypes.typeName,
                       itemAttachments.contentType
                FROM itemAttachments
                JOIN items child ON child.itemID = itemAttachments.itemID
                LEFT JOIN itemTypes ON itemTypes.itemTypeID = child.itemTypeID
                WHERE child.key = ?
                  AND itemAttachments.path IS NOT NULL
                """,
                (item_key,),
            )

    attachments = []
    for row in rows:
        resolved = _resolve_attachment_path(row["path"], row["key"])
        is_pdf = (
            str(row["contentType"] or "").lower() == "application/pdf"
            or str(resolved or "").lower().endswith(".pdf")
        )
        attachments.append(
            {
                "attachment_key": row["key"],
                "raw_path": row["path"],
                "resolved_path": resolved,
                "file_uri": _file_uri(resolved),
                "exists": bool(resolved and os.path.exists(resolved)),
                "is_pdf": is_pdf,
                "content_type": row["contentType"],
                "type": row["typeName"],
            }
        )

    pdfs = [attachment for attachment in attachments if attachment["is_pdf"]]
    if not attachments:
        return {"error": f"No attachments found for item key '{item_key}'"}

    return {"item_key": item_key, "attachments": attachments, "pdfs": pdfs}


@_tool
def zotero_read_pdf_text(
    item_key: str,
    pages: Optional[str] = None,
    max_chars: int = 30000,
) -> Dict[str, Any]:
    """
    Extract text from a Zotero PDF attachment.

    `item_key` may be either a parent item key or an attachment key.
    `pages` is 1-based and accepts values like "1", "1-3", or "1-3,7".
    `max_chars` limits returned text so large PDFs do not overwhelm the client.
    """
    if PdfReader is None:
        return {"error": "pypdf is not installed. Run: pip install -r requirements.txt"}

    if max_chars < 1000:
        max_chars = 1000
    if max_chars > 100000:
        max_chars = 100000

    pdf, error = _resolve_first_existing_pdf(item_key)
    if error:
        return {"error": error}

    pdf_path = pdf["resolved_path"]
    try:
        reader = PdfReader(pdf_path)
        page_indexes = _parse_pages(pages, len(reader.pages))
    except Exception as exc:
        return {"error": f"Could not read PDF '{pdf_path}': {exc}"}

    chunks = []
    chars_used = 0
    truncated = False
    pages_read = []

    for page_index in page_indexes:
        if chars_used >= max_chars:
            truncated = True
            break

        try:
            page_text = reader.pages[page_index].extract_text() or ""
        except Exception as exc:
            page_text = f"[Text extraction failed on page {page_index + 1}: {exc}]"

        remaining = max_chars - chars_used
        if len(page_text) > remaining:
            page_text = page_text[:remaining]
            truncated = True

        chunks.append(f"\n\n--- Page {page_index + 1} ---\n{page_text.strip()}")
        chars_used += len(page_text)
        pages_read.append(page_index + 1)

    return {
        "item_key": item_key,
        "attachment_key": pdf["attachment_key"],
        "pdf_path": pdf_path,
        "file_uri": pdf["file_uri"],
        "page_count": len(reader.pages),
        "pages_requested": pages,
        "pages_read": pages_read,
        "truncated": truncated,
        "max_chars": max_chars,
        "text": "".join(chunks).strip(),
    }


@_tool
def zotero_open_pdf(item_key: str) -> Dict[str, Any]:
    """
    Open the first existing local PDF attachment with the system default app.

    Accepts either a parent item key or an attachment key.
    """
    pdf, error = _resolve_first_existing_pdf(item_key)
    if error:
        return {"error": error}

    pdf_path = pdf["resolved_path"]
    subprocess.run(["open", pdf_path], check=False)
    return {
        "opened": True,
        "item_key": item_key,
        "attachment_key": pdf["attachment_key"],
        "pdf_path": pdf_path,
        "file_uri": pdf["file_uri"],
    }


@_tool
def zotero_open(item_key: str) -> Dict[str, str]:
    """Open a Zotero item in the local Zotero app."""
    uri = f"zotero://select/library/items/{item_key}"
    subprocess.run(["open", uri], check=False)
    return {"opened": item_key, "uri": uri}


if __name__ == "__main__":
    if mcp is None:
        raise SystemExit("fastmcp is not installed. Run: . .venv/bin/activate && pip install -r requirements.txt")
    if MCP_TRANSPORT in {"http", "streamable-http", "sse"}:
        mcp.run(
            transport=MCP_TRANSPORT,
            host=MCP_HOST,
            port=MCP_PORT,
            path=MCP_PATH,
            allowed_hosts=MCP_ALLOWED_HOSTS or None,
            allowed_origins=MCP_ALLOWED_ORIGINS or None,
        )
    else:
        mcp.run(transport="stdio")
