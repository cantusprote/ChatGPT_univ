from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(prog="obsidian-mcp")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--transport", choices=["stdio", "http"], default="http")
    index = sub.add_parser("index")
    index.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    args = parser.parse_args()

    load_dotenv()
    from .app import Services, create_http_app, create_mcp
    from .config import Settings
    from .state import StateStore

    settings = Settings.from_env()
    settings.validate()
    if args.command == "status":
        state = StateStore(settings.data_path / "index-state.sqlite")
        print(json.dumps(state.stats(), ensure_ascii=False, indent=2))
        return
    services = Services(settings)
    if args.command == "index":
        print(json.dumps(services.indexer.scan(dry_run=args.dry_run).__dict__, ensure_ascii=False, indent=2))
    elif args.transport == "stdio":
        if settings.auto_index:
            services.indexer.progress = print_progress
            print_index_summary(services.indexer.scan())
        create_mcp(services).run(transport="stdio")
    else:
        if settings.auto_index:
            services.indexer.progress = print_progress
            print_index_summary(services.indexer.scan())
        import uvicorn
        uvicorn.run(create_http_app(services), host=settings.host, port=settings.port)


def print_index_summary(report) -> None:
    print(
        "Index update: "
        f"added={len(report.added)} modified={len(report.modified)} "
        f"deleted={len(report.deleted)} unchanged={report.unchanged} "
        f"failed={len(report.failed)} ocr_failed={len(report.ocr_failed)}",
        flush=True,
    )


def print_progress(message: str) -> None:
    print(f"Index progress: {message}", flush=True)


if __name__ == "__main__":
    main()
