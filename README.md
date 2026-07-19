# ChatGPT Universal Research Library

`ChatGPT_RAG_zotero`의 구조에 Obsidian 하이브리드 검색을 통합한 독립 프로그램입니다. 원본 프로젝트는 수정하거나 실행 시 참조하지 않습니다.

하나의 FastMCP 주소에서 다음 도구를 제공합니다.

- `rag_*`: 기존 PDF RAG 검색, 근거 수집, 논문 비교
- `zotero_*`: Zotero 메타데이터와 첨부파일 접근
- `obsidian_*`: Obsidian 노트 검색, 읽기, 증분 인덱스 갱신

## 데이터 위치

- PDF RAG: 기존과 같이 `~/.config/pdf-rag/chroma` 및 BM25 파일을 읽습니다.
- Zotero: `.env`의 `ZOTERO_DB`, `ZOTERO_STORAGE`를 사용합니다.
- Obsidian: `.env`의 `OBSIDIAN_VAULT_PATH`를 읽고, 이 프로젝트의 `.obsidian-mcp/chroma`에 별도 ChromaDB를 생성합니다.

## 실행

Finder에서 `chatGPT_univ.command`를 더블 클릭하거나 Terminal에서 실행합니다.

```bash
./chatGPT_univ.command
```

실행 파일 내부에서 `/Users/sim/Dev/ChatGPT_univ` 절대경로를 사용하므로, 파일을 다른 폴더에서 호출해도 동일한 통합 프로젝트가 실행됩니다. 통합 서버는 실행 창에서, ngrok 대시보드는 자동으로 열린 별도의 Terminal 창에서 작동합니다.

최초 실행 시 이 프로젝트만의 `.venv`를 만들고 의존성을 설치한 뒤 Obsidian 인덱스를 생성합니다. 이후에는 변경된 노트만 갱신합니다. 화면에 표시된 ngrok HTTPS 주소와 `/mcp`를 ChatGPT에 한 번 등록하면 됩니다.

서버 로그는 `.logs/unified-server.log`에 기록됩니다. ngrok은 별도의 Terminal 창에서 주소, 연결 상태, 요청 수를 보여주는 기본 대시보드로 실행됩니다.
