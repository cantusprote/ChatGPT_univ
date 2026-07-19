#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/sim/Dev/ChatGPT_univ"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
REQUIREMENTS="${PROJECT_DIR}/requirements.txt"
INSTALL_STAMP="${PROJECT_DIR}/.install-stamp"
SERVER_LOG="${PROJECT_DIR}/.logs/unified-server.log"
NGROK_BIN="/opt/homebrew/bin/ngrok"
MCP_PORT="9000"

# Always resolve every relative setting (including OBSIDIAN_DATA_PATH) from the
# project directory, regardless of the folder used to launch this file.
cd "${PROJECT_DIR}"
/bin/mkdir -p "${PROJECT_DIR}/.logs"

if [[ ! -x "${NGROK_BIN}" ]]; then
  echo "오류: ngrok을 찾을 수 없습니다: ${NGROK_BIN}"
  read -r "?Enter 키를 누르면 창을 닫습니다. "
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "통합 프로그램 전용 Python 환경을 만듭니다."
  PYTHON_BOOTSTRAP=$(command -v python3 || true)
  if [[ -z "${PYTHON_BOOTSTRAP}" ]]; then
    echo "오류: Python 3를 찾을 수 없습니다."
    exit 1
  fi
  "${PYTHON_BOOTSTRAP}" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' || {
    echo "오류: Python 3.11 이상이 필요합니다."
    exit 1
  }
  "${PYTHON_BOOTSTRAP}" -m venv "${VENV_DIR}"
fi

CURRENT_HASH=$(/usr/bin/shasum -a 256 "${REQUIREMENTS}" | /usr/bin/awk '{print $1}')
INSTALLED_HASH=$(/bin/cat "${INSTALL_STAMP}" 2>/dev/null || true)
if [[ "${CURRENT_HASH}" != "${INSTALLED_HASH}" ]]; then
  echo "필요한 Python 패키지를 설치합니다. 최초 실행은 시간이 걸릴 수 있습니다."
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install -r "${REQUIREMENTS}"
  echo "${CURRENT_HASH}" > "${INSTALL_STAMP}"
fi

SERVER_PID=""
NGROK_PID=""
cleanup() {
  if [[ -n "${NGROK_PID}" ]] && kill -0 "${NGROK_PID}" 2>/dev/null; then
    kill "${NGROK_PID}" 2>/dev/null || true
    wait "${NGROK_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if /usr/sbin/lsof -nP -iTCP:${MCP_PORT} -sTCP:LISTEN >/dev/null 2>&1; then
  echo "오류: ${MCP_PORT}번 포트가 이미 사용 중입니다. 기존 서버를 종료한 뒤 다시 실행하세요."
  read -r "?Enter 키를 누르면 창을 닫습니다. "
  exit 1
fi

echo "PDF RAG, Zotero, Obsidian 통합 서버를 시작합니다."
echo "Obsidian 변경분을 확인하고 ChromaDB를 갱신하는 동안 기다려 주세요."
PYTHONPATH="${PROJECT_DIR}/src" "${PYTHON_BIN}" "${PROJECT_DIR}/unified_server.py" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

LAST_PROGRESS=""
for attempt in {1..7200}; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "오류: 통합 서버가 시작 중 종료되었습니다."
    /usr/bin/tail -n 60 "${SERVER_LOG}"
    exit 1
  fi
  HTTP_STATUS=$(/usr/bin/curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${MCP_PORT}/mcp" 2>/dev/null || true)
  if [[ "${HTTP_STATUS}" == "406" ]]; then
    break
  fi
  CURRENT_PROGRESS=$(/usr/bin/grep 'Index progress:' "${SERVER_LOG}" 2>/dev/null | /usr/bin/tail -n 1 || true)
  if [[ -n "${CURRENT_PROGRESS}" && "${CURRENT_PROGRESS}" != "${LAST_PROGRESS}" ]]; then
    echo "${CURRENT_PROGRESS}"
    LAST_PROGRESS="${CURRENT_PROGRESS}"
  fi
  /bin/sleep 1
done

if [[ "${HTTP_STATUS:-}" != "406" ]]; then
  echo "오류: 통합 서버가 제한 시간 안에 준비되지 않았습니다."
  /usr/bin/tail -n 60 "${SERVER_LOG}"
  exit 1
fi

INDEX_SUMMARY=$(/usr/bin/grep 'Index update:' "${SERVER_LOG}" 2>/dev/null | /usr/bin/tail -n 1 || true)
[[ -n "${INDEX_SUMMARY}" ]] && echo "${INDEX_SUMMARY}"

if /usr/sbin/lsof -nP -iTCP:4040 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "오류: 다른 ngrok 프로세스가 이미 실행 중입니다. 기존 ngrok을 종료한 뒤 다시 실행하세요."
  exit 1
fi

echo "새 Terminal 창에서 ngrok을 시작합니다."
/usr/bin/osascript \
  -e 'tell application "Terminal"' \
  -e "do script \"cd '${PROJECT_DIR}' && exec '${NGROK_BIN}' http '${MCP_PORT}'\"" \
  -e 'activate' \
  -e 'end tell' >/dev/null

for attempt in {1..30}; do
  NGROK_PID=$(/usr/sbin/lsof -nP -t -iTCP:4040 -sTCP:LISTEN 2>/dev/null | /usr/bin/head -n 1 || true)
  [[ -n "${NGROK_PID}" ]] && break
  /bin/sleep 1
done

if [[ -z "${NGROK_PID}" ]]; then
  echo "오류: 새 Terminal 창의 ngrok 프로세스를 확인할 수 없습니다."
  echo "새로 열린 ngrok Terminal 창의 오류 메시지를 확인하세요."
  exit 1
fi

PUBLIC_URL=""
for attempt in {1..30}; do
  if ! kill -0 "${NGROK_PID}" 2>/dev/null; then
    echo "오류: ngrok이 시작 중 종료되었습니다."
    echo "새로 열린 ngrok Terminal 창의 오류 메시지를 확인하세요."
    exit 1
  fi
  PUBLIC_URL=$(/usr/bin/curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | "${PYTHON_BIN}" -c 'import json,sys; d=json.load(sys.stdin); print(next((t["public_url"] for t in d.get("tunnels", []) if t.get("proto") == "https"), ""))' 2>/dev/null || true)
  [[ -n "${PUBLIC_URL}" ]] && break
  /bin/sleep 1
done

if [[ -z "${PUBLIC_URL}" ]]; then
  echo "오류: ngrok HTTPS 주소를 확인할 수 없습니다."
  echo "새로 열린 ngrok Terminal 창의 오류 메시지를 확인하세요."
  exit 1
fi

echo "============================================================"
echo "ChatGPT에 아래 주소 하나만 등록하세요:"
echo
echo "${PUBLIC_URL}/mcp"
echo
echo "Authentication은 No authentication으로 선택하세요."
echo "PDF RAG, Zotero, Obsidian 도구가 모두 이 주소에서 작동합니다."
echo "ngrok은 별도의 Terminal 창에서 실행 중입니다."
echo "두 서비스를 종료하려면 이 창에서 Control-C를 누르세요."
echo "============================================================"

while true; do
  if ! kill -0 "${NGROK_PID}" 2>/dev/null; then
    echo "ngrok이 종료되었습니다. 통합 서버도 종료합니다."
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "통합 서버가 종료되었습니다. ngrok도 종료합니다."
    break
  fi
  /bin/sleep 2
done
