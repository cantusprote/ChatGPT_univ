import asyncio

from obsidian_mcp.app import ProxyHostASGI


async def downstream(scope, receive, send):
    await send({"type": "http.response.start", "status": 204, "headers": []})
    host = dict(scope.get("headers", [])).get(b"host", b"")
    await send({"type": "http.response.body", "body": scope["path"].encode() + b"|" + host})


def call(path, headers=None):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http", "path": path, "raw_path": path.encode(),
        "headers": (headers or []) + [(b"host", b"example.ngrok-free.dev")],
    }
    asyncio.run(ProxyHostASGI(downstream)(scope, receive, send))
    return messages


def test_proxy_rewrites_external_host_without_authentication():
    result = call("/mcp")
    assert result[0]["status"] == 204
    assert result[1]["body"] == b"/mcp|127.0.0.1:8000"
