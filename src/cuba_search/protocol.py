"""MCP JSON-RPC protocol layer — ported from cuba-memorys/protocol.py.

Handles stdin/stdout transport, request routing.
Signal handling for graceful shutdown.
"""
import asyncio
import json
import logging
import signal
import sys
from typing import Any

from cuba_search import __version__
from cuba_search.constants import TOOL_DEFINITIONS
from cuba_search.handlers import HANDLERS

logger = logging.getLogger("cuba-search.protocol")


async def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Route a JSON-RPC request to the appropriate handler.

    Args:
        request: Parsed JSON-RPC request dict.

    Returns:
        JSON-RPC response dict, or None for notifications.
    """
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return _rpc_result(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "cuba-search", "version": __version__},
        })

    if method == "notifications/initialized":
        logger.info("Client initialized — cuba-search v%s ready", __version__)
        return None

    if method == "tools/list":
        return _rpc_result(req_id, {"tools": TOOL_DEFINITIONS})

    if method == "tools/call":
        return await _handle_tool_call(req_id, params)

    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }

    return None


def _rpc_result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-RPC success response.

    Args:
        req_id: Request ID to echo back.
        result: Result payload.

    Returns:
        Complete JSON-RPC response.
    """
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


async def _handle_tool_call(
    req_id: Any,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a tool call to its handler.

    Args:
        req_id: Request ID.
        params: Tool call parameters (name, arguments).

    Returns:
        JSON-RPC response with tool result.
    """
    tool_name = params.get("name", "")
    tool_args = params.get("arguments", {})

    handler = HANDLERS.get(tool_name)
    if not handler:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{
                    "type": "text",
                    "text": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                }],
                "isError": True,
            },
        }

    try:
        result_text = await handler(tool_args)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": result_text}],
                "isError": False,
            },
        }
    except Exception:
        logger.exception("Tool '%s' raised an exception", tool_name)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "error": "internal_error",
                        "message": "Internal server error",
                    }),
                }],
                "isError": True,
            },
        }


async def main() -> None:
    """Run the MCP server over stdin/stdout.

    Ported from cuba-memorys/protocol.py with same event loop pattern.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[cuba-search] %(message)s",
        stream=sys.stderr,
    )

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_event.set)

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

    write_transport, _write_protocol = await loop.connect_write_pipe(
        lambda: asyncio.Protocol(), sys.stdout.buffer,
    )

    try:
        while not shutdown_event.is_set():
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if not line:
                break

            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue

            try:
                request = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            response = await handle_request(request)

            if response is not None:
                response_bytes = (
                    json.dumps(response, ensure_ascii=False, default=str).encode("utf-8")
                    + b"\n"
                )
                write_transport.write(response_bytes)

    except (ConnectionError, BrokenPipeError, EOFError):
        pass
    finally:
        logger.info("Shutting down gracefully...")
        write_transport.close()
