"""stdlib HTTP API for the control actor (no aiohttp dependency).

Routes:
    GET  /healthz            200 "ok"
    GET  /metrics            Prometheus text (see ``ControlFacade.metrics_text``)
    GET  /status             JSON (pool status + orchestrator events + raw metric snapshot)
    POST /scale              body {"standalone": N, "release": bool (optional)} -> JSON request result

The handler talks to a ``ControlFacade`` (duck-typed) so it can be tested without Ray:
    facade.metrics_text() -> str
    facade.status() -> dict
    facade.scale(standalone:int, release:bool|None) -> dict
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol


class ControlFacade(Protocol):
    def metrics_text(self) -> str: ...
    def status(self) -> dict[str, Any]: ...
    def scale(self, standalone: int, release: bool | None = None) -> dict[str, Any]: ...


def make_handler(facade: ControlFacade) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "rl-pool-control/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            return

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                if path == "/healthz":
                    self._send(200, b"ok\n", "text/plain")
                elif path == "/metrics":
                    self._send(200, facade.metrics_text().encode(), "text/plain; version=0.0.4")
                elif path == "/status":
                    self._json(200, facade.status())
                else:
                    self._json(404, {"error": "not found", "path": path})
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                n = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(n) if n else b"{}"
                body = json.loads(raw or b"{}")
            except Exception as e:  # noqa: BLE001
                self._json(400, {"error": f"bad json: {e}"})
                return
            try:
                if path == "/scale":
                    if "standalone" not in body:
                        self._json(400, {"error": "missing 'standalone'"})
                        return
                    result = facade.scale(int(body["standalone"]), body.get("release"))
                    self._json(200, result)
                else:
                    self._json(404, {"error": "not found", "path": path})
            except (ValueError, TypeError) as e:
                self._json(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


class ControlHTTPServer:
    """ThreadingHTTPServer on a daemon thread. ``port=0`` picks a free port (tests)."""

    def __init__(self, facade: ControlFacade, host: str = "0.0.0.0", port: int = 8787) -> None:
        self._server = ThreadingHTTPServer((host, port), make_handler(facade))
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="rl-pool-http", daemon=True)

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> "ControlHTTPServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
