"""Explorador interactivo del espacio latente de una corrida entrenada.

Uso, desde TP5:
    python -m src.explorer "results/ej2_vae_nsynth/base__seed=7"
    python -m src.explorer "results/ej1a_baseline/base__seed=42" --port 8010

Abre una página local con el mapa del espacio latente: al hacer clic o
arrastrar se decodifica ese punto y, si el dataset es de audio, se escucha.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from src.experiments.runner import load_run
from src.explorer.session import ExplorerSession, InvalidRequest

# Solo accesible desde esta máquina.
HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAXIMUM_REQUEST_BYTES = 4096
PAGE_PATH = Path(__file__).with_name("page.html")


def build_handler(session: ExplorerSession) -> type[BaseHTTPRequestHandler]:
    page = PAGE_PATH.read_bytes()

    class ExplorerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (nombre impuesto por http.server)
            if self.path == "/":
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", page)
            elif self.path == "/api/state":
                self._send_json(HTTPStatus.OK, session.describe())
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "No existe"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/decode":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "No existe"})
                return
            try:
                self._send_json(HTTPStatus.OK, session.decode(self._read_json()))
            except InvalidRequest as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def log_message(self, format: str, *arguments: object) -> None:
            """Silencia el registro por pedido: al arrastrar llegan decenas por segundo."""

        def _read_json(self) -> object:
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError as error:
                raise InvalidRequest("Falta Content-Length") from error
            if not 0 < length <= MAXIMUM_REQUEST_BYTES:
                raise InvalidRequest("Tamaño de pedido inválido")
            try:
                return json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise InvalidRequest("El cuerpo no es JSON válido") from error

        def _send_json(self, status: HTTPStatus, content: object) -> None:
            self._send(status, "application/json", json.dumps(content).encode("utf-8"))

        def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return ExplorerHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Explorador del espacio latente")
    parser.add_argument("run", type=Path, help="carpeta de una corrida entrenada")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="no abrir el navegador")
    arguments = parser.parse_args()

    model, dataset = load_run(arguments.run)
    # Un solo hilo: las capas del modelo guardan estado entre forward y backward.
    server = HTTPServer((HOST, arguments.port), build_handler(ExplorerSession(model, dataset)))
    address = f"http://{HOST}:{arguments.port}/"
    print(f"Explorador en {address} (Ctrl+C para salir)", flush=True)
    if not arguments.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
