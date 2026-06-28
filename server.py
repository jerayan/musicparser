#!/usr/bin/env python3
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
ALLOWED_WRITE_FILES = {
    "app.js",
    "style.css",
    "index.html",
}


class LiveHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path != "/api/files":
            self.send_json({"error": "Not found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            rel_path = unquote(str(payload.get("path", ""))).strip().replace("\\", "/")
            content = str(payload.get("content", ""))

            if rel_path not in ALLOWED_WRITE_FILES:
                self.send_json({"error": f"Writing {rel_path!r} is not allowed"}, status=400)
                return

            target = (ROOT / rel_path).resolve()
            if ROOT not in target.parents and target != ROOT:
                self.send_json({"error": "Path escapes workspace"}, status=400)
                return

            target.write_text(content, encoding="utf-8")
            self.send_json({"ok": True, "path": rel_path, "bytes": len(content.encode("utf-8"))})
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), LiveHandler)
    print(f"MusicParser live server: http://{host}:{port}")
    server.serve_forever()
