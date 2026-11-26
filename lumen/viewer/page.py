"""The viewer: one HTML page with everything inside it (no network, no install).

``write_page`` saves it as a file you can open or send. ``serve`` hands the same page out over
HTTP, for a machine you reach from another device (pass ``host="0.0.0.0"`` to listen on every
interface).
"""

from __future__ import annotations

import http.server
import json
from importlib import resources
from pathlib import Path

from lumen.viewer.scene import Anatomy, resolve, scene_dict


def _static(name: str) -> str:
    return resources.files("lumen.viewer").joinpath("static", name).read_text(encoding="utf-8")


def render_page(scene: dict, replay: dict | None = None) -> str:
    """The complete page as text."""
    data = json.dumps({"scene": scene, "replay": replay}, separators=(",", ":"))
    data = data.replace("</", "<\\/")                   # keep "</script>" in names from ending the block
    title = str(scene.get("label", "anatomy")).replace("<", "&lt;").replace(">", "&gt;")
    page = _static("index.html")
    for key, value in (("__CSS__", _static("viewer.css")), ("__JS__", _static("viewer.js")),
                       ("__TITLE__", title), ("__DATA__", data)):
        page = page.replace(key, value, 1)
    return page


LOOKS = ("monitor", "anatomy", "glass")


def build(anatomy: str | Anatomy, replay: dict | None = None, target: str | None = None,
          look: str = "monitor") -> str:
    """The page for an anatomy (and optionally a replay). ``look`` is the 3D pane's starting
    style: "monitor" (dark, see-through red), "anatomy" (solid, light page) or "glass"
    (glowing, nearly clear walls); the page can switch between them."""
    if look not in LOOKS:
        raise ValueError(f"look must be one of {', '.join(LOOKS)}")
    a = anatomy if isinstance(anatomy, Anatomy) else resolve(anatomy)
    scene = scene_dict(a, target or (replay or {}).get("target"))
    scene["look"] = look
    return render_page(scene, replay)


def write_page(page: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(page, encoding="utf-8")
    return p


def serve(page: str, host: str = "127.0.0.1", port: int = 8765, *, say=print):
    """Serve the page until interrupted."""
    body = page.encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server's naming)
            if self.path.split("?")[0] not in ("/", "/index.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    shown = host if host not in ("0.0.0.0", "::") else "<this machine's address>"
    say(f"Lumen viewer on http://{shown}:{server.server_address[1]}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return server
