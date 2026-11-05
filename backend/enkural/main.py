"""En Kural server: FastAPI app serving the API and the single-page UI."""
from __future__ import annotations

import argparse
import logging
import os
import threading
from contextlib import asynccontextmanager

from . import config

config.ensure_dirs()  # must run before huggingface_hub is imported anywhere

from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402

from . import security  # noqa: E402
from .api.routes import router  # noqa: E402
from .jobs import manager  # noqa: E402

log = logging.getLogger("enkural")

CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "media-src 'self' blob:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'")


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Content-Security-Policy"] = CSP
        resp.headers["Permissions-Policy"] = "microphone=(self)"
        return resp


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.start()
    token = security.get_token()
    url = f"http://{config.HOST if config.HOST != '0.0.0.0' else '127.0.0.1'}:{config.PORT}/"
    if token:
        url += f"?token={token}"
    if os.environ.get("ENKURAL_LAUNCHER") != "1":  # start.sh prints (and opens) the link itself
        print(f"\n  En Kural is running.  Open this link in your browser:\n\n    {url}\n", flush=True)
    if os.environ.get("ENKURAL_PRELOAD", "0") == "1":
        def warm():
            try:
                from .vc.registry import get_engine

                get_engine(config.DEFAULT_ENGINE)
            except Exception as e:  # noqa: BLE001
                log.warning("Engine preload failed: %s", e)

        threading.Thread(target=warm, name="enkural-preload", daemon=True).start()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="En Kural", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_middleware(SecurityHeaders)
    app.include_router(router, prefix="/api", dependencies=[Depends(security.require_token)])

    @app.get("/")
    def index():
        resp = FileResponse(config.FRONTEND_DIR / "index.html", media_type="text/html")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/healthz")
    def health():
        return JSONResponse({"ok": True})

    app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR)), name="static")

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="En Kural – local voice conversion server")
    parser.add_argument("--host", default=config.HOST)
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--preload", action="store_true", help="load the conversion model at start-up")
    args = parser.parse_args()
    if args.preload:
        os.environ["ENKURAL_PRELOAD"] = "1"
    config.HOST, config.PORT = args.host, args.port
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: binding to a non-local address. Put the app behind HTTPS (reverse proxy) "
              "before exposing it; voice data is sensitive.", flush=True)
    logging.basicConfig(level=os.environ.get("ENKURAL_LOG", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
