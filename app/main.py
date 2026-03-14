from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

from app.core.config import get_settings
from app.api.routes import query_router, ingest_router, feedback_router, status_router

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌍 CulturaRAG starting up …")
    logger.info("Embedding model will load on first request (lazy loading).")
    yield
    logger.info("CulturaRAG shutting down.")

def create_app() -> FastAPI:
    app = FastAPI(
        title="CulturaRAG",
        description="AI Knowledge Explorer for World Cultures & Languages",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(query_router,    prefix="/api/v1")
    app.include_router(ingest_router,   prefix="/api/v1")
    app.include_router(feedback_router, prefix="/api/v1")
    app.include_router(status_router,   prefix="/api/v1")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        return """
        <html><body>
        <h1>🌍 CulturaRAG</h1>
        <p><a href="/docs">Open API Docs</a></p>
        </body></html>
        """
    return app

app = create_app()
