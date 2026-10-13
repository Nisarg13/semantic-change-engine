"""REST API (FastAPI).

Endpoints:
    GET  /health      Status and model names.
    POST /v1/detect   The verdict, in the output format of the assignment.
    POST /v1/analyze  The verdict, the changed units, the agent trace and the report.

Errors: 422 if the request is not valid, 502 if the evaluation agent fails, and 503 if
the Claude API key is missing or not valid.

Run:   uvicorn semantic_change.api:create_app --factory --port 8000 --env-file .env
Docs:  http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from semantic_change import __version__
from semantic_change.agent import AgentError, LLMNotConfiguredError
from semantic_change.models import Analysis, ChangeRequest, ChangeResult
from semantic_change.pipeline import SemanticChangePipeline

# Error responses of the detection endpoints, for the OpenAPI documentation.
_ERRORS: dict[int | str, dict[str, Any]] = {
    502: {"description": "The evaluation agent failed (Claude refused, or an API error)."},
    503: {"description": "The Claude API key is missing or not valid."},
}


def create_app(pipeline: SemanticChangePipeline | None = None) -> FastAPI:
    """Create the FastAPI application.

    ``pipeline`` replaces the default pipeline, for example in tests. The models load at
    startup, before the server accepts requests.
    """
    pipeline = pipeline or SemanticChangePipeline()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        pipeline.warmup()
        yield

    app = FastAPI(
        title="Semantic Change Detection Engine",
        version=__version__,
        lifespan=lifespan,
        description=(
            "Compares two versions of a text and reports the changes in meaning. The "
            "pipeline: spaCy clause segmentation, sentence embeddings, Hungarian alignment "
            "and a Claude evaluation agent."
        ),
    )

    @app.exception_handler(ValueError)
    async def invalid_input(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(LLMNotConfiguredError)
    async def not_configured(_: Request, exc: LLMNotConfiguredError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(AgentError)
    async def agent_failed(_: Request, exc: AgentError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        """Return the service status and the names of the configured models."""
        settings = pipeline.settings
        return {
            "status": "ok",
            "version": __version__,
            "spacy_model": settings.spacy_model,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
        }

    @app.post("/v1/detect", response_model=ChangeResult, tags=["detection"], responses=_ERRORS)
    def detect(request: ChangeRequest) -> ChangeResult:
        """Compare two texts and return the verdict in the output format of the assignment.

        If the wording changed, the Claude agent judges the change, which takes some
        seconds. If not, the verdict comes without a Claude call.
        """
        return pipeline.detect(request.text_a, request.text_b)

    @app.post("/v1/analyze", response_model=Analysis, tags=["detection"], responses=_ERRORS)
    def analyze(request: ChangeRequest) -> Analysis:
        """Compare two texts and return the full analysis.

        The response has the verdict, each changed unit with the finding of the agent,
        the unit counts, the agent trace with the token counts, and the Markdown report.
        """
        return pipeline.analyze(request.text_a, request.text_b)

    return app
