"""
server.py  —  FastAPI REST server.

Endpoints
---------
GET  /health       → health check
POST /verify       → run MASA-SVD pipeline on a query-response pair
POST /svd-only     → return only the SVD analysis (no API calls)

Start with:  python scripts/run_api.py
Then visit:  http://localhost:8000/docs   (interactive Swagger UI)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (VerifyRequest, VerifyResponse, HealthResponse,
                             Breakdown, SVDAnalysis)
from src.pipeline import MASASVDPipeline
from src.core.svd_engine import SVDSubspaceEngine
from src.pipeline import _split_sentences

log = logging.getLogger(__name__)

# ── Global pipeline instance (loaded at startup) ─────────────────────────────
_pipeline: MASASVDPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup; clean up at shutdown."""
    global _pipeline
    log.info("Server starting — loading MASA-SVD pipeline …")
    _pipeline = MASASVDPipeline(use_nli=True)
    log.info("Pipeline ready.")
    yield
    log.info("Server shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "MASA-SVD Hallucination Detection API",
    description = "Multi-Agent Subspace Alignment with SVD for LLM hallucination mitigation.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(status="ok",
                          message="MASA-SVD API is running.")


@app.post("/verify", response_model=VerifyResponse, tags=["Verification"])
def verify(req: VerifyRequest):
    """
    Run the full MASA-SVD pipeline on a query-response pair.
    Returns verdict, risk score, and per-agent breakdown.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503,
                            detail="Pipeline not initialised yet.")
    try:
        result = _pipeline.analyze(req.query, req.llm_response,
                                   use_nli=req.use_nli)
        bd = result["breakdown"]
        return VerifyResponse(
            verdict          = result["verdict"],
            final_risk_score = result["final_risk_score"],
            action           = result["action"],
            breakdown        = Breakdown(**bd),
            svd_analysis     = SVDAnalysis(**result["svd_analysis"]),
            query            = req.query,
            llm_response     = req.llm_response,
        )
    except Exception as e:
        log.exception("Error in /verify")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/svd-only", tags=["Verification"])
def svd_only(req: VerifyRequest):
    """
    Return only the SVD subspace analysis — no API calls, instant response.
    Useful for testing the geometric component in isolation.
    """
    engine = SVDSubspaceEngine()
    ctx    = _split_sentences(req.query)
    out    = _split_sentences(req.llm_response)
    return engine.full_analysis(ctx, out)