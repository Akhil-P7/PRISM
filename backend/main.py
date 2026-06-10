"""
PRISM Backend — FastAPI Application Entrypoint
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PRISM API",
    description=(
        "Pediatric Respiratory Intelligence System — "
        "REST API for respiratory sound analysis, temporal intelligence, "
        "and retrieval-augmented clinical insight generation."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---- CORS Middleware ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint — health check."""
    return {
        "service": "PRISM API",
        "version": "0.1.0",
        "status": "operational",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}


# ---- Register API Routers ----
# TODO: Uncomment as routers are implemented
# from backend.api import datasets, recordings, events, temporal, retrieval, insights
# app.include_router(datasets.router, prefix="/api/v1", tags=["Datasets"])
# app.include_router(recordings.router, prefix="/api/v1", tags=["Recordings"])
# app.include_router(events.router, prefix="/api/v1", tags=["Events"])
# app.include_router(temporal.router, prefix="/api/v1", tags=["Temporal"])
# app.include_router(retrieval.router, prefix="/api/v1", tags=["Retrieval"])
# app.include_router(insights.router, prefix="/api/v1", tags=["Insights"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
