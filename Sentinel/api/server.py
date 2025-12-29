"""
API server setup for Sentinel.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        A configured FastAPI application.
    """
    app = FastAPI(
        title="Sentinel",
        description="A LangGraph-based multi-agent system",
        version="0.1.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    # TODO: Add workflow endpoints
    # TODO: Add agent management endpoints

    return app
