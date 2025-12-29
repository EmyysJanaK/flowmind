from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import uvicorn
from .database import engine, Base, get_db
from .api.v1 import workflow, process, optimization, analytics
from .services.workflow_service import WorkflowService
from .config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FlowMind application...")
    # Initialize database connection
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown
    logger.info("Shutting down FlowMind application...")

app = FastAPI(
    title="FlowMind API",
    description="Autonomous Workflow Intelligence Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(workflow.router, prefix="/api/v1/workflows", tags=["workflows"])
app.include_router(process.router, prefix="/api/v1/processes", tags=["processes"])
app.include_router(optimization.router, prefix="/api/v1/optimization", tags=["optimization"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time workflow updates
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

@app.get("/")
async def root():
    return {"message": "FlowMind API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)