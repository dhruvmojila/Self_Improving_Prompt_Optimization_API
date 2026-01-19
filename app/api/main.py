"""
FastAPI application main entry point.
Configures the API with CORS, routes, and lifecycle events.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.core.events import event_bus
from app.infrastructure.pixeltable import init_pixeltable, init_tables, register_all_udfs

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting Prompt Optimization API...")
    
    try:
        # Initialize Pixeltable
        logger.info("Initializing Pixeltable...")
        init_pixeltable()
        init_tables()
        register_all_udfs()
        logger.info("✓ Pixeltable initialized")
        
        # Clean old jobs
        event_bus.clear_completed_jobs(older_than_hours=24)
        logger.info("✓ Cleaned old jobs")
        
        logger.info("✅ API started successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Prompt Optimization API...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="CI/CD for Prompts - Automated prompt optimization with DSPy and Pixeltable",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Check API health status."""
    from app.infrastructure.pixeltable import get_pixeltable_client
    
    pxt_health = get_pixeltable_client().health_check()
    
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "pixeltable": pxt_health,
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root - returns basic info."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


# Import and include routers
from app.api.v1.endpoints import projects, datasets, prompts, optimization

app.include_router(
    projects.router,
    prefix=f"{settings.API_V1_PREFIX}/projects",
    tags=["Projects"]
)

app.include_router(
    datasets.router,
    prefix=f"{settings.API_V1_PREFIX}/datasets",
    tags=["Datasets"]
)

app.include_router(
    prompts.router,
    prefix=f"{settings.API_V1_PREFIX}/prompts",
    tags=["Prompts"]
)

app.include_router(
    optimization.router,
    prefix=f"{settings.API_V1_PREFIX}/optimization",
    tags=["Optimization"]
)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
