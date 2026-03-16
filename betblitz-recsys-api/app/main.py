from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import Settings
from app.metrics import MODEL_LOADED
from app.services.model_service import ModelService

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service = ModelService()
    try:
        model_service.load_artifacts(
            settings.ARTIFACT_DIR,
            num_threads=settings.PREDICT_NUM_THREADS,
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Failed to load artifacts: %s", exc)
    app.state.model_service = model_service
    MODEL_LOADED.set(1 if model_service.is_loaded() else 0)
    yield
    # Shutdown: nothing to clean up for artifact-only serving


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Prometheus: auto-instruments all routes with request count, latency, in-progress
# and exposes /metrics
Instrumentator().instrument(app).expose(app)


@app.get("/")
def root():
    return {
        "message": "BetBlitz RecSys API is running",
        "docs": "/docs",
        "health": "/health",
    }


from app.routes import recommendations, health  # noqa: E402

app.include_router(recommendations.router)
app.include_router(health.router)
