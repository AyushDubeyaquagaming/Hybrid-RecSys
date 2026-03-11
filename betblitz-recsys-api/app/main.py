from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings
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
        # Artifacts missing or corrupt — service will report unhealthy
        import logging
        logging.getLogger(__name__).warning("Failed to load artifacts: %s", exc)
    app.state.model_service = model_service
    yield
    # Shutdown: nothing to clean up for artifact-only serving


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

@app.get("/")
def root():
    return {
        "message": "BetBlitz RecSys API is running",
        "docs": "/docs",
        "health": "/health"
    }


from app.routes import recommendations, health  # noqa: E402

app.include_router(recommendations.router)
app.include_router(health.router)
