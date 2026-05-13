from src.api.routers.adversarial import router as adversarial_router
from src.api.routers.analyze import router as analyze_router
from src.api.routers.clips import router as clips_router
from src.api.routers.health import router as health_router
from src.api.routers.robustness import router as robustness_router

__all__ = [
    "adversarial_router",
    "analyze_router",
    "clips_router",
    "health_router",
    "robustness_router",
]
