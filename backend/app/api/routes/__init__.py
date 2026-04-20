from fastapi import APIRouter
from app.api.routes.voice_router import router as voice_router
from app.api.routes.claims_router import router as claims_router
from app.api.routes.analytics_router import router as analytics_router

api_router = APIRouter()

api_router.include_router(voice_router,     prefix="/voice",     tags=["Voice"])
api_router.include_router(claims_router,    prefix="/claims",    tags=["Claims"])
api_router.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
