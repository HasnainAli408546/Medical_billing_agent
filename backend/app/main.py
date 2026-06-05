import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables on startup (no Alembic needed for deploy)."""
    from app.db.database import engine
    from app.db.models import Base
    logger.info("🔄 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables ready.")
    yield

app = FastAPI(
    title="Voice-Driven Revenue Cycle Copilot",
    description="Multi-Agent AI System for Healthcare Billing Automation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Revenue Cycle Copilot API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
