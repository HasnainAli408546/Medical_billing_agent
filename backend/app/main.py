from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import api_router

app = FastAPI(
    title="Voice-Driven Revenue Cycle Copilot",
    description="Multi-Agent AI System for Healthcare Billing Automation",
    version="1.0.0",
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
