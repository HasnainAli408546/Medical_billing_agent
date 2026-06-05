#!/bin/bash
set -e

echo "🔄 Running database migrations..."
python -c "
from app.db.database import engine
from app.db.models import Base
Base.metadata.create_all(bind=engine)
print('✅ Database tables created/verified.')
"

echo "🚀 Starting FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
