import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Dict, Any
from datetime import datetime, timedelta

from app.db.database import get_db
from app.db.models import Patient, Claim, Denial, AgentLog

logger = logging.getLogger(__name__)
router = APIRouter()


# ══════════════════════════════════════════════════════════════
#  GET /analytics/summary
#  Main dashboard stats endpoint
# ══════════════════════════════════════════════════════════════

@router.get("/summary")
def get_analytics_summary(db: Session = Depends(get_db)):
    """
    Returns aggregated statistics for the analytics dashboard:
    - Total claims, approval/denial counts, denial rate
    - Average denial probability
    - Daily claims trend (last 14 days)
    - Top denial reasons
    - Claims by status breakdown
    """
    try:
        # ── Core counts ───────────────────────────────────────
        total_claims    = db.query(Claim).count()
        total_patients  = db.query(Patient).count()
        total_denials   = db.query(Denial).count()

        # Claims by validation status
        valid_claims   = db.query(Claim).filter(Claim.status == "valid").count()
        invalid_claims = db.query(Claim).filter(Claim.status == "invalid").count()
        draft_claims   = db.query(Claim).filter(Claim.status == "draft").count()

        # ── Denial probability stats ──────────────────────────
        denial_stats = db.query(
            func.avg(Denial.probability).label("avg_prob"),
            func.max(Denial.probability).label("max_prob"),
            func.min(Denial.probability).label("min_prob"),
        ).first()

        avg_denial_prob = round(float(denial_stats.avg_prob or 0), 4)

        # High-risk claims (probability > 0.65)
        high_risk_count = db.query(Denial).filter(Denial.probability > 0.65).count()
        medium_risk_count = db.query(Denial).filter(
            Denial.probability > 0.45, Denial.probability <= 0.65
        ).count()
        low_risk_count = db.query(Denial).filter(Denial.probability <= 0.45).count()

        # ── Denial rate ───────────────────────────────────────
        denial_rate = round((invalid_claims / total_claims * 100) if total_claims > 0 else 0, 1)

        # ── Daily claims trend (last 14 days) ─────────────────
        fourteen_days_ago = datetime.utcnow() - timedelta(days=14)
        daily_claims = db.query(
            func.date(Claim.created_at).label("date"),
            func.count(Claim.id).label("count"),
        ).filter(
            Claim.created_at >= fourteen_days_ago
        ).group_by(
            func.date(Claim.created_at)
        ).order_by(
            func.date(Claim.created_at)
        ).all()

        trend_data = [
            {"date": str(row.date), "claims": row.count}
            for row in daily_claims
        ]

        # ── Recent claims (last 10) ───────────────────────────
        recent_claims = db.query(Claim).order_by(
            desc(Claim.created_at)
        ).limit(10).all()

        recent_list = []
        for claim in recent_claims:
            denial = db.query(Denial).filter(Denial.claim_id == claim.id).first()
            patient = db.query(Patient).filter(Patient.id == claim.patient_id).first()
            recent_list.append({
                "claim_id":           claim.id,
                "patient_name":       patient.name if patient else "Unknown",
                "diagnosis":          claim.diagnosis,
                "procedure":          claim.procedure,
                "icd_code":           claim.icd_code,
                "cpt_code":           claim.cpt_code,
                "status":             claim.status,
                "denial_probability": round(denial.probability, 3) if denial else None,
                "created_at":         claim.created_at.isoformat() if claim.created_at else None,
            })

        # ── Agent usage stats ─────────────────────────────────
        agent_usage = db.query(
            AgentLog.agent_name,
            func.count(AgentLog.id).label("count")
        ).group_by(AgentLog.agent_name).all()

        agent_stats = [
            {"agent": row.agent_name, "calls": row.count}
            for row in agent_usage
        ]

        return {
            "summary": {
                "total_claims":       total_claims,
                "total_patients":     total_patients,
                "valid_claims":       valid_claims,
                "invalid_claims":     invalid_claims,
                "draft_claims":       draft_claims,
                "denial_rate":        denial_rate,
                "avg_denial_prob":    avg_denial_prob,
                "high_risk_claims":   high_risk_count,
                "medium_risk_claims": medium_risk_count,
                "low_risk_claims":    low_risk_count,
            },
            "trend":          trend_data,
            "recent_claims":  recent_list,
            "agent_stats":    agent_stats,
        }

    except Exception as e:
        logger.error(f"Analytics query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")


# ══════════════════════════════════════════════════════════════
#  GET /analytics/model-info
#  Serve ML model metadata
# ══════════════════════════════════════════════════════════════

@router.get("/model-info")
def get_model_info():
    """Returns the trained ML model's performance metrics."""
    import os, json
    META_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
        "backend", "models", "model_metadata.json"
    )
    try:
        with open(META_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Model metadata not found. Run train_model.py first."}
