import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from app.db.database import get_db
from app.db.models import Patient, Claim, Denial, AgentLog
from app.agents.graph import agent_orchestrator
from app.agents.state import ClaimState
from app.schemas.claims import ProcessClaimRequest, ProcessClaimResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/generate", response_model=ProcessClaimResponse)
def generate_claim(request: ProcessClaimRequest, db: Session = Depends(get_db)):
    """
    Advanced Endpoint:
    1. Receives raw transcription.
    2. Feeds it through the LangGraph Multi-Agent Orchestrator.
    3. Interprets the heavily analyzed results.
    4. Commits the Patient, Claim, Denial Risk, and transparent Agent Logs straight to PostgreSQL.
    """
    logger.info("Initializing multi-agent pipeline for claim generation.")
    
    # 1. Initialize LangGraph State
    initial_state = {
        "input_text": request.transcribed_text,
        "intent": "",
        "extracted_data": {},
        "coding_data": {},
        "validation_status": "",
        "validation_errors": [],
        "denial_probability": 0.0,
        "denial_reason": "",
        "correction_suggestions": [],
        "final_claim": {},
        "explanation": "",
        "agent_logs": []
    }
    
    # 2. Invoke Orchestrator (Synchronous for now, production could use Celery/BackgroundTasks for huge graphs)
    try:
        final_state = agent_orchestrator.invoke(initial_state)
    except Exception as e:
        logger.error(f"LangGraph execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Pipeline failed: {str(e)}")
        
    extracted = final_state.get("extracted_data", {})
    patient_name = extracted.get("patient_name") or "Unknown Patient"
    # robust casting for age — LLM may return null, a string "45", or a float
    try:
        age = int(extracted.get("age") or 0)
    except (ValueError, TypeError):
        age = 0
    
    # 3. Secure Database Transactions
    try:
        # A. Resolve or Create Secondary Patient Profile
        patient = db.query(Patient).filter(Patient.name == patient_name).first()
        if not patient:
            patient = Patient(name=patient_name, age=age, insurance_provider="Auto-Assigned")
            db.add(patient)
            db.flush() # Flush to get ID early
            
        # B. Construct and Record the core Claim
        final_claim_data = final_state.get("final_claim", {})
        db_claim = Claim(
            patient_id=patient.id,
            diagnosis=extracted.get("diagnosis", "Unknown"),
            procedure=extracted.get("procedure", "Unknown"),
            icd_code=final_claim_data.get("icd_code", "N/A"),
            cpt_code=final_claim_data.get("cpt_code", "N/A"),
            status=final_state.get("validation_status", "draft")
        )
        db.add(db_claim)
        db.flush()
        
        # C. Store Machine Learning Risk Predictions (Denial)
        db_denial = Denial(
            claim_id=db_claim.id,
            reason=final_state.get("denial_reason", "Low Risk"),
            probability=final_state.get("denial_probability", 0.0)
        )
        db.add(db_denial)
        
        # D. Store Agent Audit Logs for Transparent Execution (Extreme Production Level)
        logs = final_state.get("agent_logs", [])
        for log in logs:
            db_log = AgentLog(
                claim_id=db_claim.id,
                agent_name=log.get("agent_name", "Unknown"),
                input_data={"data": str(log.get("input", ""))}, # stringify to avoid JSON cast errors from weird LLM objects
                output_data={"data": str(log.get("output", ""))}
            )
            db.add(db_log)
            
        # Commit the monolithic transaction
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database commitment failed. transaction rolled back: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database persistence failed: {str(e)}")

    # 4. API Return Object
    return ProcessClaimResponse(
        claim_id=db_claim.id,
        patient_name=patient.name,
        status=final_state.get("validation_status", "error"),
        explanation=final_state.get("explanation", ""),
        risk_score=final_state.get("denial_probability", 1.0),
        final_claim=final_state.get("final_claim", {}),
        correction_suggestions=final_state.get("correction_suggestions", []),
        agent_logs=final_state.get("agent_logs", [])
    )


# ══════════════════════════════════════════════════════════════
#  GET /claims  — List all claims with denial risk
# ══════════════════════════════════════════════════════════════

@router.get("/")
def list_claims(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """Returns paginated list of all claims with patient and denial info."""
    claims = db.query(Claim).order_by(Claim.created_at.desc()).offset(skip).limit(limit).all()

    result = []
    for claim in claims:
        patient = db.query(Patient).filter(Patient.id == claim.patient_id).first()
        denial  = db.query(Denial).filter(Denial.claim_id == claim.id).first()
        result.append({
            "claim_id":           claim.id,
            "patient_name":       patient.name if patient else "Unknown",
            "patient_age":        patient.age  if patient else None,
            "diagnosis":          claim.diagnosis,
            "procedure":          claim.procedure,
            "icd_code":           claim.icd_code,
            "cpt_code":           claim.cpt_code,
            "status":             claim.status,
            "denial_probability": round(denial.probability, 3) if denial else None,
            "denial_reason":      denial.reason if denial else None,
            "created_at":         claim.created_at.isoformat() if claim.created_at else None,
        })

    return {"total": len(result), "claims": result}


# ══════════════════════════════════════════════════════════════
#  GET /claims/{claim_id}  — Single claim with agent logs
# ══════════════════════════════════════════════════════════════

@router.get("/{claim_id}")
def get_claim_detail(claim_id: int, db: Session = Depends(get_db)):
    """Returns full detail of a single claim including agent audit logs."""
    claim = db.query(Claim).filter(Claim.id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")

    patient = db.query(Patient).filter(Patient.id == claim.patient_id).first()
    denial  = db.query(Denial).filter(Denial.claim_id == claim_id).first()
    logs    = db.query(AgentLog).filter(AgentLog.claim_id == claim_id).order_by(AgentLog.timestamp).all()

    return {
        "claim_id":     claim.id,
        "patient": {
            "name": patient.name if patient else "Unknown",
            "age":  patient.age  if patient else None,
            "insurance_provider": patient.insurance_provider if patient else None,
        },
        "claim": {
            "diagnosis":  claim.diagnosis,
            "procedure":  claim.procedure,
            "icd_code":   claim.icd_code,
            "cpt_code":   claim.cpt_code,
            "status":     claim.status,
            "created_at": claim.created_at.isoformat() if claim.created_at else None,
        },
        "denial": {
            "probability": round(denial.probability, 3) if denial else None,
            "reason":      denial.reason if denial else None,
            "corrected":   denial.corrected if denial else False,
        } if denial else None,
        "agent_logs": [
            {
                "agent":     log.agent_name,
                "input":     log.input_data,
                "output":    log.output_data,
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            }
            for log in logs
        ],
    }

