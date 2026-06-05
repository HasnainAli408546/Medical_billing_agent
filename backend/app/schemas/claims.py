from pydantic import BaseModel, Field
from typing import Dict, Any, List

class ProcessClaimRequest(BaseModel):
    transcribed_text: str = Field(..., description="The raw medical text or transcription to process.")

class ProcessClaimResponse(BaseModel):
    claim_id: int
    patient_name: str
    status: str
    explanation: str
    risk_score: float
    final_claim: Dict[str, Any]
    correction_suggestions: List[str]
    agent_logs: List[Dict[str, Any]] = []
