from typing import TypedDict, Annotated, List, Dict, Any
import operator

class ClaimState(TypedDict):
    input_text: str
    intent: str
    extracted_data: Dict[str, Any]
    coding_data: Dict[str, Any]
    validation_status: str  # e.g., "valid", "invalid"
    validation_errors: List[str]
    denial_probability: float
    denial_reason: str
    correction_suggestions: List[str]
    final_claim: Dict[str, Any]
    explanation: str
    # Keep track of agent observations and logs iteratively
    agent_logs: Annotated[List[Dict[str, Any]], operator.add]
