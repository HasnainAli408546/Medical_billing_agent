from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from app.agents.state import ClaimState
from app.agents.llm_setup import get_llm
from app.services.ml_service import predict_denial
from app.services import rag_service
import json
import logging
import re

logger = logging.getLogger(__name__)

def extract_json(text: str) -> dict:
    """Safely extracts JSON from LLM responses even if wrapped in markdown blocks."""
    print(f"\n[🤖 LLM RAW OUTPUT] --->\n{text}\n<---")
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception as e:
        print(f"[⚠️ JSON PARSE ERROR]: {e}")
        raise ValueError("Failed to parse JSON")

# Utility to log agent actions
def log_action(state: ClaimState, agent_name: str, input_data: str, output_data: Any) -> ClaimState:
    log_entry = {
        "agent_name": agent_name,
        "input": input_data,
        "output": output_data
    }
    return {"agent_logs": [log_entry]}


def intent_agent(state: ClaimState) -> ClaimState:
    """Detects whether the input is a new claim, query, or correction."""
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "Determine the user intent from this text. Reply ONLY with one of: NEW_CLAIM, QUERY, CORRECTION.\nText: {text}"
    )
    chain = prompt | llm
    response = chain.invoke({"text": state["input_text"]})
    intent = response.content.strip()
    
    return {
        "intent": intent,
        **log_action(state, "Intent Agent", state["input_text"], intent)
    }

def extraction_agent(state: ClaimState) -> ClaimState:
    """Extracts structured data from the medical voice note."""
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "Extract Patient Name, Age, Diagnosis, and Procedure from the following text.\n"
        "Return ONLY a valid JSON object with keys: patient_name, age, diagnosis, procedure.\n"
        "Text: {text}"
    )
    chain = prompt | llm
    response = chain.invoke({"text": state["input_text"]})
    
    try:
        extracted_data = extract_json(response.content)
    except Exception:
        extracted_data = {"error": "Failed to extract structured data"}

    return {
        "extracted_data": extracted_data,
        **log_action(state, "Extraction Agent", state["input_text"], extracted_data)
    }

def coding_agent(state: ClaimState) -> ClaimState:
    """
    Production-Grade Coding Agent.
    STEP 1: Performs semantic search over thousands of official codes using RAG.
    STEP 2: LLM adjudicates the best match based on clinical context.
    """
    diagnosis_query = state["extracted_data"].get("diagnosis", "Unknown")
    procedure_query = state["extracted_data"].get("procedure", "Unknown")
    
    # ── RETRIEVAL: Search official code sets ─────────────────
    icd_candidates = rag_service.search_codes(diagnosis_query, code_type="ICD", top_k=5)
    cpt_candidates = rag_service.search_codes(procedure_query, code_type="CPT", top_k=5)
    
    # ── GENERATION: LLM picks the best match ─────────────────
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "You are an expert medical coder. Based on the clinical note, pick the most accurate "
        "ICD-10-CM code and CPT code from the candidates provided.\n\n"
        "Clinical Note: {note}\n"
        "Extracted Diagnosis: {diag}\n"
        "Extracted Procedure: {proc}\n\n"
        "ICD-10 CANDIDATES:\n{icd_list}\n\n"
        "CPT CANDIDATES:\n{cpt_list}\n\n"
        "Return ONLY a valid JSON with keys: icd_code, cpt_code, rationale."
    )
    
    icd_text = "\n".join([f"- {c['code']}: {c['description']}" for c in icd_candidates])
    cpt_text = "\n".join([f"- {c['code']}: {c['description']}" for c in cpt_candidates])
    
    chain = prompt | llm
    response = chain.invoke({
        "note": state["input_text"],
        "diag": diagnosis_query,
        "proc": procedure_query,
        "icd_list": icd_text,
        "cpt_list": cpt_text
    })
    
    try:
        coding_data = extract_json(response.content)
    except Exception:
        # Fallback to top result if LLM fails
        coding_data = {
            "icd_code": icd_candidates[0]["code"] if icd_candidates else "UNKNOWN",
            "cpt_code": cpt_candidates[0]["code"] if cpt_candidates else "UNKNOWN",
            "rationale": "Fallback to top semantic match."
        }

    return {
        "coding_data": coding_data,
        **log_action(state, "Coding Agent (Production)", f"Search: {diagnosis_query} / {procedure_query}", coding_data)
    }

def validation_agent(state: ClaimState) -> ClaimState:
    """
    RAG Validation Agent.
    STEP 1: Retrieves the most relevant billing rules from FAISS.
    STEP 2: Augments the claim with the retrieved rules.
    STEP 3: LLM adjudicates → VALID / NEEDS_REVIEW / INVALID.
    """
    extracted = state.get("extracted_data", {})
    coding    = state.get("coding_data",   {})

    icd_code        = coding.get("icd_code",  "Unknown")
    cpt_code        = coding.get("cpt_code",  "Unknown")

    claim_for_rag = {
        "icd_code":         icd_code,
        "icd_description":  extracted.get("diagnosis",         "Unknown"),
        "cpt_code":         cpt_code,
        "cpt_description":  extracted.get("procedure",         "Unknown"),
        "patient_age":      extracted.get("age",               "Unknown"),
        "claim_type":       extracted.get("claim_type",        "Outpatient"),
        "submission_method":extracted.get("submission_method", "Online"),
        "specialty":        extracted.get("specialty",         "General"),
        "claim_amount":     extracted.get("claim_amount",      0),
        "prior_auth":       extracted.get("prior_auth",        "Not specified"),
        "clinical_note":    state.get("input_text",            "None"),
    }

    try:
        result = rag_service.validate_claim(claim_for_rag)
    except FileNotFoundError:
        logger.warning("FAISS index not found — using fallback validation.")
        result = rag_service.validate_claim_fallback(icd_code, cpt_code)
    except Exception as e:
        logger.error(f"RAG validation error: {e}")
        result = rag_service.validate_claim_fallback(icd_code, cpt_code)

    # Map RAG status to agent state fields
    status = "valid" if result["status"] == "VALID" else "invalid"
    errors = result.get("errors", []) + result.get("warnings", [])

    return {
        "validation_status": status,
        "validation_errors": errors,
        **log_action(
            state,
            "Validation Agent (RAG)",
            f"ICD: {icd_code} | CPT: {cpt_code}",
            {
                "rag_status":      result["status"],
                "confidence":      result.get("confidence"),
                "pipeline":        result.get("pipeline"),
                "matched_rule":    result.get("matched_rule_id"),
                "recommendation":  result.get("recommendation"),
                "retrieved_rules": result.get("retrieved_rules", []),
            }
        )
    }

def prediction_agent(state: ClaimState) -> ClaimState:
    """
    Denial Prediction Agent — powered by XGBoost (AUC 0.97).
    Extracts relevant features from the agent state and calls
    the ML service to get a real denial probability score.
    """
    extracted = state.get("extracted_data", {})
    coding    = state.get("coding_data",   {})

    # Build claim dict from current agent state
    claim_input = {
        "patient_age":               extracted.get("age", 35),
        "patient_gender":            extracted.get("gender", "M"),
        "patient_income":            extracted.get("patient_income", 50_000),
        "patient_marital_status":    extracted.get("marital_status", "Single"),
        "patient_employment_status": extracted.get("employment_status", "Employed"),
        "claim_amount":              extracted.get("claim_amount", 3_000),
        "claim_type":                extracted.get("claim_type", "Outpatient"),
        "claim_submission_method":   extracted.get("submission_method", "Online"),
        "provider_specialty":        extracted.get("specialty", "General"),
        "diagnosis_code":            coding.get("icd_code", "Z00.00"),
        "procedure_code":            coding.get("cpt_code", "99213"),
    }

    # Boost probability if validation already found errors
    # (invalid code = known denial trigger)
    if state.get("validation_status") == "invalid":
        claim_input["patient_income"]  = min(claim_input["patient_income"], 24_000)
        claim_input["claim_type"]      = "Emergency"
        claim_input["claim_submission_method"] = "Paper"

    try:
        result = predict_denial(claim_input)
        logger.info(f"ML prediction: prob={result['denial_probability']}, tier={result['risk_tier']}")
    except Exception as e:
        logger.error(f"Prediction agent error: {e}")
        result = {
            "denial_probability": 0.50,
            "is_denied":          False,
            "risk_tier":          "MEDIUM",
            "denial_reason":      "Prediction service unavailable — manual review recommended.",
            "model_used":         "fallback",
        }

    return {
        "denial_probability": result["denial_probability"],
        "denial_reason":      result["denial_reason"],
        **log_action(
            state,
            "Prediction Agent",
            str(claim_input),
            {
                "probability": result["denial_probability"],
                "risk_tier":   result["risk_tier"],
                "is_denied":   result["is_denied"],
                "model":       result["model_used"],
            }
        )
    }

def correction_agent(state: ClaimState) -> ClaimState:
    """Suggests actionable fixes when denial probability is high."""
    suggestions = []
    prob   = state.get("denial_probability", 0)
    reason = state.get("denial_reason", "")

    if prob > 0.65:
        suggestions.append("⚠️ HIGH RISK: Recommend manual review before submission.")

    if "Paper" in reason and "Emergency" in reason:
        suggestions.append("Switch submission method to Electronic/Online for Emergency claims.")

    if "underinsured" in reason.lower() or "medicaid" in reason.lower():
        suggestions.append("Verify patient's current insurance coverage and eligibility before submission.")

    if "pre-authorization" in reason.lower():
        suggestions.append("Obtain pre-authorization from insurer before billing specialist procedure.")

    if "claim amount" in reason.lower() and "routine" in reason.lower():
        suggestions.append("Review itemized billing — high amount on Routine visit may trigger upcoding audit.")

    if "phone" in reason.lower() and "inpatient" in reason.lower():
        suggestions.append("Resubmit Inpatient claim via Electronic or Online channel with full documentation.")

    if "icd" in reason.lower() or state.get("validation_status") == "invalid":
        suggestions.append(f"Review ICD-10 code for: {state.get('extracted_data', {}).get('diagnosis', 'Unknown')}")

    if not suggestions and prob > 0.45:
        suggestions.append("Moderate denial risk — double-check the diagnosis-procedure code pairing.")

    return {
        "correction_suggestions": suggestions,
        **log_action(state, "Correction Agent", reason, suggestions)
    }

def explanation_agent(state: ClaimState) -> ClaimState:
    """Generates a human readable summary of the decisions."""
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "Summarize the structured data, coding, validation and denial risk into a short human readable explanation.\n"
        "Data: {data}"
    )
    chain = prompt | llm
    response = chain.invoke({"data": str(state)})
    
    # Bundle the final claim definition
    final_claim = {
        **state.get("extracted_data", {}),
        **state.get("coding_data", {}),
        "status": state.get("validation_status"),
        "risk_score": state.get("denial_probability")
    }

    return {
        "explanation": response.content,
        "final_claim": final_claim,
        **log_action(state, "Explanation Agent", "Finalizing Claim", response.content)
    }
