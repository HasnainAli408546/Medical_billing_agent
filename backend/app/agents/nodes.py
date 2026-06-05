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
        "Extract billing data from this clinical note. Return ONLY a JSON object like this example:\n"
        '{{"patient_name":"John Doe","age":55,"gender":"M","diagnosis":"Acute appendicitis, Hypertension, Type 2 Diabetes",'
        '"procedure":"Laparoscopic appendectomy","specialty":"General Surgery",'
        '"claim_type":"Inpatient","submission_method":"Electronic","claim_amount":null,'
        '"prior_auth":"Obtained","marital_status":"Married","employment_status":"Employed"}}\n\n'
        "CRITICAL RULES:\n"
        "1. NEVER invent or assume financial data, claim amounts, or patient income if it is not explicitly provided in the text. Leave them as null if not stated.\n"
        "2. For 'procedure', only extract procedures that were EXECUTED in the current note. Do NOT extract procedures listed in the 'Plan' or 'Recommendations' section that are intended for the future.\n"
        "3. For 'diagnosis', extract the Primary Diagnosis as well as all active Secondary/Comorbid Diagnoses (PMH) that affect patient care.\n"
        "4. claim_type must be Outpatient, Inpatient, or Emergency.\n"
        "5. If prior_auth is not stated but the procedure is major surgery, set it to Obtained.\n"
        "6. submission_method defaults to Electronic.\n"
        "7. Infer gender, specialty, marital_status, employment_status from context if possible.\n"
        "8. Look closely for age in formats like '42-year-old' and extract it as an integer.\n\n"
        "Clinical Note:\n{text}"
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
    diagnosis_raw = state["extracted_data"].get("diagnosis", "Unknown")
    procedure_raw = state["extracted_data"].get("procedure", "Unknown")
    
    # Context-Aware Query Expansion
    note_text = state.get("input_text", "").lower()
    diag_modifiers = []
    if "unknown" in note_text or "idiopathic" in note_text:
        diag_modifiers.append("unknown etiology unspecified")
        
    proc_modifiers = []
    if "consultation" in procedure_raw.lower() or "visit" in procedure_raw.lower():
        proc_modifiers.append("Office or other outpatient visit for evaluation and management")
        
    diagnosis_query = f"{diagnosis_raw} {' '.join(diag_modifiers)}".strip()
    procedure_query = f"{procedure_raw} {' '.join(proc_modifiers)}".strip()
    
    # ── RETRIEVAL: Search official code sets ─────────────────
    # Expanded search radius to 25 to ensure the LLM sees unspecified and specific variants
    icd_candidates = rag_service.search_codes(diagnosis_query, code_type="ICD", top_k=25)
    cpt_candidates = rag_service.search_codes(procedure_query, code_type="CPT", top_k=25)
    
    # ── GENERATION: LLM picks the best match ─────────────────
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "You are an expert medical coder. Based on the clinical note, determine the most accurate "
        "ICD-10-CM code and ALL applicable CPT codes from the provided candidates.\n\n"
        "CRITICAL CODING PRINCIPLES:\n"
        "1. Evaluate the candidates carefully. You must select the code with the highest level of specificity explicitly supported by the clinical note.\n"
        "2. If the clinical note lacks specific details regarding etiology, anatomy, or severity, you MUST select an 'unspecified' variant.\n"
        "3. Do not infer clinical details that are not documented.\n"
        "4. For evaluation and management, prefer standard E/M codes (e.g., 99203, 99204, 99213) over outdated consultation codes if appropriate.\n"
        "5. If none of the provided candidates are accurate, use your expert knowledge to provide the correct code.\n"
        "6. Only assign CPT codes for procedures that were EXECUTED during the encounter. Do NOT assign CPT codes for planned future procedures.\n"
        "7. Ensure CPT codes are age-appropriate based on the patient's age in the note.\n"
        "8. Extract the primary ICD-10 code as icd_code. If there are secondary diagnoses, include them in a secondary_icd_codes list.\n"
        "9. You MUST include ALL relevant CPT codes for the executed procedures in the cpt_codes list. Do not limit to just one code.\n\n"
        "Clinical Note: {note}\n"
        "Extracted Diagnosis: {diag}\n"
        "Extracted Procedure: {proc}\n\n"
        "ICD-10 CANDIDATES (Top 25):\n{icd_list}\n\n"
        "CPT CANDIDATES (Top 25):\n{cpt_list}\n\n"
        "Return ONLY a valid JSON with keys: icd_code, secondary_icd_codes (list), cpt_codes (list), rationale."
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
            "secondary_icd_codes": [],
            "cpt_codes": [cpt_candidates[0]["code"]] if cpt_candidates else ["UNKNOWN"],
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
    cpt_codes_list  = coding.get("cpt_codes",  ["Unknown"])
    cpt_code        = ", ".join(cpt_codes_list) if isinstance(cpt_codes_list, list) else str(cpt_codes_list)

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
    # Use `or` to coalesce None → default (LLM may return null for fields)
    claim_input = {
        "patient_age":               extracted.get("age") or 35,
        "patient_gender":            extracted.get("gender") or "M",
        "patient_income":            extracted.get("patient_income") or 50_000,
        "patient_marital_status":    extracted.get("marital_status") or "Single",
        "patient_employment_status": extracted.get("employment_status") or "Employed",
        "claim_amount":              extracted.get("claim_amount") or 3_000,
        "claim_type":                extracted.get("claim_type") or "Outpatient",
        "claim_submission_method":   extracted.get("submission_method") or "Online",
        "provider_specialty":        extracted.get("specialty") or "General",
        "diagnosis_code":            coding.get("icd_code") or "Z00.00",
        "procedure_code":            coding.get("cpt_codes", ["99213"])[0] if isinstance(coding.get("cpt_codes"), list) and len(coding.get("cpt_codes")) > 0 else str(coding.get("cpt_codes", "99213")),
    }

    # Note: Previously this block would poison features when validation
    # was invalid (forcing income=$24k, type=Emergency, method=Paper),
    # which guaranteed ~99% denial. Removed so the ML model can give
    # an honest prediction based on actual extracted data.
    # The validation status is still available in the final claim output.

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
    """Uses LLM to suggest actionable fixes based on validation errors and denial risks."""
    prob   = state.get("denial_probability", 0)
    reason = state.get("denial_reason", "")
    val_status = state.get("validation_status", "valid")
    val_errors = state.get("validation_errors", [])
    
    # If claim is perfectly healthy and low risk, no need for heavy LLM call
    if val_status == "valid" and prob < 0.40 and not val_errors:
        return {
            "correction_suggestions": ["No corrections needed. Claim appears clean."],
            **log_action(state, "Correction Agent", "Clean claim", ["No corrections needed."])
        }

    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "You are an expert medical billing auditor. Review the following claim validation errors and denial risk factors, "
        "and provide 1-3 highly specific, actionable recommendations for the billing team to fix the claim.\n\n"
        "Validation Status: {status}\n"
        "Validation Errors: {errors}\n"
        "Denial Probability: {prob}%\n"
        "Denial Risk Factors: {reason}\n\n"
        "CRITICAL RULES:\n"
        "1. Base your suggestions strictly on the Validation Errors and Denial Risk Factors provided.\n"
        "2. Do NOT hallucinate errors. For example, if there is no error about an ICD code, do not suggest reviewing the ICD code.\n"
        "3. Provide actionable steps (e.g., 'Obtain prior authorization', 'Verify E/M code level').\n"
        "4. Output ONLY a valid JSON list of strings. Do not use markdown blocks, just the raw JSON list.\n"
        'Example: ["Obtain prior authorization for the consultation.", "Review CPT code for correct E/M level."]'
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "status": val_status,
        "errors": json.dumps(val_errors),
        "prob": round(prob * 100, 1),
        "reason": reason
    })
    
    try:
        suggestions = extract_json(response.content)
        if not isinstance(suggestions, list):
            suggestions = [str(suggestions)]
    except Exception as e:
        logger.error(f"Correction Agent LLM parsing failed: {e}")
        suggestions = ["⚠️ HIGH RISK: Recommend manual review before submission due to validation or prediction warnings."]

    return {
        "correction_suggestions": suggestions,
        **log_action(state, "Correction Agent (AI)", f"Errors: {len(val_errors)} | Risk: {prob:.2f}", suggestions)
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
    
    coding_data = state.get("coding_data", {})
    cpt_codes_list = coding_data.get("cpt_codes", ["Unknown"])
    cpt_val = ", ".join(cpt_codes_list) if isinstance(cpt_codes_list, list) else str(cpt_codes_list)

    final_claim = {
        **state.get("extracted_data", {}),
        **coding_data,
        "cpt_code": cpt_val,
        "status": state.get("validation_status"),
        "risk_score": state.get("denial_probability")
    }

    return {
        "explanation": response.content,
        "final_claim": final_claim,
        **log_action(state, "Explanation Agent", "Finalizing Claim", response.content)
    }
