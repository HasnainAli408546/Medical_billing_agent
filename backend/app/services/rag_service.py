"""
==============================================================
  RAG Validation Service
  Voice-Driven Revenue Cycle Copilot
==============================================================
  Full RAG pipeline for billing rule validation:

  STEP 1 — RETRIEVAL:
    Encode the incoming claim as a query vector.
    Search FAISS index → top-K most relevant billing rules.

  STEP 2 — AUGMENTATION:
    Format retrieved rules + claim data as a structured prompt.

  STEP 3 — GENERATION:
    LLM (Groq LLaMA-3) reads the rules and adjudicates
    the claim → returns VALID / NEEDS_REVIEW / INVALID
    with specific errors and recommendations.
==============================================================
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_DIR  = os.path.join(BASE_DIR, "models", "faiss_index")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH  = os.path.join(INDEX_DIR, "metadata.json")

# ── Singleton State ────────────────────────────────────────────
_rules_index    = None
_rules_metadata = None
_med_index      = None
_med_metadata   = None
_embedder       = None

# New Paths for Medical Reference
MEDICAL_INDEX_DIR  = os.path.join(BASE_DIR, "models", "medical_index")
MEDICAL_INDEX_PATH = os.path.join(MEDICAL_INDEX_DIR, "index.faiss")
MEDICAL_META_PATH  = os.path.join(MEDICAL_INDEX_DIR, "metadata.json")


# ══════════════════════════════════════════════════════════════
#  1. LOAD ARTIFACTS (once at startup)
# ══════════════════════════════════════════════════════════════

def _load_rag_artifacts():
    """Load FAISS indices + metadata + embedding model once."""
    global _rules_index, _rules_metadata, _med_index, _med_metadata, _embedder

    if _rules_index is not None and _med_index is not None and _embedder is not None:
        return  # Already loaded

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install faiss-cpu sentence-transformers")

    # 1. Load Billing Rules Index
    if os.path.exists(INDEX_PATH):
        logger.info("🔍 Loading Billing Rules FAISS index...")
        _rules_index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r") as f:
            _rules_metadata = json.load(f)
    else:
        logger.warning(f"⚠️ Billing Rules FAISS index not found at {INDEX_PATH}")

    # 2. Load Medical Medical Reference Index
    if os.path.exists(MEDICAL_INDEX_PATH):
        logger.info("🔍 Loading Medical Reference FAISS index...")
        _med_index = faiss.read_index(MEDICAL_INDEX_PATH)
        with open(MEDICAL_META_PATH, "r") as f:
            _med_metadata = json.load(f)
    else:
        logger.warning(f"⚠️ Medical Reference FAISS index not found at {MEDICAL_INDEX_PATH}")

    # 3. Load Embedder
    if _embedder is None:
        embed_model = "all-MiniLM-L6-v2"
        hf_token = os.getenv("HF_TOKEN")
        _embedder   = SentenceTransformer(embed_model, token=hf_token)
    
    logger.info("✅ RAG services ready")


# ══════════════════════════════════════════════════════════════
#  2. RETRIEVAL — Find most relevant billing rules
# ══════════════════════════════════════════════════════════════

def _retrieve_rules(query: str, top_k: int = 3) -> List[Dict]:
    """
    Embed the query and search FAISS for the top_k most
    semantically similar billing rules.
    """
    if _rules_index is None:
        return []

    query_vec = _embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    scores, indices = _rules_index.search(query_vec, k=top_k)
    rules = _rules_metadata["rules"]

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(rules):
            rule = rules[idx].copy()
            rule["_similarity_score"] = round(float(score), 4)
            results.append(rule)

    return results


def _build_query(claim: Dict[str, Any]) -> str:
    """Build semantic search query from claim fields."""
    parts = []
    if claim.get("icd_code"):
        parts.append(f"ICD {claim['icd_code']}")
    if claim.get("icd_description"):
        parts.append(claim["icd_description"])
    if claim.get("cpt_code"):
        parts.append(f"CPT {claim['cpt_code']}")
    if claim.get("cpt_description"):
        parts.append(claim["cpt_description"])
    if claim.get("specialty"):
        parts.append(claim["specialty"])
    if claim.get("claim_type"):
        parts.append(f"{claim['claim_type']} claim")
    return " ".join(parts) if parts else "general billing validation"


# ══════════════════════════════════════════════════════════════
#  3. AUGMENTATION + GENERATION — LLM adjudication
# ══════════════════════════════════════════════════════════════

VALIDATION_PROMPT = """You are an expert healthcare billing compliance officer.

A medical claim has been submitted. Review it against the retrieved billing rules below
and determine if it is VALID, NEEDS_REVIEW, or INVALID.

═══════════════════════════════════════════════
SUBMITTED CLAIM
═══════════════════════════════════════════════
Diagnosis (ICD):   {icd_code} — {icd_description}
Procedure (CPT):   {cpt_code} — {cpt_description}
Patient Age:       {patient_age}
Claim Type:        {claim_type}
Submission Method: {submission_method}
Specialty:         {specialty}
Claim Amount:      ${claim_amount}
Prior Auth:        {prior_auth}
Raw Clinical Note: {clinical_note}

═══════════════════════════════════════════════
RETRIEVED BILLING RULES (most relevant)
═══════════════════════════════════════════════
{rules_text}

═══════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════
Based ONLY on the retrieved rules above, adjudicate this claim.
Return a JSON object with this EXACT structure:

{{
  "status": "VALID" | "NEEDS_REVIEW" | "INVALID",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "errors": ["list of specific billing rule violations found"],
  "warnings": ["list of items that need documentation or may cause denial"],
  "recommendation": "one clear actionable recommendation for the billing team",
  "matched_rule_id": "RULE-XXX or null if no close match found"
}}

Rules for status:
- VALID: claim meets all billing requirements from the retrieved rules
- NEEDS_REVIEW: claim is potentially valid but requires documentation or prior auth
- INVALID: claim clearly violates a billing rule (wrong code pair, frequency exceeded, etc.)

Return ONLY the JSON object, no other text."""


def _format_rules_text(rules: List[Dict]) -> str:
    """Format retrieved rules into readable text for the LLM prompt."""
    lines = []
    for i, rule in enumerate(rules, 1):
        lines.append(f"Rule {i} [{rule['id']}] (Similarity: {rule['_similarity_score']:.3f})")
        lines.append(f"  Diagnosis: {rule['icd_description']} ({rule['icd_code']})")
        lines.append(f"  Procedure: {rule['cpt_description']} ({rule['cpt_code']})")
        lines.append(f"  Rule: {rule['billing_rule']}")
        lines.append(f"  Prior Auth Required: {rule.get('requires_prior_auth', 'Unknown')}")
        lines.append(f"  Frequency Limit: {rule.get('frequency_limit', 'Not specified')}")
        lines.append(f"  Denial Risk: {rule.get('denial_risk', 'Unknown')}")
        lines.append("")
    return "\n".join(lines)


def _call_llm_for_validation(prompt: str) -> Dict:
    """Call the LLM and parse structured JSON response."""
    try:
        from app.agents.llm_setup import get_llm
        llm  = get_llm()
        resp = llm.invoke(prompt)
        text = resp.content.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        return json.loads(text)

    except json.JSONDecodeError as e:
        logger.warning(f"LLM returned non-JSON response: {e}")
        return {
            "status":          "NEEDS_REVIEW",
            "confidence":      "LOW",
            "errors":          [],
            "warnings":        ["LLM validation response could not be parsed — manual review recommended."],
            "recommendation":  "Verify claim manually with billing team.",
            "matched_rule_id": None,
        }
    except Exception as e:
        logger.error(f"LLM validation call failed: {e}")
        return {
            "status":          "NEEDS_REVIEW",
            "confidence":      "LOW",
            "errors":          [],
            "warnings":        [f"Validation service error: {str(e)}"],
            "recommendation":  "Manual review required.",
            "matched_rule_id": None,
        }


# ══════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════

def validate_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full RAG validation pipeline for a billing claim.

    Args:
        claim: dict with claim fields:
            icd_code, icd_description, cpt_code, cpt_description,
            patient_age, claim_type, submission_method, specialty,
            claim_amount, prior_auth

    Returns:
        {
            "status":           str,   # VALID / NEEDS_REVIEW / INVALID
            "confidence":       str,   # HIGH / MEDIUM / LOW
            "errors":           list,
            "warnings":         list,
            "recommendation":   str,
            "matched_rule_id":  str,
            "retrieved_rules":  list,  # top-K rules retrieved
            "pipeline":         str    # "rag" or "fallback"
        }
    """
    _load_rag_artifacts()

    # ── STEP 1: RETRIEVAL ─────────────────────────────────
    query          = _build_query(claim)
    retrieved_rules = _retrieve_rules(query, top_k=3)
    logger.info(f"RAG retrieved {len(retrieved_rules)} rules for query: '{query[:60]}...'")

    # ── STEP 2: AUGMENTATION ──────────────────────────────
    rules_text = _format_rules_text(retrieved_rules)
    prompt     = VALIDATION_PROMPT.format(
        icd_code        = claim.get("icd_code",        "Unknown"),
        icd_description = claim.get("icd_description", "Unknown"),
        cpt_code        = claim.get("cpt_code",        "Unknown"),
        cpt_description = claim.get("cpt_description", "Unknown"),
        patient_age     = claim.get("patient_age",     "Unknown"),
        claim_type      = claim.get("claim_type",      "Unknown"),
        submission_method = claim.get("submission_method", "Unknown"),
        specialty       = claim.get("specialty",        "Unknown"),
        claim_amount    = claim.get("claim_amount",     0),
        prior_auth      = claim.get("prior_auth",      "Not specified"),
        clinical_note   = claim.get("clinical_note",   "None provided"),
        rules_text      = rules_text,
    )

    # ── STEP 3: GENERATION ────────────────────────────────
    result = _call_llm_for_validation(prompt)

    # Add pipeline metadata
    result["retrieved_rules"] = [
        {
            "rule_id":     r["id"],
            "icd":         f"{r['icd_code']} — {r['icd_description']}",
            "cpt":         f"{r['cpt_code']} — {r['cpt_description']}",
            "denial_risk": r.get("denial_risk", "Unknown"),
            "similarity":  r["_similarity_score"],
        }
        for r in retrieved_rules
    ]
    result["pipeline"] = "rag"

    logger.info(f"RAG validation complete: status={result['status']}, confidence={result.get('confidence')}")
    return result


def validate_claim_fallback(icd_code: str, cpt_code: str) -> Dict[str, Any]:
    """
    Rule-based fallback when FAISS index is not available.
    Used during development before build_faiss_index.py is run.
    """
    status = "NEEDS_REVIEW"
    errors = []

    if icd_code in ("UNKNOWN", "Unknown", ""):
        status = "INVALID"
        errors.append("Missing or unknown ICD-10 diagnosis code.")

    if cpt_code in ("UNKNOWN", "Unknown", ""):
        status = "INVALID"
        errors.append("Missing or unknown CPT procedure code.")

    return {
        "status":           status,
        "confidence":       "LOW",
        "errors":           errors,
        "warnings":         ["RAG validation unavailable — using rule-based fallback."],
        "recommendation":   "Run build_faiss_index.py to enable full RAG validation.",
        "matched_rule_id":  None,
        "retrieved_rules":  [],
        "pipeline":         "fallback",
    }


def search_codes(query: str, code_type: str = "ICD", top_k: int = 5) -> List[Dict]:
    """
    Search for professional medical codes (ICD or CPT) using semantic search.
    
    Args:
        query: The clinical description (e.g. "lung infection")
        code_type: "ICD" or "CPT"
        top_k: Number of results to return
    """
    _load_rag_artifacts()
    
    if _med_index is None:
        logger.warning("Medical index not found, cannot search codes.")
        return []
        
    query_vec = _embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    
    # We search more items and then filter by code_type
    # code_type mapping: "ICD" -> "I", "CPT" -> "C"
    target_type = "I" if code_type == "ICD" else "C"
    
    scores, indices = _med_index.search(query_vec, k=top_k * 5)
    all_metadata = _med_metadata["m"]
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(all_metadata):
            meta = all_metadata[idx]
            if meta["t"] == target_type:
                results.append({
                    "code": meta["c"],
                    "description": meta["d"],
                    "_similarity_score": float(score)
                })
                if len(results) >= top_k:
                    break
                    
    return results
