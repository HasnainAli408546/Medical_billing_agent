"""
==============================================================
  ML Service — Denial Prediction
  Voice-Driven Revenue Cycle Copilot
==============================================================
  Loads the trained XGBoost model and label encoders once
  at startup (singleton pattern) and exposes a predict()
  function for the Prediction Agent in nodes.py.
==============================================================
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "denial_model.pkl")
ENC_PATH   = os.path.join(BASE_DIR, "models", "label_encoders.pkl")

# ── Singleton Model State ──────────────────────────────────────
_model          = None
_encoder_bundle = None


def _load_artifacts():
    """Load model + encoders from disk once. Cached in module-level globals."""
    global _model, _encoder_bundle

    if _model is not None:
        return  # Already loaded

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run: python scripts/relabel_with_cms_rules.py && python scripts/train_model.py"
        )

    logger.info("🤖 Loading XGBoost denial prediction model...")
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    with open(ENC_PATH, "rb") as f:
        _encoder_bundle = pickle.load(f)

    logger.info(f"✅ Model loaded. Features: {_encoder_bundle['feature_cols']}")


def _build_feature_row(claim: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a raw claim dict into a feature DataFrame the model can predict on.

    Expected claim keys (all optional — defaults applied for missing):
        patient_age, patient_gender, patient_income, patient_marital_status,
        patient_employment_status, claim_amount, claim_type, claim_submission_method,
        provider_specialty, diagnosis_code, procedure_code, claim_month
    """
    bundle = _encoder_bundle

    # ── Raw values with safe defaults ────────────────────────────
    age               = float(claim.get("patient_age", 35))
    income            = float(claim.get("patient_income", 50_000))
    amount            = float(claim.get("claim_amount", 3_000))
    claim_month       = int(claim.get("claim_month", 6))

    gender            = str(claim.get("patient_gender", "M"))
    specialty         = str(claim.get("provider_specialty", "General"))
    marital_status    = str(claim.get("patient_marital_status", "Single"))
    employment_status = str(claim.get("patient_employment_status", "Employed"))
    claim_type        = str(claim.get("claim_type", "Outpatient"))
    submission_method = str(claim.get("claim_submission_method", "Online"))
    diagnosis_code    = str(claim.get("diagnosis_code", "Z00.00"))
    procedure_code    = str(claim.get("procedure_code", "99213"))

    # ── Label Encode categoricals ─────────────────────────────────
    encoders = bundle["label_encoders"]

    def safe_encode(encoder, value):
        """Handle unseen labels gracefully."""
        try:
            return encoder.transform([value])[0]
        except ValueError:
            # Unseen label → use mode class (0)
            return 0

    gender_enc     = safe_encode(encoders["PatientGender"],            gender)
    specialty_enc  = safe_encode(encoders["ProviderSpecialty"],        specialty)
    marital_enc    = safe_encode(encoders["PatientMaritalStatus"],     marital_status)
    employ_enc     = safe_encode(encoders["PatientEmploymentStatus"],  employment_status)
    type_enc       = safe_encode(encoders["ClaimType"],                claim_type)
    method_enc     = safe_encode(encoders["ClaimSubmissionMethod"],    submission_method)

    # ── Frequency Encode high-cardinality codes ───────────────────
    freq_maps = bundle["freq_maps"]
    diag_freq = freq_maps["DiagnosisCode"].get(diagnosis_code, 0.0001)
    proc_freq = freq_maps["ProcedureCode"].get(procedure_code, 0.0001)

    # ── Assemble feature vector (must match training order) ────────
    feature_cols = bundle["feature_cols"]
    values = {
        "PatientGender_enc":            gender_enc,
        "ProviderSpecialty_enc":        specialty_enc,
        "PatientMaritalStatus_enc":     marital_enc,
        "PatientEmploymentStatus_enc":  employ_enc,
        "ClaimType_enc":                type_enc,
        "ClaimSubmissionMethod_enc":    method_enc,
        "DiagnosisCode_freq":           diag_freq,
        "ProcedureCode_freq":           proc_freq,
        "ClaimAmount":                  amount,
        "PatientAge":                   age,
        "PatientIncome":                income,
        "ClaimMonth":                   claim_month,
    }

    row_df = pd.DataFrame([[values[col] for col in feature_cols]], columns=feature_cols)
    return row_df


def _get_risk_tier(prob: float) -> str:
    """Map probability to human-readable risk tier."""
    if prob >= 0.75:
        return "HIGH"
    elif prob >= 0.45:
        return "MEDIUM"
    else:
        return "LOW"


def _get_denial_reason(claim: Dict[str, Any], prob: float) -> str:
    """Generate a human-readable denial reason based on top risk factors."""
    reasons = []

    employment = claim.get("patient_employment_status", "Employed")
    if employment in ("Unemployed", "Student"):
        reasons.append(f"Patient employment status ({employment}) suggests potential coverage gap")

    claim_type = claim.get("claim_type", "")
    method = claim.get("claim_submission_method", "")
    if method == "Paper" and claim_type == "Emergency":
        reasons.append("Paper submission on Emergency claim — high processing error risk")

    income = float(claim.get("patient_income", 99999))
    if income < 25_000:
        reasons.append(f"Low patient income (${income:,.0f}) — likely underinsured or Medicaid gap")
    elif income < 50_000:
        reasons.append(f"Below-average income (${income:,.0f}) — partial coverage risk")

    amount = float(claim.get("claim_amount", 0))
    if amount > 8_000 and claim_type == "Routine":
        reasons.append(f"High claim amount (${amount:,.0f}) on Routine visit — medical necessity review likely")

    specialty = claim.get("provider_specialty", "")
    if specialty in ("Cardiology", "Neurology", "Orthopedics") and claim_type == "Outpatient":
        reasons.append(f"{specialty} outpatient claim likely requires pre-authorization")

    if method == "Phone" and claim_type == "Inpatient":
        reasons.append("Inpatient claim submitted by phone — insufficient documentation channel")

    if not reasons:
        if prob >= 0.45:
            return "Claim flagged for review — combination of risk factors detected."
        return "No major denial risk factors identified."

    return " | ".join(reasons)


# ══════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════

def predict_denial(claim: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction function called by the Prediction Agent.

    Args:
        claim: dict with claim fields (snake_case keys)

    Returns:
        {
            "denial_probability": float,   # 0.0 – 1.0
            "is_denied": bool,
            "risk_tier": str,              # LOW / MEDIUM / HIGH
            "denial_reason": str,
            "model_used": str
        }
    """
    _load_artifacts()

    try:
        row_df = _build_feature_row(claim)
        prob   = float(_model.predict_proba(row_df)[0][1])
        prob   = round(prob, 4)
    except Exception as e:
        logger.error(f"ML prediction failed: {e}")
        # Graceful fallback — rule-based estimate
        income = float(claim.get("patient_income", 99999))
        amount = float(claim.get("claim_amount", 0))
        prob   = 0.65 if (income < 30_000 or amount > 8_000) else 0.20

    risk_tier     = _get_risk_tier(prob)
    denial_reason = _get_denial_reason(claim, prob)

    return {
        "denial_probability": prob,
        "is_denied":          prob >= 0.45,
        "risk_tier":          risk_tier,
        "denial_reason":      denial_reason,
        "model_used":         "XGBoost (CMS-rule trained)",
    }
