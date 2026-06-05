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
_explainer      = None


def _load_artifacts():
    """Load model + encoders from disk once. Cached in module-level globals."""
    global _model, _encoder_bundle, _explainer

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
    
    try:
        import shap
        _explainer = shap.TreeExplainer(_model)
        logger.info("✅ SHAP Explainer initialized.")
    except Exception as e:
        logger.warning(f"⚠️ SHAP Explainer failed to initialize: {e}")


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
    # Use `or` to coalesce None → default (LLM may return null for missing fields)
    age               = float(claim.get("patient_age") or 35)
    income            = float(claim.get("patient_income") or 50_000)
    amount            = float(claim.get("claim_amount") or 3_000)
    claim_month       = int(claim.get("claim_month") or 6)

    gender            = str(claim.get("patient_gender") or "M")
    specialty         = str(claim.get("provider_specialty") or "General")
    marital_status    = str(claim.get("patient_marital_status") or "Single")
    employment_status = str(claim.get("patient_employment_status") or "Employed")
    claim_type        = str(claim.get("claim_type") or "Outpatient")
    submission_method = str(claim.get("claim_submission_method") or "Online")
    diagnosis_code    = str(claim.get("diagnosis_code") or "Z00.00")
    procedure_code    = str(claim.get("procedure_code") or "99213")

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


def _get_denial_reason(claim: Dict[str, Any], prob: float, row_df: pd.DataFrame = None) -> str:
    """Generate a human-readable denial reason based on top risk factors from SHAP."""
    global _explainer
    if row_df is None or _explainer is None:
        if prob >= 0.45:
            return "Claim flagged for review — combination of risk factors detected."
        return "No major denial risk factors identified."
        
    try:
        # Calculate SHAP values for this specific claim
        shap_values = _explainer.shap_values(row_df)
        # XGBoost binary classification usually returns a single array or list. Safely extract:
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        
        feature_names = row_df.columns.tolist()
        feature_impacts = []
        
        for i, feat_name in enumerate(feature_names):
            impact = sv[i]
            if impact > 0: # Only care about features that INCREASED the denial risk
                pretty_name = feat_name.replace('_enc', '').replace('_freq', '')
                
                # Fetch original human-readable value from the claim dict
                mapping = {
                    "PatientGender": "patient_gender",
                    "ProviderSpecialty": "provider_specialty",
                    "PatientMaritalStatus": "patient_marital_status",
                    "PatientEmploymentStatus": "patient_employment_status",
                    "ClaimType": "claim_type",
                    "ClaimSubmissionMethod": "claim_submission_method",
                    "DiagnosisCode": "diagnosis_code",
                    "ProcedureCode": "procedure_code",
                    "ClaimAmount": "claim_amount",
                    "PatientAge": "patient_age",
                    "PatientIncome": "patient_income",
                    "ClaimMonth": "claim_month"
                }
                
                orig_key = mapping.get(pretty_name)
                val_str = str(claim.get(orig_key, row_df.iloc[0][feat_name]))
                
                # Format currency nicely
                if pretty_name in ["ClaimAmount", "PatientIncome"] and orig_key in claim:
                    try:
                        val_str = f"${float(claim[orig_key]):,.0f}"
                    except:
                        pass
                    
                feature_impacts.append((pretty_name, val_str, impact))
                
        # Sort by highest mathematical impact on the model
        feature_impacts.sort(key=lambda x: x[2], reverse=True)
        top_factors = feature_impacts[:3]
        
        if prob >= 0.45 and top_factors:
            factor_strings = [f"{name} ({val})" for name, val, _ in top_factors]
            return f"High risk flagged by AI based on: {', '.join(factor_strings)}."
        elif prob >= 0.45:
            return "Claim flagged for review — combination of risk factors detected."
        else:
            return "No major denial risk factors identified."
            
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return "Claim flagged for review — combination of risk factors detected."


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
        row_df = None

    risk_tier     = _get_risk_tier(prob)
    denial_reason = _get_denial_reason(claim, prob, row_df)

    return {
        "denial_probability": prob,
        "is_denied":          prob >= 0.45,
        "risk_tier":          risk_tier,
        "denial_reason":      denial_reason,
        "model_used":         "XGBoost (CMS-rule trained)",
    }
