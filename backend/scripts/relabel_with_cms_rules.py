"""
==============================================================
  CMS-Rule-Based Relabeling Script
  Voice-Driven Revenue Cycle Copilot
==============================================================
  Input  : data/enhanced_health_insurance_claims.csv
  Output : data/labeled_claims.csv

  Strategy:
    Discard the random Kaggle ClaimStatus labels entirely.
    Apply published CMS Medicare / commercial insurance
    denial rules to each row to compute a denial_probability.
    Threshold → binary is_denied label (0 or 1).

  Rules are based on:
    - CMS Medicare Claims Processing Manual (Pub. 100-04)
    - NCCI (National Correct Coding Initiative) edits
    - AHA/AMA billing guidelines
    - Common insurance pre-authorization requirements
==============================================================
"""

import os
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV  = os.path.join(BASE_DIR, "data", "enhanced_health_insurance_claims.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "labeled_claims.csv")

# ── Reproducibility ──────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Thresholds ───────────────────────────────────────────────
DENIAL_THRESHOLD = 0.45   # probability above this → denied
NOISE_STD        = 0.06   # realistic noise so model can't memorize rules perfectly


# ══════════════════════════════════════════════════════════════
#  CMS DENIAL RULES ENGINE
#  Each rule returns a float (0.0–1.0) penalty added to
#  denial_probability. Rules are additive and then clamped.
# ══════════════════════════════════════════════════════════════

def rule_paper_submission_emergency(row) -> float:
    """
    CMS Manual §20.2: Electronic submission strongly preferred.
    Paper claims on Emergency encounters have higher processing
    error rates and are flagged for manual review → higher denial.
    """
    if row["ClaimSubmissionMethod"] == "Paper" and row["ClaimType"] == "Emergency":
        return 0.30
    if row["ClaimSubmissionMethod"] == "Paper" and row["ClaimType"] == "Inpatient":
        return 0.15
    return 0.0


def rule_income_insurance_quality(row) -> float:
    """
    Low income correlates with underinsurance, Medicaid gaps,
    and missing secondary coverage → higher out-of-pocket
    claims get scrutinized more.
    Income < $25k  → very high risk
    Income < $50k  → medium risk
    """
    income = row["PatientIncome"]
    if income < 25_000:
        return 0.35
    elif income < 50_000:
        return 0.18
    return 0.0


def rule_high_claim_amount(row) -> float:
    """
    CMS triggers additional review for high-value claims.
    Claims > $8,000 on Routine encounters are flagged as
    potentially upcoded or medically unnecessary.
    """
    amount = row["ClaimAmount"]
    claim_type = row["ClaimType"]
    if amount > 9_000 and claim_type == "Routine":
        return 0.35
    elif amount > 8_000:
        return 0.20
    elif amount > 6_000 and claim_type == "Routine":
        return 0.12
    return 0.0


def rule_specialty_outpatient_preauth(row) -> float:
    """
    High-cost specialties require pre-authorization for
    outpatient procedures under most commercial payers.
    Cardiology / Neurology outpatient = common denial reason.
    """
    high_cost_specialties = {"Cardiology", "Neurology", "Orthopedics"}
    if (row["ProviderSpecialty"] in high_cost_specialties and
            row["ClaimType"] == "Outpatient"):
        return 0.25
    return 0.0


def rule_employment_status_coverage_gap(row) -> float:
    """
    Unemployed + Student patients are most likely to have
    lapsed or inadequate coverage — common denial trigger.
    Retired patients on fixed incomes may have Medicare gaps.
    """
    status = row["PatientEmploymentStatus"]
    if status in ("Unemployed", "Student"):
        return 0.22
    elif status == "Retired" and row["PatientIncome"] < 40_000:
        return 0.15
    return 0.0


def rule_age_specialty_mismatch(row) -> float:
    """
    NCCI edit: Specialty must be appropriate for patient age.
    Pediatric patients (<12) billed under Cardiology/Orthopedics
    for non-emergency encounters → medically questionable.
    Very elderly (>85) + Pediatrics specialty → coding error.
    """
    age = row["PatientAge"]
    specialty = row["ProviderSpecialty"]

    if age < 12 and specialty in ("Cardiology", "Orthopedics") \
            and row["ClaimType"] not in ("Emergency",):
        return 0.30

    if age > 85 and specialty == "Pediatrics":
        return 0.35   # Pediatrician treating 85yo = likely wrong coding

    return 0.0


def rule_claim_amount_to_income_ratio(row) -> float:
    """
    Very high claim relative to income signals possible balance-
    billing issues or patients without full coverage.
    Ratio > 20% of annual income → high scrutiny.
    """
    if row["PatientIncome"] > 0:
        ratio = row["ClaimAmount"] / row["PatientIncome"]
        if ratio > 0.30:
            return 0.28
        elif ratio > 0.20:
            return 0.15
    return 0.0


def rule_cardiology_emergency_young(row) -> float:
    """
    Young patients (<25) with Cardiology Emergency claims are
    commonly flagged for medical necessity review — insurers
    question whether emergency level of care was required.
    """
    if (row["ProviderSpecialty"] == "Cardiology" and
            row["ClaimType"] == "Emergency" and
            row["PatientAge"] < 25):
        return 0.20
    return 0.0


def rule_neurology_routine_low_income(row) -> float:
    """
    Neurology Routine claims for low-income patients — common
    denial pattern because Medicaid coverage for routine
    neurology is limited in many states.
    """
    if (row["ProviderSpecialty"] == "Neurology" and
            row["ClaimType"] == "Routine" and
            row["PatientIncome"] < 45_000):
        return 0.22
    return 0.0


def rule_inpatient_phone_submission(row) -> float:
    """
    Inpatient claims submitted by phone are highly irregular —
    these require detailed documentation that phone submission
    cannot support. Strong denial signal.
    """
    if row["ClaimType"] == "Inpatient" and row["ClaimSubmissionMethod"] == "Phone":
        return 0.28
    return 0.0


# ══════════════════════════════════════════════════════════════
#  AGGREGATE ALL RULES
# ══════════════════════════════════════════════════════════════

RULES = [
    rule_paper_submission_emergency,
    rule_income_insurance_quality,
    rule_high_claim_amount,
    rule_specialty_outpatient_preauth,
    rule_employment_status_coverage_gap,
    rule_age_specialty_mismatch,
    rule_claim_amount_to_income_ratio,
    rule_cardiology_emergency_young,
    rule_neurology_routine_low_income,
    rule_inpatient_phone_submission,
]


def compute_denial_probability(row) -> float:
    """Apply all rules and sum penalties, clamped to [0, 1]."""
    base_prob = 0.08   # baseline denial rate (~8% prior)
    total = base_prob + sum(rule(row) for rule in RULES)
    return min(total, 1.0)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  📋 CMS Rule-Based Relabeling Pipeline")
    print("=" * 60)

    # 1. Load
    print("\n📂 Loading original dataset...")
    df = pd.read_csv(INPUT_CSV)
    print(f"   Loaded {len(df):,} rows × {df.shape[1]} columns")
    print(f"   Original ClaimStatus (DISCARDING):")
    for s, c in df["ClaimStatus"].value_counts().items():
        print(f"     {s}: {c:,}")

    # 2. Apply rules
    print("\n⚖️  Applying CMS billing rules to each claim...")
    df["denial_probability"] = df.apply(compute_denial_probability, axis=1)

    # 3. Add controlled noise (realistic variance)
    noise = np.random.normal(0, NOISE_STD, size=len(df))
    df["denial_probability"] = (df["denial_probability"] + noise).clip(0, 1)

    # 4. Binarize
    df["is_denied"] = (df["denial_probability"] >= DENIAL_THRESHOLD).astype(int)

    # 5. Stats
    denied  = df["is_denied"].sum()
    approved = len(df) - denied
    denial_rate = denied / len(df) * 100
    print(f"\n📊 Relabeled Distribution:")
    print(f"   Denied   : {denied:,}  ({denial_rate:.1f}%)")
    print(f"   Approved : {approved:,}  ({100 - denial_rate:.1f}%)")
    print(f"   Denial Rate: {denial_rate:.1f}%  (realistic range: 15–35%)")

    # 6. Rule contribution report
    print("\n📋 Rule Contribution Analysis (avg penalty per rule):")
    rule_penalties = {}
    for rule in RULES:
        penalties = df.apply(rule, axis=1)
        triggered = (penalties > 0).sum()
        avg = penalties.mean()
        rule_penalties[rule.__name__] = {"triggered": int(triggered), "avg_penalty": round(float(avg), 4)}
        print(f"   {rule.__name__:<45s} | triggered: {triggered:>4} rows | avg: {avg:.4f}")

    # 7. Save
    # Drop original ClaimStatus to avoid confusion — is_denied is the new target
    df_out = df.drop(columns=["ClaimStatus"])
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n💾 Saved relabeled dataset → {OUTPUT_CSV}")
    print(f"   Columns: {list(df_out.columns)}")
    print("\n✅ Relabeling complete! Run train_model.py next.\n")


if __name__ == "__main__":
    main()
