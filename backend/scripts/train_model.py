"""
==============================================================
  Denial Prediction Model — Training Script
  Voice-Driven Revenue Cycle Copilot
==============================================================
  Dataset  : CMS-Rule-Relabeled Enhanced Health Insurance Claims
  Strategy : Kaggle features + CMS billing rules → meaningful denial labels
  Model    : XGBoost Binary Classifier
  Target   : is_denied → 1 (Denied) vs 0 (Approved)
  Output   : backend/models/denial_model.pkl
             backend/models/label_encoders.pkl
==============================================================
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "labeled_claims.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "denial_model.pkl")
ENC_PATH   = os.path.join(MODEL_DIR, "label_encoders.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_metadata.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  FEATURE CONFIGURATION
# ─────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "PatientGender",
    "ProviderSpecialty",
    "PatientMaritalStatus",
    "PatientEmploymentStatus",
    "ClaimType",
    "ClaimSubmissionMethod",
]

NUMERICAL_FEATURES = [
    "ClaimAmount",
    "PatientAge",
    "PatientIncome",
]

# DiagnosisCode and ProcedureCode are high-cardinality — we'll use frequency encoding
HIGH_CARDINALITY = ["DiagnosisCode", "ProcedureCode"]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + HIGH_CARDINALITY
TARGET       = "ClaimStatus"


# ─────────────────────────────────────────────
#  1. LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   ✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────
#  2. EXPLORE & VALIDATE
# ─────────────────────────────────────────────
def explore_data(df: pd.DataFrame):
    print("\n📊 Dataset Overview:")
    print(f"   Rows:    {len(df):,}")
    print(f"\n   is_denied Distribution (CMS-relabeled):")
    for val, count in df["is_denied"].value_counts().items():
        label = "Denied" if val == 1 else "Approved"
        pct = count / len(df) * 100
        print(f"     {label:10s} ({val}) → {count:>5,} ({pct:.1f}%)")
    print(f"\n   Denial rate : {df['is_denied'].mean()*100:.1f}%")
    print(f"   Missing values: {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────
#  3. PREPROCESS
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    print("\n⚙️  Preprocessing...")
    df = df.copy()

    # --- Target is already set by relabel_with_cms_rules.py ---
    # is_denied column: 1 = Denied, 0 = Approved
    # No mapping needed. Just validate.
    assert "is_denied" in df.columns, "Run relabel_with_cms_rules.py first!"
    df["is_denied"] = df["is_denied"].astype(int)

    denied_count   = df["is_denied"].sum()
    approved_count = len(df) - denied_count
    print(f"   Class balance → Denied: {denied_count:,} | Approved: {approved_count:,} | Denial rate: {denied_count/len(df)*100:.1f}%")

    # --- Frequency Encoding for High-Cardinality Codes ---
    freq_maps = {}
    for col in HIGH_CARDINALITY:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)
        freq_maps[col] = freq.to_dict()
    print(f"   Frequency-encoded: {HIGH_CARDINALITY}")

    # --- Label Encoding for Categoricals ---
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # --- Extract Month from ClaimDate for temporal signal ---
    df["ClaimDate"] = pd.to_datetime(df["ClaimDate"], errors="coerce")
    df["ClaimMonth"] = df["ClaimDate"].dt.month.fillna(0).astype(int)

    print(f"   Label-encoded:     {CATEGORICAL_FEATURES}")

    return df, label_encoders, freq_maps


# ─────────────────────────────────────────────
#  4. BUILD FEATURE MATRIX
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame):
    feature_cols = (
        [col + "_enc"  for col in CATEGORICAL_FEATURES] +
        [col + "_freq" for col in HIGH_CARDINALITY] +
        NUMERICAL_FEATURES +
        ["ClaimMonth"]
    )
    X = df[feature_cols]
    y = df["is_denied"]
    print(f"\n🧩 Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    return X, y, feature_cols


# ─────────────────────────────────────────────
#  5. TRAIN XGBOOST
# ─────────────────────────────────────────────
def train_model(X_train, y_train):
    print("\n🤖 Training XGBoost Classifier...")

    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   scale_pos_weight = {scale_pos_weight:.2f}  (handles class imbalance)")

    model = XGBClassifier(
        n_estimators       = 300,
        max_depth          = 6,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        scale_pos_weight   = scale_pos_weight,
        use_label_encoder  = False,
        eval_metric        = "logloss",
        random_state       = 42,
        n_jobs             = -1,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"   5-Fold CV AUC scores: {cv_scores.round(3)}")
    print(f"   Mean CV AUC:          {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
#  6. EVALUATE
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, feature_cols):
    print("\n📈 Evaluation Results:")

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,  zero_division=0)
    recall    = recall_score(y_test, y_pred,     zero_division=0)
    f1        = f1_score(y_test, y_pred,          zero_division=0)
    auc       = roc_auc_score(y_test, y_pred_prob)

    print(f"   Accuracy  : {acc:.4f}")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    print(f"   F1 Score  : {f1:.4f}")
    print(f"   ROC-AUC   : {auc:.4f}")

    print(f"\n{'─'*50}")
    print("   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Denied", "Denied"]))

    # Feature Importance (Top 10)
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("   Top 10 Feature Importances:")
    print(feat_imp_df.head(10).to_string(index=False))

    return {
        "accuracy":  round(float(acc),       4),
        "precision": round(float(precision),  4),
        "recall":    round(float(recall),     4),
        "f1_score":  round(float(f1),         4),
        "roc_auc":   round(float(auc),        4),
    }


# ─────────────────────────────────────────────
#  7. SAVE ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(model, label_encoders, freq_maps, feature_cols, metrics):
    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save label encoders + freq maps together
    encoder_bundle = {
        "label_encoders":      label_encoders,
        "freq_maps":           freq_maps,
        "feature_cols":        feature_cols,
        "categorical_features": CATEGORICAL_FEATURES,
        "high_cardinality":    HIGH_CARDINALITY,
        "numerical_features":  NUMERICAL_FEATURES,
    }
    with open(ENC_PATH, "wb") as f:
        pickle.dump(encoder_bundle, f)

    # Save metadata/metrics as JSON for the API to serve
    meta = {
        "model_type":    "XGBoostClassifier",
        "target":        "is_denied (1=Denied, 0=Not Denied)",
        "features":      feature_cols,
        "num_features":  len(feature_cols),
        "metrics":       metrics,
        "training_note": "Trained on Kaggle Enhanced Health Insurance Claims Dataset, relabeled using CMS Medicare billing rules (10 domain rules). Binary: is_denied=1 (Denied), is_denied=0 (Approved).",
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n💾 Saved artifacts:")
    print(f"   Model   → {MODEL_PATH}")
    print(f"   Encoders→ {ENC_PATH}")
    print(f"   Metadata→ {META_PATH}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🏥 Denial Prediction Model — Training Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_data()

    # 2. Explore
    explore_data(df)

    # 3. Preprocess
    df, label_encoders, freq_maps = preprocess(df)

    # 4. Feature matrix
    X, y, feature_cols = build_features(df)

    # 5. Split  (80% train, 20% test — stratified on the target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # 6. Train
    model = train_model(X_train, y_train)

    # 7. Evaluate
    metrics = evaluate_model(model, X_test, y_test, feature_cols)

    # 8. Save
    save_artifacts(model, label_encoders, freq_maps, feature_cols, metrics)

    print("\n✅ Training complete!")
    print(f"   ROC-AUC achieved: {metrics['roc_auc']}")
    print("   Model is ready to serve predictions via the Prediction Agent.\n")


if __name__ == "__main__":
    main()
