import os
import sys
from datetime import datetime, timedelta
import random

# Ensure we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import SessionLocal
from app.db.models import Patient, Claim, Denial, AgentLog

def seed_db():
    db = SessionLocal()

    # Clear existing data just in case this is run multiple times
    print("Clearing existing data...")
    db.query(AgentLog).delete()
    db.query(Denial).delete()
    db.query(Claim).delete()
    db.query(Patient).delete()
    db.commit()

    print("Generating realistic patients...")
    patients_data = [
        {"name": "James Sullivan", "age": 45, "insurance_provider": "BlueCross"},
        {"name": "Maria Garcia", "age": 62, "insurance_provider": "Medicare"},
        {"name": "Robert Chen", "age": 34, "insurance_provider": "Aetna"},
        {"name": "Sarah Jenkins", "age": 28, "insurance_provider": "Cigna"},
        {"name": "William Ford", "age": 75, "insurance_provider": "Medicare"},
        {"name": "Linda Nguyen", "age": 51, "insurance_provider": "UnitedHealthcare"},
        {"name": "David Smith", "age": 42, "insurance_provider": "BlueCross"},
        {"name": "Emma Thompson", "age": 22, "insurance_provider": "Aetna"}
    ]

    patient_objs = []
    for p in patients_data:
        pat = Patient(**p)
        db.add(pat)
        patient_objs.append(pat)
    db.commit()

    print("Generating realistic claims based on CMS rules...")
    # Scenarios mapping closely to our knowledge base and real-world billing
    scenarios = [
        {
            "diagnosis": "Essential Hypertension", "procedure": "Electrocardiogram (ECG)",
            "icd": "I10", "cpt": "93000", "status": "valid", "prob": 0.25, "reason": "Low Risk: ECG routinely justified if symptoms documented."
        },
        {
            "diagnosis": "Type 2 Diabetes", "procedure": "HbA1c test",
            "icd": "E11.9", "cpt": "83036", "status": "valid", "prob": 0.15, "reason": "Low Risk: Standard frequency monitoring."
        },
        {
            "diagnosis": "Type 2 Diabetes", "procedure": "HbA1c test",
            "icd": "E11.9", "cpt": "83036", "status": "invalid", "prob": 0.88, "reason": "High Risk: High frequency billing > 4 times per year without documentation."
        },
        {
            "diagnosis": "Acute Bronchitis", "procedure": "Chest X-Ray",
            "icd": "J20.9", "cpt": "71045", "status": "invalid", "prob": 0.92, "reason": "High Risk: Routine X-ray for bronchitis without pneumonia indicators is often denied."
        },
        {
            "diagnosis": "Unspecified Abdominal Pain", "procedure": "CT Scan Abdomen",
            "icd": "R10.4", "cpt": "74176", "status": "invalid", "prob": 0.78, "reason": "High Risk: Requires documented failure of conservative treatment. Pre-auth missing."
        },
        {
            "diagnosis": "Lower Back Pain", "procedure": "Physical Therapy Evaluation",
            "icd": "M54.5", "cpt": "97161", "status": "valid", "prob": 0.12, "reason": "Low Risk: First 10 visits automatically covered."
        },
        {
            "diagnosis": "Major Depressive Disorder", "procedure": "Psychotherapy, 60 min",
            "icd": "F32.1", "cpt": "90837", "status": "valid", "prob": 0.22, "reason": "Low Risk: Under 20 sessions limit per year."
        },
        {
            "diagnosis": "Primary osteoarthritis, right knee", "procedure": "Total Knee Arthroplasty",
            "icd": "M17.11", "cpt": "27447", "status": "draft", "prob": 0.65, "reason": "Medium Risk: Surgery requires well-documented conservative treatment failure."
        },
        {
            "diagnosis": "Encounter for immunization", "procedure": "Influenza vaccine",
            "icd": "Z23", "cpt": "90686", "status": "valid", "prob": 0.05, "reason": "Very Low Risk: Preventive care, fully covered."
        },
        {
            "diagnosis": "Chest pain, unspecified", "procedure": "Emergency Department visit, high complexity",
            "icd": "R07.9", "cpt": "99285", "status": "invalid", "prob": 0.72, "reason": "Medium-High Risk: High MDM criteria not fully supported by extracted text logic."
        },
        {
            "diagnosis": "Obstructive Sleep Apnea", "procedure": "CPAP device",
            "icd": "G47.33", "cpt": "E0601", "status": "draft", "prob": 0.55, "reason": "Medium Risk: Requires 90-day compliance documentation."
        },
        {
            "diagnosis": "General explicit adult checkup", "procedure": "Preventive visit",
            "icd": "Z00.00", "cpt": "99395", "status": "valid", "prob": 0.10, "reason": "Low Risk: Standard annual wellness cover."
        }
    ]

    base_date = datetime.utcnow()
    # Generate 25 claims spread across the last 14 days
    for i in range(25):
        pat = random.choice(patient_objs)
        scn = random.choice(scenarios)
        
        # Slight jitter to probabilities
        prob = max(0.01, min(0.99, scn["prob"] + random.uniform(-0.08, 0.08)))
        
        days_ago = random.randint(0, 14)
        hours_ago = random.randint(0, 23)
        created_at = base_date - timedelta(days=days_ago, hours=hours_ago)

        claim = Claim(
            patient_id=pat.id,
            diagnosis=scn["diagnosis"],
            procedure=scn["procedure"],
            icd_code=scn["icd"],
            cpt_code=scn["cpt"],
            status=scn["status"],
            created_at=created_at
        )
        db.add(claim)
        db.flush()

        denial = Denial(
            claim_id=claim.id,
            reason=scn["reason"],
            probability=prob,
            corrected=False,
            created_at=created_at
        )
        db.add(denial)

        # Generate fake agent logs to make the trace interface light up
        agents = ["Intent Agent", "Extraction Agent", "Coding Agent", "Validation Agent (RAG)", "Prediction Agent (ML)", "Correction Agent"]
        for idx, agent in enumerate(agents):
            log = AgentLog(
                claim_id=claim.id,
                agent_name=agent,
                input_data={"system_step": f"Entering {agent}"},
                output_data={"result": f"Processed logic for {agent}", "confidence": round(random.uniform(0.8, 1.0), 2)},
                timestamp=created_at + timedelta(seconds=idx * 2)
            )
            db.add(log)

    db.commit()
    print("✅ Database successfully seeded with realistic CMS claims, denials, and agent traces!")
    db.close()

if __name__ == "__main__":
    seed_db()
