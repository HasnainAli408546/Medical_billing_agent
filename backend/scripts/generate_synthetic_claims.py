import csv
import random
import os
from faker import Faker

fake = Faker()

# Data configuration
NUM_RECORDS = 300
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_claims.csv")

# Medical mappings (tied to our knowledge base domains)
DIAGNOSES = [
    ("Hypertension", "I10", "93000"),
    ("Type 2 Diabetes", "E11.9", "83036"),
    ("Acute Bronchitis", "J20.9", "71045"),
    ("Abdominal Pain", "R10.4", "74176"),
    ("Lower Back Pain", "M54.5", "97161")
]

INSURANCE_PROVIDERS = ["Medicare", "BlueCross", "UnitedHealthcare", "Medicaid", "Aetna", "None"]

def generate_claim():
    # Basic info
    patient_id = fake.uuid4()
    patient_age = random.randint(18, 90)
    insurance = random.choice(INSURANCE_PROVIDERS)
    
    # Medical info
    diagnosis_name, icd_code, cpt_code = random.choice(DIAGNOSES)
    
    # Introduce intentional "errors" or risky parameters for the ML model to learn from
    # 1. Missing Authorization
    has_authorization = random.choices([True, False], weights=[0.7, 0.3])[0]
    
    # 2. Risk Engine Logic (This explicitly trains the ML model on patterns)
    denial_probability_base = 0.05
    
    # Penalize "None" insurance heavily
    if insurance == "None":
         denial_probability_base += 0.80
         
    # Penalize missing authorization for expensive procedures (CT Scans)
    if not has_authorization and cpt_code == "74176":
        denial_probability_base += 0.60
        
    # Medicare strict on HbA1c frequency (randomly simulate it here)
    if insurance == "Medicare" and cpt_code == "83036" and random.random() < 0.2:
        denial_probability_base += 0.50
        
    # Introduce small random noise
    denial_probability_base += random.uniform(0.0, 0.1)
    
    is_denied = 1 if denial_probability_base > 0.6 else 0
    
    # Assign readable denial reasons
    reason = "Approved"
    if is_denied:
        if insurance == "None":
            reason = "Missing Coverage Data"
        elif not has_authorization:
            reason = "Pre-authorization Required"
        else:
            reason = "Exceeds Frequency Limit or Invalid Coding"

    return {
        "patient_id": patient_id,
        "age": patient_age,
        "insurance_type": insurance,
        "diagnosis": diagnosis_name,
        "icd_code": icd_code,
        "cpt_code": cpt_code,
        "has_authorization": has_authorization,
        "is_denied": is_denied,
        "denial_reason": reason
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    claims = [generate_claim() for _ in range(NUM_RECORDS)]
    
    headers = claims[0].keys()
    
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(claims)
        
    print(f"✅ Successfully generated {NUM_RECORDS} synthetic medical claims at {OUTPUT_FILE}")
    print("These claims contain deterministic rules for the ML model to pick up on (e.g. Missing authorization on CT Scans = Denial).")

if __name__ == "__main__":
    main()
