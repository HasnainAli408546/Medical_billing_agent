import os
import urllib.request
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR  = os.path.join(BASE_DIR, "data", "reference")
os.makedirs(REF_DIR, exist_ok=True)

# ── URLs (Verified Real Datasets) ─────────────────────────────
ICD_URL = "https://raw.githubusercontent.com/Bobrovskiy/ICD-10-CSV/master/2020/diagnosis.csv"
CPT_URL = "https://raw.githubusercontent.com/SoftmedTanzania/tanzania-openhim-hdr/master/cpt.csv"

def download_file(url, filename):
    path = os.path.join(REF_DIR, filename)
    print(f"📥 Downloading {filename} from {url}...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"✅ Saved to {path}")
        return path
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return None

def process_icd(path):
    print("🧹 Cleaning ICD-10 data...")
    # The source file has: Id, Code, CodeWithSeparator, ShortDescription, LongDescription, HippaCovered, Deleted
    df = pd.read_csv(path)
    # Standardize to our format: code, description, specialty
    df = df.rename(columns={"Code": "code", "LongDescription": "description"})
    df["specialty"] = "General" 
    df = df[["code", "description", "specialty"]]
    output_path = os.path.join(REF_DIR, "icd10_reference.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Processed {len(df)} ICD-10 codes.")

def process_cpt(path):
    print("🧹 Cleaning CPT data...")
    # The SoftmedTanzania file has: CATEGORY, CONSULATIONS AND GENERAL SERVICE, Unnamed: 2, ...
    df = pd.read_csv(path)
    
    # Identify code and description columns by position since headers are non-standard
    # Column 0: Code, Column 1: Description
    df = df.iloc[:, [0, 1]]
    df.columns = ["code", "description"]
    
    # Filter out rows that are actually headers or subcategories (non-numeric codes usually)
    # We want rows where the first column looks like a CPT code (5 characters, or alphanumeric)
    def is_valid_code(val):
        val = str(val).strip()
        if not val or val.lower() == "nan" or "subcategory" in val.lower() or "category" in val.lower():
            return False
        return True

    df = df[df["code"].apply(is_valid_code)]
    df["category"] = "Specialty"
    
    output_path = os.path.join(REF_DIR, "cpt_reference.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned and processed {len(df)} valid CPT codes.")

if __name__ == "__main__":
    print("🚀 Pulling Production-Grade Medical Code Sets...")
    
    icd_path = download_file(ICD_URL, "raw_icd.csv")
    if icd_path:
        process_icd(icd_path)
        
    cpt_path = download_file(CPT_URL, "raw_cpt.csv")
    if cpt_path:
        process_cpt(cpt_path)
        
    print("\n🎉 DONE! Data is now official and comprehensive.")
