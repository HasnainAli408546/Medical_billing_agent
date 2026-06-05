import os
import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import sys

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR    = os.path.join(BASE_DIR, "data", "reference")
INDEX_DIR  = os.path.join(BASE_DIR, "models", "medical_index")
os.makedirs(INDEX_DIR, exist_ok=True)

# ── Model ──────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"

def main():
    print("🚀 Building Production Medical Index (Large Scale)...")
    
    # 1. Load Data
    try:
        icd_df = pd.read_csv(os.path.join(REF_DIR, "icd10_reference.csv"))
        cpt_df = pd.read_csv(os.path.join(REF_DIR, "cpt_reference.csv"))
    except Exception as e:
        print(f"❌ Error loading CSVs: {e}")
        print("   Ensure you ran generate_master_reference.py first.")
        sys.exit(1)
        
    documents = []
    metadata  = []
    
    print(f"📝 Preparing {len(icd_df)} ICD and {len(cpt_df)} CPT codes...")
    
    # Process ICD Codes (Limit to 10k for demo performance, or keep all for production)
    # Let's keep all to satisfy the "not missing cases" requirement
    for _, row in icd_df.iterrows():
        desc = str(row['description'])
        text = f"ICD: {row['code']} | {desc}"
        documents.append(text)
        metadata.append({"t": "I", "c": row['code'], "d": desc})
        
    for _, row in cpt_df.iterrows():
        desc = str(row['description'])
        text = f"CPT: {row['code']} | {desc}"
        documents.append(text)
        metadata.append({"t": "C", "c": row['code'], "d": desc})
        
    print(f"📊 Total documents to index: {len(documents)}")
    
    # 2. Embed
    print(f"🤖 Loading {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL, token=os.getenv("HF_TOKEN"))
    
    print("🔢 Generating embeddings (This may take 2-5 minutes for 80k+ codes)...")
    embeddings = model.encode(
        documents, 
        batch_size=128, 
        show_progress_bar=True, 
        normalize_embeddings=True
    )
    
    # 3. Build FAISS
    print("🗄️ Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    
    # 4. Save
    print("💾 Saving artifacts...")
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "metadata.json"), "w") as f:
        json.dump({"m": metadata}, f) # Compressed keys for smaller file size
        
    print(f"✅ Production Medical Index built successfully!")

if __name__ == "__main__":
    main()
