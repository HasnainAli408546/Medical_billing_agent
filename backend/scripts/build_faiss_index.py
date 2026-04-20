"""
==============================================================
  FAISS Vector Index Builder
  Voice-Driven Revenue Cycle Copilot
==============================================================
  Input  : data/knowledge_base.json  (40 billing rules)
  Output : models/faiss_index/
              ├── index.faiss          (vector index)
              └── metadata.json        (rule metadata for retrieval)

  Embedding Model: sentence-transformers/all-MiniLM-L6-v2
    - Free, local, no API key required
    - 384-dimensional embeddings
    - Fast inference (~14k sentences/sec on CPU)

  Each rule is embedded as a rich text combining:
    ICD description + CPT description + billing rule + specialty
  This maximises semantic search relevance.
==============================================================
"""

import os
import json
import pickle
import numpy as np

# ── Imports ───────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("❌ Missing packages. Run:")
    print("   pip install sentence-transformers faiss-cpu")
    exit(1)

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH    = os.path.join(BASE_DIR, "data",   "knowledge_base.json")
INDEX_DIR  = os.path.join(BASE_DIR, "models", "faiss_index")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH  = os.path.join(INDEX_DIR, "metadata.json")

os.makedirs(INDEX_DIR, exist_ok=True)

# ── Embedding Model ────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # 80MB, free, local


def build_document_text(rule: dict) -> str:
    """
    Create a rich text document from a billing rule for embedding.
    Combines the most semantically relevant fields.
    """
    parts = [
        f"Diagnosis: {rule.get('icd_description', '')}",
        f"ICD Code: {rule.get('icd_code', '')}",
        f"Procedure: {rule.get('cpt_description', '')}",
        f"CPT Code: {rule.get('cpt_code', '')}",
        f"Specialty: {rule.get('specialty', '')}",
        f"Category: {rule.get('category', '')}",
        f"Billing Rule: {rule.get('billing_rule', '')}",
        f"Denial Risk: {rule.get('denial_risk', '')}",
    ]
    return " | ".join(parts)


def main():
    print("=" * 60)
    print("  🔍 Building FAISS Vector Index for RAG Pipeline")
    print("=" * 60)

    # 1. Load knowledge base
    print(f"\n📂 Loading knowledge base from {KB_PATH}...")
    with open(KB_PATH, "r") as f:
        rules = json.load(f)
    print(f"   ✅ Loaded {len(rules)} billing rules")

    # 2. Load embedding model
    print(f"\n🤖 Loading embedding model: {EMBED_MODEL}")
    print("   (First run downloads ~80MB — subsequent runs use cache)")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"   ✅ Model loaded | Embedding dim: {model.get_sentence_embedding_dimension()}")

    # 3. Build document texts & embed
    print("\n📝 Building document texts...")
    documents = [build_document_text(rule) for rule in rules]

    print("🔢 Generating embeddings...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via dot product
        convert_to_numpy=True,
    )
    print(f"   ✅ Embeddings shape: {embeddings.shape}")

    # 4. Build FAISS index (Inner Product = cosine similarity since normalized)
    print("\n🗄️  Building FAISS index...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # IP = Inner Product (cosine on normalized vecs)
    index.add(embeddings.astype(np.float32))
    print(f"   ✅ FAISS index built | Vectors stored: {index.ntotal}")

    # 5. Save FAISS index
    faiss.write_index(index, INDEX_PATH)
    print(f"   💾 FAISS index saved → {INDEX_PATH}")

    # 6. Save metadata (for retrieval lookup)
    # Store full rule objects so we can return rich context after retrieval
    metadata = {
        "embed_model":   EMBED_MODEL,
        "num_rules":     len(rules),
        "embedding_dim": int(dim),
        "rules":         rules,
        "documents":     documents,   # raw text (for debugging/display)
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   💾 Metadata saved     → {META_PATH}")

    # 7. Quick sanity check
    print("\n🔬 Sanity check — test query: 'hypertension ECG denied'")
    test_vec = model.encode(
        ["hypertension ECG electrocardiogram denied"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    scores, indices = index.search(test_vec, k=3)
    print("   Top 3 matching rules:")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        rule = rules[idx]
        print(f"   [{rank}] Score: {score:.4f} | {rule['id']} | {rule['icd_description']} + {rule['cpt_description']}")

    print("\n✅ FAISS index build complete!")
    print("   Run the backend to activate the RAG Validation Agent.\n")


if __name__ == "__main__":
    main()
