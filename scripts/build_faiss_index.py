import json
import os
import numpy as np
import faiss

EMBEDDINGS_PATH = "data/processed/nudge_embeddings.json"
INDEX_PATH      = "data/processed/nudge.index"
METADATA_PATH   = "data/processed/nudge_metadata.json"

def build_index():
    with open(EMBEDDINGS_PATH) as f:
        nudges = json.load(f)

    embeddings = np.array(
        [n["embedding"] for n in nudges],
        dtype=np.float32
    )

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vecs
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index built → {INDEX_PATH}")
    print(f"  {index.ntotal} vectors, {dim} dims")

    metadata = [{
        "nudge_id":  n["nudge_id"],
        "title":     n["title"],
        "archetype": n["archetype"],
        "category":  n["category"],
    } for n in nudges]

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"metadata saved  → {METADATA_PATH}")

def test_retrieval():
    with open(EMBEDDINGS_PATH) as f:
        nudges = json.load(f)

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    print("\n── retrieval test ──")
    print("query: investor user embedding (nudge n01 as proxy)")

    query_vec = np.array(
        [nudges[0]["embedding"]],
        dtype=np.float32
    )
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, 5)

    print("top 5 results:")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        m = metadata[idx]
        print(f"  {rank}. [{m['nudge_id']}] {m['title'][:45]:<45} "
              f"score={dist:.3f} archetype={m['archetype']}")

if __name__ == "__main__":
    build_index()
    test_retrieval()
    print("\nFAISS index ready.")