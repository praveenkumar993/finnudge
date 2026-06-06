import json
import numpy as np
import pandas as pd
import sqlite3
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

INPUT_PATH  = "data/processed/users.json"
OUTPUT_PATH = "data/processed/features.json"
DB_PATH     = "data/processed/finnudge.db"

CATEGORIES = [
    "mutual_fund", "stocks", "sip", "gold",
    "fd", "recurring_deposit", "savings", "insurance",
    "food", "shopping", "entertainment", "travel",
    "electricity", "rent", "mobile_recharge", "ott",
    "emi", "loan_repayment", "credit_card", "bnpl"
]

ARCHETYPES = ["investor", "saver", "spender", "bill_payer", "borrower"]
CITIES     = [
    "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"
]

def extract_features(user):
    txns     = user["transactions"]
    features = user["features"]

    # ── basic features ───────────────────────────────────────────────
    avg_amount      = features.get("avg_amount", 0)
    max_amount      = features.get("max_amount", 0)
    txn_count       = features.get("txn_count_6m", 0)
    top_cat_ratio   = features.get("top_cat_ratio", 0)
    unique_cats     = features.get("unique_categories", 0)
    avg_hour        = features.get("avg_hour", 12)

    # ── category distribution (20-dim vector) ────────────────────────
    cat_counts = {c: 0 for c in CATEGORIES}
    for t in txns:
        c = t["category"]
        if c in cat_counts:
            cat_counts[c] += 1
    total = max(sum(cat_counts.values()), 1)
    cat_vector = [round(cat_counts[c] / total, 4) for c in CATEGORIES]

    # ── time features ────────────────────────────────────────────────
    hours      = [t["hour"] for t in txns]
    morning    = sum(1 for h in hours if 6  <= h < 12) / max(len(hours), 1)
    afternoon  = sum(1 for h in hours if 12 <= h < 17) / max(len(hours), 1)
    evening    = sum(1 for h in hours if 17 <= h < 24) / max(len(hours), 1)
    weekend    = sum(1 for t in txns if t["day_of_week"] >= 5) / max(len(txns), 1)

    # ── amount buckets ───────────────────────────────────────────────
    amounts    = [t["amount"] for t in txns]
    small      = sum(1 for a in amounts if a < 1000)   / max(len(amounts), 1)
    medium     = sum(1 for a in amounts if 1000 <= a < 10000) / max(len(amounts), 1)
    large      = sum(1 for a in amounts if a >= 10000) / max(len(amounts), 1)

    # ── city one-hot (10-dim) ────────────────────────────────────────
    city_vec   = [1 if user["city"] == c else 0 for c in CITIES]

    # ── archetype label (for training) ──────────────────────────────
    archetype_idx = ARCHETYPES.index(user["archetype"])

    # ── assemble full vector ─────────────────────────────────────────
    vector = (
        [
            avg_amount / 50000,        # normalize to 0-1
            max_amount / 50000,
            txn_count  / 100,
            top_cat_ratio,
            unique_cats / 20,
            avg_hour   / 24,
            morning,
            afternoon,
            evening,
            weekend,
            small,
            medium,
            large,
            user["age"] / 60,
            user["salary_day"] / 31,
        ]
        + cat_vector                   # 20 dims
        + city_vec                     # 10 dims
    )

    return {
        "user_id":       user["user_id"],
        "archetype":     user["archetype"],
        "archetype_idx": archetype_idx,
        "vector":        vector,
        "vector_dim":    len(vector),
        "relevant_nudges": user["relevant_nudges"],
        "city":          user["city"],
        "age":           user["age"],
    }

def build_feature_matrix(users):
    records = []
    for u in users:
        try:
            r = extract_features(u)
            records.append(r)
        except Exception as e:
            print(f"skipping {u['user_id']}: {e}")
    return records

def save_features(records):
    os.makedirs("data/processed", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"saved {len(records)} feature vectors → {OUTPUT_PATH}")
    print(f"vector dimension: {records[0]['vector_dim']}")

def save_to_sqlite(records):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_features (
            user_id       TEXT PRIMARY KEY,
            archetype     TEXT,
            archetype_idx INTEGER,
            vector        TEXT,
            vector_dim    INTEGER
        )
    """)
    for r in records:
        cur.execute("""
            INSERT OR REPLACE INTO user_features
            VALUES (?, ?, ?, ?, ?)
        """, (
            r["user_id"],
            r["archetype"],
            r["archetype_idx"],
            json.dumps(r["vector"]),
            r["vector_dim"],
        ))
    conn.commit()
    conn.close()
    print(f"saved feature vectors → sqlite")

def print_summary(records):
    df = pd.DataFrame([{
        "user_id":   r["user_id"],
        "archetype": r["archetype"],
        "vec_dim":   r["vector_dim"],
    } for r in records])

    print("\n── archetype distribution ──")
    print(df["archetype"].value_counts().to_string())
    print(f"\n── vector dimension: {records[0]['vector_dim']} ──")
    print("  15 behavioral + 20 category + 10 city = 45 dims")

if __name__ == "__main__":
    print("loading users...")
    with open(INPUT_PATH) as f:
        users = json.load(f)

    print(f"engineering features for {len(users)} users...")
    records = build_feature_matrix(users)

    save_features(records)
    save_to_sqlite(records)
    print_summary(records)
    print("\nDay 2 feature engineering complete.")