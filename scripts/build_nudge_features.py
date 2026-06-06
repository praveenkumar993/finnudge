import json
import os
import sqlite3

DB_PATH     = "data/processed/finnudge.db"
OUTPUT_PATH = "data/processed/nudge_features.json"

ARCHETYPES = ["investor", "saver", "spender", "bill_payer", "borrower"]
CATEGORIES = ["investment", "savings", "alert", "budgeting",
              "reminder", "emi", "rewards", "insurance", "offers"]

NUDGES = [
    {"id": "n01", "title": "Start ₹500 SIP in Nifty 50",          "archetype": "investor",   "category": "investment", "urgency": 0.6, "value": 0.9},
    {"id": "n02", "title": "Enable auto-invest on salary day",     "archetype": "investor",   "category": "investment", "urgency": 0.5, "value": 0.85},
    {"id": "n03", "title": "Your FD is maturing in 7 days",        "archetype": "saver",      "category": "savings",    "urgency": 0.9, "value": 0.8},
    {"id": "n04", "title": "Open a recurring deposit today",       "archetype": "saver",      "category": "savings",    "urgency": 0.4, "value": 0.7},
    {"id": "n05", "title": "Your spend is 20% above last month",   "archetype": "spender",    "category": "alert",      "urgency": 0.8, "value": 0.75},
    {"id": "n06", "title": "Set a monthly food budget",            "archetype": "spender",    "category": "budgeting",  "urgency": 0.3, "value": 0.65},
    {"id": "n07", "title": "Pay electricity bill — due in 3 days", "archetype": "bill_payer", "category": "reminder",   "urgency": 0.95,"value": 0.9},
    {"id": "n08", "title": "Recharge your Jio plan today",         "archetype": "bill_payer", "category": "reminder",   "urgency": 0.85,"value": 0.8},
    {"id": "n09", "title": "EMI due on 5th — ensure balance",      "archetype": "borrower",   "category": "emi",        "urgency": 0.95,"value": 0.9},
    {"id": "n10", "title": "Close your BNPL early — save interest","archetype": "borrower",   "category": "emi",        "urgency": 0.6, "value": 0.8},
    {"id": "n11", "title": "Gold prices dipped — good time to buy","archetype": "investor",   "category": "investment", "urgency": 0.7, "value": 0.85},
    {"id": "n12", "title": "Activate credit card reward points",   "archetype": "spender",    "category": "rewards",    "urgency": 0.5, "value": 0.7},
    {"id": "n13", "title": "Term insurance reminder",              "archetype": "saver",      "category": "insurance",  "urgency": 0.6, "value": 0.8},
    {"id": "n14", "title": "UPI cashback offer — pay bills",       "archetype": "bill_payer", "category": "offers",     "urgency": 0.7, "value": 0.75},
    {"id": "n15", "title": "Top up your emergency fund",           "archetype": "saver",      "category": "savings",    "urgency": 0.5, "value": 0.75},
    {"id": "n16", "title": "Stocks in your watchlist dropped 5%",  "archetype": "investor",   "category": "investment", "urgency": 0.8, "value": 0.8},
    {"id": "n17", "title": "Upgrade to premium credit card",       "archetype": "spender",    "category": "offers",     "urgency": 0.4, "value": 0.65},
    {"id": "n18", "title": "Pre-pay part of your home loan",       "archetype": "borrower",   "category": "emi",        "urgency": 0.5, "value": 0.85},
    {"id": "n19", "title": "Travel insurance for your trip",       "archetype": "spender",    "category": "insurance",  "urgency": 0.6, "value": 0.7},
    {"id": "n20", "title": "Save tax — invest in ELSS before March","archetype": "investor",  "category": "investment", "urgency": 0.9, "value": 0.95},
]

def build_nudge_vector(nudge):
    archetype_vec = [1 if nudge["archetype"] == a else 0 for a in ARCHETYPES]
    category_vec  = [1 if nudge["category"]  == c else 0 for c in CATEGORIES]
    vector = archetype_vec + category_vec + [nudge["urgency"], nudge["value"]]
    return vector

def build_all_nudge_features():
    records = []
    for n in NUDGES:
        vector = build_nudge_vector(n)
        records.append({
            "nudge_id":  n["id"],
            "title":     n["title"],
            "archetype": n["archetype"],
            "category":  n["category"],
            "urgency":   n["urgency"],
            "value":     n["value"],
            "vector":    vector,
            "vector_dim": len(vector),
        })
    return records

def save_to_json(records):
    os.makedirs("data/processed", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"saved {len(records)} nudge vectors → {OUTPUT_PATH}")
    print(f"nudge vector dimension: {records[0]['vector_dim']}")
    print("  5 archetype + 9 category + 2 scores = 16 dims")

def save_to_sqlite(records):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS nudge_features (
            nudge_id  TEXT PRIMARY KEY,
            title     TEXT,
            archetype TEXT,
            category  TEXT,
            vector    TEXT,
            vector_dim INTEGER
        )
    """)
    for r in records:
        cur.execute("""
            INSERT OR REPLACE INTO nudge_features
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            r["nudge_id"],
            r["title"],
            r["archetype"],
            r["category"],
            json.dumps(r["vector"]),
            r["vector_dim"],
        ))
    conn.commit()
    conn.close()
    print(f"saved nudge vectors → sqlite")

def print_summary(records):
    print("\n── nudge catalog ──")
    for r in records:
        print(f"  {r['nudge_id']} | {r['archetype']:<12} | {r['category']:<12} | {r['title'][:40]}")

if __name__ == "__main__":
    print("building nudge feature vectors...")
    records = build_all_nudge_features()
    save_to_json(records)
    save_to_sqlite(records)
    print_summary(records)
    print("\nnudge features complete.")