import json
import random
import sqlite3
import os
from datetime import datetime, timedelta

random.seed(42)

# ── archetypes ──────────────────────────────────────────────────────────────
ARCHETYPES = {
    "investor": {
        "categories": ["mutual_fund", "stocks", "sip", "gold"],
        "avg_amount":  (5000, 50000),
        "txn_per_month": (8, 20),
        "active_hours": [9, 10, 11, 20, 21],
        "salary_day_spike": True,
    },
    "saver": {
        "categories": ["fd", "recurring_deposit", "savings", "insurance"],
        "avg_amount":  (2000, 15000),
        "txn_per_month": (4, 10),
        "active_hours": [8, 12, 19, 20],
        "salary_day_spike": True,
    },
    "spender": {
        "categories": ["food", "shopping", "entertainment", "travel"],
        "avg_amount":  (200, 5000),
        "txn_per_month": (20, 50),
        "active_hours": [12, 13, 19, 20, 21, 22],
        "salary_day_spike": False,
    },
    "bill_payer": {
        "categories": ["electricity", "rent", "mobile_recharge", "ott"],
        "avg_amount":  (500, 8000),
        "txn_per_month": (6, 12),
        "active_hours": [9, 10, 18, 19],
        "salary_day_spike": False,
    },
    "borrower": {
        "categories": ["emi", "loan_repayment", "credit_card", "bnpl"],
        "avg_amount":  (3000, 30000),
        "txn_per_month": (5, 10),
        "active_hours": [10, 11, 15, 16],
        "salary_day_spike": True,
    },
}

# ── nudge catalog ────────────────────────────────────────────────────────────
NUDGES = [
    {"id": "n01", "title": "Start ₹500 SIP in Nifty 50",         "archetype": "investor",   "category": "investment"},
    {"id": "n02", "title": "Enable auto-invest on salary day",    "archetype": "investor",   "category": "investment"},
    {"id": "n03", "title": "Your FD is maturing in 7 days",       "archetype": "saver",      "category": "savings"},
    {"id": "n04", "title": "Open a recurring deposit today",      "archetype": "saver",      "category": "savings"},
    {"id": "n05", "title": "Your spend is 20% above last month",  "archetype": "spender",    "category": "alert"},
    {"id": "n06", "title": "Set a monthly food budget",           "archetype": "spender",    "category": "budgeting"},
    {"id": "n07", "title": "Pay electricity bill — due in 3 days","archetype": "bill_payer", "category": "reminder"},
    {"id": "n08", "title": "Recharge your Jio plan today",        "archetype": "bill_payer", "category": "reminder"},
    {"id": "n09", "title": "EMI due on 5th — ensure balance",     "archetype": "borrower",   "category": "emi"},
    {"id": "n10", "title": "Close your BNPL early — save interest","archetype": "borrower",  "category": "emi"},
    {"id": "n11", "title": "Gold prices dipped — good time to buy","archetype": "investor",  "category": "investment"},
    {"id": "n12", "title": "Activate credit card reward points",  "archetype": "spender",    "category": "rewards"},
    {"id": "n13", "title": "Term insurance reminder",             "archetype": "saver",      "category": "insurance"},
    {"id": "n14", "title": "UPI cashback offer — pay bills",      "archetype": "bill_payer", "category": "offers"},
    {"id": "n15", "title": "Top up your emergency fund",          "archetype": "saver",      "category": "savings"},
    {"id": "n16", "title": "Stocks in your watchlist dropped 5%", "archetype": "investor",   "category": "investment"},
    {"id": "n17", "title": "Upgrade to premium credit card",      "archetype": "spender",    "category": "offers"},
    {"id": "n18", "title": "Pre-pay part of your home loan",      "archetype": "borrower",   "category": "emi"},
    {"id": "n19", "title": "Travel insurance for your trip",      "archetype": "spender",    "category": "insurance"},
    {"id": "n20", "title": "Save tax — invest in ELSS before March","archetype": "investor", "category": "investment"},
]

INDIAN_CITIES = [
    "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"
]

def random_date(days_back=180):
    return datetime.now() - timedelta(days=random.randint(0, days_back))

def generate_transactions(archetype_key, archetype, n_months=6):
    txns = []
    for month in range(n_months):
        count = random.randint(*archetype["txn_per_month"])
        for _ in range(count):
            category = random.choice(archetype["categories"])
            amount   = round(random.uniform(*archetype["avg_amount"]), 2)
            hour     = random.choice(archetype["active_hours"])
            date     = datetime.now() - timedelta(days=month * 30 + random.randint(0, 29))
            date     = date.replace(hour=hour, minute=random.randint(0, 59))
            txns.append({
                "category": category,
                "amount":   amount,
                "hour":     hour,
                "day_of_week": date.weekday(),
                "month":    date.month,
                "timestamp": date.isoformat(),
            })
    return txns

def compute_features(archetype_key, txns):
    if not txns:
        return {}
    amounts    = [t["amount"] for t in txns]
    categories = [t["category"] for t in txns]
    hours      = [t["hour"] for t in txns]
    cat_counts = {c: categories.count(c) for c in set(categories)}
    top_cat    = max(cat_counts, key=cat_counts.get)
    return {
        "avg_amount":        round(sum(amounts) / len(amounts), 2),
        "max_amount":        round(max(amounts), 2),
        "txn_count_6m":      len(txns),
        "top_category":      top_cat,
        "top_cat_ratio":     round(cat_counts[top_cat] / len(txns), 3),
        "avg_hour":          round(sum(hours) / len(hours), 1),
        "unique_categories": len(set(categories)),
        "archetype":         archetype_key,
    }

def assign_nudges(archetype_key):
    matching  = [n for n in NUDGES if n["archetype"] == archetype_key]
    other     = [n for n in NUDGES if n["archetype"] != archetype_key]
    chosen    = random.sample(matching, min(3, len(matching)))
    chosen   += random.sample(other, min(2, len(other)))
    return [n["id"] for n in chosen]

def generate_dataset(n_users=5000):
    users = []
    archetype_keys = list(ARCHETYPES.keys())

    for i in range(n_users):
        archetype_key = random.choice(archetype_keys)
        archetype     = ARCHETYPES[archetype_key]
        txns          = generate_transactions(archetype_key, archetype)
        features      = compute_features(archetype_key, txns)
        nudge_ids     = assign_nudges(archetype_key)
        salary_day    = random.randint(1, 10)

        user = {
            "user_id":       f"user_{i+1:04d}",
            "city":          random.choice(INDIAN_CITIES),
            "age":           random.randint(22, 55),
            "salary_day":    salary_day,
            "archetype":     archetype_key,
            "features":      features,
            "transactions":  txns,
            "relevant_nudges": nudge_ids,
            "created_at":    random_date(365).isoformat(),
        }
        users.append(user)

        if (i + 1) % 500 == 0:
            print(f"  generated {i+1} users...")

    return users

def save_to_json(users, path="data/processed/users.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(users, f, indent=2)
    print(f"saved {len(users)} users → {path}")

def save_to_sqlite(users, path="data/processed/finnudge.db"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            city TEXT,
            age INTEGER,
            salary_day INTEGER,
            archetype TEXT,
            features TEXT,
            relevant_nudges TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS nudges (
            id TEXT PRIMARY KEY,
            title TEXT,
            archetype TEXT,
            category TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            nudge_id TEXT,
            action TEXT,
            ab_group TEXT,
            timestamp TEXT
        )
    """)

    for u in users:
        cur.execute("""
            INSERT OR REPLACE INTO users
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            u["user_id"],
            u["city"],
            u["age"],
            u["salary_day"],
            u["archetype"],
            json.dumps(u["features"]),
            json.dumps(u["relevant_nudges"]),
            u["created_at"],
        ))

    for n in NUDGES:
        cur.execute("""
            INSERT OR REPLACE INTO nudges VALUES (?, ?, ?, ?)
        """, (n["id"], n["title"], n["archetype"], n["category"]))

    conn.commit()
    conn.close()
    print(f"saved to sqlite → {path}")

if __name__ == "__main__":
    print("generating 5000 users...")
    users = generate_dataset(5000)
    save_to_json(users)
    save_to_sqlite(users)
    print("done. Day 1 data generation complete.")