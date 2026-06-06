import json

INPUT_PATH = "data/processed/users.json"

RULES = {
    "investor": {
        "categories": ["mutual_fund", "stocks", "sip", "gold"],
        "min_ratio":  0.50,
    },
    "saver": {
        "categories": ["fd", "recurring_deposit", "savings", "insurance"],
        "min_ratio":  0.45,
    },
    "spender": {
        "categories": ["food", "shopping", "entertainment", "travel"],
        "min_ratio":  0.50,
    },
    "bill_payer": {
        "categories": ["electricity", "rent", "mobile_recharge", "ott"],
        "min_ratio":  0.45,
    },
    "borrower": {
        "categories": ["emi", "loan_repayment", "credit_card", "bnpl"],
        "min_ratio":  0.40,
    },
}

def classify(user):
    txns  = user["transactions"]
    total = max(len(txns), 1)
    cats  = [t["category"] for t in txns]

    scores = {}
    for archetype, rule in RULES.items():
        count  = sum(1 for c in cats if c in rule["categories"])
        ratio  = count / total
        scores[archetype] = ratio

    predicted = max(scores, key=scores.get)
    actual    = user["archetype"]
    correct   = predicted == actual
    return predicted, actual, correct, scores

def evaluate(users):
    results  = [classify(u) for u in users]
    correct  = sum(1 for _, _, c, _ in results if c)
    accuracy = correct / len(results)

    print(f"\n── archetype classifier accuracy ──")
    print(f"  total users : {len(users)}")
    print(f"  correct     : {correct}")
    print(f"  accuracy    : {accuracy:.1%}")

    print(f"\n── per archetype breakdown ──")
    archetypes = ["investor", "saver", "spender", "bill_payer", "borrower"]
    for arch in archetypes:
        arch_results = [(p, a, c, s) for p, a, c, s in results if a == arch]
        if not arch_results:
            continue
        arch_correct = sum(1 for _, _, c, _ in arch_results if c)
        arch_acc     = arch_correct / len(arch_results)
        print(f"  {arch:<12} → {arch_acc:.1%}  ({arch_correct}/{len(arch_results)})")

    print(f"\n── sample predictions ──")
    for i, (predicted, actual, correct, scores) in enumerate(results[:5]):
        status = "✓" if correct else "✗"
        print(f"  user_{i+1:04d} | actual: {actual:<12} predicted: {predicted:<12} {status}")

    return accuracy

if __name__ == "__main__":
    with open(INPUT_PATH) as f:
        users = json.load(f)

    accuracy = evaluate(users)

    if accuracy >= 0.80:
        print(f"\nclassifier is solid — ready for Day 3 training.")
    else:
        print(f"\naccuracy low — check category assignments in generate_users.py")