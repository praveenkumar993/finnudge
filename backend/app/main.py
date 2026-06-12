import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import sqlite3
import json
import numpy as np
from backend.app.services.recommender import recommender
from backend.app.services.online_learner import online_learner
from backend.app.core.config import settings
from backend.app.services.ab_testing import ab_testing

app = FastAPI(
    title="FinNudge API",
    version="1.0.0",
    description="Real-time financial personalization engine"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    import os
    import subprocess

    # generate data if not present (first deploy)
    if not os.path.exists("data/processed/users.json"):
        print("generating data for first deploy...")
        subprocess.run(["python", "scripts/generate_users.py"])
        subprocess.run(["python", "scripts/feature_engineering.py"])
        subprocess.run(["python", "scripts/build_nudge_features.py"])
        subprocess.run(["python", "scripts/train_model.py"])
        subprocess.run(["python", "scripts/build_faiss_index.py"])
        print("data generation complete")

    recommender.load()


# ── schemas ──────────────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    user_id:  str
    nudge_id: str
    action:   str        # click | skip | dismiss
    ab_group: Optional[str] = "control"

class RecommendRequest(BaseModel):
    user_id:       str
    top_k:         Optional[int] = 3
    current_action: Optional[str] = "opened_app"


# ── endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":    "ok",
        "version":   settings.VERSION,
        "model_loaded": recommender.is_loaded,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not recommender.is_loaded:
        raise HTTPException(503, "model not loaded yet")

    ab_group = ab_testing.assign_group(req.user_id)

    if ab_group == "treatment":
        result = recommender.recommend(req.user_id, req.top_k,req.current_action or "opened_app")
    else:
        # control group gets random nudges
        all_nudges = recommender.get_all_nudges()
        sample     = random.sample(all_nudges,
                                   req.top_k or settings.TOP_K)
        result = {
            "user_id":    req.user_id,
            "archetype":  "unknown",
            "nudges": [{
                "rank":        i + 1,
                "nudge_id":    n["nudge_id"],
                "title":       n["title"],
                "archetype":   n["archetype"],
                "category":    n["category"],
                "score":       0.5,
                "explanation": "Showing available nudges",
            } for i, n in enumerate(sample)],
            "latency_ms": 0,
            "source":     "control_random",
        }

    for nudge in result.get("nudges", []):
        ab_testing.log_event(
            req.user_id,
            nudge["nudge_id"],
            "impression",
            ab_group
        )
    result["ab_group"] = ab_group
    return result


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if not recommender.is_loaded:
        raise HTTPException(503, "model not loaded yet")

    # get current vector
    user_data = recommender.user_cache.get(req.user_id)
    if not user_data:
        raise HTTPException(404, f"user {req.user_id} not found")

    old_vec = user_data["vector"]
    new_vec = online_learner.update(old_vec, req.nudge_id, req.action)
    shift   = online_learner.get_shift_magnitude(old_vec, new_vec)

    # update in-memory embedding
    recommender.update_user_embedding(req.user_id, new_vec)

    # log event
    recommender.log_event(
        req.user_id, req.nudge_id, req.action, req.ab_group
    )

    return {
        "user_id":         req.user_id,
        "nudge_id":        req.nudge_id,
        "action":          req.action,
        "embedding_shift": round(shift, 6),
        "updated":         True,
    }


@app.get("/nudges")
def get_nudges():
    return {
        "nudges": recommender.get_all_nudges(),
        "total":  len(recommender.get_all_nudges()),
    }


@app.get("/metrics/summary")
def metrics_summary():
    conn = sqlite3.connect(settings.DB_PATH)
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM events")
    total_events = cur.fetchone()[0]

    cur.execute("""
        SELECT action, COUNT(*) as cnt
        FROM events
        GROUP BY action
    """)
    action_counts = {row[0]: row[1] for row in cur.fetchall()}

    cur.execute("""
        SELECT nudge_id, COUNT(*) as cnt
        FROM events
        WHERE action = 'click'
        GROUP BY nudge_id
        ORDER BY cnt DESC
        LIMIT 5
    """)
    top_nudges = [{"nudge_id": r[0], "clicks": r[1]}
                  for r in cur.fetchall()]

    cur.execute("""
        SELECT ab_group, action, COUNT(*) as cnt
        FROM events
        GROUP BY ab_group, action
    """)
    ab_raw = cur.fetchall()
    conn.close()

    # CTR per ab group
    ab_stats = {}
    for group, action, cnt in ab_raw:
        if group not in ab_stats:
            ab_stats[group] = {"impressions": 0, "clicks": 0}
        if action == "impression":
            ab_stats[group]["impressions"] += cnt
        elif action == "click":
            ab_stats[group]["clicks"] += cnt

    for group in ab_stats:
        imp = ab_stats[group]["impressions"]
        clk = ab_stats[group]["clicks"]
        ab_stats[group]["ctr"] = round(clk / max(imp, 1), 4)

    return {
        "total_events":  total_events,
        "action_counts": action_counts,
        "top_nudges":    top_nudges,
        "ab_stats":      ab_stats,
        "model": {
            "recall_at_3": 0.479,
            "recall_at_5": 0.628,
            "recall_at_1": 0.156,
            "random_baseline_at_3": 0.15,
            "improvement": "3.2x over random",
        }
    }


@app.get("/metrics/history")
def metrics_history():
    conn = sqlite3.connect(settings.DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        SELECT
            DATE(timestamp) as day,
            COUNT(*) as events,
            SUM(CASE WHEN action='click' THEN 1 ELSE 0 END) as clicks
        FROM events
        GROUP BY DATE(timestamp)
        ORDER BY day DESC
        LIMIT 30
    """)
    rows = cur.fetchall()
    conn.close()
    return {
        "history": [
            {"date": r[0], "events": r[1], "clicks": r[2]}
            for r in rows
        ]
    }


@app.get("/user/{user_id}")
def get_user(user_id: str):
    user = recommender.user_cache.get(user_id)
    if not user:
        raise HTTPException(404, f"user {user_id} not found")
    return {
        "user_id":   user_id,
        "archetype": user["archetype"],
        "city":      user["city"],
        "age":       user["age"],
    }

@app.get("/abtest/results")
def abtest_results():
    return ab_testing.get_results()


@app.get("/abtest/assign/{user_id}")
def abtest_assign(user_id: str):
    group = ab_testing.assign_group(user_id)
    return {"user_id": user_id, "ab_group": group}

@app.get("/user/{user_id}/personalization")
def get_personalization(user_id: str):
    if not recommender.is_loaded:
        raise HTTPException(503, "model not loaded yet")
    score = recommender.get_personalization_score(user_id)
    return {"user_id": user_id, **score}


@app.get("/metrics/personalization")
def personalization_distribution():
    """
    Returns personalization score distribution
    across all users — for dashboard visualization
    """
    if not recommender.is_loaded:
        raise HTTPException(503, "model not loaded yet")

    sample_users = list(recommender.user_cache.keys())[:500]
    scores = []
    for uid in sample_users:
        s = recommender.get_personalization_score(uid)
        scores.append(s["score"])

    scores_arr = np.array(scores)
    return {
        "mean":   round(float(np.mean(scores_arr)),  2),
        "median": round(float(np.median(scores_arr)), 2),
        "min":    round(float(np.min(scores_arr)),   2),
        "max":    round(float(np.max(scores_arr)),   2),
        "buckets": {
            "new_user":    int(np.sum(scores_arr < 20)),
            "learning":    int(np.sum((scores_arr >= 20) &
                                      (scores_arr < 50))),
            "personalized":int(np.sum((scores_arr >= 50) &
                                      (scores_arr < 80))),
            "highly":      int(np.sum(scores_arr >= 80)),
        }
    }