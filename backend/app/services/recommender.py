import json
import time
import numpy as np
import faiss
import torch
import sqlite3
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.app.models.two_tower import TwoTowerModel
from backend.app.core.config import settings

class RecommendationService:
    """
    Loads trained two-tower model + FAISS index
    Serves top-K nudge recommendations per user
    Tracks latency per request
    """

    def __init__(self):
        self.model        = None
        self.index        = None
        self.nudge_meta   = None
        self.nudge_feats  = None
        self.user_cache   = {}   # user_id → feature vector
        self.is_loaded    = False

    def load(self):
        print("loading two-tower model...")
        checkpoint  = torch.load(
            settings.MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )
        self.model  = TwoTowerModel(
            user_input_dim=45,
            nudge_input_dim=16,
            embedding_dim=settings.EMBEDDING_DIM
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        print(f"  model loaded — best Recall@3: "
              f"{checkpoint.get('recall_at_3', 'N/A')}")

        print("loading FAISS index...")
        self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
        print(f"  index loaded — {self.index.ntotal} nudges")

        with open(settings.NUDGE_META_PATH) as f:
            self.nudge_meta = json.load(f)

        with open(settings.NUDGE_FEAT_PATH) as f:
            self.nudge_feats = {
                n["nudge_id"]: n for n in json.load(f)
            }

        self._load_user_cache()
        self.is_loaded = True
        print("recommendation service ready.")

    def _load_user_cache(self):
        with open(settings.USER_FEAT_PATH) as f:
            users = json.load(f)
        for u in users:
            self.user_cache[u["user_id"]] = {
                "vector":    u["vector"],
                "archetype": u["archetype"],
                "city":      u["city"],
                "age":       u["age"],
            }
        print(f"  user cache loaded — {len(self.user_cache)} users")

    def _get_user_vector(self, user_id: str):
        if user_id in self.user_cache:
            return self.user_cache[user_id]["vector"]
        return None

    def _get_user_embedding(self, vector: list) -> np.ndarray:
        with torch.no_grad():
            vec = torch.tensor([vector], dtype=torch.float32)
            emb = self.model.get_user_embedding(vec)
            emb = emb.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(emb)
        return emb

    def recommend(self, user_id: str, top_k: int = None) -> dict:
        start = time.time()
        k     = top_k or settings.TOP_K

        vector = self._get_user_vector(user_id)

        # cold start — use archetype default embedding
        if vector is None:
            return self._cold_start_recommend(user_id, k)

        user_emb = self._get_user_embedding(vector)
        distances, indices = self.index.search(user_emb, k)

        results = []
        for rank, (dist, idx) in enumerate(
            zip(distances[0], indices[0]), 1
        ):
            meta = self.nudge_meta[idx]
            results.append({
                "rank":      rank,
                "nudge_id":  meta["nudge_id"],
                "title":     meta["title"],
                "archetype": meta["archetype"],
                "category":  meta["category"],
                "score":     round(float(dist), 4),
                "explanation": self._explain(
                    user_id, meta["archetype"], meta["category"]
                ),
            })

        latency_ms = round((time.time() - start) * 1000, 2)

        user_info = self.user_cache.get(user_id, {})
        return {
            "user_id":    user_id,
            "archetype":  user_info.get("archetype", "unknown"),
            "nudges":     results,
            "latency_ms": latency_ms,
            "source":     "two_tower",
        }

    def _cold_start_recommend(self, user_id: str, k: int) -> dict:
        """fallback for unknown users — return top general nudges"""
        fallback = self.nudge_meta[:k]
        results  = [{
            "rank":        i + 1,
            "nudge_id":    n["nudge_id"],
            "title":       n["title"],
            "archetype":   n["archetype"],
            "category":    n["category"],
            "score":       0.5,
            "explanation": "New user — showing popular nudges",
        } for i, n in enumerate(fallback)]

        return {
            "user_id":    user_id,
            "archetype":  "unknown",
            "nudges":     results,
            "latency_ms": 0,
            "source":     "cold_start",
        }

    def _explain(self, user_id: str, nudge_archetype: str,
                 nudge_category: str) -> str:
        user   = self.user_cache.get(user_id, {})
        arch   = user.get("archetype", "")
        city   = user.get("city", "")

        if arch == nudge_archetype:
            templates = {
                "investment": f"Recommended based on your "
                              f"investment activity in {city}",
                "savings":    "Matches your consistent saving pattern",
                "emi":        "Based on your loan repayment history",
                "reminder":   "You regularly pay bills on time",
                "alert":      "Your spending pattern triggered this",
                "budgeting":  "Based on your monthly spend trends",
                "rewards":    "You have unclaimed rewards available",
                "insurance":  "Matches your financial safety profile",
                "offers":     "Personalised offer based on your usage",
            }
            return templates.get(nudge_category,
                                 "Recommended based on your profile")
        return "Recommended based on your financial activity"

    def update_user_embedding(self, user_id: str,
                               new_vector: list):
        """called by online learner after interaction"""
        if user_id in self.user_cache:
            self.user_cache[user_id]["vector"] = new_vector

    def get_all_nudges(self) -> list:
        return self.nudge_meta

    def log_event(self, user_id: str, nudge_id: str,
                  action: str, ab_group: str):
        conn = sqlite3.connect(settings.DB_PATH)
        cur  = conn.cursor()
        from datetime import datetime
        cur.execute("""
            INSERT INTO events
            (user_id, nudge_id, action, ab_group, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, nudge_id, action,
              ab_group, datetime.now().isoformat()))
        conn.commit()
        conn.close()


# singleton
recommender = RecommendationService()