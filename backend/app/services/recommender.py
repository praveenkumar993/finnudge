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

        self.nudge_feats = {}
        with open(settings.NUDGE_FEAT_PATH) as f:
            for n in json.load(f):
                self.nudge_feats[n["nudge_id"]] = n
        print(f"  nudge features loaded for MMR")

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

    def recommend(self, user_id: str, top_k: int = None, current_action: str = "opened_app") -> dict:
        start = time.time()
        k     = top_k or settings.TOP_K

        vector = self._get_user_vector(user_id)

        # cold start — use archetype default embedding
        if vector is None:
            return self._cold_start_recommend(user_id, k)

        user_emb = self._get_user_embedding(vector)

        # fetch more candidates than needed for MMR reranking
        fetch_k  = min(k * 3, self.index.ntotal)
        distances, indices = self.index.search(user_emb, fetch_k)

        # MMR reranking for diversity
        reranked = self._mmr_rerank(
            user_emb       = user_emb[0],
            candidate_indices   = indices[0].tolist(),
            candidate_distances = distances[0].tolist(),
            k              = k,
            lambda_param   = 0.7,
        )

        results = []
        for rank, (idx, dist) in enumerate(reranked, 1):
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
                "context_boosted": False,
            })
        results = self._apply_context_boost(results, current_action)

        results = self._apply_fatigue_filter(results, user_id)

        latency_ms = round((time.time() - start) * 1000, 2)

        user_info = self.user_cache.get(user_id, {})
        persona_score = self.get_personalization_score(user_id)

        return {
            "user_id":    user_id,
            "archetype":  user_info.get("archetype", "unknown"),
            "nudges":     results,
            "latency_ms": latency_ms,
            "source":     "two_tower",
            "personalization":  persona_score,
            "context":         current_action,
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
        """
        Dynamic explanations using actual user stats.
        Much more personal than template strings.
        """
        user    = self.user_cache.get(user_id, {})
        arch    = user.get("archetype", "")
        city    = user.get("city", "")
        age     = user.get("age", 0)

        # pull real feature stats if available
        vector  = user.get("vector", [])
        avg_amt = round(vector[0] * 50000) if vector else 0
        txn_cnt = round(vector[2] * 100)   if vector else 0

        # format amount nicely
        def fmt_amount(amt):
            if amt >= 100000:
                return f"₹{amt/100000:.1f}L"
            elif amt >= 1000:
                return f"₹{amt/1000:.0f}K"
            return f"₹{amt}"

        amt_str = fmt_amount(avg_amt)

        if arch != nudge_archetype:
            return "Recommended based on your financial activity"

        # dynamic explanations by category
        explanations = {
            "investment": [
                f"Your avg investment is {amt_str} — "
                f"time to grow it further",
                f"Based on {txn_cnt} investment transactions "
                f"in the last 6 months",
                f"Investors in {city} like you are acting on this",
            ],
            "savings": [
                f"You've been saving consistently — "
                f"here's your next step",
                f"Based on your saving pattern "
                f"across {txn_cnt} transactions",
                f"Smart move for a {age}-year-old "
                f"building long-term wealth",
            ],
            "emi": [
                f"Your avg EMI payment is {amt_str} — "
                f"stay ahead of it",
                f"Based on your loan repayment history "
                f"in {city}",
                f"Borrowers who act early save "
                f"significantly on interest",
            ],
            "reminder": [
                f"You've paid {txn_cnt} bills on time — "
                f"keep the streak going",
                f"Based on your bill payment pattern",
                f"Don't break your on-time payment streak",
            ],
            "alert": [
                f"Your spending avg is {amt_str} per transaction"
                f" — here's what to watch",
                f"Based on {txn_cnt} spending transactions",
                f"Spenders in {city} with your profile "
                f"benefit from this",
            ],
            "budgeting": [
                f"Your {txn_cnt} transactions show "
                f"a pattern worth optimising",
                f"Based on your monthly spending behaviour",
                f"Setting limits helped users like you "
                f"save 18% monthly",
            ],
            "rewards": [
                f"You have unclaimed rewards "
                f"from {txn_cnt} transactions",
                f"Most {city} users with your profile "
                f"miss this",
                f"Takes 30 seconds — "
                f"worth {fmt_amount(avg_amt // 10)} avg",
            ],
            "insurance": [
                f"At {age}, this is the right time "
                f"to secure your finances",
                f"Based on your financial profile in {city}",
                f"Users your age who act now "
                f"pay 40% lower premiums",
            ],
            "offers": [
                f"Personalised offer based on your "
                f"{txn_cnt} transactions",
                f"Exclusive to {city} users "
                f"with your spending pattern",
                f"Your {amt_str} avg spend qualifies "
                f"you for this",
            ],
        }

        options = explanations.get(
            nudge_category,
            [f"Recommended based on your activity in {city}"]
        )

        # pick explanation based on user_id hash for consistency
        idx = sum(ord(c) for c in user_id) % len(options)
        return options[idx]

    def _apply_context_boost(self, nudges: list,
                              current_action: str) -> list:
        """
        Boosts nudge scores based on what the user just did.
        
        current_action → which categories get boosted:
        
        paid_bill       → savings, investment (money freed up)
        checked_stocks  → investment (already in investment mindset)
        checked_balance → alert, budgeting (awareness moment)
        received_salary → investment, savings, emi (salary day)
        opened_app      → no boost (neutral session)
        made_payment    → rewards, offers (post-payment moment)
        """
        CONTEXT_BOOSTS = {
            "paid_bill": {
                "savings":    0.15,
                "investment": 0.10,
            },
            "checked_stocks": {
                "investment": 0.20,
            },
            "checked_balance": {
                "alert":      0.15,
                "budgeting":  0.10,
            },
            "received_salary": {
                "investment": 0.15,
                "savings":    0.15,
                "emi":        0.20,
            },
            "made_payment": {
                "rewards":    0.20,
                "offers":     0.15,
            },
            "opened_app": {},   # no boost
        }

        boosts = CONTEXT_BOOSTS.get(current_action, {})
        if not boosts:
            return nudges

        # apply boost to matching categories
        for nudge in nudges:
            category = nudge.get("category", "")
            if category in boosts:
                original_score   = nudge["score"]
                nudge["score"]   = round(
                    min(original_score + boosts[category], 1.0), 4
                )
                nudge["context_boosted"] = True
                nudge["boost_amount"]    = boosts[category]
            else:
                nudge["context_boosted"] = False

        # re-sort by boosted score
        nudges.sort(key=lambda x: x["score"], reverse=True)

        # update ranks after re-sort
        for i, nudge in enumerate(nudges):
            nudge["rank"] = i + 1

        return nudges

    def _get_nudge_impression_counts(self, user_id: str) -> dict:
        """
        Returns how many times each nudge was shown
        to this user without a click — fatigue counter
        """
        conn = sqlite3.connect(settings.DB_PATH)
        cur  = conn.cursor()

        # impressions per nudge for this user
        cur.execute("""
            SELECT nudge_id, COUNT(*) as impressions
            FROM events
            WHERE user_id = ?
            AND   action  = 'impression'
            GROUP BY nudge_id
        """, (user_id,))
        impressions = {row[0]: row[1] for row in cur.fetchall()}

        # clicks per nudge for this user
        cur.execute("""
            SELECT nudge_id, COUNT(*) as clicks
            FROM events
            WHERE user_id = ?
            AND   action  = 'click'
            GROUP BY nudge_id
        """, (user_id,))
        clicks = {row[0]: row[1] for row in cur.fetchall()}
        conn.close()

        # unseen impressions = impressions - clicks
        # if user clicked, fatigue resets
        fatigue = {}
        for nudge_id, imp_count in impressions.items():
            click_count       = clicks.get(nudge_id, 0)
            fatigue[nudge_id] = max(imp_count - click_count, 0)

        return fatigue

    def _apply_fatigue_filter(self, results: list,
                               user_id: str,
                               fatigue_threshold: int = 3) -> list:
        """
        Filters out nudges shown too many times without a click.
        
        fatigue_threshold = 3:
        If user has seen a nudge 3+ times without clicking,
        replace it with the next best non-fatigued nudge.
        
        This prevents the system from spamming the same nudge
        repeatedly — a real production requirement.
        """
        fatigue_counts = self._get_nudge_impression_counts(user_id)

        # separate fatigued vs fresh nudges
        fresh_nudges   = []
        fatigued_nudges = []

        for nudge in results:
            nid   = nudge["nudge_id"]
            count = fatigue_counts.get(nid, 0)
            if count >= fatigue_threshold:
                nudge["fatigued"]          = True
                nudge["fatigue_impressions"] = count
                fatigued_nudges.append(nudge)
            else:
                nudge["fatigued"]          = False
                nudge["fatigue_impressions"] = count
                fresh_nudges.append(nudge)

        # if all fresh — return as is
        if len(fatigued_nudges) == 0:
            return results

        # if some fatigued — fetch replacements from full catalog
        if len(fresh_nudges) < len(results):
            needed      = len(results) - len(fresh_nudges)
            shown_ids   = {n["nudge_id"] for n in results}

            # get all nudges not already in results
            replacements = [
                n for n in self.nudge_meta
                if n["nudge_id"] not in shown_ids
                and fatigue_counts.get(n["nudge_id"], 0)
                    < fatigue_threshold
            ]

            # add up to needed replacements
            for i, rep in enumerate(replacements[:needed]):
                fresh_nudges.append({
                    "rank":              len(fresh_nudges) + 1,
                    "nudge_id":         rep["nudge_id"],
                    "title":            rep["title"],
                    "archetype":        rep["archetype"],
                    "category":         rep["category"],
                    "score":            0.4,
                    "explanation":      "Fresh recommendation for you",
                    "context_boosted":  False,
                    "fatigued":         False,
                    "fatigue_impressions": 0,
                    "is_replacement":   True,
                })

        # re-rank
        for i, nudge in enumerate(fresh_nudges[:len(results)]):
            nudge["rank"] = i + 1

        return fresh_nudges[:len(results)]

    def get_personalization_score(self, user_id: str) -> dict:
        """
        Measures how much a user's embedding has drifted
        from their archetype's average embedding.
        
        Score 0-100:
        0-20  → new user, barely personalized
        20-50 → some personalization
        50-80 → well personalized
        80+   → heavily adapted to this specific user
        """
        user_data = self.user_cache.get(user_id)
        if not user_data:
            return {"score": 0, "label": "new user"}

        user_vec   = np.array(user_data["vector"])
        archetype  = user_data["archetype"]

        # compute archetype center — average vector of all
        # users with same archetype
        archetype_vecs = [
            np.array(u["vector"])
            for u in self.user_cache.values()
            if u["archetype"] == archetype
        ]

        if not archetype_vecs:
            return {"score": 0, "label": "unknown"}

        archetype_center = np.mean(archetype_vecs, axis=0)

        # cosine distance from archetype center
        # higher distance = more personalized
        dot     = np.dot(user_vec, archetype_center)
        norm    = (np.linalg.norm(user_vec) *
                   np.linalg.norm(archetype_center))
        cosine_sim  = dot / max(norm, 1e-8)
        cosine_dist = 1 - cosine_sim

        # scale to 0-100
        # max expected drift after online learning ~ 0.15
        score = min(int((cosine_dist / 0.15) * 100), 100)

        if score < 20:
            label = "new user"
        elif score < 50:
            label = "learning your preferences"
        elif score < 80:
            label = "well personalized"
        else:
            label = "highly personalized"

        return {
            "score":     score,
            "label":     label,
            "archetype": archetype,
            "drift":     round(float(cosine_dist), 6),
        }

    def _mmr_rerank(self, user_emb: np.ndarray,
                    candidate_indices: list,
                    candidate_distances: list,
                    k: int,
                    lambda_param: float = 0.7) -> list:
        """
        Maximal Marginal Relevance reranking.
        Balances relevance with diversity.
        
        lambda_param: 0.7 = 70% relevance, 30% diversity
        """
        if len(candidate_indices) <= k:
            return list(zip(candidate_indices, candidate_distances))

        # get nudge embeddings for all candidates
        nudge_embs = []
        for idx in candidate_indices:
            vec = torch.tensor(
                [list(self.nudge_feats[
                    self.nudge_meta[idx]["nudge_id"]
                ]["vector"])],
                dtype=torch.float32
            )
            with torch.no_grad():
                emb = self.model.get_nudge_embedding(vec)
            nudge_embs.append(emb.cpu().numpy()[0])

        selected        = []
        selected_embs   = []
        remaining       = list(range(len(candidate_indices)))

        while len(selected) < k and remaining:
            best_score = -np.inf
            best_idx   = None

            for r in remaining:
                # relevance score (from FAISS distance)
                relevance = candidate_distances[r]

                # diversity score
                if not selected_embs:
                    diversity = 1.0
                else:
                    sims_to_selected = [
                        float(np.dot(nudge_embs[r], sel_emb))
                        for sel_emb in selected_embs
                    ]
                    diversity = 1 - max(sims_to_selected)

                mmr_score = (lambda_param * relevance +
                             (1 - lambda_param) * diversity)

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx   = r

            selected.append(candidate_indices[best_idx])
            selected_embs.append(nudge_embs[best_idx])
            remaining.remove(best_idx)

        distances_map = {
            candidate_indices[i]: candidate_distances[i]
            for i in range(len(candidate_indices))
        }
        return [(idx, distances_map[idx]) for idx in selected]

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