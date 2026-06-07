import numpy as np
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.app.core.config import settings

class OnlineLearner:
    """
    Updates user embeddings in real time based on interactions.
    
    When user clicks a nudge  → pull their vector toward that nudge type
    When user skips a nudge   → push their vector away slightly
    
    This is the online learning layer — no full retraining needed.
    Uses simple gradient-free embedding shift (production-viable approach
    for low-latency systems where River SGD would add too much latency).
    """

    def __init__(self, lr: float = None):
        self.lr          = lr or settings.LEARNING_RATE
        self.noise_decay = settings.NOISE_DECAY
        self.nudge_feats = {}
        self._load_nudge_features()

    def _load_nudge_features(self):
        with open(settings.NUDGE_FEAT_PATH) as f:
            nudges = json.load(f)
        self.nudge_feats = {
            n["nudge_id"]: n for n in nudges
        }
        print(f"online learner loaded {len(self.nudge_feats)} nudge features")

    def update(self, user_vector: list, nudge_id: str,
               action: str) -> list:
        """
        action: 'click'  → move user vector toward nudge features
                'skip'   → move user vector away slightly
                'dismiss'→ stronger push away
        """
        if nudge_id not in self.nudge_feats:
            return user_vector

        nudge_vec  = np.array(
            self.nudge_feats[nudge_id]["vector"], dtype=np.float32
        )
        user_vec   = np.array(user_vector, dtype=np.float32)

        action_lr  = {
            "click":   +self.lr,
            "skip":    -self.lr * 0.3,
            "dismiss": -self.lr * 0.6,
        }.get(action, 0.0)

        # nudge vectors are 16-dim, user vectors are 45-dim
        # update only the overlapping behavioral dims (first 15)
        overlap_dims = min(len(nudge_vec), 15)
        delta        = nudge_vec[:overlap_dims] - user_vec[:overlap_dims]
        user_vec[:overlap_dims] += action_lr * delta

        # clip to valid range [0, 1]
        user_vec = np.clip(user_vec, 0.0, 1.0)
        return user_vec.tolist()

    def get_shift_magnitude(self, old_vec: list,
                             new_vec: list) -> float:
        """how much did the embedding move — for logging"""
        old = np.array(old_vec)
        new = np.array(new_vec)
        return float(np.linalg.norm(new - old))


# singleton
online_learner = OnlineLearner()