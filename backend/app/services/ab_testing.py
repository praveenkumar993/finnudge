import sqlite3
import numpy as np
from scipy import stats
from datetime import datetime
from backend.app.core.config import settings


class ABTestingService:
    """
    A/B Testing Framework for FinNudge
    
    Control group   → random nudges (no personalization)
    Treatment group → two-tower personalized nudges
    
    Measures:
    - CTR (click-through rate) per group
    - Statistical significance (p-value)
    - Confidence interval
    - Lift percentage
    """

    def __init__(self):
        self.control_group   = "control"
        self.treatment_group = "treatment"

    def assign_group(self, user_id: str) -> str:
        """
        Consistent hash-based assignment.
        Same user always gets same group — deterministic.
        50/50 split between control and treatment.
        """
        hash_val = sum(ord(c) for c in user_id)
        return self.treatment_group if hash_val % 2 == 0 \
               else self.control_group

    def log_event(self, user_id: str, nudge_id: str,
                  action: str, ab_group: str):
        conn = sqlite3.connect(settings.DB_PATH)
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO events
            (user_id, nudge_id, action, ab_group, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, nudge_id, action,
              ab_group, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_results(self) -> dict:
        conn = sqlite3.connect(settings.DB_PATH)
        cur  = conn.cursor()

        # get per-user CTR for each group
        cur.execute("""
            SELECT
                ab_group,
                user_id,
                SUM(CASE WHEN action='impression' THEN 1 ELSE 0 END) as impressions,
                SUM(CASE WHEN action='click'      THEN 1 ELSE 0 END) as clicks
            FROM events
            WHERE ab_group IN ('control', 'treatment')
            AND   action   IN ('impression', 'click')
            GROUP BY ab_group, user_id
            HAVING impressions > 0
        """)
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return self._empty_results()

        # separate by group
        control_data   = [(r[2], r[3]) for r in rows
                          if r[0] == "control"]
        treatment_data = [(r[2], r[3]) for r in rows
                          if r[0] == "treatment"]

        if not control_data or not treatment_data:
            return self._empty_results()

        # CTR per user
        control_ctrs   = [c / max(i, 1)
                          for i, c in control_data]
        treatment_ctrs = [c / max(i, 1)
                          for i, c in treatment_data]

        control_ctr   = np.mean(control_ctrs)
        treatment_ctr = np.mean(treatment_ctrs)

        # Welch's t-test — does not assume equal variance
        t_stat, p_value = stats.ttest_ind(
            treatment_ctrs,
            control_ctrs,
            equal_var=False
        )

        # 95% confidence interval for treatment CTR
        n   = len(treatment_ctrs)
        se  = stats.sem(treatment_ctrs)
        ci  = stats.t.interval(
            0.95, df=n-1,
            loc=treatment_ctr,
            scale=se
        )

        # lift = how much better is treatment vs control
        lift = ((treatment_ctr - control_ctr) /
                max(control_ctr, 0.001)) * 100

        # significance threshold: p < 0.05
        is_significant = bool(p_value < 0.05)

        return {
            "control": {
                "users":       len(control_data),
                "impressions": sum(i for i, _ in control_data),
                "clicks":      sum(c for _, c in control_data),
                "ctr":         round(float(control_ctr), 4),
            },
            "treatment": {
                "users":       len(treatment_data),
                "impressions": sum(i for i, _ in treatment_data),
                "clicks":      sum(c for _, c in treatment_data),
                "ctr":         round(float(treatment_ctr), 4),
            },
            "stats": {
                "lift_percent":     round(float(lift), 2),
                "p_value":          round(float(p_value), 4),
                "t_statistic":      round(float(t_stat), 4),
                "is_significant":   is_significant,
                "confidence_level": "95%",
                "ci_lower":         round(float(ci[0]), 4),
                "ci_upper":         round(float(ci[1]), 4),
                "verdict": (
                    "✓ Personalization wins — statistically significant"
                    if is_significant and lift > 0
                    else "⏳ Not enough data yet"
                    if not is_significant
                    else "✗ No significant improvement"
                ),
            }
        }

    def _empty_results(self):
        return {
            "control":   {"users": 0, "impressions": 0,
                          "clicks": 0, "ctr": 0},
            "treatment": {"users": 0, "impressions": 0,
                          "clicks": 0, "ctr": 0},
            "stats": {
                "lift_percent":   0,
                "p_value":        1.0,
                "is_significant": False,
                "verdict":        "No data yet"
            }
        }


# singleton
ab_testing = ABTestingService()