import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import random
import sqlite3
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from backend.app.services.recommender import recommender
from backend.app.services.ab_testing  import ab_testing

random.seed(42)

N_USERS        = 200   # simulate 200 users
N_SESSIONS     = 5     # each user has 5 sessions
DB_PATH        = "data/processed/finnudge.db"

# realistic CTR by archetype — treatment group clicks more
ARCHETYPE_CTR = {
    "investor":   {"treatment": 0.45, "control": 0.20},
    "saver":      {"treatment": 0.40, "control": 0.18},
    "spender":    {"treatment": 0.35, "control": 0.15},
    "bill_payer": {"treatment": 0.50, "control": 0.22},
    "borrower":   {"treatment": 0.42, "control": 0.19},
}


def clear_existing_events():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("DELETE FROM events")
    conn.commit()
    conn.close()
    print("cleared existing events")


def simulate():
    recommender.load()

    # pick 200 random users
    all_user_ids = list(recommender.user_cache.keys())
    selected     = random.sample(all_user_ids, N_USERS)

    total_impressions = 0
    total_clicks      = 0

    print(f"\nsimulating {N_USERS} users × "
          f"{N_SESSIONS} sessions...")

    for user_id in selected:
        user_data = recommender.user_cache[user_id]
        archetype = user_data["archetype"]
        ab_group  = ab_testing.assign_group(user_id)
        ctr_rates = ARCHETYPE_CTR[archetype]

        for session in range(N_SESSIONS):
            # get recommendations
            if ab_group == "treatment":
                result  = recommender.recommend(user_id, top_k=3)
                nudges  = result["nudges"]
            else:
                # control → random nudges
                all_nudges = recommender.get_all_nudges()
                sample     = random.sample(all_nudges, 3)
                nudges     = [{
                    "nudge_id": n["nudge_id"],
                    "title":    n["title"],
                } for n in sample]

            # simulate timestamp spread over last 30 days
            days_ago  = random.randint(0, 29)
            hours_ago = random.randint(0, 23)
            ts = (datetime.now()
                  - timedelta(days=days_ago, hours=hours_ago)
                  ).isoformat()

            # log impressions
            for nudge in nudges:
                ab_testing.log_event(
                    user_id,
                    nudge["nudge_id"],
                    "impression",
                    ab_group
                )
                total_impressions += 1

            # simulate clicks based on CTR
            click_rate = ctr_rates[ab_group]
            for nudge in nudges:
                if random.random() < click_rate:
                    ab_testing.log_event(
                        user_id,
                        nudge["nudge_id"],
                        "click",
                        ab_group
                    )
                    total_clicks += 1

                    # online learning update on click
                    if ab_group == "treatment":
                        from backend.app.services.online_learner \
                            import online_learner
                        vec     = recommender.user_cache\
                                  [user_id]["vector"]
                        new_vec = online_learner.update(
                            vec, nudge["nudge_id"], "click"
                        )
                        recommender.update_user_embedding(
                            user_id, new_vec
                        )

    print(f"simulation complete:")
    print(f"  total impressions : {total_impressions}")
    print(f"  total clicks      : {total_clicks}")
    print(f"  overall CTR       : "
          f"{total_clicks/max(total_impressions,1):.1%}")

    # print A/B results
    results = ab_testing.get_results()
    print(f"\n── A/B test results ──")
    print(f"  control   CTR : "
          f"{results['control']['ctr']:.1%} "
          f"({results['control']['users']} users)")
    print(f"  treatment CTR : "
          f"{results['treatment']['ctr']:.1%} "
          f"({results['treatment']['users']} users)")
    print(f"  lift          : "
          f"{results['stats']['lift_percent']}%")
    print(f"  p-value       : "
          f"{results['stats']['p_value']}")
    print(f"  significant   : "
          f"{results['stats']['is_significant']}")
    print(f"  verdict       : "
          f"{results['stats']['verdict']}")


if __name__ == "__main__":
    clear_existing_events()
    simulate()