import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from backend.app.services.recommender import recommender
from backend.app.services.online_learner import online_learner

def test():
    # load everything
    recommender.load()

    # test 1 — known investor user
    print("\n── test 1: investor user ──")
    result = recommender.recommend("user_0001", top_k=3)
    print(f"user     : {result['user_id']}")
    print(f"archetype: {result['archetype']}")
    print(f"latency  : {result['latency_ms']}ms")
    for n in result["nudges"]:
        print(f"  {n['rank']}. [{n['nudge_id']}] "
              f"{n['title'][:45]:<45} score={n['score']}")
        print(f"     → {n['explanation']}")

    # test 2 — unknown user (cold start)
    print("\n── test 2: unknown user (cold start) ──")
    result2 = recommender.recommend("user_9999", top_k=3)
    print(f"source: {result2['source']}")
    for n in result2["nudges"]:
        print(f"  {n['rank']}. {n['title']}")

    # test 3 — online learning update
    print("\n── test 3: online learning ──")
    user    = "user_0001"
    vec     = recommender.user_cache[user]["vector"]
    new_vec = online_learner.update(vec, "n09", "click")
    shift   = online_learner.get_shift_magnitude(vec, new_vec)
    print(f"user clicked EMI nudge n09")
    print(f"embedding shift magnitude: {shift:.6f}")
    recommender.update_user_embedding(user, new_vec)

    # re-recommend after update
    result3 = recommender.recommend(user, top_k=3)
    print("nudges after click on EMI nudge:")
    for n in result3["nudges"]:
        print(f"  {n['rank']}. [{n['nudge_id']}] {n['title'][:45]}")

    print("\nall tests passed.")

if __name__ == "__main__":
    test()