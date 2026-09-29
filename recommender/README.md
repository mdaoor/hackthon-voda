# Personalization tool (v6 recommender)

`personalization_model_v6.pkl` (128 MB) is the v6 model behind `submission_v6.csv` (public leaderboard 0.26946), packaged for agent use. Verified: it reproduces `submission_v6.csv` exactly for all 6,000 test customers.

## Requirements
- Python with `numpy`, `pandas`, `scipy`, `lightgbm` (4.x).
- The code in `src/v2/` must be importable when unpickling. The pickle stores the fitted state, not the code: `personalization_tool.py`, `feats.py` and `core.py` are all needed.

## Load
```python
import sys
sys.path.insert(0, r"D:/recommender_system/src/v2")
from personalization_tool import load_tool

tool = load_tool(r"D:/recommender_system/tool/personalization_model_v6.pkl")
```

## Calls

| Call | Use | Speed |
|---|---|---|
| `tool.recommend(user_id="U024000", k=5)` | Known customer (all 30,000 train and test customers are stored) | ~0.4 s |
| `tool.recommend(user_id="U024000", events=[...])` | Known customer + events from the current session | ~0.4 s |
| `tool.recommend(profile={...}, events=[...])` | New customer | ~0.4 s |
| `tool.recommend_batch(user_ids, k=5)` | Many known customers | ~16 ms per customer |
| `tool.describe()` | Model card: task, inputs, quality, drivers, limitations | instant |

Each recommendation is a dict:
```json
{"rank": 1, "product_id": "I0662", "product_name": "Gaming 13", "category": "gaming",
 "domain": "entertainment", "price": 80.92, "offer_discount": 0.2, "consensus_rank": 1.0,
 "reasons": ["previously engaged (5 views, 1 carts, 1 favourites; last 19 days ago)",
             "28% of recent activity is in gaming", "20% offer discount", "matches preferred quality tier"]}
```
`consensus_rank` is the product's average position across the ensemble's four models; 1.0 means all of them ranked it first. Pass `explain=False` to skip the reasons.

Event format: `{"product_id": "I0740", "event_type": "cart", "event_date": "2026-05-30", "app_section": "shop"}`. Event types are view, cart, favorite, purchase, redeem, return and cancel. Dates must be before the 2026-06-01 cutoff. Example: a cart plus a favourite moves a product from rank 50 to rank 5.

## Registering it with an agent
`tool_schema.json` is a ready-made tool definition (name, description, JSON input schema). Route its arguments straight to `tool.recommend(**args)` and return the list as the tool result.

## Behaviour and limits
- **Eligibility is always enforced:** launched by the cutoff, available in the customer's region, compatible with their device OS, and not an already-owned non-repeatable item.
- **Scores are as of 2026-06-01,** using history strictly before that date; recent popularity reflects May 2026.
- **Recommendation quality depends on history.** About 0.36 for customers with 80+ events, 0.15 with 1–5 events, and about 0.04 with no history (profile and declared interests only).
- **Rebuilding.** Run `python final.py <lambdarank|binary> <seed>` for seeds 0 and 1 (saves the boosters), then `python build_tool.py`, then `python verify_tool.py`.
