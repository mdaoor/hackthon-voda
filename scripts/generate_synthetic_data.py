"""Generate a synthetic dataset in the same shape as the challenge files.

Used for local development and demos until the organisers' full datasets
(train, products, interactions, recommender output) are dropped into data/.
Deterministic (seeded). Semicolon-delimited like the starter CSVs.

    python scripts/generate_synthetic_data.py            # writes data/synthetic/
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic"
REGIONS = ["north", "south", "east", "west", "central", "coastal"]
OS_LIST = ["android", "ios"]
TAGS = ["premium", "social", "practical", "compact", "family", "eco", "smart",
        "classic", "sporty", "cozy", "minimal", "durable", "portable", "gaming"]

# domain -> category -> (label, price range, repeatable, subscription, os_specific)
CATALOGUE = {
    "electronics": {
        "audio": ("Audio", (35, 260), 0, 0, False),
        "smartphone": ("Smartphone", (80, 900), 0, 0, True),
        "wearable": ("Wearable", (40, 420), 0, 0, True),
        "tablet": ("Tablet", (120, 750), 0, 0, True),
        "laptop": ("Laptop", (350, 1600), 0, 0, False),
        "tv": ("TV", (180, 1400), 0, 0, False),
        "smart_home": ("Smart Home", (20, 240), 0, 0, False),
        "phone_accessory": ("Phone Accessory", (8, 70), 1, 0, True),
    },
    "home": {
        "furniture": ("Furniture", (60, 1100), 0, 0, False),
        "kitchen": ("Kitchen", (15, 380), 0, 0, False),
        "appliance": ("Appliance", (90, 950), 0, 0, False),
        "bedding": ("Bedding", (20, 260), 1, 0, False),
        "lighting": ("Lighting", (12, 180), 1, 0, False),
        "cleaning": ("Cleaning", (5, 60), 1, 0, False),
        "decor": ("Decor", (10, 150), 1, 0, False),
    },
    "lifestyle": {
        "fitness": ("Fitness", (15, 480), 0, 0, False),
        "travel": ("Travel", (25, 320), 0, 0, False),
        "beauty": ("Beauty", (6, 90), 1, 0, False),
    },
    "services": {
        "streaming": ("Streaming Plan", (5, 25), 1, 1, False),
        "mobile_plan": ("Mobile Plan", (10, 60), 1, 1, False),
        "device_care": ("Device Care", (4, 20), 1, 1, False),
    },
}

STARTER_PRODUCTS = [  # identical to the starter sample so behaviour matches the starter tests
    ["I0000", "electronics", "audio", "B00", "Audio 01", "91.8", "1", "premium social", "0", "0", "2025/10/01", " ".join(REGIONS), "any", "0.0", "Synthetic audio offering"],
    ["I0001", "electronics", "audio", "B01", "Audio 02", "120.03", "2", "social practical", "0", "0", "2025/10/01", " ".join(REGIONS), "any", "0.3", "Synthetic audio offering"],
    ["I0002", "electronics", "audio", "B02", "Audio 03", "118.07", "3", "practical compact", "0", "0", "2025/10/01", " ".join(REGIONS), "any", "0.0", "Synthetic audio offering"],
    ["I0003", "electronics", "audio", "B03", "Audio 04", "173.79", "4", "premium family", "0", "0", "2025/10/01", " ".join(REGIONS), "any", "0.1", "Synthetic audio offering"],
]

PRODUCT_COLS = ["product_id", "domain", "category", "brand_id", "product_name", "price", "quality_tier",
                "style_tags", "is_repeatable", "is_subscription", "available_from", "available_regions",
                "compatible_os", "offer_discount", "description"]
CUSTOMER_COLS = ["user_id", "snapshot_date", "region", "age_band", "household_size", "membership_tier",
                 "device_type", "device_os", "tenure_months", "monthly_budget", "preferred_quality",
                 "language", "marketing_opt_in", "owned_items", "declared_interests", "relevant_items"]
INTERACTION_COLS = ["event_id", "user_id", "product_id", "app_section", "event_type", "event_date",
                    "browse_duration_sec", "session_depth", "related_event_id", "transaction_value", "quantity"]


def write(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(cols)
        w.writerows(rows)


def main(seed=7, n_customers=40):
    rnd = random.Random(seed)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- products ----------------
    products = [list(r) for r in STARTER_PRODUCTS]
    pid = 4
    for domain, cats in CATALOGUE.items():
        for cat, (label, (lo, hi), rep, sub, os_specific) in cats.items():
            count = 12 if domain != "services" else 5
            start = 5 if cat == "audio" else 1
            for i in range(start, start + count):
                q = rnd.randint(1, 5)
                price = round(lo + (hi - lo) * (((q - 1) / 4) ** 1.7) * rnd.uniform(0.6, 1.15)
                              + rnd.uniform(0, (hi - lo) * 0.08), 2)
                price = max(lo, min(hi, price))
                tags = " ".join(rnd.sample(TAGS, 2))
                regions = " ".join(REGIONS) if rnd.random() < 0.8 else " ".join(sorted(rnd.sample(REGIONS, 3)))
                os_ = rnd.choice(OS_LIST) if os_specific and rnd.random() < 0.6 else "any"
                avail = "2025/%02d/01" % rnd.randint(1, 12)
                if rnd.random() < 0.06:
                    avail = "2026/06/01"  # not launched on the demo date
                disc = rnd.choice(["0.0"] * 6 + ["0.1", "0.15", "0.2", "0.25", "0.3"])
                desc = f"Synthetic {label.lower()} offering"
                products.append([f"I{pid:04d}", domain, cat, f"B{rnd.randint(0, 29):02d}", f"{label} {i:02d}",
                                 f"{price:.2f}", str(q), tags, str(rep), str(sub), avail, regions, os_, disc, desc])
                pid += 1
    write(OUT / "products.csv", PRODUCT_COLS, products)
    by_cat = {}
    for p in products:
        by_cat.setdefault(p[2], []).append(p)

    # ---------------- customers (train) ----------------
    interests_pool = ["music", "gaming", "fitness", "cooking", "travel", "home", "tech", "beauty", "family", "movies"]
    customers = [
        ["U000000", "2026/04/01", "west", "35-49", "5", "basic", "entry", "ios", "45", "21.01", "2.0", "lang_c", "1", "I0012 I0016 I0022 I0042", "", "I0001"],
        ["U000001", "2026/04/01", "north", "25-34", "3", "basic", "entry", "android", "37", "67.69", "4.0", "lang_c", "0", "", "", "I0003"],
    ]
    for u in range(2, n_customers):
        owned = " ".join(rnd.choice(products)[0] for _ in range(rnd.randint(0, 4)))
        customers.append([
            f"U{u:06d}", "2026/04/01", rnd.choice(REGIONS), rnd.choice(["18-24", "25-34", "35-49", "50-64", "65+"]),
            str(rnd.randint(1, 6)), rnd.choice(["basic", "basic", "silver", "gold", "platinum"]),
            rnd.choice(["entry", "mid", "flagship"]), rnd.choice(OS_LIST), str(rnd.randint(1, 120)),
            f"{rnd.uniform(20, 400):.2f}", f"{rnd.randint(1, 5)}.0", rnd.choice(["lang_a", "lang_b", "lang_c"]),
            rnd.choice(["0", "1", "1"]), owned, " ".join(rnd.sample(interests_pool, rnd.randint(0, 3))),
            " ".join(rnd.choice(products)[0] for _ in range(3)),  # ML target labels - must stay hidden
        ])
    write(OUT / "train.csv", CUSTOMER_COLS, customers)

    # ---------------- interactions ----------------
    rows, eid = [], 1
    bundles = [("smartphone", "phone_accessory"), ("smartphone", "audio"), ("smartphone", "device_care"),
               ("furniture", "lighting"), ("furniture", "decor"), ("appliance", "kitchen"),
               ("bedding", "decor"), ("tv", "streaming"), ("tv", "audio"), ("laptop", "phone_accessory"),
               ("fitness", "wearable"), ("kitchen", "cleaning")]
    start = date(2025, 10, 1)
    for c in customers:
        uid = c[0]
        for _ in range(rnd.randint(15, 60)):
            a, b = rnd.choice(bundles)
            d = start + timedelta(days=rnd.randint(0, 181))  # bundle items share a shopping day
            for cat in (a, b) if rnd.random() < 0.55 else (a,):
                p = rnd.choice(by_cat[cat])
                section = rnd.choice(["home", "shop", "shop", "shop", "rewards"])
                journey = ["view"]
                r = rnd.random()
                if r < 0.35:
                    journey.append("add_to_cart")
                if r < 0.18:
                    journey.append("purchase")
                if rnd.random() < 0.08:
                    journey.append("wishlist")
                if section == "rewards" and rnd.random() < 0.4:
                    journey.append("redeem_reward")
                parent = ""
                for et in journey:
                    value = qty = ""
                    if et == "purchase":
                        qty = "1"
                        value = f"{float(p[5]) * (1 - float(p[13])):.2f}"
                    rows.append([f"E{eid:07d}", uid, p[0], section, et, d.strftime("%Y/%m/%d"),
                                 str(rnd.randint(5, 600)), str(rnd.randint(1, 12)), parent, value, qty])
                    parent = f"E{eid:07d}"
                    eid += 1
    write(OUT / "interactions.csv", INTERACTION_COLS, rows)

    # ---------------- recommender output ----------------
    rec_rows = []
    for c in customers:
        pool = rnd.sample(products, 12)
        for rank, p in enumerate(pool, 1):
            rec_rows.append([c[0], str(rank), p[0], f"{1 - rank / 20:.3f}"])
    write(OUT / "recommendations.csv", ["user_id", "rank", "product_id", "score"], rec_rows)
    print(f"products={len(products)} customers={len(customers)} interactions={len(rows)} recs={len(rec_rows)} -> {OUT}")


if __name__ == "__main__":
    main()
