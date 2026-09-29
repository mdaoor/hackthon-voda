"""Read-only knowledge layer around the starter Store.

* Catalogue   - search, eligibility (delegated to the starter's Store.product), final prices.
* Profiles    - sanitized customer profile (drops ML target labels such as relevant_items).
* Interactions- behaviour insights across Home / Shop / Rewards + category co-occurrence.
* Recommender - top-five product inference from a trusted pickled model.
* Personalizer- explainable scoring used by search, recommendations and cross-sell.

Interaction columns are resolved by aliases because the organisers only
describe the fields, not the headers.
"""
from __future__ import annotations

import csv
import pickle
import re
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

from starter.store import ToolError, money

HIDDEN_PROFILE_FIELDS = {"relevant_items"}  # ML target labels - never shown to the model


# --------------------------------------------------------------------------- utils
def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        sample = f.read(20000)
        f.seek(0)
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
        except csv.Error:
            delim = ";"
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(f, delimiter=delim)]


def resolve(columns, exact=(), contains=()):
    lower = {c.lower().strip(): c for c in columns}
    for name in exact:
        if name in lower:
            return lower[name]
    for frag in contains:
        for lc, original in lower.items():
            if frag in lc:
                return original
    return None


def parse_date(value: str):
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10].replace("/", "-"))
    except ValueError:
        return None


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def norm(text: str) -> str:
    text = (text or "").lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


SYNONYMS = {
    "phone": ["smartphone", "phone", "mobile"], "phones": ["smartphone", "phone", "mobile"],
    "smartphone": ["smartphone", "phone", "mobile"], "mobile": ["smartphone", "phone", "mobile"],
    "handset": ["smartphone", "phone"], "headphones": ["audio"], "earbuds": ["audio"], "speaker": ["audio"],
    "headset": ["audio"], "music": ["audio"], "watch": ["wearable"], "smartwatch": ["wearable"],
    "tracker": ["wearable", "fitness"], "television": ["tv"], "computer": ["laptop"], "notebook": ["laptop"],
    "sofa": ["furniture"], "couch": ["furniture"], "chair": ["furniture"], "table": ["furniture"],
    "desk": ["furniture"], "bed": ["furniture", "bedding"], "fridge": ["appliance"], "washer": ["appliance"],
    "microwave": ["appliance", "kitchen"], "cookware": ["kitchen"], "pan": ["kitchen"], "kettle": ["kitchen"],
    "lamp": ["lighting"], "light": ["lighting"], "bulb": ["lighting"], "sheets": ["bedding"], "pillow": ["bedding"],
    "accessory": ["accessory"], "case": ["accessory"], "charger": ["accessory"], "cable": ["accessory"], "gym": ["fitness"],
    "luggage": ["travel"], "suitcase": ["travel"], "skincare": ["beauty"], "netflix": ["streaming"],
    "tv subscription": ["streaming"], "plan": ["plan"], "data": ["mobile plan"], "insurance": ["care"],
    "home": ["home"], "house": ["home"], "apartment": ["home"],
}

# Fallback complements when interaction co-occurrence is thin (keyword on category name).
COMPLEMENT_HINTS = {
    "smartphone": ["accessory", "audio", "care", "wearable", "plan"], "phone": ["accessory", "audio", "care"],
    "laptop": ["accessory", "audio", "care"], "tablet": ["accessory", "audio"], "tv": ["streaming", "audio", "smart home"],
    "audio": ["accessory", "streaming"], "wearable": ["fitness", "accessory"], "furniture": ["lighting", "decor", "bedding"],
    "appliance": ["kitchen", "cleaning"], "kitchen": ["cleaning", "appliance"], "bedding": ["decor", "lighting"],
    "fitness": ["wearable", "audio"], "travel": ["accessory", "audio"], "smart home": ["lighting", "audio"],
}


# --------------------------------------------------------------------------- catalogue
class Catalogue:
    def __init__(self, store):
        self.store = store
        self.products = store.products  # same dict the starter uses -> identical prices/eligibility

    @staticmethod
    def final_price(p) -> Decimal:
        return Decimal(money(Decimal(p["price"]) * (1 - Decimal(p.get("offer_discount") or "0"))))

    def exists(self, pid):
        return pid in self.products

    def eligibility(self, user_id, pid):
        try:
            self.store.product(user_id, pid)
            return True, None, None
        except ToolError as e:
            return False, e.code, e.message

    def card(self, pid, user_id=None, extra=None):
        p = self.products[pid]
        disc = to_float(p.get("offer_discount"))
        out = {
            "product_id": pid, "name": p["product_name"], "domain": p["domain"], "category": p["category"],
            "brand": p.get("brand_id", ""), "list_price": money(p["price"]), "discount_pct": round(disc * 100),
            "final_price": money(self.final_price(p)), "quality_tier": int(to_float(p.get("quality_tier"))),
            "style_tags": (p.get("style_tags") or "").split(), "subscription": p.get("is_subscription") == "1",
            "repeatable": p.get("is_repeatable") == "1", "description": p.get("description", ""),
        }
        if user_id:
            ok, code, message = self.eligibility(user_id, pid)
            out["eligible"] = ok
            if not ok:
                out["ineligible_reason"] = code
                out["ineligible_message"] = message
        if extra:
            out.update(extra)
        return out

    def overview(self):
        """domain -> category -> {count, min, max} over final prices."""
        if getattr(self, "_overview", None) is None:
            tree = defaultdict(dict)
            for p in self.products.values():
                price = float(self.final_price(p))
                c = tree[p["domain"]].setdefault(p["category"], {"count": 0, "min": price, "max": price,
                                                                "subscription": p.get("is_subscription") == "1"})
                c["count"] += 1
                c["min"], c["max"] = min(c["min"], price), max(c["max"], price)
            self._overview = {d: dict(sorted(cats.items())) for d, cats in sorted(tree.items())}
        return self._overview

    def overview_text(self):
        lines = []
        for domain, cats in self.overview().items():
            parts = [f"{c} ({v['count']}, {v['min']:.0f}-{v['max']:.0f}{', subscription' if v['subscription'] else ''})"
                     for c, v in cats.items()]
            lines.append(f"- {domain}: " + "; ".join(parts))
        return "\n".join(lines)

    def resolve_category(self, text):
        """Map free text ('phones', 'smart-home', 'home') to catalogue categories/domains.
        Exact category/domain matches win; synonyms next; partial word matches last."""
        if not text:
            return set(), set()
        raw = norm(text)
        t = singular(raw)
        all_cats = {p["category"] for p in self.products.values()}
        all_doms = {p["domain"] for p in self.products.values()}
        exact_c = {c for c in all_cats if t in (norm(c), singular(norm(c)))}
        exact_d = {d for d in all_doms if t in (norm(d), singular(norm(d)))}
        if exact_c or exact_d:
            return exact_c, exact_d
        syn = set(SYNONYMS.get(raw, [])) | set(SYNONYMS.get(t, []))
        syn_c = {c for c in all_cats for s_ in syn if s_ in (norm(c), singular(norm(c)))}
        syn_d = {d for d in all_doms for s_ in syn if s_ in (norm(d), singular(norm(d)))}
        if syn_c or syn_d:
            return syn_c, syn_d
        terms = {t} | syn
        part_c = {c for c in all_cats for term in terms if term and term in norm(c).split()}
        if not part_c:
            part_c = {c for c in all_cats for term in terms if len(term) > 3 and term in norm(c)}
        return part_c, set()


# --------------------------------------------------------------------------- profiles
class Profiles:
    def __init__(self, store):
        self.store = store

    def exists(self, user_id):
        return user_id in self.store.customers

    def get(self, user_id):
        raw = self.store.customer(user_id)
        return {k: v for k, v in raw.items() if k not in HIDDEN_PROFILE_FIELDS}

    def sample_ids(self, n=6):
        return list(self.store.customers)[:n]


# --------------------------------------------------------------------------- interactions
def classify_event(event_type: str) -> str:
    e = (event_type or "").lower()
    if any(k in e for k in ("purchase", "buy", "order", "transaction", "checkout", "paid")):
        return "purchase"
    if any(k in e for k in ("cart", "basket")):
        return "cart"
    if any(k in e for k in ("wish", "save", "like", "fav")):
        return "wishlist"
    if any(k in e for k in ("redeem", "reward", "claim", "point", "coupon")):
        return "reward"
    if "search" in e:
        return "search"
    return "view"


KIND_WEIGHT = {"purchase": 4.0, "cart": 2.5, "wishlist": 2.0, "reward": 1.5, "view": 1.0, "search": 0.5}


class Interactions:
    def __init__(self, path, catalogue: Catalogue):
        self.catalogue = catalogue
        self.by_user = defaultdict(list)
        self.cat_cooc = defaultdict(Counter)
        self.popularity = Counter()
        self.available = False
        self.columns = {}
        if path and Path(path).is_file():
            self._load(Path(path))

    def _load(self, path):
        rows = read_rows(path)
        if not rows:
            return
        cols = list(rows[0].keys())
        c = self.columns = {
            "user": resolve(cols, ("user_id", "customer_id"), ("user", "customer")),
            "product": resolve(cols, ("product_id", "item_id"), ("product", "item")),
            "section": resolve(cols, ("app_section", "section"), ("section", "page", "screen")),
            "type": resolve(cols, ("event_type", "type", "action"), ("event_type", "type", "action")),
            "date": resolve(cols, ("event_date", "date", "timestamp", "event_time"), ("date", "time")),
            "duration": resolve(cols, (), ("duration", "dwell", "seconds")),
            "depth": resolve(cols, (), ("depth",)),
            "value": resolve(cols, ("transaction_value", "amount", "revenue"), ("transaction_value", "amount", "revenue", "value", "spend")),
        }
        if not c["user"] or not c["product"]:
            return
        for r in rows:
            uid, pid = r.get(c["user"], ""), r.get(c["product"], "")
            if not uid:
                continue
            kind = classify_event(r.get(c["type"], "") if c["type"] else "")
            ev = {"product_id": pid, "section": (r.get(c["section"], "") if c["section"] else "").lower(),
                  "event_type": r.get(c["type"], "") if c["type"] else "", "kind": kind,
                  "date": parse_date(r.get(c["date"], "")) if c["date"] else None,
                  "duration": to_float(r.get(c["duration"])) if c["duration"] else 0.0,
                  "depth": to_float(r.get(c["depth"])) if c["depth"] else 0.0,
                  "value": to_float(r.get(c["value"])) if c["value"] else 0.0}
            self.by_user[uid].append(ev)
            if pid in self.catalogue.products:
                self.popularity[pid] += KIND_WEIGHT[kind]
        missions = []  # (user, day) groups approximate a shopping session
        for uid, events in self.by_user.items():
            events.sort(key=lambda e: e["date"] or date.min)
            by_day = defaultdict(set)
            for e in events:
                if e["kind"] in ("purchase", "cart", "wishlist") and e["product_id"] in self.catalogue.products:
                    by_day[e["date"]].add(self.catalogue.products[e["product_id"]]["category"])
            missions.extend(cats for cats in by_day.values())
        single, pair = Counter(), Counter()
        for cats in missions:
            single.update(cats)
            for a in cats:
                for b in cats:
                    if a != b:
                        pair[(a, b)] += 1
        n = len(missions) or 1
        for (a, b), n_ab in pair.items():
            if n_ab >= 2:
                lift = n_ab * n / (single[a] * single[b])
                if lift > 1.2:
                    self.cat_cooc[a][b] = round(lift, 2)
        self.available = True

    def category_affinity(self, user_id):
        scores = Counter()
        for e in self.by_user.get(user_id, []):
            p = self.catalogue.products.get(e["product_id"])
            if p:
                scores[p["category"]] += KIND_WEIGHT[e["kind"]]
        total = sum(scores.values()) or 1
        return {k: v / total for k, v in scores.items()}

    def insights(self, user_id, owned=()):
        events = self.by_user.get(user_id, [])
        if not events:
            return {"available": self.available, "events": 0}
        prods = self.catalogue.products
        name = lambda pid: prods[pid]["product_name"] if pid in prods else pid  # noqa: E731
        purchased = [e for e in events if e["kind"] == "purchase"]
        bought_ids = {e["product_id"] for e in purchased} | set(owned)
        carted_open, seen = [], set()
        for e in reversed(events):
            pid = e["product_id"]
            if e["kind"] == "cart" and pid not in bought_ids and pid not in seen and pid in prods:
                seen.add(pid)
                carted_open.append({"product_id": pid, "name": name(pid), "date": str(e["date"] or "")})
        recent_views, seen = [], set()
        for e in reversed(events):
            if e["kind"] == "view" and e["product_id"] not in seen and e["product_id"] in prods:
                seen.add(e["product_id"])
                recent_views.append({"product_id": e["product_id"], "name": name(e["product_id"]),
                                     "category": prods[e["product_id"]]["category"], "date": str(e["date"] or "")})
            if len(recent_views) >= 6:
                break
        affinity = sorted(self.category_affinity(user_id).items(), key=lambda kv: -kv[1])[:5]
        rewards = [e for e in events if e["section"] == "rewards" or e["kind"] == "reward"]
        return {
            "available": True, "events": len(events),
            "period": f"{events[0]['date']} to {events[-1]['date']}",
            "sections": dict(Counter(e["section"] or "unknown" for e in events)),
            "event_kinds": dict(Counter(e["kind"] for e in events)),
            "top_categories": [{"category": k, "share": round(v, 2)} for k, v in affinity],
            "recent_views": recent_views,
            "open_cart_items": carted_open[:5],
            "recent_purchases": [{"product_id": e["product_id"], "name": name(e["product_id"]), "date": str(e["date"] or "")}
                                 for e in purchased[-5:]][::-1],
            "total_spend": round(sum(e["value"] for e in purchased), 2),
            "rewards_activity": {"events": len(rewards), "redemptions": sum(1 for e in rewards if e["kind"] == "reward"),
                                 "last": str(rewards[-1]["date"]) if rewards else None},
            "avg_browse_seconds": round(sum(e["duration"] for e in events) / len(events)),
            "avg_session_depth": round(sum(e["depth"] for e in events) / len(events), 1),
        }


# --------------------------------------------------------------------------- recommender
class Recommender:
    """Runs a trusted pickled recommendation model and caches its top-five output.

    Preferred model contract: ``recommend(user_id, n=5)``. For compatibility,
    ``recommend_for_user``, ``predict_for_user`` and callable objects are also
    accepted. A collaborative-filtering ``predict(user_id, product_id)`` model
    is supported by scoring the catalogue. Ranked results may contain IDs,
    ``(ID, score)`` pairs, or dictionaries with product/item IDs and scores.

    Pickle can execute code while loading. Only configure artifacts produced by
    and transferred from a trusted source.
    """

    def __init__(self, path, catalogue: Catalogue):
        self.ranked: dict[str, list[str]] = {}
        self.source = "none"
        self.catalogue = catalogue
        self.model = None
        if path and Path(path).is_file():
            try:
                self._load(Path(path))
                self.source = Path(path).name
            except Exception as exc:  # never fail startup on an unexpected layout
                print(f"[recommender] could not parse {path}: {exc}")

    def _load(self, path):
        if path.suffix.lower() not in (".pkl", ".pickle"):
            raise ValueError("recommendation model must be a .pkl or .pickle file")
        with open(path, "rb") as artifact:
            self.model = pickle.load(artifact)

    @staticmethod
    def _normalise(result):
        if hasattr(result, "tolist"):
            result = result.tolist()
        if isinstance(result, dict):
            for key in ("recommendations", "product_ids", "item_ids", "results"):
                if key in result:
                    result = result[key]
                    break
            else:
                result = [item for item, _ in sorted(result.items(), key=lambda pair: to_float(pair[1]), reverse=True)]
        if isinstance(result, str):
            result = [result]
        if not isinstance(result, (list, tuple)):
            try:
                result = list(result)
            except TypeError as exc:
                raise TypeError("model recommendation output must be an iterable of product IDs") from exc
        scored = []
        for position, item in enumerate(result):
            score = None
            if isinstance(item, dict):
                pid = next((item.get(k) for k in ("product_id", "item_id", "id") if item.get(k) is not None), None)
                score = next((item.get(k) for k in ("score", "prediction", "probability") if item.get(k) is not None), None)
            elif isinstance(item, (list, tuple)):
                pid = item[0] if item else None
                score = item[1] if len(item) > 1 else None
            else:
                pid = item
            if pid is not None:
                scored.append((position, str(pid), score))
        if scored and all(score is not None for _, _, score in scored):
            scored.sort(key=lambda row: to_float(row[2]), reverse=True)
        return list(dict.fromkeys(pid for _, pid, _ in scored))[:5]

    def _infer(self, user_id):
        model = self.model
        if model is None:
            return []
        method = next((getattr(model, name) for name in ("recommend", "recommend_for_user", "predict_for_user")
                       if callable(getattr(model, name, None))), None)
        if method is None and callable(model):
            method = model
        if method is None:
            predict = getattr(model, "predict", None)
            if not callable(predict):
                raise TypeError("pickled model must provide recommend(user_id, n=5), predict(user_id, product_id), or be callable")
            scored = []
            for product_id in self.catalogue.products:
                prediction = predict(user_id, product_id)
                score = getattr(prediction, "est", prediction)
                scored.append((product_id, float(score)))
            return self._normalise(scored)
        try:
            result = method(user_id, n=5)
        except TypeError:
            try:
                result = method(user_id, 5)
            except TypeError:
                result = method(user_id)
        return self._normalise(result)

    def for_user(self, user_id):
        if user_id not in self.ranked:
            try:
                self.ranked[user_id] = self._infer(user_id)
            except Exception as exc:
                print(f"[recommender] inference failed for {user_id}: {exc}")
                self.ranked[user_id] = []
        return [p for p in self.ranked[user_id] if p in self.catalogue.products]


# --------------------------------------------------------------------------- personalisation
class Personalizer:
    def __init__(self, catalogue: Catalogue, profiles: Profiles, interactions: Interactions, recommender: Recommender):
        self.catalogue, self.profiles, self.interactions, self.recommender = catalogue, profiles, interactions, recommender
        top = self.interactions.popularity.most_common(1)
        self._pop_max = top[0][1] if top else 1

    def score(self, user_id, pid, budget=None, affinity=None, rec_rank=None):
        p = self.catalogue.products[pid]
        prof = self.profiles.get(user_id)
        s, reasons = 0.0, []
        if rec_rank is None:
            recs = self.recommender.for_user(user_id)
            rec_rank = recs.index(pid) + 1 if pid in recs else None
        if rec_rank:
            s += 3.0 * max(0.2, 1 - (rec_rank - 1) / 12)
            reasons.append(f"ranked #{rec_rank} by your recommender")
        pref = to_float(prof.get("preferred_quality"), 0)
        q = to_float(p.get("quality_tier"), 0)
        if pref:
            diff = abs(q - pref)
            s += 1.5 - 0.6 * diff
            if diff <= 0.5:
                reasons.append("matches the quality level you usually prefer")
        text = norm(" ".join([p["category"], p["domain"], p.get("style_tags", ""), p.get("description", "")]))
        for interest in (prof.get("declared_interests") or "").split():
            if norm(interest) in text:
                s += 0.8
                reasons.append(f"fits your interest in {interest}")
        affinity = affinity if affinity is not None else self.interactions.category_affinity(user_id)
        if affinity.get(p["category"], 0) > 0:
            s += 2.0 * affinity[p["category"]]
            if affinity[p["category"]] >= 0.12:
                reasons.append(f"you've been exploring {p['category'].replace('_', ' ')}")
        hh = int(to_float(prof.get("household_size"), 1))
        if hh >= 3 and "family" in (p.get("style_tags") or ""):
            s += 0.5
            reasons.append(f"family-friendly for a household of {hh}")
        disc = to_float(p.get("offer_discount"))
        if disc > 0:
            s += disc * 2
            reasons.append(f"{round(disc * 100)}% off right now")
        price = float(self.catalogue.final_price(p))
        if budget:
            if price <= budget:
                s += 0.6
                reasons.append("within your budget")
            else:
                s -= 1.0
        s += 0.4 * self.interactions.popularity.get(pid, 0) / self._pop_max
        return round(s, 3), reasons[:3]

    # ---- main search used by the agent tools and the Shop tab
    def search(self, user_id, query=None, domain=None, category=None, min_price=None, max_price=None,
               min_quality=None, tags=None, exclude_ids=(), include_ineligible=False, sort_by="relevance", limit=6):
        cat = self.catalogue
        exclude = set(exclude_ids)
        cats, doms = set(), set()
        if category:
            cats, d2 = cat.resolve_category(category)
            doms |= d2 if not cats else set()
            if not cats and not d2:
                return {"results": [], "total_matches": 0,
                        "note": f"No catalogue category matches '{category}'. Available: "
                                + ", ".join(sorted({p['category'] for p in cat.products.values()}))}
        if domain:
            _, d2 = cat.resolve_category(domain)
            doms |= d2 or {domain}
        q_terms = []
        if query:
            for w in norm(query).split():
                w = singular(w)
                if len(w) > 2:
                    q_terms.append({w, *SYNONYMS.get(w, [])})
        tag_set = {norm(t) for t in (tags or [])}
        affinity = self.interactions.category_affinity(user_id)
        recs = self.recommender.for_user(user_id)
        rec_pos = {p: i + 1 for i, p in enumerate(recs)}

        matched, hidden = [], Counter()
        for pid, p in cat.products.items():
            if pid in exclude:
                continue
            if cats and p["category"] not in cats:
                continue
            if doms and p["domain"] not in doms:
                continue
            text = norm(" ".join([p["product_name"], p["category"], p["domain"], p.get("brand_id", ""),
                                  p.get("style_tags", ""), p.get("description", "")]))
            text_score = 0.0
            if q_terms:
                hits = sum(1 for alts in q_terms if any(a in text for a in alts))
                if hits == 0:
                    continue
                text_score = hits / len(q_terms)
            ptags = set((p.get("style_tags") or "").split())
            if tag_set and not tag_set & ptags:
                continue
            price = float(cat.final_price(p))
            if max_price is not None and price > max_price:
                continue
            if min_price is not None and price < min_price:
                continue
            if min_quality is not None and to_float(p.get("quality_tier")) < min_quality:
                continue
            ok, code, _ = cat.eligibility(user_id, pid)
            if not ok and not include_ineligible:
                hidden[code] += 1
                continue
            s, reasons = self.score(user_id, pid, budget=max_price, affinity=affinity, rec_rank=rec_pos.get(pid))
            matched.append((s + 3 * text_score, price, pid, reasons))

        key = {"price_asc": lambda m: (m[1], -m[0]), "price_desc": lambda m: (-m[1], -m[0]),
               "quality": lambda m: (-to_float(cat.products[m[2]].get("quality_tier")), -m[0]),
               "discount": lambda m: (-to_float(cat.products[m[2]].get("offer_discount")), -m[0])}.get(
            sort_by, lambda m: -m[0])
        matched.sort(key=key)
        results = [cat.card(pid, user_id, {"fit_score": s, "why": reasons}) for s, _, pid, reasons in matched[:limit]]
        out = {"results": results, "total_matches": len(matched)}
        if hidden:
            out["hidden_as_ineligible"] = dict(hidden)
        if not matched:
            out["note"] = "No eligible products match these filters. Consider relaxing price or category."
        return out

    def recommendations(self, user_id, category=None, max_price=None, exclude_ids=(), limit=5):
        """Recommender output first (eligible + within constraints); topped up by the heuristic scorer."""
        cats = self.catalogue.resolve_category(category)[0] if category else set()
        picks, exclude = [], set(exclude_ids)
        affinity = self.interactions.category_affinity(user_id)
        for rank, pid in enumerate(self.recommender.for_user(user_id), 1):
            p = self.catalogue.products[pid]
            if pid in exclude or (cats and p["category"] not in cats):
                continue
            if max_price is not None and float(self.catalogue.final_price(p)) > max_price:
                continue
            if not self.catalogue.eligibility(user_id, pid)[0]:
                continue
            s, reasons = self.score(user_id, pid, budget=max_price, affinity=affinity, rec_rank=rank)
            picks.append(self.catalogue.card(pid, user_id, {"source": "recommender", "rec_rank": rank, "why": reasons}))
            if len(picks) >= limit:
                break
        if len(picks) < limit:
            extra = self.search(user_id, category=category, max_price=max_price,
                                exclude_ids=exclude | {p["product_id"] for p in picks}, limit=limit - len(picks))
            for r in extra["results"]:
                r["source"] = "personalised_ranking"
                picks.append(r)
        return {"results": picks, "recommender_source": self.recommender.source}

    def complementary(self, user_id, base_ids, max_price=None, exclude_ids=(), limit=3):
        cat = self.catalogue
        base_cats = {cat.products[b]["category"] for b in base_ids if b in cat.products}
        if not base_cats:
            return {"results": [], "note": "No base product to complement."}
        candidates = Counter()
        for bc in base_cats:
            for other, n in self.interactions.cat_cooc.get(bc, {}).items():
                if other not in base_cats:
                    candidates[other] += n
        evidence = {c: "customers who buy %s often add this" % "/".join(sorted(base_cats)).replace("_", " ")
                    for c in candidates}
        all_cats = {p["category"] for p in cat.products.values()}
        for bc in base_cats:
            for key, hints in COMPLEMENT_HINTS.items():
                if key in norm(bc):
                    for h in hints:
                        for c in all_cats:
                            if h in norm(c) and c not in base_cats:
                                candidates[c] += 1.0
                                evidence.setdefault(c, "commonly paired with %s" % norm(bc))
        if max_price is None:  # a cross-sell shouldn't cost more than what it complements
            max_price = 0.75 * max(float(cat.final_price(cat.products[b])) for b in base_ids if b in cat.products)
        exclude = set(exclude_ids) | set(base_ids)
        out = []
        for c, _ in candidates.most_common():
            res = self.search(user_id, category=c, max_price=max_price, exclude_ids=exclude, limit=1)["results"]
            if res:
                res[0]["why"] = [evidence.get(c, "pairs well")] + res[0].get("why", [])[:2]
                out.append(res[0])
            if len(out) >= limit:
                break
        return {"results": out, "base_categories": sorted(base_cats)}

    def proactive_nudges(self, user_id, insights, limit=3):
        """Deterministic conversation starters derived from data (used on Home + greeting)."""
        prof = self.profiles.get(user_id)
        nudges = []
        for item in insights.get("open_cart_items", [])[:3]:
            pid = item["product_id"]
            if self.catalogue.eligibility(user_id, pid)[0]:
                c = self.catalogue.card(pid, user_id)
                deal = f" and it's {c['discount_pct']}% off now" if c["discount_pct"] else ""
                nudges.append({"type": "open_cart", "product_id": pid,
                               "text": f"You left {c['name']} in your cart on {item['date']}{deal}."})
                break
        recs = self.recommendations(user_id, limit=3)["results"]
        deals = [r for r in recs if r["discount_pct"]]
        if deals:
            d = deals[0]
            nudges.append({"type": "offer", "product_id": d["product_id"],
                           "text": f"{d['name']} is picked for you and is {d['discount_pct']}% off ({d['final_price']})."})
        recent = insights.get("recent_purchases") or []
        if recent:
            comp = self.complementary(user_id, [recent[0]["product_id"]], limit=1)["results"]
            if comp:
                nudges.append({"type": "complement", "product_id": comp[0]["product_id"],
                               "text": f"Goes well with your recent {recent[0]['name']}: {comp[0]['name']} ({comp[0]['final_price']})."})
        top = insights.get("top_categories") or []
        if top and len(nudges) < limit:
            nudges.append({"type": "interest", "category": top[0]["category"],
                           "text": f"You've been browsing {top[0]['category'].replace('_', ' ')} lately."})
        if prof.get("marketing_opt_in") != "1":
            nudges = [n for n in nudges if n["type"] in ("open_cart", "interest")]
        return nudges[:limit]
