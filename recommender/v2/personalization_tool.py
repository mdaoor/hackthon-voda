"""Personalization tool: the v6 recommender (public LB 0.26946) packaged for agent use.

Load:
    import sys; sys.path.insert(0, r"D:/recommender_system/src/v2")
    from personalization_tool import load_tool
    tool = load_tool(r"D:/recommender_system/tool/personalization_model_v6.pkl")

Use:
    tool.recommend(user_id="U024000", k=5)                 # known customer, stored history
    tool.recommend(user_id="U024000", events=[...], k=5)   # + fresh events from the current session
    tool.recommend(profile={...}, events=[...], k=5)       # brand-new customer
    tool.recommend_batch(["U024000", "U024001"], k=5)      # many customers at once (faster per customer)
    tool.describe()                                        # model card: inputs, metrics, drivers

Every call returns ranked, eligible products (region, device OS, launch date, already-owned
durables are enforced) with a score and plain-language reasons. Scoring is "as of" the model's
cutoff (2026-06-01): history strictly before it is used, recent popularity reflects May 2026.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from core import eligible, day_of
from feats import build_block

EVENT_TYPES = ["view", "cart", "favorite", "purchase", "redeem", "return", "cancel"]
SECTIONS = ["home", "shop", "rewards"]
PROFILE_DEFAULTS = {
    "region": "central", "age_band": "25-34", "household_size": 2, "membership_tier": "basic",
    "device_type": "unknown", "device_os": "android", "tenure_months": 12, "monthly_budget": np.nan,
    "preferred_quality": np.nan, "language": "lang_a", "marketing_opt_in": 0, "owned_items": np.nan,
    "declared_interests": np.nan,
}
REL_FEATS = ["assoc1", "assoc2", "tridge", "cos", "content", "knn_hist100", "knn_hist400", "knn_lab200"]


def _z(a):
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)


def _stage1(F):
    return (_z(F["assoc2"]) + 0.5 * _z(F["content"]) + 0.5 * _z(F["tridge"]) + 0.3 * _z(F["cos"])
            + 0.3 * _z(F["exp_unseen"]) + 0.7 * _z(F["knn_hist400"]))


class PersonalizationModel:
    """Two-stage recommender: stage-1 blend picks 200 candidates, then 2 LightGBM LambdaRank +
    2 LightGBM binary rankers are rank-averaged."""

    def __init__(self, products, catalog, unsup, label_model, booster_strings, feature_cols, new_items,
                 history, profiles, snapshot_date="2026-06-01", n_candidates=200, card=None):
        self.products = products
        self.cat = catalog
        self.un = unsup
        self.lm = label_model
        self.booster_strings = booster_strings
        self.cols = feature_cols
        self.new_items = np.asarray(new_items)
        self.history = history          # compact interaction log of all 30k known customers
        self.profiles = profiles        # one row per known customer
        self.snapshot_date = snapshot_date
        self.n_candidates = n_candidates
        self.card = card or {}
        self._boosters = None

    # ---- persistence -------------------------------------------------------------------
    def __getstate__(self):
        s = self.__dict__.copy()
        s["_boosters"] = None          # boosters are rebuilt from their model strings on load
        return s

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=5)

    @property
    def boosters(self):
        if self._boosters is None:
            self._boosters = [lgb.Booster(model_str=s) for s in self.booster_strings]
        return self._boosters

    # ---- inputs ------------------------------------------------------------------------
    def _users_frame(self, user_ids=None, profile=None):
        if user_ids is not None:
            missing = [u for u in user_ids if u not in self.profiles.index]
            if missing and profile is None:
                raise KeyError(f"Unknown customer(s) {missing[:5]}; pass profile=... for new customers.")
            rows = []
            for u in user_ids:
                if u in self.profiles.index:
                    r = self.profiles.loc[u].to_dict()
                else:
                    r = {**PROFILE_DEFAULTS, **(profile or {})}
                r["user_id"] = u
                rows.append(r)
            users = pd.DataFrame(rows)
        else:
            r = {**PROFILE_DEFAULTS, **(profile or {})}
            r["user_id"] = r.get("user_id", "NEW_CUSTOMER")
            users = pd.DataFrame([r])
        users["snapshot_date"] = self.snapshot_date
        for c, v in PROFILE_DEFAULTS.items():
            if c not in users:
                users[c] = v
        users["owned_items"] = users.owned_items.map(lambda x: " ".join(x) if isinstance(x, (list, tuple)) else x)
        users["declared_interests"] = users.declared_interests.map(lambda x: " ".join(x) if isinstance(x, (list, tuple)) else x)
        return users.reset_index(drop=True)

    def _events_frame(self, users, events):
        h = self.history[self.history.user_id.isin(set(users.user_id))]
        h = h.assign(user_id=h.user_id.astype(str), event_type=h.event_type.astype(str), app_section=h.app_section.astype(str))
        if events is not None and len(events):
            e = pd.DataFrame(events).copy()
            if "user_id" not in e:
                if len(users) != 1:
                    raise ValueError("events need a user_id column when scoring several customers")
                e["user_id"] = users.user_id.iloc[0]
            bad = set(e.event_type) - set(EVENT_TYPES)
            if bad:
                raise ValueError(f"unknown event_type {bad}; use {EVENT_TYPES}")
            e["item"] = self.cat.pidx.get_indexer(e.product_id)
            if (e.item < 0).any():
                raise ValueError(f"unknown product_id(s): {list(e.product_id[e.item < 0])[:5]}")
            e["day"] = [day_of(d) for d in e.event_date]
            if (e.day >= day_of(self.snapshot_date)).any():
                raise ValueError(f"event_date must be before the model cutoff {self.snapshot_date}")
            e["app_section"] = e.get("app_section", "home")
            for c in ("dwell_seconds", "discount_rate", "quantity"):
                e[c] = e[c] if c in e else np.nan
            h = pd.concat([h, e[h.columns]], ignore_index=True)
        h = h.copy()
        h["day"] = h.day.astype(np.int16); h["item"] = h.item.astype(np.int16)
        return h

    # ---- scoring -----------------------------------------------------------------------
    def _score(self, users, events):
        inter = self._events_frame(users, events)
        F, item, user = build_block(users, inter, self.cat, self.lm, self.un, self.new_items)
        elig = eligible(users, self.products)
        s1 = np.where(elig, _stage1(F), -np.inf)
        kk = int(min(self.n_candidates, elig.sum(1).min()))
        cand = np.argpartition(-s1, kk - 1, axis=1)[:, :kk]
        rows = np.repeat(np.arange(len(users)), kk); cols = cand.ravel()
        tab = {name: M[rows, cols] for name, M in F.items()}
        tab["s1"] = s1[rows, cols]
        tab["s1_rank"] = np.argsort(np.argsort(-np.take_along_axis(s1, cand, 1), 1), 1).ravel().astype(np.float32)
        for name in REL_FEATS:
            M = np.take_along_axis(F[name], cand, 1)
            tab[name + "_rel"] = (M - M.max(1, keepdims=True)).ravel()
        df = pd.DataFrame(tab)
        for c in item.columns:
            df[c] = item[c].to_numpy()[cols]
        for c in user.columns:
            df["u_" + c] = user[c].to_numpy()[rows]
        X = df[self.cols].to_numpy(np.float32)
        total = np.zeros((len(users), self.cat.n), np.float32)
        for b in self.boosters:
            S = np.full((len(users), self.cat.n), -1e9, np.float32)
            S[rows, cols] = b.predict(X)
            total += -np.argsort(np.argsort(-S, 1), 1).astype(np.float32)
        total[~elig] = -np.inf
        return total, F, elig

    def _reasons(self, F, u, j):
        p = self.products.iloc[j]
        out = []
        n_seen = int(F["own_view"][u, j] + F["own_cart"][u, j] + F["own_fav"][u, j] + F["own_buy"][u, j])
        if n_seen:
            acts = [f"{int(F[k][u, j])} {lbl}" for k, lbl in
                    [("own_view", "views"), ("own_cart", "carts"), ("own_fav", "favourites"), ("own_buy", "purchases")] if F[k][u, j] > 0]
            out.append(f"previously engaged ({', '.join(acts)}; last {int(F['own_last'][u, j])} days ago)")
        if F["cat_share"][u, j] >= 0.15:
            out.append(f"{F['cat_share'][u, j]:.0%} of recent activity is in {p.category}")
        if F["declared"][u, j] > 0:
            out.append(f"declared interest in {p.category}")
        if p.offer_discount >= 0.15:
            out.append(f"{p.offer_discount:.0%} offer discount")
        if F["quality_gap_pref"][u, j] == 0:
            out.append("matches preferred quality tier")
        if j in set(self.new_items.tolist()):
            out.append("new launch (May 2026)")
        if not out:
            out.append("similar customers engage with it")
        return out

    @staticmethod
    def _top_k(total, k):
        """Same selection as the submission code (core.top5): argpartition, then stable sort of the k."""
        k = min(int(k), total.shape[1] - 1)
        part = np.argpartition(-total, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(total, part, 1), axis=1, kind="stable")
        return np.take_along_axis(part, order, 1)

    def _format(self, total, F, users, k, explain):
        top = self._top_k(total, k)
        n_models = len(self.booster_strings)
        res = {}
        for u, uid in enumerate(users.user_id):
            recs = []
            for r, j in enumerate(top[u]):
                if not np.isfinite(total[u, j]):
                    break
                p = self.products.iloc[j]
                rec = {"rank": r + 1, "product_id": p.product_id, "product_name": p.product_name,
                       "category": p.category, "domain": p.domain, "price": float(p.price),
                       "offer_discount": float(p.offer_discount),
                       # average rank position across the ensemble's models (1.0 = first in every model)
                       "consensus_rank": float(-total[u, j] / n_models + 1)}
                if explain:
                    rec["reasons"] = self._reasons(F, u, j)
                recs.append(rec)
            res[uid] = recs
        return res

    # ---- public API --------------------------------------------------------------------
    def recommend(self, user_id=None, profile=None, events=None, k=5, explain=True):
        """Top-k products for one customer.

        user_id : known customer id (history and profile are looked up), or None for a new customer.
        profile : dict of profile fields (see describe()["profile_fields"]); overrides nothing for known ids.
        events  : optional list/DataFrame of extra events with product_id, event_type, event_date
                  (YYYY-MM-DD, before the cutoff), optional app_section / dwell_seconds / discount_rate / quantity.
        k       : number of products (<= 200).
        Returns a list of dicts ranked best first.
        """
        users = self._users_frame([user_id] if user_id is not None else None, profile)
        total, F, _ = self._score(users, events)
        return self._format(total, F, users, k, explain)[users.user_id.iloc[0]]

    def recommend_batch(self, user_ids, k=5, explain=False):
        """Top-k for many known customers. Returns {user_id: [recommendations...]}."""
        users = self._users_frame(list(user_ids))
        out = {}
        for s in range(0, len(users), 2000):
            part = users.iloc[s:s + 2000].reset_index(drop=True)
            total, F, _ = self._score(part, None)
            out.update(self._format(total, F, part, k, explain))
        return out

    def describe(self):
        """Model card for the agent: what the tool does, inputs, quality, main drivers."""
        return self.card


def load_tool(path):
    """Unpickle a PersonalizationModel (this module, core.py and feats.py must be importable)."""
    with open(path, "rb") as f:
        return pickle.load(f)
