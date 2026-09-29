"""Shared v2 utilities: cached data loading, eligibility, fast vectorised metric,
multi-channel history matrices."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

DATA = Path(__file__).resolve().parents[2] / "data"
CACHE = Path(__file__).resolve().parents[2] / "cache"
try:  # training-time cache; skipped on read-only deployments (e.g. AWS Lambda)
    CACHE.mkdir(exist_ok=True)
except OSError:
    pass
NEW_LAUNCH = "2026-05-01"


def load():
    pq = CACHE / "inter.pkl"
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    products = pd.read_csv(DATA / "products.csv")
    if pq.exists():
        inter = pd.read_pickle(pq)
    else:
        inter = pd.read_csv(DATA / "interactions.csv")
        inter["day"] = (pd.to_datetime(inter.event_date) - pd.Timestamp("2026-01-01")).dt.days.astype(np.int16)
        pidx = pd.Index(products.product_id)
        inter["item"] = pidx.get_indexer(inter.product_id).astype(np.int16)
        inter.to_pickle(pq)
    return train, test, products, inter


def day_of(date_str):
    return (pd.Timestamp(date_str) - pd.Timestamp("2026-01-01")).days


def eligible(users, products):
    regions = [set(str(x).split()) for x in products.available_regions]
    pid = pd.Index(products.product_id)
    avail = products.available_from.to_numpy()
    compat = products.compatible_os.to_numpy()
    reg_names = sorted({r for s in regions for r in s} | set(users.region))
    reg_mat = {r: np.array([r in s for s in regions]) for r in reg_names}
    out = np.zeros((len(users), len(products)), dtype=bool)
    for k, r in enumerate(users.itertuples()):
        m = (avail <= r.snapshot_date) & reg_mat[r.region] & ((compat == "any") | (compat == r.device_os))
        if isinstance(r.owned_items, str):
            m[pid.get_indexer(r.owned_items.split())] = False
        out[k] = m
    return out


def label_matrix(users, n_items, pidx):
    rows, cols = [], []
    for i, s in enumerate(users.relevant_items.fillna("")):
        for p in s.split():
            rows.append(i); cols.append(pidx.get_loc(p))
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(users), n_items))


_DISC = 1 / np.log2(np.arange(2, 7))
_IDEAL = np.r_[0, np.cumsum(_DISC)]


def top5(scores, elig):
    s = np.where(elig, scores, -np.inf)
    part = np.argpartition(-s, 5, axis=1)[:, :5]
    order = np.argsort(-np.take_along_axis(s, part, 1), axis=1, kind="stable")
    return np.take_along_axis(part, order, 1)


def metric(scores, elig, Y, per_user=False):
    """Exact competition score. Y: csr label matrix."""
    t = top5(scores, elig)
    Yd = Y.toarray().astype(bool) if sparse.issparse(Y) else Y
    hits = np.take_along_axis(Yd, t, 1)
    nrel = Yd.sum(1)
    ndcg = np.where(nrel > 0, hits @ _DISC / np.maximum(_IDEAL[np.minimum(nrel, 5)], 1e-9), 0)
    prec = hits.sum(1) / 5
    s = 0.5 * ndcg + 0.5 * prec
    return s if per_user else float(s.mean())


EV = ["view", "cart", "favorite", "purchase", "redeem", "return", "cancel"]


def hist_tensor(users, inter, n_items, snap_day=None, min_day=None, drop_items=None):
    """Return dict of csr matrices per channel with raw decayed counts, for events strictly
    before each user's snapshot (or snap_day override)."""
    uidx = pd.Index(users.user_id)
    r = uidx.get_indexer(inter.user_id)
    h = inter[r >= 0]
    r = r[r >= 0]
    cut = np.array([day_of(d) for d in users.snapshot_date]) if snap_day is None else np.full(len(users), snap_day)
    age = cut[r] - h.day.to_numpy()
    keep = age > 0
    if min_day is not None:
        keep &= h.day.to_numpy() >= min_day
    if drop_items is not None:
        keep &= ~np.isin(h.item.to_numpy(), drop_items)
    h = h[keep]; r = r[keep]; age = age[keep]
    return h, r, age
