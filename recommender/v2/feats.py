"""v2 feature builder.

build_block(target, label_users, all_users, cold_items, ...) computes a dict of dense
(n_target x n_items) float32 feature matrices plus per-user / per-item vectors.

Cold-item simulation: items in `cold_items` are treated exactly like the real new
products at test time -- no label information and no history in `label_users`, and in
every other history only events from the final `cold_window` days before each user's
snapshot are kept (new products launched one month before the test snapshot).
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import solve

from core import day_of, label_matrix

EVW = {"view": 1., "cart": 2., "favorite": 4., "purchase": 3., "redeem": 3., "return": 0., "cancel": 0.}
TYPES = {"view": ["view"], "cart": ["cart"], "fav": ["favorite"], "buy": ["purchase", "redeem"],
         "ref": ["return", "cancel"]}


class Catalog:
    def __init__(self, products):
        self.p = products
        self.n = len(products)
        self.pidx = pd.Index(products.product_id)
        self.cat_codes, self.cats = pd.factorize(products.category, sort=True)
        self.dom_codes, self.doms = pd.factorize(products.domain, sort=True)
        self.brand_codes, self.brands = pd.factorize(products.domain + "_" + products.brand_id, sort=True)
        self.C = np.eye(len(self.cats), dtype=np.float32)[self.cat_codes]          # items x cats
        self.D = np.eye(len(self.doms), dtype=np.float32)[self.dom_codes]
        self.B = np.eye(len(self.brands), dtype=np.float32)[self.brand_codes]
        self.T = products.style_tags.str.get_dummies(sep=" ").to_numpy(np.float32)  # items x tags
        self.tag_names = products.style_tags.str.get_dummies(sep=" ").columns
        med = products.groupby("category").price.transform("median").to_numpy()
        self.rel_price = np.log(products.price.to_numpy() / med).astype(np.float32)
        self.cat_med = med
        self.quality = products.quality_tier.to_numpy().astype(np.float32)
        self.family = self.T[:, list(self.tag_names).index("family")]
        # content neighbour table for cold imputation: same category, similar price/quality/tags
        self._nn = {}

    def neighbours(self, j, known, k=5):
        key = (j, known.tobytes().__hash__())
        if key in self._nn:
            return self._nn[key]
        p = self.p
        cand = np.flatnonzero((self.cat_codes == self.cat_codes[j]) & known)
        dist = (np.abs(self.rel_price[cand] - self.rel_price[j])
                + 0.25 * np.abs(self.quality[cand] - self.quality[j])
                - 0.3 * (self.T[cand] @ self.T[j])
                - 0.15 * (self.brand_codes[cand] == self.brand_codes[j]))
        nn = cand[np.argsort(dist, kind="stable")[:k]]
        self._nn[key] = nn
        return nn


def filter_events(inter, users, cold_items=None, cold_window=31, snap_override=None, cold_users=None):
    """Events of `users` strictly before their snapshot; cold items only inside the last
    cold_window days. Returns (h, row, age)."""
    uidx = pd.Index(users.user_id)
    r = uidx.get_indexer(inter.user_id)
    m = r >= 0
    h = inter[m]; r = r[m]
    snap = np.array([day_of(d) for d in users.snapshot_date]) if snap_override is None else snap_override
    age = snap[r] - h.day.to_numpy()
    keep = age > 0
    if cold_items is not None and len(cold_items):
        is_cold = np.isin(h.item.to_numpy(), cold_items)
        ok = age <= cold_window
        if cold_users is not None:
            ok &= h.user_id.isin(set(cold_users)).to_numpy()
        keep &= ~is_cold | ok
    return h[keep], r[keep], age[keep]


def csr(r, c, w, shape):
    x = sparse.csr_matrix((w.astype(np.float64), (r, c)), shape=shape)
    x.sum_duplicates()
    x.eliminate_zeros()
    return x


def lognorm(x):
    x = x.copy(); x.data = np.log1p(x.data)
    rs = np.maximum(np.asarray(x.sum(1)).ravel(), 1)
    return sparse.csr_matrix(x.multiply(1 / rs[:, None]))


def hist_mats(h, r, age, n_users, n_items, hl=45.):
    """Channel matrices for ridge / features."""
    et = h.event_type.to_numpy(); it = h.item.to_numpy()
    dec = np.exp(-age / hl)
    w_all = np.array([EVW[e] for e in et]) * dec
    X = csr(r, it, w_all, (n_users, n_items))
    ch = {}
    for g, ts in TYPES.items():
        m = np.isin(et, ts)
        ch[g] = csr(r[m], it[m], dec[m], (n_users, n_items))
    return X, ch


def multi_channel(ch):
    tot = ch["view"] + ch["cart"] + ch["fav"] + ch["buy"]
    rs = np.log1p(np.maximum(np.asarray(tot.sum(1)).ravel(), 1))
    out = []
    for g in ["view", "cart", "fav", "buy", "ref"]:
        x = ch[g].copy(); x.data = np.log1p(x.data)
        out.append(sparse.csr_matrix(x.multiply(1 / rs[:, None])))
    return sparse.hstack(out).tocsr()


def ridge(X, Y, lam):
    G = (X.T @ X); G = G.toarray() if sparse.issparse(G) else np.asarray(G)
    G = G.astype(np.float64); G.flat[::G.shape[0] + 1] += lam
    R = (X.T @ Y); R = R.toarray() if sparse.issparse(R) else np.asarray(R)
    return solve(G, R, assume_a="pos", check_finite=False).astype(np.float32)


def impute_cols(Wm, cat, cold, known):
    Wm = Wm.copy()
    for j in cold:
        Wm[:, j] = Wm[:, cat.neighbours(j, known)].mean(1)
    return Wm


def impute_rows(Wm, cat, cold, known, n_blocks=1):
    """Rows index input items; ridge weight rows for input items never seen are ~0.
    Transfer from content neighbours (per block for multi-channel inputs)."""
    Wm = Wm.copy()
    n = cat.n
    for b in range(n_blocks):
        for j in cold:
            nn = cat.neighbours(j, known)
            Wm[b * n + j] = Wm[b * n + nn].mean(0)
    return Wm


def impute_vec(v, cat, cold, known):
    v = v.copy()
    for j in cold:
        v[j] = v[cat.neighbours(j, known)].mean()
    return v


def l2rows(x):
    x = sparse.csr_matrix(x)
    nr = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    return sparse.csr_matrix(sparse.diags(1 / np.maximum(nr, 1e-9)) @ x)


def knn_scores(Q, B, T, ks, alpha=2.0, chunk=1000, self_idx=None):
    """User-KNN: for each row of Q, sum T-rows of its top-k most similar B-rows (cosine^alpha).
    self_idx[i] = row of B that is query i itself (excluded), or -1."""
    out = {k: np.zeros((Q.shape[0], T.shape[1]), np.float32) for k in ks}
    BT = B.T.tocsc()
    for s0 in range(0, Q.shape[0], chunk):
        sim = (Q[s0:s0 + chunk] @ BT).toarray().astype(np.float32)
        if self_idx is not None:
            si = self_idx[s0:s0 + chunk]
            ok = si >= 0
            sim[np.flatnonzero(ok), si[ok]] = -1.0
        for k in ks:
            idx = np.argpartition(-sim, k, axis=1)[:, :k]
            w = np.take_along_axis(sim, idx, 1) ** alpha
            rows = np.repeat(np.arange(sim.shape[0]), k)
            W = sparse.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(sim.shape[0], B.shape[0]))
            out[k][s0:s0 + chunk] = np.asarray((W @ T).todense() if sparse.issparse(W @ T) else W @ T)
    return out


class Unsup:
    """Label-free statistics from every customer's history (train + test cohorts)."""

    def __init__(self, all_users, inter, cat, cold_items, target_ids):
        """Cold items are only observed in the target cohort's recent history (as the real
        new products are only observed in the test cohort's final month)."""
        n = cat.n
        h, r, age = filter_events(inter, all_users, cold_items, cold_users=np.asarray(target_ids))
        is_tgt = all_users.user_id.isin(set(target_ids)).to_numpy()
        X, ch = hist_mats(h, r, age, len(all_users), n)
        Xn = lognorm(X)
        # for user-KNN over every customer's history
        self.ids = all_users.user_id.to_numpy()
        self.Xn = Xn
        df = np.asarray((X > 0).sum(0)).ravel()
        self.idf = np.log(len(all_users) / (1 + df)).astype(np.float32)
        # item-item cosine on normalised engagement
        G = (Xn.T @ Xn).toarray()
        d = np.sqrt(np.clip(np.diag(G), 1e-12, None))
        S = G / d[:, None] / d[None, :]
        np.fill_diagonal(S, 0)
        self.cos = S.astype(np.float32)
        # recent popularity per snapshot cohort (share of events in last 30 / 14 days)
        snaps = np.array([day_of(d) for d in all_users.snapshot_date])
        self.recent = {}
        for sd in np.unique(snaps):
            it = h.item.to_numpy(); a = age
            w = np.array([EVW[e] for e in h.event_type.to_numpy()])
            rp = {}
            for win in (14, 30):
                m = (snaps[r] == sd) & (a <= win) & is_tgt[r]
                v = np.bincount(it[m], weights=w[m], minlength=n)
                rp[win] = v / max(v.sum(), 1) * n
            # item "buy share" of events in last 30 days (conversion proxy)
            m = (snaps[r] == sd) & (a <= 30) & is_tgt[r]
            evs = np.bincount(it[m], minlength=n)
            buys = np.bincount(it[m & np.isin(h.event_type.to_numpy(), ["purchase", "redeem", "favorite", "cart"])],
                               minlength=n)
            rp["conv"] = (buys + 1) / (evs + 3)
            self.recent[sd] = rp
        # self-supervised temporal ridge: history before (snap-14) -> items touched in last 14 days
        cut = 14
        m_in = age > cut
        m_out = age <= cut
        et = h.event_type.to_numpy()
        # inputs: channels using age shifted by cut
        hi = h[m_in]; ri = r[m_in]; ai = age[m_in] - cut
        _, chi = hist_mats(hi, ri, ai, len(all_users), n)
        Xi = multi_channel(chi)
        pos = m_out & np.isin(et, ["view", "cart", "favorite", "purchase", "redeem"])
        Yo = csr(r[pos], h.item.to_numpy()[pos], np.ones(pos.sum()), (len(all_users), n))
        Yo.data[:] = 1.0
        has = np.asarray(Xi.sum(1)).ravel() > 0
        self.tridge = ridge(Xi[has], Yo[has], 3.0)


def user_profile_feats(users, h, r, age, cat):
    """Per-user dense aggregates (n_users x k) and user x category / tag / brand shares."""
    n_u = len(users)
    it = h.item.to_numpy(); et = h.event_type.to_numpy()
    w = np.array([EVW[e] for e in et]) * np.exp(-age / 45.)
    X = csr(r, it, w, (n_u, cat.n))
    Xl = X.copy(); Xl.data = np.log1p(Xl.data)
    rs = np.maximum(np.asarray(Xl.sum(1)).ravel(), 1e-9)
    cat_share = np.asarray(Xl @ cat.C) / rs[:, None]
    dom_share = np.asarray(Xl @ cat.D) / rs[:, None]
    tag_share = np.asarray(Xl @ cat.T) / rs[:, None]
    brand_share = np.asarray(Xl @ cat.B) / rs[:, None]
    # recent (14 day) category share
    m = age <= 14
    Xr = csr(r[m], it[m], w[m], (n_u, cat.n)); Xr.data = np.log1p(Xr.data)
    rsr = np.maximum(np.asarray(Xr.sum(1)).ravel(), 1e-9)
    cat_share14 = np.asarray(Xr @ cat.C) / rsr[:, None]
    # intent category share (cart/fav/buy)
    m = np.isin(et, ["cart", "favorite", "purchase", "redeem"])
    Xi = csr(r[m], it[m], w[m], (n_u, cat.n)); Xi.data = np.log1p(Xi.data)
    rsi = np.maximum(np.asarray(Xi.sum(1)).ravel(), 1e-9)
    cat_share_int = np.asarray(Xi @ cat.C) / rsi[:, None]
    # raw count per category
    Xc = csr(r, it, np.ones(len(r)), (n_u, cat.n))
    cat_cnt = np.asarray(Xc @ cat.C)
    # last touch age per category
    cat_last = np.full((n_u, len(cat.cats)), 120., np.float32)
    ci = cat.cat_codes[it]
    np.minimum.at(cat_last, (r, ci), age.astype(np.float32))
    # price / quality preference: weighted avg of rel_price and quality overall and per category
    wsum = np.bincount(r, weights=w, minlength=n_u)
    up = np.bincount(r, weights=w * cat.rel_price[it], minlength=n_u) / np.maximum(wsum, 1e-9)
    uq = np.bincount(r, weights=w * cat.quality[it], minlength=n_u) / np.maximum(wsum, 1e-9)
    has = wsum > 0
    up[~has] = np.nan; uq[~has] = np.nan
    wc = np.zeros((n_u, len(cat.cats))); np.add.at(wc, (r, ci), w)
    pc = np.zeros_like(wc); np.add.at(pc, (r, ci), w * cat.rel_price[it])
    ucp = np.where(wc > 0, pc / np.maximum(wc, 1e-9), np.nan)
    # paid price level
    m = np.isin(et, ["purchase", "redeem"])
    paid = np.bincount(r[m], weights=cat.rel_price[it[m]], minlength=n_u) / np.maximum(np.bincount(r[m], minlength=n_u), 1)
    paid[np.bincount(r[m], minlength=n_u) == 0] = np.nan
    # activity
    act = pd.DataFrame({
        "n_events": np.bincount(r, minlength=n_u),
        "n_items": np.asarray((Xc > 0).sum(1)).ravel(),
        "n_cats": (cat_cnt > 0).sum(1),
        "last_age": pd.Series(age).groupby(r).min().reindex(range(n_u)).fillna(120).to_numpy(),
        "n_recent14": np.bincount(r[age <= 14], minlength=n_u),
        "n_buy": np.bincount(r[np.isin(et, ["purchase", "redeem"])], minlength=n_u),
        "n_fav": np.bincount(r[et == "favorite"], minlength=n_u),
        "n_ref": np.bincount(r[np.isin(et, ["return", "cancel"])], minlength=n_u),
        "frac_home": np.bincount(r[h.app_section.to_numpy() == "home"], minlength=n_u) / np.maximum(np.bincount(r, minlength=n_u), 1),
        "frac_rewards": np.bincount(r[h.app_section.to_numpy() == "rewards"], minlength=n_u) / np.maximum(np.bincount(r, minlength=n_u), 1),
        "u_price": up, "u_quality": uq, "u_paid_price": paid,
    })
    # transaction traits: discount sensitivity, basket quantity
    mb = np.isin(et, ["purchase", "redeem"])
    nb = np.bincount(r[mb], minlength=n_u)
    hd = np.bincount(r[mb], weights=h.discount_rate.fillna(0).to_numpy()[mb], minlength=n_u) / np.maximum(nb, 1)
    act["hist_disc"] = np.where(nb > 0, hd, np.nan)
    hq = np.bincount(r[mb], weights=h.quantity.fillna(1).to_numpy()[mb], minlength=n_u) / np.maximum(nb, 1)
    act["hist_qty"] = np.where(nb > 0, hq, np.nan)
    return dict(X=X, cat_share=cat_share, dom_share=dom_share, tag_share=tag_share, brand_share=brand_share,
                cat_share14=cat_share14, cat_share_int=cat_share_int, cat_cnt=cat_cnt, cat_last=cat_last,
                ucp=ucp, act=act)


def own_item_feats(h, r, age, n_u, n_i):
    it = h.item.to_numpy(); et = h.event_type.to_numpy()
    out = {}
    for g, ts in TYPES.items():
        m = np.isin(et, ts)
        out[f"own_{g}"] = csr(r[m], it[m], np.ones(m.sum()), (n_u, n_i)).toarray().astype(np.float32)
    last = np.full((n_u, n_i), 120., np.float32)
    np.minimum.at(last, (r, it), age.astype(np.float32))
    out["own_last"] = last
    m = np.isin(et, ["cart", "favorite", "purchase", "redeem"])
    last_i = np.full((n_u, n_i), 120., np.float32)
    np.minimum.at(last_i, (r[m], it[m]), age[m].astype(np.float32))
    out["own_last_intent"] = last_i
    w = np.array([EVW[e] for e in et]) * np.exp(-age / 45.)
    out["own_w"] = csr(r, it, w, (n_u, n_i)).toarray().astype(np.float32)
    dw = h.dwell_seconds.fillna(0).to_numpy()
    out["own_dwell"] = csr(r, it, dw, (n_u, n_i)).toarray().astype(np.float32)
    sec = h.app_section.to_numpy()
    for s_ in ["home", "shop", "rewards"]:
        m = sec == s_
        out[f"own_{s_}"] = csr(r[m], it[m], np.ones(m.sum()), (n_u, n_i)).toarray().astype(np.float32)
    first = np.zeros((n_u, n_i), np.float32)
    np.maximum.at(first, (r, it), age.astype(np.float32))
    out["own_first"] = first
    # distinct active days with the item
    key = pd.DataFrame({"r": r, "i": it, "d": age}).drop_duplicates()
    out["own_days"] = csr(key.r.to_numpy(), key.i.to_numpy(), np.ones(len(key)), (n_u, n_i)).toarray().astype(np.float32)
    # net acquisition: purchases minus refunds
    out["own_net_buy"] = out["own_buy"] - out["own_ref"]
    return out


class LabelModel:
    """Everything fit on labelled customers: ridge associations + label popularity."""

    def __init__(self, label_users, inter, cat, cold_items):
        n = cat.n
        cold_items = np.asarray(cold_items if cold_items is not None else [], dtype=int)
        known = np.ones(n, bool); known[cold_items] = False
        # label users never saw cold items (they did not exist before their snapshot)
        h, r, age = filter_events(inter, label_users)
        keep = ~np.isin(h.item.to_numpy(), cold_items)
        h, r, age = h[keep], r[keep], age[keep]
        Y = label_matrix(label_users, n, cat.pidx).tolil()
        if len(cold_items):
            Y[:, cold_items] = 0
        Y = Y.tocsr(); Y.eliminate_zeros()
        X, ch = hist_mats(h, r, age, len(label_users), n)
        Xs = lognorm(X)
        W1 = ridge(Xs, Y, 1.0)
        Xm = multi_channel(ch)
        W2 = ridge(Xm, Y, 3.0)
        if len(cold_items):
            W1 = impute_cols(impute_rows(W1, cat, cold_items, known), cat, cold_items, known)
            W2 = impute_cols(impute_rows(W2, cat, cold_items, known, 5), cat, cold_items, known)
        self.W1, self.W2 = W1, W2
        self.Xs, self.Y = Xs, Y
        lp = np.asarray(Y.sum(0)).ravel().astype(np.float32) / len(label_users) * 100
        # label rate given engagement: P(label | user touched item before)
        touched = (X > 0).astype(np.float32)
        tl = np.asarray(touched.multiply(Y).sum(0)).ravel()
        tn = np.asarray(touched.sum(0)).ravel()
        rep = ((tl + 1) / (tn + 10)).astype(np.float32)
        # untouched label rate
        ul = np.asarray(Y.sum(0)).ravel() - tl
        un = len(label_users) - tn
        unt = ((ul + 1) / (un + 10) * 100).astype(np.float32)
        if len(cold_items):
            lp = impute_vec(lp, cat, cold_items, known)
            rep = impute_vec(rep, cat, cold_items, known)
            unt = impute_vec(unt, cat, cold_items, known)
        self.label_pop, self.touch_rate, self.untouch_rate = lp, rep, unt
        # user-category -> label-category transition (24x24) : P(label in cat | history cat share)
        cs = np.asarray(Xs @ cat.C)
        yc = np.asarray(Y @ cat.C)
        self.catW = ridge(cs, yc, 1.0)


def build_block(target, inter, cat, lm: LabelModel, un: Unsup, cold_items):
    """Dense feature matrices for target users (all items)."""
    n_u, n = len(target), cat.n
    h, r, age = filter_events(inter, target, cold_items)
    prof = user_profile_feats(target, h, r, age, cat)
    own = own_item_feats(h, r, age, n_u, n)
    X, ch = hist_mats(h, r, age, n_u, n)
    Xs = lognorm(X)
    Xm = multi_channel(ch)
    F = {}
    F["assoc1"] = np.asarray(Xs @ lm.W1)
    F["assoc2"] = np.asarray(Xm @ lm.W2)
    F["tridge"] = np.asarray(Xm @ un.tridge)
    F["cos"] = np.asarray(Xs @ un.cos)
    # cosine from the last 14 days only
    m = age <= 14
    Xr = lognorm(csr(r[m], h.item.to_numpy()[m], np.array([EVW[e] for e in h.event_type.to_numpy()[m]]), (n_u, n)))
    F["cos14"] = np.asarray(Xr @ un.cos)
    F["catW"] = (np.asarray(Xs @ cat.C) @ lm.catW) @ cat.C.T
    # user-KNN collaborative filtering
    Q = l2rows(Xs.multiply(un.idf[None, :]))
    Bh = l2rows(un.Xn.multiply(un.idf[None, :]))
    self_idx = pd.Index(un.ids).get_indexer(target.user_id)
    kh = knn_scores(Q, Bh, un.Xn, (100, 400), self_idx=self_idx)
    F["knn_hist100"], F["knn_hist400"] = kh[100], kh[400]
    Bl = l2rows(lm.Xs.multiply(un.idf[None, :]))
    kl = knn_scores(Q, Bl, lm.Y, (200,))
    F["knn_lab200"] = kl[200]
    cc = cat.cat_codes
    F["cat_share"] = prof["cat_share"][:, cc]
    F["cat_share14"] = prof["cat_share14"][:, cc]
    F["cat_share_int"] = prof["cat_share_int"][:, cc]
    F["cat_cnt"] = prof["cat_cnt"][:, cc]
    F["cat_last"] = prof["cat_last"][:, cc]
    F["dom_share"] = prof["dom_share"][:, cat.dom_codes]
    F["brand_share"] = prof["brand_share"][:, cat.brand_codes]
    F["tag_score"] = prof["tag_share"] @ cat.T.T
    # Bayesian category preference: history share shrunk towards declared interests
    nev = prof["act"].n_events.to_numpy()[:, None]
    dec0 = target.declared_interests.fillna("").str.get_dummies(sep=" ").reindex(columns=cat.cats, fill_value=0).to_numpy(np.float32)
    prior = (dec0 + 0.05) / (dec0 + 0.05).sum(1, keepdims=True)
    post = (prof["cat_share"] * nev + 5 * prior) / (nev + 5)
    F["cat_post"] = post[:, cc]
    F["log_cat_post"] = np.log(post[:, cc] + 1e-3)
    # item appeal relative to its category (within-category popularity / discount rank)
    lp = pd.Series(lm.label_pop)
    F["pop_rank_cat"] = np.broadcast_to(lp.groupby(cc).rank(pct=True).to_numpy(np.float32), (n_u, n))
    F["disc_rank_cat"] = np.broadcast_to(pd.Series(cat.p.offer_discount.to_numpy()).groupby(cc).rank(pct=True).to_numpy(np.float32), (n_u, n))
    F["exp_unseen"] = F["log_cat_post"] + np.log(lm.untouch_rate + 1e-3)[None, :]
    declared = target.declared_interests.fillna("").str.get_dummies(sep=" ").reindex(columns=cat.cats, fill_value=0).to_numpy(np.float32)
    F["declared"] = declared[:, cc]
    ucp = prof["ucp"][:, cc]
    up = prof["act"].u_price.to_numpy()[:, None]
    F["price_gap_cat"] = np.abs(np.where(np.isnan(ucp), up, ucp) - cat.rel_price[None, :])
    F["price_gap_user"] = np.abs(up - cat.rel_price[None, :])
    budget = target.monthly_budget.to_numpy()
    F["budget_gap"] = np.abs(np.log(cat.p.price.to_numpy()[None, :] / (np.nan_to_num(budget, nan=60)[:, None] / 60 * cat.cat_med[None, :])))
    F["budget_gap"][np.isnan(budget)] = np.nan
    # signed versions: label rate peaks slightly *below* the budget-scaled category median
    F["budget_ratio"] = np.log(cat.p.price.to_numpy()[None, :] / (np.nan_to_num(budget, nan=60)[:, None] / 60 * cat.cat_med[None, :]))
    F["budget_ratio"][np.isnan(budget)] = np.nan
    F["price_diff_user"] = cat.rel_price[None, :] - np.where(np.isnan(ucp), up, ucp)
    F["quality_gap_user"] = np.abs(prof["act"].u_quality.to_numpy()[:, None] - cat.quality[None, :])
    pq = target.preferred_quality.to_numpy()
    F["quality_gap_pref"] = np.abs(pq[:, None] - cat.quality[None, :])
    F["quality_diff_pref"] = cat.quality[None, :] - pq[:, None]
    for k, v in own.items():
        F[k] = v
    # content score from the starter (strong for sparse users)
    catpref = np.asarray(Xs @ cat.C) + 0.01 + 0.08 * declared
    catpref /= catpref.sum(1, keepdims=True)
    F["content"] = (1.5 * np.log(catpref @ cat.C.T + 0.002) + 1.5 * (np.asarray(Xs @ cat.T) @ cat.T.T)
                    - 0.8 * np.nan_to_num(F["budget_gap"], nan=0.5) - 0.3 * np.abs(np.nan_to_num(pq, nan=3)[:, None] - cat.quality[None, :]))
    F["disc_x_sens"] = cat.p.offer_discount.to_numpy()[None, :] * np.nan_to_num(prof["act"].hist_disc.to_numpy(), nan=0.09)[:, None]
    F["family_x_hh"] = cat.family[None, :] * target.household_size.to_numpy()[:, None]
    F["sub_x_tier"] = cat.p.is_subscription.to_numpy()[None, :] * pd.Categorical(target.membership_tier, categories=["basic", "plus", "premium"]).codes[:, None]
    F = {k: np.asarray(v, dtype=np.float32) for k, v in F.items()}

    snaps = np.array([day_of(d) for d in target.snapshot_date])
    assert len(np.unique(snaps)) == 1
    rp = un.recent[snaps[0]]
    item = pd.DataFrame({
        "label_pop": lm.label_pop, "touch_rate": lm.touch_rate, "untouch_rate": lm.untouch_rate,
        "rpop14": rp[14], "rpop30": rp[30], "rconv": rp["conv"],
        "price": cat.p.price.to_numpy(), "rel_price": cat.rel_price, "quality": cat.quality,
        "discount": cat.p.offer_discount.to_numpy(), "repeatable": cat.p.is_repeatable.to_numpy(),
        "subscription": cat.p.is_subscription.to_numpy(), "n_regions": cat.p.available_regions.str.split().str.len().to_numpy(),
        "item_cat": cat.cat_codes, "item_dom": cat.dom_codes,
        "is_cold": np.isin(np.arange(n), cold_items).astype(np.float32),
    })
    user = prof["act"].copy()
    user["household"] = target.household_size.to_numpy()
    user["tier"] = pd.Categorical(target.membership_tier, categories=["basic", "plus", "premium"]).codes
    user["budget"] = budget
    user["pref_q"] = pq
    user["tenure"] = target.tenure_months.to_numpy()
    user["n_declared"] = declared.sum(1)
    user["age_band"] = pd.Categorical(target.age_band).codes
    user["device_type"] = pd.Categorical(target.device_type).codes
    return F, item.astype(np.float32), user.astype(np.float32)
