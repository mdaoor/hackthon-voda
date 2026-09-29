"""Tools exposed to the LLM and their execution.

Supplied starter functions (get_basket, add_to_basket, update_basket,
remove_from_basket, prepare_checkout, get_order) are registered with their
original schemas from starter/tool_schemas.json and executed through the
starter's `tool_adapter.call_tool`, which binds user_id from the session.

`confirm_order` is NOT model-visible. The model may only call `place_order`
(no arguments); the application then calls the starter's `confirm_order` with
customer_confirmed=True if - and only if - the customer's own latest message
was an explicit confirmation of a checkout summary shown in an earlier turn
(see confirmation.py) or the customer pressed the Confirm button.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import STARTER_DIR
from starter import order_tools
from starter.store import ToolError
from starter.tool_adapter import call_tool

STARTER_TOOLS = {"get_basket", "add_to_basket", "update_basket", "remove_from_basket", "prepare_checkout", "get_order"}
BASKET_MUTATIONS = {"add_to_basket", "update_basket", "remove_from_basket"}
MAX_PRESENTED_LISTS = 4


def _spec(name, description, properties=None, required=None):
    return {"toolSpec": {"name": name, "description": description, "inputSchema": {"json": {
        "type": "object", "properties": properties or {}, "required": required or []}}}}


def starter_specs():
    schemas = json.loads((STARTER_DIR / "tool_schemas.json").read_text())
    out = []
    for s in schemas:
        params = {k: v for k, v in s["parameters"].items() if k != "additionalProperties"}
        out.append({"toolSpec": {"name": s["name"], "description": s["description"], "inputSchema": {"json": params}}})
    return out


ID_LIST = {"type": "array", "items": {"type": "string"}}

COMPANION_SPECS = [
    _spec("search_products",
          "Search the catalogue for products eligible for THIS customer (region, device OS, launch date, "
          "ownership are checked). Returns final prices after discount, quality tier, tags and why each fits. "
          "Rejected products are excluded automatically. Use category names from the CATALOGUE MAP.",
          {"query": {"type": "string", "description": "Free-text keywords, e.g. 'compact speaker'."},
           "category": {"type": "string", "description": "Catalogue category (preferred), e.g. 'smartphone'."},
           "domain": {"type": "string", "description": "Catalogue domain, e.g. 'home'."},
           "max_price": {"type": "number", "description": "Max final price per item. Defaults to the customer's stated per-item budget."},
           "min_price": {"type": "number"},
           "min_quality": {"type": "integer", "description": "Minimum quality tier 1-5."},
           "tags": {**ID_LIST, "description": "Style tags to match, e.g. ['family','compact']."},
           "sort_by": {"type": "string", "enum": ["relevance", "price_asc", "price_desc", "quality", "discount"]},
           "include_ineligible": {"type": "boolean", "description": "Also return ineligible items with reason (to explain unavailability)."},
           "ignore_budget": {"type": "boolean", "description": "Do not apply the stored budget automatically."},
           "limit": {"type": "integer", "minimum": 1, "maximum": 10}}),
    _spec("get_recommendations",
          "Personalised picks: the recommender model's ranked output for this customer (eligible only), topped up "
          "with profile/history-based ranking. Filter by category and budget to fit the current request.",
          {"category": {"type": "string"}, "max_price": {"type": "number"},
           "ignore_budget": {"type": "boolean"}, "limit": {"type": "integer", "minimum": 1, "maximum": 8}}),
    _spec("get_product_details", "Full facts and eligibility for specific products.",
          {"product_ids": {**ID_LIST, "maxItems": 6}}, ["product_ids"]),
    _spec("compare_products", "Side-by-side comparison of 2-4 products with the key differences highlighted.",
          {"product_ids": {**ID_LIST, "minItems": 2, "maxItems": 4}}, ["product_ids"]),
    _spec("get_complementary_products",
          "Cross-sell: products that pair well with a product (or the current basket when product_id is omitted), "
          "based on what similar customers buy together. Eligible and budget-aware.",
          {"product_id": {"type": "string"}, "max_price": {"type": "number"},
           "limit": {"type": "integer", "minimum": 1, "maximum": 4}}),
    _spec("present_options",
          "REQUIRED before describing products to the customer. Registers the exact ordered list you are about to "
          "present (becomes 'option 1, 2, 3' for later references) and shows product cards in the app.",
          {"title": {"type": "string", "description": "Short heading, e.g. 'Smartphones under 100'."},
           "product_ids": {**ID_LIST, "minItems": 1, "maxItems": 5},
           "purpose": {"type": "string", "enum": ["recommendation", "comparison", "cross_sell", "alternatives"]}},
          ["title", "product_ids"]),
    _spec("update_customer_memory",
          "Save what the customer tells you: goal, budget changes, preferences, dislikes, rejected products. "
          "Call it as soon as they state or change something.",
          {"goal": {"type": "string"},
           "budget": {"type": "number", "description": "Budget amount the customer stated."},
           "budget_scope": {"type": "string", "enum": ["per_item", "total"],
                            "description": "per_item (default) = max price per product; total = whole basket."},
           "clear_budget": {"type": "boolean"},
           "preferences_add": {**ID_LIST, "description": "e.g. ['prefers android', 'likes compact devices']"},
           "dislikes_add": ID_LIST,
           "rejected_product_ids": ID_LIST,
           "rejection_reason": {"type": "string"},
           "unreject_product_ids": ID_LIST,
           "note": {"type": "string"}}),
    _spec("get_customer_activity",
          "Detailed behaviour history across Home, Shop and Rewards: recent views, open cart items, purchases, "
          "category interests, rewards activity.", {}),
    _spec("place_order",
          "Place the simulated order for the checkout summary already shown to the customer. Only call this when the "
          "customer's latest message explicitly confirms that summary. The application verifies the confirmation "
          "itself and will refuse otherwise.", {}),
]


def all_specs():
    return COMPANION_SPECS + starter_specs()


@dataclass
class TurnState:
    user_id: str
    turn: int
    channel: str = "text"
    user_confirmed: bool = False
    session: dict = field(default_factory=dict)
    long_term: dict = field(default_factory=dict)
    actions: list = field(default_factory=list)
    cards: list = field(default_factory=list)
    checkout: dict | None = None
    order: dict | None = None
    basket_changed: bool = False


class ToolExecutor:
    def __init__(self, ctx):
        self.ctx = ctx  # AppContext

    # ------------------------------------------------------------------ helpers
    def _budget(self, st: TurnState, args):
        if args.get("max_price") is not None:
            return float(args["max_price"])
        if args.get("ignore_budget"):
            return None
        s = st.session
        if s.get("budget") and s.get("budget_scope", "per_item") == "per_item":
            return float(s["budget"])
        if s.get("budget") and s.get("budget_scope") == "total":
            basket = self.ctx.store.get_basket(st.user_id)
            remaining = float(s["budget"]) - float(basket["total"])
            return max(remaining, 0.0)
        return None

    def _excluded(self, st: TurnState):
        return set(st.session.get("rejected", {}))

    def _validate_ids(self, ids):
        known = [i for i in ids if self.ctx.catalogue.exists(i)]
        unknown = [i for i in ids if not self.ctx.catalogue.exists(i)]
        return known, unknown

    def _compact(self, card):
        keep = ("product_id", "name", "category", "final_price", "list_price", "discount_pct", "quality_tier",
                "style_tags", "subscription", "eligible", "ineligible_reason", "why", "source", "rec_rank",
                "model_rank", "consensus_rank")
        return {k: card[k] for k in keep if k in card}

    def _recommendation_event(self, st, product_id, event_type, app_section="shop"):
        event = {"product_id": product_id, "event_type": event_type,
                 "event_date": self.ctx.config.demo_date, "app_section": app_section}
        st.session["recommendation_events"] = (st.session.get("recommendation_events", []) + [event])[-100:]
        self.ctx.recommender.events_changed(st.user_id, st.session["recommendation_events"])

    def _record_basket_delta(self, st, before, after):
        old = {row["product_id"]: row["quantity"] for row in before.get("items", [])}
        new = {row["product_id"]: row["quantity"] for row in after.get("items", [])}
        for pid in sorted(set(old) | set(new)):
            if new.get(pid, 0) > old.get(pid, 0):
                self._recommendation_event(st, pid, "cart")
            elif new.get(pid, 0) < old.get(pid, 0):
                self._recommendation_event(st, pid, "cancel")

    # ------------------------------------------------------------------ dispatch
    def run(self, name: str, args: dict, st: TurnState) -> tuple[dict, bool]:
        args = args or {}
        try:
            if name in STARTER_TOOLS:
                result = self._starter(name, args, st)
            else:
                handler = getattr(self, f"t_{name}", None)
                if handler is None:
                    result = {"ok": False, "error": {"code": "UNKNOWN_TOOL", "message": f"{name} is not available."}}
                else:
                    result = handler(args, st)
        except ToolError as e:
            result = {"ok": False, "error": {"code": e.code, "message": e.message}}
        except Exception as e:  # defensive: surface as a tool error, never crash the turn
            result = {"ok": False, "error": {"code": "INTERNAL_ERROR", "message": str(e)[:300]}}
        ok = bool(result.get("ok"))
        st.actions.append({"tool": name, "input": args, "ok": ok, "summary": self._summarize(name, result)})
        self.ctx.memory.log(st.user_id, "tool", {"tool": name, "input": args, "ok": ok,
                                                 "error": result.get("error")})
        return result, ok

    @staticmethod
    def _summarize(name, result):
        if not result.get("ok"):
            return f"{result['error']['code']}: {result['error']['message']}"
        d = result.get("data", {})
        if isinstance(d, dict):
            if "results" in d:
                return f"{len(d['results'])} result(s)"
            if "items" in d and "total" in d:
                return f"basket: {len(d['items'])} item(s), total {d['total']}"
            if "checkout_id" in d:
                return f"checkout ready, total {d['summary']['total']}"
            if "order_id" in d:
                return f"order {d['order_id']}"
        return "done"

    # ------------------------------------------------------------------ starter tools
    def _starter(self, name, args, st: TurnState):
        before = self.ctx.store.get_basket(st.user_id) if name in BASKET_MUTATIONS else None
        result = call_tool(self.ctx.store, st.user_id, name, args)  # user_id from session, never from the model
        if not result["ok"]:
            if name == "prepare_checkout" and result["error"]["code"] == "SUBSCRIPTION_TERMS_MISSING":
                result["hint"] = "Tell the customer subscriptions can't be checked out yet; offer to remove them."
            return result
        data = result["data"]
        if name in BASKET_MUTATIONS:
            self._record_basket_delta(st, before, data)
            st.basket_changed = True
            pending = st.session.get("pending_checkout")
            if pending and pending.get("revision") != data.get("revision"):
                st.session["pending_checkout"] = None
                result["checkout_summary_invalidated"] = True
                result["hint"] = "The basket changed, so the earlier checkout summary is void. Call prepare_checkout again and ask for fresh confirmation."
            st.checkout = None
        if name == "prepare_checkout":
            st.session["pending_checkout"] = {"checkout_id": data["checkout_id"], "revision": data["summary"]["revision"],
                                              "turn": st.turn, "total": data["summary"]["total"]}
            st.checkout = data
            result["hint"] = ("Present this summary (items, quantities, unit prices, total, simulated) and ask the "
                              "customer to confirm explicitly. Do not call place_order in this same reply.")
        return result

    # ------------------------------------------------------------------ discovery tools
    def t_search_products(self, a, st):
        budget = self._budget(st, a)
        data = self.ctx.personalizer.search(
            st.user_id, query=a.get("query"), domain=a.get("domain"), category=a.get("category"),
            min_price=a.get("min_price"), max_price=budget, min_quality=a.get("min_quality"), tags=a.get("tags"),
            exclude_ids=self._excluded(st), include_ineligible=bool(a.get("include_ineligible")),
            sort_by=a.get("sort_by") or "relevance", limit=int(a.get("limit") or 6))
        data["results"] = [self._compact(r) for r in data["results"]]
        data["applied_max_price"] = budget
        if self._excluded(st):
            data["excluded_rejected"] = sorted(self._excluded(st))
        return {"ok": True, "data": data}

    def t_get_recommendations(self, a, st):
        budget = self._budget(st, a)
        data = self.ctx.personalizer.recommendations(st.user_id, category=a.get("category"), max_price=budget,
                                                     exclude_ids=self._excluded(st), limit=int(a.get("limit") or 5))
        data["results"] = [self._compact(r) for r in data["results"]]
        data["applied_max_price"] = budget
        return {"ok": True, "data": data}

    def t_get_product_details(self, a, st):
        known, unknown = self._validate_ids(a.get("product_ids") or [])
        data = {"products": [self.ctx.catalogue.card(pid, st.user_id) for pid in known]}
        if unknown:
            data["unknown_product_ids"] = unknown
        return {"ok": True, "data": data}

    def t_compare_products(self, a, st):
        known, unknown = self._validate_ids(a.get("product_ids") or [])
        if len(known) < 2:
            return {"ok": False, "error": {"code": "INVALID_ARGUMENTS", "message": f"Need 2-4 known products; unknown: {unknown}"}}
        cards = [self.ctx.catalogue.card(pid, st.user_id) for pid in known]
        for c in cards:
            c["fit_score"], c["why"] = self.ctx.personalizer.score(st.user_id, c["product_id"], budget=self._budget(st, {}))
        cheapest = min(cards, key=lambda c: float(c["final_price"]))
        best_q = max(cards, key=lambda c: c["quality_tier"])
        best_fit = max(cards, key=lambda c: c["fit_score"])
        return {"ok": True, "data": {"products": [self._compact(c) | {"fit_score": c["fit_score"]} for c in cards],
                                     "highlights": {"cheapest": cheapest["product_id"], "highest_quality": best_q["product_id"],
                                                    "best_fit_for_customer": best_fit["product_id"]},
                                     "unknown_product_ids": unknown}}

    def t_get_complementary_products(self, a, st):
        base = [a["product_id"]] if a.get("product_id") else [i["product_id"] for i in self.ctx.store.get_basket(st.user_id)["items"]]
        basket_ids = [i["product_id"] for i in self.ctx.store.get_basket(st.user_id)["items"]]
        data = self.ctx.personalizer.complementary(st.user_id, base, max_price=self._budget(st, a),
                                                   exclude_ids=self._excluded(st) | set(basket_ids),
                                                   limit=int(a.get("limit") or 2))
        data["results"] = [self._compact(r) for r in data["results"]]
        if self.ctx.profiles.get(st.user_id).get("marketing_opt_in") != "1":
            data["note"] = "Customer has not opted in to marketing: only mention if clearly useful for their stated goal."
        return {"ok": True, "data": data}

    def t_present_options(self, a, st):
        known, unknown = self._validate_ids(a.get("product_ids") or [])
        if not known:
            return {"ok": False, "error": {"code": "UNKNOWN_PRODUCT", "message": f"None of these product IDs exist: {unknown}"}}
        cards = [self.ctx.catalogue.card(pid, st.user_id) for pid in known]
        for c in cards:
            c["why"] = self.ctx.personalizer.score(st.user_id, c["product_id"], budget=self._budget(st, {}))[1]
        entry = {"turn": st.turn, "title": a.get("title") or "Options", "purpose": a.get("purpose") or "recommendation",
                 "options": [{"n": i + 1, "product_id": c["product_id"], "name": c["name"], "final_price": c["final_price"]}
                             for i, c in enumerate(cards)]}
        st.session["presented"] = (st.session.get("presented", []) + [entry])[-MAX_PRESENTED_LISTS:]
        st.cards.append({"title": entry["title"], "purpose": entry["purpose"], "products": cards})
        data = {"presented": entry["options"],
                "eligibility_warnings": [f"{c['name']}: {c['ineligible_reason']}" for c in cards if not c["eligible"]]}
        if unknown:
            data["unknown_product_ids"] = unknown
        return {"ok": True, "data": data}

    def t_update_customer_memory(self, a, st):
        s, lt = st.session, st.long_term
        changed = []
        if a.get("goal"):
            s["goal"] = a["goal"]
            if a["goal"] not in lt["goals"]:
                lt["goals"] = (lt["goals"] + [a["goal"]])[-5:]
            changed.append("goal")
        if a.get("clear_budget"):
            s["budget"] = None
            changed.append("budget cleared")
        if a.get("budget") is not None:
            s["budget"] = float(a["budget"])
            s["budget_scope"] = a.get("budget_scope") or "per_item"
            changed.append(f"budget={s['budget']} ({s['budget_scope']})")
        for key in ("preferences", "dislikes"):
            for item in a.get(f"{key}_add") or []:
                if item not in s[key]:
                    s[key].append(item)
                if item not in lt[key]:
                    lt[key] = (lt[key] + [item])[-15:]
                changed.append(f"{key}+{item}")
        for pid in a.get("rejected_product_ids") or []:
            if self.ctx.catalogue.exists(pid):
                s["rejected"][pid] = a.get("rejection_reason") or "customer rejected"
                changed.append(f"rejected {pid}")
        for pid in a.get("unreject_product_ids") or []:
            s["rejected"].pop(pid, None)
            changed.append(f"unrejected {pid}")
        if a.get("note"):
            s["notes"] = (s["notes"] + [a["note"]])[-10:]
            changed.append("note")
        return {"ok": True, "data": {"saved": changed, "budget": s.get("budget"), "budget_scope": s.get("budget_scope"),
                                     "rejected": s["rejected"]}}

    def t_get_customer_activity(self, a, st):
        prof = self.ctx.profiles.get(st.user_id)
        return {"ok": True, "data": self.ctx.interactions.insights(st.user_id, (prof.get("owned_items") or "").split())}

    def t_place_order(self, a, st):
        pending = st.session.get("pending_checkout")
        if not pending:
            return {"ok": False, "error": {"code": "NO_PENDING_CHECKOUT",
                                           "message": "No checkout summary is awaiting confirmation. Call prepare_checkout and show the summary first."}}
        if pending["turn"] >= st.turn:
            return {"ok": False, "error": {"code": "CONFIRMATION_REQUIRED",
                                           "message": "The summary was prepared in this reply. Show it and wait for the customer's explicit confirmation."}}
        if not st.user_confirmed:
            return {"ok": False, "error": {"code": "CONFIRMATION_REQUIRED",
                                           "message": "The customer's latest message is not an explicit confirmation. Ask them to confirm the summary (or use the Confirm button)."}}
        return self.confirm(st, pending["checkout_id"])

    # ------------------------------------------------------------------ trusted confirmation path
    def confirm(self, st: TurnState, checkout_id: str):
        """Application-only call of the starter's confirm_order with customer_confirmed=True."""
        try:
            order = order_tools.confirm_order(self.ctx.store, st.user_id, checkout_id=checkout_id, customer_confirmed=True)
        except ToolError as e:
            if e.code in ("CHECKOUT_CHANGED", "CHECKOUT_NOT_FOUND"):
                st.session["pending_checkout"] = None
            return {"ok": False, "error": {"code": e.code, "message": e.message}}
        st.session["pending_checkout"] = None
        st.session["last_order"] = {"order_id": order["order_id"], "total": order["total"],
                                    "items": [i["product_name"] for i in order["items"]]}
        st.long_term["orders"] = (st.long_term.get("orders", []) + [st.session["last_order"]])[-10:]
        for item in order["items"]:
            self._recommendation_event(st, item["product_id"], "purchase", "checkout")
        st.order, st.checkout, st.basket_changed = order, None, True
        return {"ok": True, "data": order}
