"""Companion agent: a single tool-using agent on Amazon Bedrock (Converse API).

Per turn:
  1. load conversation + session memory + long-term memory for the customer (isolated by user_id)
  2. decide - deterministically, from the customer's own words - whether this turn is an explicit
     confirmation of a previously shown checkout summary
  3. loop: build system prompt with live context -> model -> execute tool calls -> repeat
  4. persist memory, compact long histories into a rolling summary, return reply + UI payload
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from .confirmation import is_explicit_confirmation
from .config import Config
from .data_layer import Catalogue, Interactions, Personalizer, Profiles, Recommender
from .memory import MemoryStore
from .prompts import GREETING_TRIGGER, SUMMARY_PROMPT, build_system
from .tools import ToolExecutor, TurnState, all_specs

log = logging.getLogger("companion.agent")


def _ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AppContext:
    def __init__(self, config: Config, llm=None):
        from starter.store import Store

        self.config = config
        self.store = Store(db_path=config.store_db, products_path=config.products_path,
                           customers_path=config.customers_path, demo_date=config.demo_date, currency=config.currency)
        self.catalogue = Catalogue(self.store)
        self.profiles = Profiles(self.store)
        self.interactions = Interactions(config.interactions_path, self.catalogue)
        self.recommender = Recommender(config.recommendations_path, self.catalogue)
        self.personalizer = Personalizer(self.catalogue, self.profiles, self.interactions, self.recommender)
        self.memory = MemoryStore(config.memory_db)
        if llm is None:
            from .llm import BedrockLLM
            llm = BedrockLLM(config)
        self.llm = llm
        self.executor = ToolExecutor(self)
        self.tools = all_specs()
        self._insights = {}
        self.agent = CompanionAgent(self)

    def insights(self, user_id):
        if user_id not in self._insights:
            owned = (self.profiles.get(user_id).get("owned_items") or "").split()
            self._insights[user_id] = self.interactions.insights(user_id, owned)
        return self._insights[user_id]

    def data_status(self):
        return {"products": len(self.catalogue.products), "customers": len(self.store.customers),
                "interactions_loaded": self.interactions.available,
                "interaction_columns": self.interactions.columns,
                "recommender_source": self.recommender.source,
                "recommender_users": len(self.recommender.ranked),
                "files": {k: str(getattr(self.config, k)) for k in
                          ("products_path", "customers_path", "interactions_path", "recommendations_path")}}


class CompanionAgent:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    # ------------------------------------------------------------------ main turn
    def chat(self, user_id: str, text: str, channel: str = "text", hidden: bool = False) -> dict:
        ctx = self.ctx
        with ctx.memory.lock(user_id):
            conv = ctx.memory.load_conversation(user_id)
            session = ctx.memory.load_session(user_id)
            long_term = ctx.memory.load_long_term(user_id)
            turn = conv["turn"] + 1
            pending = session.get("pending_checkout")
            confirmed = (not hidden and bool(pending) and pending["turn"] < turn and is_explicit_confirmation(text))
            st = TurnState(user_id=user_id, turn=turn, channel=channel, user_confirmed=confirmed,
                           session=session, long_term=long_term)
            messages = list(conv["messages"]) + [{"role": "user", "content": [{"text": text}]}]
            hint = ("The customer's latest message IS an explicit confirmation of the pending checkout summary; "
                    "you may call place_order." if confirmed else None)
            if not hidden:
                conv["transcript"].append({"role": "user", "text": text, "channel": channel, "ts": _ts()})

            reply, error = "", None
            started = time.time()
            try:
                reply = self._loop(st, messages, conv["summary"], hint)
                conv["messages"] = messages
                conv["turn"] = turn
            except Exception as exc:  # AI service problems must not lose the customer's state
                log.exception("turn failed")
                error = str(exc)[:300]
                if st.actions:
                    reply = ("I hit an AI service error after completing part of your request. "
                             "Any successful actions shown below were kept, and your basket is saved. "
                             "Please review them before trying again.")
                else:
                    reply = ("I couldn't reach the AI service just now, so I didn't perform any new actions. "
                             "Your basket is saved - please try again in a moment.")
                # Keep the attempted turn and any completed tool results in the
                # model history. Dropping them would make later turns disagree
                # with the persisted basket/session state.
                messages.append({"role": "assistant", "content": [{"text": reply}]})
                conv["messages"] = messages
                conv["turn"] = turn
            ctx.memory.save_session(user_id, session)
            ctx.memory.save_long_term(user_id, long_term)
            entry = {"role": "assistant", "text": reply, "channel": channel, "ts": _ts(), "actions": st.actions,
                     "cards": st.cards, "checkout": st.checkout, "order": st.order, "error": error}
            conv["transcript"].append(entry)
            if not error:
                self._compact(conv)
            ctx.memory.save_conversation(user_id, conv)
            ctx.memory.log(user_id, "turn", {"channel": channel, "confirmed": confirmed, "tools": [a["tool"] for a in st.actions],
                                             "seconds": round(time.time() - started, 2), "error": error})
            return {**entry, "basket": ctx.store.get_basket(user_id), "pending_checkout": session.get("pending_checkout"),
                    "memory": self.memory_view(session, long_term)}

    def _loop(self, st: TurnState, messages: list, summary: str, hint: str | None) -> str:
        ctx = self.ctx
        for _ in range(ctx.config.max_tool_steps):
            basket = ctx.store.get_basket(st.user_id)
            system = build_system(ctx, st.user_id, st.session, st.long_term, summary, st.channel, basket, hint)
            resp = ctx.llm.converse(system, messages, ctx.tools)
            msg = resp["output"]["message"]
            msg["content"] = [b for b in msg.get("content", []) if b] or [{"text": "..."}]
            messages.append(msg)
            uses = [b["toolUse"] for b in msg["content"] if "toolUse" in b]
            if not uses:
                return "".join(b.get("text", "") for b in msg["content"]).strip() or "Is there anything else I can help with?"
            results = []
            with_status = "anthropic" in str(getattr(ctx.llm, "active_model", "anthropic"))
            for tu in uses:
                result, ok = ctx.executor.run(tu["name"], tu.get("input") or {}, st)
                block = {"toolUseId": tu["toolUseId"], "content": [{"json": json.loads(json.dumps(result, default=str))}]}
                if with_status:
                    block["status"] = "success" if ok else "error"
                results.append({"toolResult": block})
            messages.append({"role": "user", "content": results})
        fallback = "I've done several steps on that - let me know how you'd like to continue."
        messages.append({"role": "assistant", "content": [{"text": fallback}]})
        return fallback

    # ------------------------------------------------------------------ session start / greeting
    def start(self, user_id: str) -> dict:
        # Serialise initial greeting creation too. The lock is re-entrant
        # because chat() protects the complete agent turn independently.
        with self.ctx.memory.lock(user_id):
            conv = self.ctx.memory.load_conversation(user_id)
            if not conv["transcript"]:
                result = self.chat(user_id, GREETING_TRIGGER, channel="text", hidden=True)
                if result.get("error"):
                    self._deterministic_greeting(user_id)
                conv = self.ctx.memory.load_conversation(user_id)
            return conv

    def _deterministic_greeting(self, user_id):
        nudges = self.ctx.personalizer.proactive_nudges(user_id, self.ctx.insights(user_id))
        text = "Hi! I'm your lifestyle companion. " + (nudges[0]["text"] + " " if nudges else "") + \
               "What are you planning today?"
        conv = self.ctx.memory.load_conversation(user_id)
        conv["transcript"] = [{"role": "assistant", "text": text, "channel": "text", "ts": _ts(), "actions": [], "cards": []}]
        conv["messages"] = []
        self.ctx.memory.save_conversation(user_id, conv)

    # ------------------------------------------------------------------ trusted UI actions
    def confirm_via_ui(self, user_id: str, checkout_id: str) -> dict:
        ctx = self.ctx
        with ctx.memory.lock(user_id):
            conv = ctx.memory.load_conversation(user_id)
            session = ctx.memory.load_session(user_id)
            long_term = ctx.memory.load_long_term(user_id)
            pending = session.get("pending_checkout")
            st = TurnState(user_id=user_id, turn=conv["turn"] + 1, channel="ui", user_confirmed=True,
                           session=session, long_term=long_term)
            if not pending or pending["checkout_id"] != checkout_id:
                result = {"ok": False, "error": {"code": "CHECKOUT_OUTDATED",
                                                 "message": "This summary is out of date. Ask me to prepare the checkout again."}}
            else:
                result = ctx.executor.confirm(st, checkout_id)
            if result["ok"]:
                o = result["data"]
                reply = (f"Done - your simulated order {o['order_id']} is placed. "
                         f"Total {o['total']} {o['currency']} for {len(o['items'])} item(s). No payment was taken.")
            else:
                reply = f"I couldn't place the order: {result['error']['message']}"
            st.actions.append({"tool": "confirm_order (app, customer pressed Confirm)", "input": {"checkout_id": checkout_id},
                               "ok": result["ok"], "summary": reply})
            conv["messages"] += [{"role": "user", "content": [{"text": "[App event] I pressed the Confirm order button on the checkout summary."}]},
                                 {"role": "assistant", "content": [{"text": reply}]}]
            conv["turn"] += 1
            conv["transcript"] += [{"role": "user", "text": "Confirm order", "channel": "ui", "ts": _ts()},
                                   {"role": "assistant", "text": reply, "channel": "ui", "ts": _ts(),
                                    "actions": st.actions, "cards": [], "order": st.order}]
            ctx.memory.save_session(user_id, session)
            ctx.memory.save_long_term(user_id, long_term)
            ctx.memory.save_conversation(user_id, conv)
            ctx.memory.log(user_id, "ui_confirm", {"ok": result["ok"], "checkout_id": checkout_id})
            return {"ok": result["ok"], "reply": reply, "order": st.order, "error": result.get("error"),
                    "basket": ctx.store.get_basket(user_id)}

    def record_ui_event(self, user_id: str, text: str, basket_revision=None):
        with self.ctx.memory.lock(user_id):
            session = self.ctx.memory.load_session(user_id)
            session["ui_events"] = (session.get("ui_events", []) + [text])[-10:]
            pending = session.get("pending_checkout")
            if pending and basket_revision is not None and pending["revision"] != basket_revision:
                session["pending_checkout"] = None
            self.ctx.memory.save_session(user_id, session)

    def cancel_checkout(self, user_id: str):
        with self.ctx.memory.lock(user_id):
            session = self.ctx.memory.load_session(user_id)
            session["pending_checkout"] = None
            session["ui_events"] = (session.get("ui_events", []) + ["Customer dismissed the checkout summary"])[-10:]
            self.ctx.memory.save_session(user_id, session)

    # ------------------------------------------------------------------ memory compaction
    def _compact(self, conv):
        cfg = self.ctx.config
        msgs = conv["messages"]
        if len(msgs) <= cfg.max_history_messages:
            return
        target = len(msgs) - cfg.keep_recent_messages
        cut = next((i for i in range(target, len(msgs))
                    if msgs[i]["role"] == "user" and all("text" in b for b in msgs[i]["content"])), None)
        if not cut:
            return
        old, keep = msgs[:cut], msgs[cut:]
        lines = []
        for m in old:
            for b in m["content"]:
                if "text" in b:
                    who = "Customer" if m["role"] == "user" else "Assistant"
                    if not b["text"].startswith("(The customer just opened"):
                        lines.append(f"{who}: {b['text'][:600]}")
                elif "toolUse" in b:
                    lines.append(f"[tool {b['toolUse']['name']} {json.dumps(b['toolUse'].get('input'))[:200]}]")
                elif "toolResult" in b:
                    lines.append(f"[result {json.dumps(b['toolResult']['content'])[:300]}]")
        try:
            prompt = SUMMARY_PROMPT.format(previous=conv.get("summary") or "(none)", transcript="\n".join(lines))
            r = self.ctx.llm.converse("You write compact, factual memory summaries.",
                                      [{"role": "user", "content": [{"text": prompt}]}], None, max_tokens=500,
                                      model_id=cfg.summary_model_id)
            summary = "".join(b.get("text", "") for b in r["output"]["message"]["content"]).strip()
        except Exception:
            summary = ((conv.get("summary") or "") + " | " + " / ".join(l for l in lines if l.startswith("Customer"))[-800:]).strip(" |")
        conv["summary"], conv["messages"] = summary, keep

    # ------------------------------------------------------------------ views
    @staticmethod
    def memory_view(session, long_term):
        return {"goal": session.get("goal"), "budget": session.get("budget"), "budget_scope": session.get("budget_scope"),
                "preferences": session.get("preferences"), "dislikes": session.get("dislikes"),
                "rejected": session.get("rejected"), "last_presented": (session.get("presented") or [None])[-1],
                "long_term": long_term}
