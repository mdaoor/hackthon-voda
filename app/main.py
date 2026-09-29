"""HTTP API + static web UI.  Run:  uvicorn app.main:app --host 0.0.0.0 --port 8080"""
from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import AppContext
from .config import ROOT, Config
from .voice import SUPPORTED_LANGUAGES, Voice, text_language

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("companion")

CTX: AppContext | None = None
VOICE: Voice | None = None
USER_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")


def ctx() -> AppContext:
    global CTX, VOICE
    if CTX is None:
        cfg = Config()
        CTX = AppContext(cfg)
        VOICE = Voice(cfg)
        log.info("data: %s", CTX.data_status())
    return CTX


@asynccontextmanager
async def lifespan(_app):
    ctx()
    yield


app = FastAPI(title="AI Lifestyle Companion", version="1.0", lifespan=lifespan)


def valid_user(user_id: str) -> str:
    uid = (user_id or "").strip()
    if not USER_ID_RE.match(uid) or not ctx().profiles.exists(uid):
        raise HTTPException(404, f"We couldn't find customer ID '{uid[:40]}'. Check the ID and enter a valid one.")
    return uid


# ------------------------------------------------------------------ models
class StartIn(BaseModel):
    user_id: str


class ChatIn(BaseModel):
    user_id: str
    message: str = Field(min_length=1, max_length=2000)
    channel: str = "text"
    voice_language: str | None = None
    voice_languages: list[str] = Field(default_factory=list)


class ConfirmIn(BaseModel):
    user_id: str
    checkout_id: str


class BasketIn(BaseModel):
    user_id: str
    product_id: str
    quantity: int = 1


class SpeakIn(BaseModel):
    text: str = Field(max_length=4000)
    language: str | None = None


# ------------------------------------------------------------------ meta
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def config():
    c = ctx()
    return {"currency": c.config.currency, "demo_date": c.config.demo_date, "stt": VOICE.stt_mode,
            "tts": c.config.tts_provider, "sample_ids": c.profiles.sample_ids() if c.config.show_sample_ids else [],
            "model": getattr(c.llm, "active_model", None),
            "voice_languages": [{"code": "en-US", "label": "English"}, {"code": "ar-SA", "label": "العربية"}],
            "configured_voice_languages": list(VOICE.languages),
            "voice_auto_detection": VOICE.auto_language_available}


@app.get("/api/status")
def status(check_llm: bool = False):
    c = ctx()
    out = {"data": c.data_status()}
    if check_llm:
        out["llm"] = c.llm.health()
    return out


# ------------------------------------------------------------------ session + chat
def customer_view(uid):
    c = ctx()
    p = c.profiles.get(uid)
    return {"user_id": uid, "region": p.get("region"), "age_band": p.get("age_band"), "household_size": p.get("household_size"),
            "membership_tier": p.get("membership_tier"), "device": f"{p.get('device_type')} {p.get('device_os')}",
            "device_os": p.get("device_os"), "tenure_months": p.get("tenure_months"), "monthly_budget": p.get("monthly_budget"),
            "preferred_quality": p.get("preferred_quality"), "marketing_opt_in": p.get("marketing_opt_in") == "1",
            "interests": (p.get("declared_interests") or "").split(),
            "owned": [c.catalogue.card(o) for o in (p.get("owned_items") or "").split() if c.catalogue.exists(o)]}


@app.post("/api/session/start")
def session_start(body: StartIn):
    uid = valid_user(body.user_id)
    c = ctx()
    conv = c.agent.start(uid)
    session = c.memory.load_session(uid)
    return {"customer": customer_view(uid), "transcript": conv["transcript"], "basket": c.store.get_basket(uid),
            "pending_checkout": session.get("pending_checkout"),
            "memory": c.agent.memory_view(session, c.memory.load_long_term(uid))}


@app.post("/api/session/reset")
def session_reset(body: StartIn):
    uid = valid_user(body.user_id)
    c = ctx()
    c.memory.reset_conversation(uid)
    c.recommender.events_changed(uid, [])
    return session_start(body)


@app.post("/api/chat")
def chat(body: ChatIn):
    uid = valid_user(body.user_id)
    channel = "voice" if body.channel == "voice" else "text"
    if body.voice_language is not None and body.voice_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(422, f"Unsupported voice language '{body.voice_language}'.")
    if any(language not in SUPPORTED_LANGUAGES for language in body.voice_languages):
        raise HTTPException(422, "voice_languages contains an unsupported language.")
    return ctx().agent.chat(uid, body.message.strip(), channel=channel,
                            voice_language=body.voice_language, voice_languages=body.voice_languages)


@app.get("/api/customer/{user_id}/memory")
def memory(user_id: str):
    uid = valid_user(user_id)
    c = ctx()
    return {**c.agent.memory_view(c.memory.load_session(uid), c.memory.load_long_term(uid)),
            "events": c.memory.recent_events(uid, 30)}


# ------------------------------------------------------------------ checkout (trusted, model-free)
@app.post("/api/checkout/confirm")
def checkout_confirm(body: ConfirmIn):
    uid = valid_user(body.user_id)
    return ctx().agent.confirm_via_ui(uid, body.checkout_id)


@app.post("/api/checkout/cancel")
def checkout_cancel(body: StartIn):
    uid = valid_user(body.user_id)
    ctx().agent.cancel_checkout(uid)
    return {"ok": True}


# ------------------------------------------------------------------ basket panel (direct UI edits via starter functions)
def _basket_op(uid, name, args, note):
    from starter.tool_adapter import call_tool
    c = ctx()
    # Use the same per-customer lock as agent turns and checkout confirmation.
    # This prevents a panel edit from racing between prepare_checkout and the
    # session-memory update that records its basket revision.
    with c.memory.lock(uid):
        before = c.store.get_basket(uid)
        res = call_tool(c.store, uid, name, args)
        if res["ok"]:
            c.agent.record_ui_event(uid, note, res["data"]["revision"])
            old = {row["product_id"]: row["quantity"] for row in before["items"]}
            new = {row["product_id"]: row["quantity"] for row in res["data"]["items"]}
            events = []
            for pid in sorted(set(old) | set(new)):
                if new.get(pid, 0) != old.get(pid, 0):
                    events.append({"product_id": pid,
                                   "event_type": "cart" if new.get(pid, 0) > old.get(pid, 0) else "cancel",
                                   "event_date": c.config.demo_date, "app_section": "basket"})
            c.agent.record_recommendation_events(uid, events)
            c.memory.log(uid, "ui_basket", {"tool": name, "args": args})
        return res


@app.get("/api/basket/{user_id}")
def basket(user_id: str):
    uid = valid_user(user_id)
    return ctx().store.get_basket(uid)


@app.post("/api/basket/update")
def basket_update(body: BasketIn):
    uid = valid_user(body.user_id)
    name = ctx().catalogue.products.get(body.product_id, {}).get("product_name", body.product_id)
    return _basket_op(uid, "update_basket", {"product_id": body.product_id, "quantity": body.quantity},
                      f"Customer set {name} quantity to {body.quantity} in the Basket panel")


@app.post("/api/basket/remove")
def basket_remove(body: BasketIn):
    uid = valid_user(body.user_id)
    name = ctx().catalogue.products.get(body.product_id, {}).get("product_name", body.product_id)
    return _basket_op(uid, "remove_from_basket", {"product_id": body.product_id},
                      f"Customer removed {name} in the Basket panel")


# ------------------------------------------------------------------ Home / Shop / Rewards
@app.get("/api/customer/{user_id}/home")
def home(user_id: str):
    uid = valid_user(user_id)
    c = ctx()
    ins = c.insights(uid)
    rejected = set(c.memory.load_session(uid).get("rejected", {}))
    return {"customer": customer_view(uid), "nudges": c.personalizer.proactive_nudges(uid, ins),
            "for_you": c.personalizer.recommendations(uid, limit=5, exclude_ids=rejected)["results"],
            "activity": {k: ins.get(k) for k in ("events", "top_categories", "recent_views", "open_cart_items", "recent_purchases")}}


@app.get("/api/customer/{user_id}/shop")
def shop(user_id: str, q: str | None = None, category: str | None = None, max_price: float | None = None,
         sort_by: str = "relevance", include_ineligible: bool = False):
    uid = valid_user(user_id)
    c = ctx()
    res = c.personalizer.search(uid, query=q, category=category, max_price=max_price, sort_by=sort_by,
                                include_ineligible=include_ineligible, limit=24)
    return {**res, "categories": c.catalogue.overview()}


@app.get("/api/customer/{user_id}/rewards")
def rewards(user_id: str):
    uid = valid_user(user_id)
    c = ctx()
    ins = c.insights(uid)
    offers = c.personalizer.search(uid, sort_by="discount", limit=40)["results"]
    offers = sorted([o for o in offers if o["discount_pct"] > 0], key=lambda o: -o["fit_score"])[:6]
    prof = c.profiles.get(uid)
    return {"membership_tier": prof.get("membership_tier"), "tenure_months": prof.get("tenure_months"),
            "marketing_opt_in": prof.get("marketing_opt_in") == "1", "rewards_activity": ins.get("rewards_activity"),
            "offers": offers, "orders": c.memory.load_long_term(uid).get("orders", [])}


# ------------------------------------------------------------------ voice
@app.post("/api/voice/transcribe")
async def voice_transcribe(request: Request, user_id: str, rate: int = 16000, language: str = "auto"):
    valid_user(user_id)
    try:
        VOICE.validate_language(language)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    pcm = await request.body()
    if len(pcm) > 16000 * 2 * 60:
        raise HTTPException(413, "Recording is too long; keep it under 60 seconds.")
    try:
        result = await VOICE.transcribe(pcm, sample_rate=rate, language=language)
    except Exception as exc:
        log.exception("transcribe failed")
        return JSONResponse({"ok": False, "error": str(exc)[:200], "fallback": "browser"}, status_code=502)
    return {"ok": True, **result}


@app.post("/api/voice/speak")
def voice_speak(body: SpeakIn):
    if body.language is not None and body.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(422, f"Unsupported voice language '{body.language}'.")
    try:
        language = body.language or text_language(body.text)
        audio = VOICE.speak(body.text, language=language)
    except Exception as exc:
        log.exception("polly failed")
        return JSONResponse({"ok": False, "error": str(exc)[:200], "fallback": "browser",
                             "language": body.language or text_language(body.text)}, status_code=502)
    return Response(content=audio, media_type="audio/mpeg", headers={"X-Voice-Language": language})


# ------------------------------------------------------------------ static UI
WEB = ROOT / "web"
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(Path(WEB) / "index.html")
