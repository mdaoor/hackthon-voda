"""End-to-end tests without AWS: a scripted fake LLM drives the real agent loop,
tools, memory and the starter's basket/order functions."""
import os
import pickle
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATA_DIR", str(ROOT / "data" / "synthetic"))

from app.config import Config  # noqa: E402
from app.agent import AppContext  # noqa: E402
from app.confirmation import is_explicit_confirmation  # noqa: E402

UID = "U000001"  # north, android, preferred quality 4, marketing opt-in 0


class PickledRecommendationModel:
    def recommend(self, user_id, n=5):
        assert user_id
        return [("I0003", 0.70), ("I0001", 0.95), ("I0002", 0.80)][:n]


class PickledScoringModel:
    def predict(self, user_id, product_id):
        assert user_id
        return {"I0001": 0.9, "I0002": 0.7}.get(product_id, 0.1)


class FakeLLM:
    """Each script step: list of (tool_name, input) tuples -> tool_use, or a str -> final text."""
    active_model = "anthropic.fake"

    def __init__(self):
        self.script, self.calls, self.last_results = [], [], []

    def queue(self, *steps):
        self.script.extend(steps)

    def converse(self, system, messages, tools=None, max_tokens=None, model_id=None):
        self.calls.append({"system": system, "messages": messages})
        last = messages[-1]
        self.last_results = [b["toolResult"]["content"][0]["json"] for b in last["content"] if "toolResult" in b]
        if tools is None:  # summarisation call
            return {"output": {"message": {"role": "assistant", "content": [{"text": "SUMMARY: customer wants a phone"}]}},
                    "stopReason": "end_turn"}
        step = self.script.pop(0)
        if callable(step):
            step = step(self.last_results)
        if isinstance(step, str):
            return {"output": {"message": {"role": "assistant", "content": [{"text": step}]}}, "stopReason": "end_turn"}
        content = [{"toolUse": {"toolUseId": f"t{len(self.calls)}_{i}", "name": n, "input": inp}} for i, (n, inp) in enumerate(step)]
        return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": "tool_use"}


@pytest.fixture()
def ctx(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("RUNTIME_DIR", tmp)
    return AppContext(Config(), llm=FakeLLM())


def phone_under(ctx, uid, budget):
    return ctx.personalizer.search(uid, category="smartphone", max_price=budget, limit=10)["results"]


def test_voice_config_defaults_and_legacy_single_language(monkeypatch):
    monkeypatch.delenv("TRANSCRIBE_LANGUAGES", raising=False)
    monkeypatch.delenv("TRANSCRIBE_LANGUAGE", raising=False)
    monkeypatch.delenv("POLLY_VOICE_ID", raising=False)
    assert Config().transcribe_languages == ("en-US", "ar-SA")
    assert Config().polly_voice == "Hala"

    monkeypatch.setenv("TRANSCRIBE_LANGUAGE", "ar-SA")
    assert Config().transcribe_languages == ("ar-SA",)

    monkeypatch.setenv("TRANSCRIBE_LANGUAGES", "en-US,ar-SA,en-US")
    assert Config().transcribe_languages == ("en-US", "ar-SA")


# ---------------------------------------------------------------- data layer
def test_ml_labels_hidden(ctx):
    assert "relevant_items" in ctx.store.customers[UID]  # present in raw train data
    assert "relevant_items" not in ctx.profiles.get(UID)  # never reaches the agent
    ctx.llm.queue("Hello!")
    ctx.agent.chat(UID, "hi")
    assert "relevant_items" not in ctx.llm.calls[-1]["system"]


def test_search_respects_eligibility_budget_and_prices(ctx):
    res = ctx.personalizer.search(UID, category="phones", max_price=300, limit=50)
    assert res["results"], "synonym 'phones' should map to smartphone"
    for r in res["results"]:
        assert r["eligible"] and float(r["final_price"]) <= 300
        ok, code, _ = ctx.catalogue.eligibility(UID, r["product_id"])
        assert ok, code
    ios_only = [pid for pid, p in ctx.catalogue.products.items() if p["compatible_os"] == "ios"]
    assert not set(ios_only) & {r["product_id"] for r in ctx.personalizer.search(UID, limit=500)["results"]}


def test_recommender_and_interactions_loaded(ctx):
    assert ctx.interactions.available and ctx.insights(UID)["events"] > 0
    recs = ctx.personalizer.recommendations(UID, limit=5)["results"]
    assert recs and all(r["eligible"] for r in recs)


def test_recommender_loads_pickle_and_runs_model(ctx, tmp_path):
    from app.data_layer import Recommender

    output = tmp_path / "recommendation_model.pkl"
    with output.open("wb") as artifact:
        pickle.dump(PickledRecommendationModel(), artifact)
    recommender = Recommender(output, ctx.catalogue)
    assert recommender.for_user("U1") == ["I0001", "I0002", "I0003"]
    assert recommender.source == "recommendation_model.pkl"


def test_recommender_can_rank_catalogue_with_scoring_model(ctx, tmp_path):
    from app.data_layer import Recommender

    output = tmp_path / "scoring_model.pickle"
    with output.open("wb") as artifact:
        pickle.dump(PickledScoringModel(), artifact)
    recommender = Recommender(output, ctx.catalogue)
    assert recommender.for_user("U1")[:2] == ["I0001", "I0002"]


# ---------------------------------------------------------------- memory + references
def test_budget_change_and_rejection_flow(ctx):
    llm = ctx.llm
    first = [None]

    def present(results):
        ids = [r["product_id"] for r in results[0]["data"]["results"][:3]]
        first[0] = ids[0]
        return [("present_options", {"title": "Phones for you", "product_ids": ids})]

    llm.queue([("update_customer_memory", {"goal": "buy a new smartphone"}),
               ("search_products", {"category": "smartphone", "ignore_budget": True})],
              lambda res: present(res[1:]), "Here are three phones.")
    out = ctx.agent.chat(UID, "I would like to buy a new smartphone. What would suit me?")
    assert out["cards"] and len(out["cards"][0]["products"]) == 3
    session = ctx.memory.load_session(UID)
    assert session["presented"][-1]["options"][0]["product_id"] == first[0]

    # "That's too expensive. My budget is now 100, and I don't want the first option."
    llm.queue(lambda _: [("update_customer_memory", {"budget": 100, "rejected_product_ids": [first[0]],
                                                      "rejection_reason": "customer does not want option 1"})],
              [("search_products", {"category": "smartphone"})], "Updated picks.")
    ctx.agent.chat(UID, "That's too expensive. My budget is now 100, and I don't want the first option.")
    res = llm.last_results[0]["data"]
    assert res["applied_max_price"] == 100.0
    assert first[0] not in {r["product_id"] for r in res["results"]}
    assert all(float(r["final_price"]) <= 100 for r in res["results"])
    assert first[0] in ctx.memory.load_session(UID)["rejected"]


def test_customers_are_isolated(ctx):
    ctx.llm.queue([("update_customer_memory", {"budget": 50})], "ok")
    ctx.agent.chat(UID, "budget 50")
    other = "U000002"
    assert ctx.memory.load_session(other)["budget"] is None
    assert ctx.memory.load_conversation(other)["messages"] == []


# ---------------------------------------------------------------- checkout gate
def _eligible_non_subscription(ctx, n=2):
    res = ctx.personalizer.search(UID, domain="home", max_price=150, limit=20)["results"]
    return [r["product_id"] for r in res if not r["subscription"]][:n]


def test_checkout_requires_explicit_confirmation(ctx):
    llm = ctx.llm
    a, b = _eligible_non_subscription(ctx)
    llm.queue([("add_to_basket", {"product_id": a}), ("add_to_basket", {"product_id": b})],
              [("prepare_checkout", {})],
              [("place_order", {})],  # model tries to place in the SAME turn -> must be refused
              "Here's your summary. Shall I place the order?")
    out = ctx.agent.chat(UID, "Add both and check out")
    refused = [x for x in out["actions"] if x["tool"] == "place_order"][0]
    assert not refused["ok"] and "CONFIRMATION_REQUIRED" in refused["summary"]
    assert out["checkout"] and ctx.memory.load_session(UID)["pending_checkout"]

    # "Remove that item before placing the order." -> not a confirmation
    llm.queue([("place_order", {})], [("remove_from_basket", {"product_id": b})], [("prepare_checkout", {})],
              "Removed. New summary - confirm?")
    out = ctx.agent.chat(UID, "Remove that item before placing the order.")
    assert not out["actions"][0]["ok"]
    assert len(ctx.store.get_basket(UID)["items"]) == 1
    assert ctx.memory.load_session(UID)["pending_checkout"]["revision"] == ctx.store.get_basket(UID)["revision"]

    llm.queue([("place_order", {})], "Order placed.")
    out = ctx.agent.chat(UID, "Yes, place the order.")
    assert out["actions"][0]["ok"] and out["order"]["status"] == "SIMULATED_PURCHASE_COMPLETED"
    assert ctx.store.get_basket(UID)["items"] == []
    assert ctx.memory.load_long_term(UID)["orders"]


def test_basket_change_invalidates_summary(ctx):
    llm = ctx.llm
    a, b = _eligible_non_subscription(ctx)
    llm.queue([("add_to_basket", {"product_id": a})], [("prepare_checkout", {})], "Confirm?")
    ctx.agent.chat(UID, "add and checkout")
    llm.queue([("add_to_basket", {"product_id": b})], "Added.")
    out = ctx.agent.chat(UID, "add another thing")
    assert ctx.memory.load_session(UID)["pending_checkout"] is None
    llm.queue([("place_order", {})], "Need a new summary.")
    out = ctx.agent.chat(UID, "yes")
    assert out["actions"][0]["summary"].startswith("NO_PENDING_CHECKOUT")


def test_ui_confirm_button(ctx):
    llm = ctx.llm
    a = _eligible_non_subscription(ctx, 1)[0]
    llm.queue([("add_to_basket", {"product_id": a})], [("prepare_checkout", {})], "Confirm?")
    out = ctx.agent.chat(UID, "buy it")
    res = ctx.agent.confirm_via_ui(UID, "CHK-wrong")
    assert not res["ok"]
    res = ctx.agent.confirm_via_ui(UID, out["checkout"]["checkout_id"])
    assert res["ok"] and res["order"]["order_id"].startswith("ORD-")
    again = ctx.agent.confirm_via_ui(UID, out["checkout"]["checkout_id"])
    assert not again["ok"]  # summary consumed


def test_confirmation_classifier():
    yes = ["Yes, place the order.", "yes", "Confirm", "ok go ahead", "Yes please place my order", "sounds good",
           "that's fine", "let's do it"]
    no = ["Remove that item before placing the order.", "yes but remove the charger", "no", "wait",
          "is that the total?", "Yes, add the case too", "place the order without the speaker", "what's the total"]
    assert all(is_explicit_confirmation(t) for t in yes), [t for t in yes if not is_explicit_confirmation(t)]
    assert not any(is_explicit_confirmation(t) for t in no), [t for t in no if is_explicit_confirmation(t)]


def test_arabic_confirmation_classifier():
    yes = ["نعم", "أيوه", "ايوه.", "أكد الطلب", "تأكيد الشراء", "تمام، نفذ الطلب"]
    no = ["لا", "أيوه بس شيل الشاحن", "نعم، لكن غير الهاتف", "أكد الطلب بدون السماعة",
          "هل أؤكد الطلب؟", "كام الإجمالي؟", "أيوه؟", "تمام لو السعر أقل"]
    assert all(is_explicit_confirmation(t) for t in yes), [t for t in yes if not is_explicit_confirmation(t)]
    assert not any(is_explicit_confirmation(t) for t in no), [t for t in no if is_explicit_confirmation(t)]


def test_voice_language_metadata_guides_prompt_and_response(ctx):
    ctx.llm.queue("إليك هاتف مناسب.")
    out = ctx.agent.chat(UID, "عايز smartphone under 100", channel="voice",
                         voice_language="ar-SA", voice_languages=["ar-SA", "en-US"])
    system = ctx.llm.calls[-1]["system"]
    assert "Dominant input language: ar-SA" in system
    assert '"ar-SA","en-US"' in system
    assert out["language"] == "ar-SA"
    transcript = ctx.memory.load_conversation(UID)["transcript"]
    assert transcript[-2]["language"] == "ar-SA"
    assert transcript[-2]["languages"] == ["ar-SA", "en-US"]


def test_subscription_blocked_with_hint(ctx):
    sub = next(pid for pid, p in ctx.catalogue.products.items()
               if p["is_subscription"] == "1" and ctx.catalogue.eligibility(UID, pid)[0])
    ctx.llm.queue([("add_to_basket", {"product_id": sub})], [("prepare_checkout", {})], "Subscriptions can't be checked out.")
    out = ctx.agent.chat(UID, "add the plan and checkout")
    assert "SUBSCRIPTION_TERMS_MISSING" in out["actions"][1]["summary"]


def test_compaction_keeps_valid_history(ctx, monkeypatch):
    ctx.config.max_history_messages, ctx.config.keep_recent_messages = 8, 4
    for i in range(8):
        ctx.llm.queue([("get_basket", {})], f"reply {i}")
        ctx.agent.chat(UID, f"message {i}")
    conv = ctx.memory.load_conversation(UID)
    assert conv["summary"].startswith("SUMMARY")
    first = conv["messages"][0]
    assert first["role"] == "user" and "text" in first["content"][0]


def test_failure_after_successful_tool_reports_and_remembers_side_effect(ctx):
    """A later model failure must not claim an earlier basket mutation vanished."""
    item = _eligible_non_subscription(ctx, 1)[0]

    def model_failure(_results):
        raise RuntimeError("temporary Bedrock failure")

    ctx.llm.queue([("add_to_basket", {"product_id": item})], model_failure)
    out = ctx.agent.chat(UID, "add it")

    assert out["error"] == "temporary Bedrock failure"
    assert out["actions"][0]["ok"]
    assert "successful actions" in out["text"]
    assert ctx.store.get_basket(UID)["items"][0]["product_id"] == item

    conv = ctx.memory.load_conversation(UID)
    assert conv["turn"] == 1
    assert conv["messages"][-1]["role"] == "assistant"
    assert "AI service error" in conv["messages"][-1]["content"][0]["text"]


# ---------------------------------------------------------------- API
def test_api_invalid_user_and_flow(ctx):
    from fastapi.testclient import TestClient
    import app.main as main
    main.CTX = ctx
    from app.voice import Voice
    main.VOICE = Voice(ctx.config)
    client = TestClient(main.app)
    r = client.post("/api/session/start", json={"user_id": "NOPE"})
    assert r.status_code == 404 and "valid" in r.json()["detail"]
    ctx.llm.queue("Welcome back!")
    r = client.post("/api/session/start", json={"user_id": UID})
    assert r.status_code == 200 and r.json()["transcript"][-1]["text"] == "Welcome back!"
    assert "relevant_items" not in str(r.json())
    for path in (f"/api/customer/{UID}/home", f"/api/customer/{UID}/shop?category=kitchen", f"/api/customer/{UID}/rewards"):
        assert client.get(path).status_code == 200, path
    ctx.llm.queue("Sure.")
    r = client.post("/api/chat", json={"user_id": UID, "message": "hello", "channel": "voice"})
    assert r.json()["channel"] == "voice"
    assert "VOICE" in ctx.llm.calls[-1]["system"]
    cfg = client.get("/api/config").json()
    assert {x["code"] for x in cfg["voice_languages"]} == {"en-US", "ar-SA"}
    assert client.post("/api/chat", json={"user_id": UID, "message": "hi", "channel": "voice",
                                           "voice_language": "fr-FR"}).status_code == 422
    assert client.post("/api/voice/speak", json={"text": "hi", "language": "fr-FR"}).status_code == 422


def test_voice_api_provider_failures_offer_browser_fallback(ctx, monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main
    from app.voice import Voice

    main.CTX = ctx
    main.VOICE = Voice(ctx.config)
    client = TestClient(main.app)

    async def fail_transcribe(*_args, **_kwargs):
        raise RuntimeError("transcribe unavailable")

    def fail_speak(*_args, **_kwargs):
        raise RuntimeError("polly unavailable")

    monkeypatch.setattr(main.VOICE, "transcribe", fail_transcribe)
    r = client.post(f"/api/voice/transcribe?user_id={UID}&language=en-US",
                    content=b"\x00" * 4000, headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 502 and r.json()["fallback"] == "browser"

    monkeypatch.setattr(main.VOICE, "speak", fail_speak)
    r = client.post("/api/voice/speak", json={"text": "مرحبا", "language": "ar-SA"})
    assert r.status_code == 502 and r.json()["fallback"] == "browser" and r.json()["language"] == "ar-SA"
