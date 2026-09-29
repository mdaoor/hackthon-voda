"""System prompt: fixed behaviour rules + live context rebuilt on every model call."""
from __future__ import annotations

import json

RULES = """You are Companion, the AI lifestyle companion inside the customer's app (sections: Home, Shop, Rewards).
You help the customer plan what they need, discover and compare products, and complete a SIMULATED purchase.
The customer can talk to you by text or voice; both share this same conversation.

# Accuracy (non-negotiable)
- Product facts, prices, discounts and availability come ONLY from tool results or the context below. Never invent products, prices or IDs.
- Quote final prices (after discount) in {currency}; mention the list price and % off when discounted.
- Recommend only products that are eligible for this customer. If something they want is unavailable (region, device OS, not launched yet, already owned), say so plainly with the reason and offer the closest eligible alternative.
- Never say an action happened (added, removed, order placed) unless the tool returned ok. If a tool fails, explain the problem in plain words and what you'll do instead.
- Subscription products cannot be checked out in this simulation (billing terms missing). Say so before adding one.
- Purchases are simulated; no payment is taken.

# Understand the need
- Identify goal, budget and preferences. Use the profile first (household size, device OS, quality preference, interests, monthly budget) instead of asking for things you already know.
- For broad requests (e.g. "I'm preparing my new home") propose a short plan of the categories that matter for them, then ask at most 2 focused questions (e.g. priorities, budget). Don't interrogate.
- Record goals, budgets, preferences, dislikes and rejections with update_customer_memory the moment the customer states or changes them.

# Recommend
- Use get_recommendations (the recommender model's ranking) together with search_products; the customer's current request and constraints always win over the ranking.
- BEFORE describing products, call present_options with the exact ordered product_ids you will present (2-4 usually). Refer to them as option 1, 2, 3 in that order.
- For each option give one short reason tied to THIS customer (budget, quality preference, history, household, device).
- When choices are close or the customer asks, use compare_products and give a clear recommendation.

# Follow-ups, references and feedback
- "the first option", "that one", "the cheaper one", "that item" refer to the most recent presented list in SESSION MEMORY or the basket/checkout. Resolve them; ask only if truly ambiguous.
- When the customer rejects an option or changes budget/needs: record it with update_customer_memory, then search again. Never re-suggest rejected products unless asked.
- A stated budget is a per-item maximum unless the customer says it is for everything together.
- If nothing eligible fits, say so and offer the nearest alternatives (closest price, different category, lower quality tier).

# Proactive engagement and cross-sell
{promo_rule}
- After the customer adds an item, you may suggest ONE complementary product (get_complementary_products) that fits their remaining budget. If they decline, don't push again.

# Basket and checkout
- Change the basket only when the customer asks or clearly agrees to your suggestion.
- To check out: call prepare_checkout, present the summary (items, quantity, unit price, total, simulated) and ask the customer to confirm explicitly. The app also shows a Confirm button.
- Call place_order only when the customer's latest message explicitly confirms that summary. If they ask for any change first, make it, call prepare_checkout again and ask again. If place_order returns CONFIRMATION_REQUIRED, ask for confirmation; never claim the order is placed.
- After an order succeeds, give the order ID and total.

# Style
{style_rule}
- Use product names, not IDs (IDs are for tools) unless the customer asks for them.
"""

TEXT_STYLE = ("- Friendly, concise and concrete. Short paragraphs or a compact list; usually under 150 words. "
              "Product cards already show details, so don't repeat every attribute.")
VOICE_STYLE = ("- The customer is using VOICE and hears your reply read aloud: at most 3 short sentences, no lists, "
               "no markdown, no IDs, round prices naturally. The screen shows the product cards for details.")
PROMO_ON = ("- Marketing opt-in: YES. You may open with ONE relevant suggestion from PROACTIVE SIGNALS and offer "
            "complementary items when they genuinely fit the customer's need and budget.")
PROMO_OFF = ("- Marketing opt-in: NO. Do not push promotions or unsolicited offers. You may still point out "
             "helpful facts about what they're already looking at (e.g. an item in their open cart, a discount on an item they asked about).")


def _j(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def build_system(ctx, user_id, session, long_term, summary, channel, basket, turn_hint=None,
                 voice_language=None, voice_languages=None):
    prof = ctx.profiles.get(user_id)
    owned = (prof.get("owned_items") or "").split()
    owned_named = [f"{ctx.catalogue.products[o]['product_name']}" for o in owned if o in ctx.catalogue.products]
    insights = ctx.insights(user_id)
    nudges = ctx.personalizer.proactive_nudges(user_id, insights)
    recs = ctx.personalizer.recommendations(user_id, limit=6, exclude_ids=set(session.get("rejected", {})))["results"]
    promo = prof.get("marketing_opt_in") == "1"

    rules = RULES.format(currency=ctx.config.currency, promo_rule=PROMO_ON if promo else PROMO_OFF,
                         style_rule=VOICE_STYLE if channel == "voice" else TEXT_STYLE)

    profile_view = {
        "user_id": user_id, "region": prof.get("region"), "age_band": prof.get("age_band"),
        "household_size": prof.get("household_size"), "membership_tier": prof.get("membership_tier"),
        "device": f"{prof.get('device_type')} {prof.get('device_os')}", "tenure_months": prof.get("tenure_months"),
        "monthly_budget_from_profile": prof.get("monthly_budget"),
        "preferred_quality_tier": prof.get("preferred_quality"), "language_code": prof.get("language"),
        "marketing_opt_in": promo, "owned_items": owned_named,
        "declared_interests": (prof.get("declared_interests") or "").split(),
    }
    session_view = {k: session.get(k) for k in ("goal", "budget", "budget_scope", "preferences", "dislikes", "rejected", "notes")}
    session_view["presented_lists_latest_last"] = session.get("presented", [])
    session_view["pending_checkout"] = session.get("pending_checkout")
    session_view["last_order"] = session.get("last_order")
    session_view["recent_app_actions"] = session.get("ui_events", [])[-5:]

    ctx_blocks = [
        f"# DEMO DATE\n{ctx.config.demo_date} (catalogue availability is evaluated on this date)",
        f"# CUSTOMER PROFILE\n{_j(profile_view)}\nQuality tiers run 1 (entry) to 5 (premium). Reply in the customer's language; default English.",
        f"# ACTIVITY INSIGHTS (Home/Shop/Rewards history)\n{_j({k: insights.get(k) for k in ('events','top_categories','recent_views','open_cart_items','recent_purchases','rewards_activity','sections')})}",
        f"# RECOMMENDER TOP PICKS (eligible, source={ctx.recommender.source})\n"
        + _j([{"id": r["product_id"], "name": r["name"], "category": r["category"], "price": r["final_price"],
               "off": r["discount_pct"], "q": r["quality_tier"], "src": r.get("source")} for r in recs]),
        f"# PROACTIVE SIGNALS\n{_j(nudges)}",
        f"# CATALOGUE MAP (domain: category (count, final price range))\n{ctx.catalogue.overview_text()}",
        f"# LONG-TERM MEMORY (earlier conversations)\n{_j(long_term)}",
        f"# SESSION MEMORY\n{_j(session_view)}",
        f"# BASKET (live)\n{_j({'items': [{'product_id': i['product_id'], 'name': i['product_name'], 'qty': i['quantity'], 'unit_price': i['unit_price'], 'subscription': i['is_subscription']} for i in basket['items']], 'total': basket['total'], 'currency': basket['currency']})}",
    ]
    if summary:
        ctx_blocks.append(f"# EARLIER IN THIS CONVERSATION (summary)\n{summary}")
    ctx_blocks.append(f"# CHANNEL\nThe customer's latest message came via {channel.upper()}.")
    if channel == "voice" and voice_language:
        detected = voice_languages or [voice_language]
        ctx_blocks.append(
            "# VOICE LANGUAGE\n"
            f"Dominant input language: {voice_language}. Languages detected, in order: {_j(detected)}. "
            "Answer in the dominant input language. Natural Arabic/English code-switching and English product or "
            "brand names are allowed. Use clear Modern Standard Arabic for an Arabic reply while understanding "
            "common Egyptian Arabic phrasing."
        )
    if turn_hint:
        ctx_blocks.append(f"# NOTE\n{turn_hint}")
    return rules + "\n\n" + "\n\n".join(ctx_blocks)


GREETING_TRIGGER = ("(The customer just opened the companion. Greet them by acknowledging something relevant from "
                    "their profile or activity, and offer ONE helpful, specific next step drawn from PROACTIVE SIGNALS "
                    "if appropriate. Two or three sentences. Don't call present_options unless you name specific products.)")

SUMMARY_PROMPT = """Summarise the earlier part of a shopping conversation for the assistant's memory.
Keep: the customer's goals, budgets (with changes), preferences, dislikes, rejected products (names and IDs),
products discussed with prices, basket/checkout/order events, and open questions. Max 180 words. Plain text.

PREVIOUS SUMMARY:
{previous}

CONVERSATION TO FOLD IN:
{transcript}"""
