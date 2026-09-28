"""Application-side confirmation gate.

The starter requires that `customer_confirmed=True` is supplied by the application
only after the customer confirms, never from a model-generated boolean. We accept
two sources of confirmation:

1. The UI "Confirm order" button (POST /api/checkout/confirm) - fully model-free.
2. A conversational turn whose *customer text* is an explicit, unconditional
   confirmation ("yes, place the order"). This is decided here with deterministic
   rules on the customer's own words, not by the LLM. Anything conditional or
   asking for a change ("remove that item before placing the order", "yes but...")
   is NOT a confirmation.
"""
import re

AFFIRM = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|confirm(ed)?|approve[d]?|go ahead|proceed|do it|"
    r"place (the |my )?order|place it|complete (the |my )?(order|purchase)|buy (it|them|now)|"
    r"order (it|them|now)|check ?out now|sounds good|that'?s (fine|right|correct)|let'?s do it|i agree)\b",
    re.I,
)
NEGATE = re.compile(
    r"\b(no|nope|not|don'?t|do not|never|wait|hold( on)?|stop|cancel|remove|delete|drop|change|swap|"
    r"replace|instead|add|also|before|but|except|without|actually|rather|cheaper|another|other|"
    r"less|more|update|modify|what|which|how|why|\?)\b|\?",
    re.I,
)


def is_explicit_confirmation(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if len(t.split()) > 20:
        return False
    return bool(AFFIRM.search(t)) and not NEGATE.search(t)
