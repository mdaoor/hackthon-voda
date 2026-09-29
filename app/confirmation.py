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

ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")
ARABIC_AFFIRM = re.compile(
    r"(?:^|\s)(?:نعم|ايوه|ايوا|اه|تمام|موافق|اوافق|اكد(?:\s+(?:الطلب|الشراء))?|"
    r"تاكيد(?:\s+(?:الطلب|الشراء))?|نفذ(?:\s+الطلب)?|اطلبه|اطلبهم)(?:\s|$)"
)
ARABIC_REJECT = re.compile(
    r"(?:^|\s)(?:لا|لأ|مش|مو|انتظر|استنى|توقف|الغي|الغاء|احذف|شيل|غير|بدل|استبدل|"
    r"اضف|زود|قبل|لكن|بس|الا|بدون|من غير|لو|اذا|ارخص|اغلى|اكثر|اقل|كام|كم|هل|"
    r"ليه|لماذا|ازاي|كيف|ايه|ما هو|ما هي|فين|اين|امتى|متى)(?:\s|$)|[؟?]"
)


def _normalise_arabic(text: str) -> str:
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.translate(str.maketrans("أإآٱؤئىة", "ااااوييه"))
    text = re.sub(r"ـ+", "", text)
    text = re.sub(r"[^\w\s؟?']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_explicit_confirmation(text: str) -> bool:
    if not text:
        return False
    t = _normalise_arabic(text.strip().lower())
    if len(t.split()) > 20:
        return False
    affirmed = bool(AFFIRM.search(t) or ARABIC_AFFIRM.search(t))
    rejected = bool(NEGATE.search(t) or ARABIC_REJECT.search(t))
    return affirmed and not rejected
