# Demo script (screen recording)

The organisers share the real test scenarios 30 minutes before the deadline. Rehearse this flow now so you only need to swap in their wording on the day. Before recording, dry-run any scenario list against the live URL:

```bash
python scripts/run_scenarios.py --url https://<service>.ecs.<region>.on.aws --user <ID> --fresh
python scripts/run_scenarios.py --url ... --user <ID> --scenario "their message 1" --scenario "their message 2"
```

## Recording checklist

- Deploy the final version first; a redeploy wipes state.
- Use Chrome, allow the microphone, and tick **Read replies aloud** for the voice part.
- Pick a customer with rich history and marketing opt-in = yes to show proactive offers. Then briefly switch to a second customer to show that contexts are separate.
- Expand **"N actions taken"** under a reply at least once so the judges see real tool calls.

## Suggested flow (about 5 minutes)

| # | Do / say | What it shows |
|---|---|---|
| 1 | Enter a wrong ID, then a valid one | Customer identification and validation |
| 2 | Read the greeting and point at Home → **Worth a look** | Proactive engagement from interactions (open cart, offers) |
| 3 | "I'm preparing my new home. Can you help me choose what I need?" | Plan by category plus at most 2 follow-up questions, using household size and budget |
| 4 | Answer the question, e.g. "Mainly kitchen and lighting, around 150 per item" | Goal and budget saved (Home → **What the companion remembers**) |
| 5 | **Voice:** "I would like to buy a new smartphone. What would suit me?" | Same conversation over voice. Cards with reasons (recommender rank, quality fit, device OS) |
| 6 | **Voice:** "That's too expensive. My budget is now 100, and I don't want the first option." | Reference to option 1 resolved, rejection plus new budget remembered, new search. If nothing fits, it says so and offers the nearest alternatives. |
| 7 | "Find something in this category within my budget." | Carries the category and budget from context |
| 8 | Click **Add to basket** on a card | Supplied `add_to_basket` runs, followed by one budget-aware complementary suggestion |
| 9 | "Add that too, then check out." | `prepare_checkout` produces a summary card with a **Confirm order** button |
| 10 | "Remove that item before placing the order." | Not treated as a confirmation. The item is removed, the old summary is voided and a new summary is shown. |
| 11 | "Yes, place the order." (or press **Confirm order**) | Order created only after explicit confirmation. Order card, Rewards → Orders. |
| 12 | Ask about something ineligible, e.g. an iOS-only item for an Android customer | Clear unavailability reason plus an alternative |
| 13 | **Change customer** to another ID, then back | Separate baskets and memory. The original context is restored. |

## If something goes wrong live

- **AI service error:** the reply says nothing was changed and the basket is kept. Just resend.
- **Transcription fails:** the UI switches to browser speech recognition; tap the mic again.
- **A subscription blocks checkout:** that is the starter's rule (`SUBSCRIPTION_TERMS_MISSING`). The companion explains it and offers to remove the subscription.
