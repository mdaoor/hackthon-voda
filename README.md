# AI Lifestyle Companion

An agentic companion for the Vodafone *AI Lifestyle Companion* challenge. It works by voice and text and helps a customer plan what they need, discover and compare products, get proactive and cross-sell suggestions, and complete a **simulated purchase** with the supplied starter functions. It runs on AWS: Amazon Bedrock, Transcribe, Polly and ECS Express Mode.

| Deliverable | Where |
|---|---|
| Agent design diagram | `docs/agent-design.svg` / `.png` (source `docs/agent-design.mmd`); explained in `docs/ARCHITECTURE.md` |
| Source code and run instructions | This repo; see Quick start and `docs/DEPLOY_AWS.md` |
| HTTPS chatbot link | Output of `./deploy/deploy_ecs_express.sh` |
| Screen recording | Follow `docs/DEMO_SCRIPT.md` |

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env && set -a && . ./.env && set +a    # adjust AWS_REGION / model IDs
python scripts/check_bedrock.py                          # confirms model, Polly, Transcribe access
uvicorn app.main:app --port 8080                         # open http://localhost:8080
```

It runs on the synthetic dataset in `data/synthetic/`, which has the challenge's column layout, until you point `DATA_DIR` at the real files (see `docs/DEPLOY_AWS.md` §3). Regenerate the synthetic set with `python scripts/generate_synthetic_data.py`.

**Tests** use a scripted fake model and need no AWS access:

```bash
pytest -q                                   # agent loop, tools, memory, confirmation gate, API
cd starter && python -m unittest -v         # the organisers' starter tests, unmodified
```

## Deploy

```bash
export AWS_REGION=<hackathon region>
./deploy/deploy_ecs_express.sh              # prints https://<service>.ecs.<region>.on.aws
```

## What's inside

```
app/
  main.py          FastAPI: session (user_id validation), chat, voice, checkout confirm, Home/Shop/Rewards/Basket APIs
  agent.py         Bedrock Converse tool loop, per-customer locking, memory compaction, trusted confirm path
  tools.py         Tool specs and executor; wraps the starter's call_tool; gated place_order
  confirmation.py  Deterministic "is this an explicit confirmation?" check on the customer's own words
  prompts.py       Behaviour rules plus live context (profile, insights, recs, memory, basket)
  data_layer.py    Catalogue, eligibility, profiles (drops relevant_items), interactions insights,
                   recommender loader (several layouts), personalised scoring, cross-sell, proactive nudges
  memory.py        SQLite: conversation + summary, session memory, long-term memory, audit log
  llm.py           Bedrock client with automatic model fallback
  voice.py         Amazon Transcribe streaming (STT) and Amazon Polly (TTS)
web/               Single-page UI, no build step: ID entry, chat, cards, checkout, voice, side panel
starter/           Organisers' AI_Companion_Starter, unmodified
data/synthetic/    Test data in the challenge format
deploy/            ECS Express Mode deploy script and IAM policies
scripts/           check_bedrock.py, run_scenarios.py, generate_synthetic_data.py
docs/              ARCHITECTURE.md, DEPLOY_AWS.md, DEMO_SCRIPT.md, diagram
```

## Challenge rules and how they're met

- **Supplied functions:** used unmodified through `tool_adapter.call_tool`. The `user_id` comes from the session, never from the model.
- **`confirm_order` is not model-visible.** The app supplies `customer_confirmed=True` only when one of these happens:
  - the customer presses **Confirm order**, or
  - the customer's own latest message unconditionally confirms a summary shown in an earlier turn.
- **Basket changes after a summary void it,** so a fresh summary and confirmation are required.
- **The first page is a `user_id` box.** Invalid IDs are asked for again.
- **`relevant_items` (ML target labels)** is removed from profiles before anything reaches the model.
- **Eligibility, prices and discounts** come from the starter's `Store`. That covers region, device OS, launch date on `DEMO_DATE`, ownership and repeatability. Unavailable items are explained with the reason.
- **Subscriptions** can't be checked out yet (starter rule). The companion says so and offers to remove them.
- **Voice and text share one conversation, memory and basket.** Switching customers loads a separate context.

## Configuration

All settings are environment variables; see `.env.example`. The most important:

| Variable | Default | Notes |
|---|---|---|
| `DATA_DIR` / `*_PATH` | `data/synthetic` | Real challenge files |
| `BEDROCK_MODEL_ID` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Use a profile available in your region |
| `BEDROCK_FALLBACK_MODEL_ID` | `us.amazon.nova-pro-v1:0` | Used automatically if the primary is unavailable |
| `STT_PROVIDER` / `TTS_PROVIDER` | `auto` / `polly` | `browser` for either forces the browser's speech features |
| `DEMO_DATE` | `2026-04-01` | Availability date used by the starter |
| `SHOW_SAMPLE_IDS` | `true` | Shows clickable demo IDs on the first page |

## Known limits

- State is SQLite on the task's disk: one task, and it resets on redeploy. For persistence across deploys, swap `memory.py` and the starter DB path onto EFS, or move memory to DynamoDB.
- Voice is push-to-talk (Transcribe, then the agent, then Polly), not full-duplex speech-to-speech. A Nova Sonic front end could be added later without changing the tools or memory.
- There is no authentication beyond the demo `user_id`, as the brief specifies.
