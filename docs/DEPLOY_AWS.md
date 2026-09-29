# Deploy to the provided AWS account

The target is **Amazon ECS Express Mode**. One command builds the container, creates the IAM roles and gives you a public HTTPS URL of the form `https://<service>.ecs.<region>.on.aws`. That URL is your chatbot link deliverable.

> AWS App Runner is closed to new customers since 30 April 2026, so it is not used here.

## 1. Prerequisites (on your laptop or AWS CloudShell)

- AWS CLI v2, recent enough to have `aws ecs create-express-gateway-service`. Run `aws --version` and update if the command is missing.
- Docker, with buildx for `--platform linux/amd64`. CloudShell has no Docker; use a laptop or a small EC2 build box.
- Credentials for the hackathon account, with permission to use ECR, ECS, IAM (create roles, `PassRole`), Bedrock, Polly and Transcribe.
- A default VPC with public subnets in the region. Express Mode uses it by default.

## 2. Check model access first (2 minutes)

```bash
export AWS_REGION=us-east-1          # the hackathon account's region
pip install boto3 amazon-transcribe
python scripts/check_bedrock.py
```

The script lists the Claude and Nova inference profiles you can use and test-calls the configured primary and fallback models with a tool definition. It also checks Polly and Transcribe. If the default model fails, pick a working ID from the list:

```bash
export BEDROCK_MODEL_ID=eu.anthropic.claude-sonnet-4-5-20250929-v1:0   # example for an EU region
export BEDROCK_FALLBACK_MODEL_ID=eu.amazon.nova-pro-v1:0
```

Inference-profile prefixes follow the region family: `us.`, `eu.`, `apac.`, or `global.` where offered. If no model works, enable model access in the Bedrock console for that region.

## 3. Put the challenge data in place

Copy the organisers' files into `data/challenge/` and point the app at them:

```bash
mkdir -p data/challenge && cp /path/to/{train,products,interactions}* /path/to/recommendation_model.pkl data/challenge/
export DATA_DIR=data/challenge
# only if the model file name differs from recommendation_model.pkl:
export RECOMMENDATION_MODEL_PATH=data/challenge/my_model.pkl
```

- **Delimiters** are auto-detected (`;`, `,`, tab, `|`). The starter `Store` requires semicolons for `products` and `train`, and so do the organiser files.
- **Interactions columns** are matched by alias. `user_id`/`customer_id`, `product_id`, `app_section`, `event_type`, `event_date`, durations, depth and transaction value are all handled.
- **Recommendation model** must be a trusted `.pkl`/`.pickle` object exposing `recommend(user_id, n=5)` or `predict(user_id, product_id)`.
- **`relevant_items`** in `train` is dropped before anything reaches the agent.
- If `DEMO_DATE` or `CURRENCY` is specified by the organisers, export those too.

Check locally before deploying:

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8080
curl localhost:8080/api/status        # counts, detected columns, recommender source
```

## 4. Deploy

```bash
./deploy/deploy_ecs_express.sh
```

The script does the following:

1. Creates the ECR repo `lifestyle-companion`, then builds and pushes a `linux/amd64` image.
2. Creates any missing IAM roles:
   - `ecsTaskExecutionRole`
   - `ecsInfrastructureRoleForExpressServices`
   - `lifestyle-companion-task-role`, with an inline policy for `bedrock:InvokeModel`, `polly:SynthesizeSpeech` and `transcribe:StartStreamTranscription` (see `deploy/iam/task-role-policy.json`)
3. Creates the Express Mode service with 1 vCPU, 2 GB, health check `/health` and exactly one task. The state lives in SQLite inside the task, so it must not scale out.
4. Prints the HTTPS URL and saves the service ARN in `deploy/.service-arn`. Re-running the script updates the same service.

These environment variables are forwarded into the container when set: `BEDROCK_*`, `SUMMARY_MODEL_ID`, `DATA_DIR`, `*_PATH`, `DEMO_DATE`, `CURRENCY`, voice settings, `SHOW_SAMPLE_IDS`.

**Verify:** open `https://<url>/api/status?check_llm=true`. You should see the data counts and `"llm": {"ok": true}`. Then open `https://<url>/`.

## 5. Things to know

- **State resets on redeploy.** Baskets, orders and memory live on the task's disk. This keeps the demo simple and fast, but record the screen video *after* your final deploy.
- **The mic needs HTTPS.** The Express URL provides it. On `http://localhost` browsers also allow the mic.
- **Voice fallbacks.** If Transcribe streaming is blocked in the account, the UI switches to the browser's speech recognition (Chrome or Edge). If Polly fails, the browser voice reads the reply.
- **Latency.** A turn with several tool calls takes about 5-15 s on Claude Sonnet. For snappier voice demos set `SUMMARY_MODEL_ID` to a Nova Lite profile; it only affects memory summaries.
- **Logs** are in CloudWatch Logs, in the log group the Express service created. Each turn logs its tools, duration and any error.
- **If an IAM role was only just created** and the first create call fails with "Unable to assume the service linked role", wait a minute and re-run.

## Fallback if Express Mode isn't available in the account

Run the same container anywhere with an HTTPS front door:

- **EC2 + CloudFront.** Start an EC2 instance with an instance profile that has the task-role policy, then run `docker run -d -p 80:8080 -e AWS_REGION=... <image>`. Create a CloudFront distribution with the instance's public DNS as an HTTP origin, caching disabled and all HTTP methods allowed. The `https://xxxx.cloudfront.net` domain is your link.
- **Existing App Runner** (only if the account used App Runner before April 2026): source image from ECR, port 8080, health check `/health`, and the task role as the instance role.
