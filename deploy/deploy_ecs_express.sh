#!/usr/bin/env bash
# Deploy the companion to Amazon ECS Express Mode -> public HTTPS URL (https://<name>.ecs.<region>.on.aws)
#
#   export AWS_REGION=us-east-1            # region of the provided hackathon account
#   export BEDROCK_MODEL_ID=...             # optional, see .env.example
#   ./deploy/deploy_ecs_express.sh
#
# Re-running builds a new image and updates the same service.
set -euo pipefail
cd "$(dirname "$0")/.."

APP="${APP_NAME:-lifestyle-companion}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
REPO="${REGISTRY}/${APP}"
TAG="$(date +%Y%m%d-%H%M%S)"
EXEC_ROLE="${EXEC_ROLE_NAME:-ecsTaskExecutionRole}"
INFRA_ROLE="${INFRA_ROLE_NAME:-ecsInfrastructureRoleForExpressServices}"
TASK_ROLE="${TASK_ROLE_NAME:-${APP}-task-role}"
ARN_FILE="deploy/.service-arn"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

retry_cmd() {  # retry_cmd command args...
  local attempt
  for attempt in 1 2 3 4 5; do
    if "$@"; then return 0; fi
    if [ "$attempt" -lt 5 ]; then
      printf 'Attempt %s failed; retrying in %ss...\n' "$attempt" "$((attempt * 2))" >&2
      sleep "$((attempt * 2))"
    fi
  done
  return 1
}

login_ecr() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if aws ecr get-login-password --region "$REGION" \
      | docker login --username AWS --password-stdin "$REGISTRY"; then
      return 0
    fi
    if [ "$attempt" -lt 5 ]; then
      printf 'ECR login attempt %s failed; retrying in %ss...\n' "$attempt" "$((attempt * 2))" >&2
      sleep "$((attempt * 2))"
    fi
  done
  return 1
}

# ---------------------------------------------------------------- image
say "ECR repository ${APP}"
aws ecr describe-repositories --repository-names "$APP" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$APP" --region "$REGION" >/dev/null
login_ecr

say "Build and push ${REPO}:${TAG} (linux/amd64)"
retry_cmd docker build --platform linux/amd64 -t "${REPO}:${TAG}" .
retry_cmd docker push "${REPO}:${TAG}"

# ---------------------------------------------------------------- IAM (skips anything that already exists)
ensure_role() {  # name trust-file managed-policy-arn|-
  if ! aws iam get-role --role-name "$1" >/dev/null 2>&1; then
    say "Create IAM role $1"
    aws iam create-role --role-name "$1" --assume-role-policy-document "file://$2" >/dev/null
    CREATED_ROLE=1
  fi
  [ "$3" != "-" ] && aws iam attach-role-policy --role-name "$1" --policy-arn "$3" 2>/dev/null || true
}
CREATED_ROLE=0
ensure_role "$EXEC_ROLE" deploy/iam/ecs-tasks-trust.json arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
ensure_role "$INFRA_ROLE" deploy/iam/ecs-infra-trust.json arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices
ensure_role "$TASK_ROLE" deploy/iam/ecs-tasks-trust.json -
aws iam put-role-policy --role-name "$TASK_ROLE" --policy-name companion-runtime \
  --policy-document file://deploy/iam/task-role-policy.json
if [ "$CREATED_ROLE" = "1" ]; then say "Waiting 20s for new IAM roles to propagate"; sleep 20; fi

EXEC_ARN="arn:aws:iam::${ACCOUNT}:role/${EXEC_ROLE}"
INFRA_ARN="arn:aws:iam::${ACCOUNT}:role/${INFRA_ROLE}"
TASK_ARN="arn:aws:iam::${ACCOUNT}:role/${TASK_ROLE}"

# ---------------------------------------------------------------- container definition
env_json() {
  local out="" first=1
  for v in AWS_REGION BEDROCK_REGION BEDROCK_MODEL_ID BEDROCK_FALLBACK_MODEL_ID SUMMARY_MODEL_ID DATA_DIR \
           PRODUCTS_PATH CUSTOMERS_PATH INTERACTIONS_PATH RECOMMENDATION_MODEL_PATH DEMO_DATE CURRENCY STT_PROVIDER \
           TRANSCRIBE_LANGUAGE TTS_PROVIDER POLLY_VOICE_ID POLLY_ENGINE SHOW_SAMPLE_IDS LLM_TEMPERATURE; do
    local val="${!v:-}"
    [ "$v" = "AWS_REGION" ] && val="$REGION"
    [ -z "$val" ] && continue
    [ $first = 1 ] && first=0 || out+=","
    out+="{\"name\":\"$v\",\"value\":\"$val\"}"
  done
  echo "[$out]"
}
CONTAINER="{\"image\":\"${REPO}:${TAG}\",\"containerPort\":8080,\"environment\":$(env_json)}"

# ---------------------------------------------------------------- create or update the Express service
SERVICE_ARN="$(cat "$ARN_FILE" 2>/dev/null || true)"
RECOVERED_SERVICE=0
# Recover when service creation succeeded but a previous local run failed
# before saving its ARN (for example, Windows Git Bash /tmp path differences).
if [ -z "$SERVICE_ARN" ]; then
  SERVICE_ARN="$(aws ecs list-services --cluster default --region "$REGION" \
    --query "serviceArns[?ends_with(@, '/${APP}')]|[0]" --output text 2>/dev/null || true)"
  [ "$SERVICE_ARN" = "None" ] && SERVICE_ARN=""
  if [ -n "$SERVICE_ARN" ]; then
    say "Recovered existing service ${SERVICE_ARN}"
    echo "$SERVICE_ARN" > "$ARN_FILE"
    RECOVERED_SERVICE=1
  fi
fi
if [ -n "$SERVICE_ARN" ] && aws ecs describe-express-gateway-service --service-arn "$SERVICE_ARN" --region "$REGION" >/dev/null 2>&1; then
  if [ "$RECOVERED_SERVICE" = "1" ]; then
    say "Monitor recovered service ${SERVICE_ARN}"
    aws ecs monitor-express-gateway-service --service-arn "$SERVICE_ARN" --region "$REGION" || true
  else
    say "Update service ${SERVICE_ARN}"
    # Prevent Git Bash from rewriting the Linux health path into C:/Program Files/Git/health.
    MSYS_NO_PATHCONV=1 aws ecs update-express-gateway-service --region "$REGION" --service-arn "$SERVICE_ARN" \
      --primary-container "$CONTAINER" --task-role-arn "$TASK_ARN" \
      --health-check-path "/health" --monitor-resources
  fi
else
  say "Create Express Mode service ${APP}"
  # ECS API values are CPU units and MiB: 1024 = 1 vCPU, 2048 = 2 GiB.
  # MSYS_NO_PATHCONV preserves /health when invoking the Windows AWS CLI from Git Bash.
  CREATE_JSON="$(MSYS_NO_PATHCONV=1 aws ecs create-express-gateway-service --region "$REGION" \
    --service-name "$APP" \
    --execution-role-arn "$EXEC_ARN" \
    --infrastructure-role-arn "$INFRA_ARN" \
    --task-role-arn "$TASK_ARN" \
    --primary-container "$CONTAINER" \
    --cpu 1024 --memory 2048 \
    --health-check-path "/health" \
    --scaling-target '{"minTaskCount":1,"maxTaskCount":1}' \
    --output json)"
  SERVICE_ARN="$(printf '%s' "$CREATE_JSON" | python3 -c \
    "import json,sys; print(json.load(sys.stdin)['service']['serviceArn'])")"
  echo "$SERVICE_ARN" > "$ARN_FILE"
  aws ecs monitor-express-gateway-service --service-arn "$SERVICE_ARN" --region "$REGION" || true
fi

say "Service"
aws ecs describe-express-gateway-service --service-arn "$SERVICE_ARN" --region "$REGION" --output json \
  | python3 -c "import json,sys,re; s=json.dumps(json.load(sys.stdin)); u=re.findall(r'(?:https://)?[a-z0-9.-]+\.ecs\.[a-z0-9-]+\.on\.aws/?', s); print('URL:', ('https://' + u[0].removeprefix('https://')) if u else 'see ECS console (Express Mode service)')"
echo "Service ARN: $SERVICE_ARN"
echo "Tip: open <URL>/api/status?check_llm=true to verify data files and Bedrock access."
