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

# ---------------------------------------------------------------- image
say "ECR repository ${APP}"
aws ecr describe-repositories --repository-names "$APP" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$APP" --region "$REGION" >/dev/null
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

say "Build and push ${REPO}:${TAG} (linux/amd64)"
docker build --platform linux/amd64 -t "${REPO}:${TAG}" .
docker push "${REPO}:${TAG}"

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
           PRODUCTS_PATH CUSTOMERS_PATH INTERACTIONS_PATH RECOMMENDATIONS_PATH DEMO_DATE CURRENCY STT_PROVIDER \
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
if [ -n "$SERVICE_ARN" ] && aws ecs describe-express-gateway-service --service-arn "$SERVICE_ARN" --region "$REGION" >/dev/null 2>&1; then
  say "Update service ${SERVICE_ARN}"
  aws ecs update-express-gateway-service --region "$REGION" --service-arn "$SERVICE_ARN" \
    --primary-container "$CONTAINER" --task-role-arn "$TASK_ARN" --monitor-resources
else
  say "Create Express Mode service ${APP}"
  aws ecs create-express-gateway-service --region "$REGION" \
    --service-name "$APP" \
    --execution-role-arn "$EXEC_ARN" \
    --infrastructure-role-arn "$INFRA_ARN" \
    --task-role-arn "$TASK_ARN" \
    --primary-container "$CONTAINER" \
    --cpu 1 --memory 2 \
    --health-check-path "/health" \
    --scaling-target '{"minTaskCount":1,"maxTaskCount":1}' \
    --output json > /tmp/${APP}-create.json
  SERVICE_ARN="$(python3 -c "import json;print(json.load(open('/tmp/${APP}-create.json'))['service']['serviceArn'])")"
  echo "$SERVICE_ARN" > "$ARN_FILE"
  aws ecs monitor-express-gateway-service --service-arn "$SERVICE_ARN" --region "$REGION" || true
fi

say "Service"
aws ecs describe-express-gateway-service --service-arn "$SERVICE_ARN" --region "$REGION" --output json \
  | python3 -c "import json,sys,re; s=json.dumps(json.load(sys.stdin)); u=re.findall(r'https://[a-z0-9.-]+\.on\.aws/?', s); print('URL:', u[0] if u else 'see ECS console (Express Mode service)')"
echo "Service ARN: $SERVICE_ARN"
echo "Tip: open <URL>/api/status?check_llm=true to verify data files and Bedrock access."
