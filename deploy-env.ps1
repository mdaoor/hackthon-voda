# Reusable, non-secret environment configuration for AWS deployment.
#
# Usage from PowerShell:
#   . .\deploy-env.ps1
#   aws sts get-caller-identity
#   & "C:\Program Files\Git\bin\bash.exe" "./deploy/deploy_ecs_express.sh"
#
# Supply AWS authentication separately through AWS_PROFILE or fresh temporary
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN values.
# Never store credentials in this file.

$Env:APP_NAME = "lifestyle-companion-v2"
$Env:AWS_REGION = "us-east-1"
$Env:AWS_DEFAULT_REGION = "us-east-1"

# Amazon Bedrock
$Env:BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
$Env:BEDROCK_FALLBACK_MODEL_ID = "us.amazon.nova-pro-v1:0"
$Env:SUMMARY_MODEL_ID = "us.amazon.nova-lite-v1:0"
$Env:LLM_TEMPERATURE = "0.3"

# Voice
$Env:STT_PROVIDER = "auto"
$Env:TRANSCRIBE_LANGUAGES = "en-US,ar-SA"
$Env:TTS_PROVIDER = "polly"
$Env:POLLY_VOICE_ID = "Hala"
$Env:POLLY_ENGINE = "neural"

# Simulation and UI
$Env:DEMO_DATE = "2026-04-01"
$Env:CURRENCY = "DEMO_UNITS"
$Env:SHOW_SAMPLE_IDS = "true"

# The Docker image contains data under /srv/data and sets RUNTIME_DIR itself.
# Remove host-specific overrides so Windows paths are never passed to Linux ECS.
Remove-Item Env:DATA_DIR -ErrorAction SilentlyContinue
Remove-Item Env:PRODUCTS_PATH -ErrorAction SilentlyContinue
Remove-Item Env:CUSTOMERS_PATH -ErrorAction SilentlyContinue
Remove-Item Env:INTERACTIONS_PATH -ErrorAction SilentlyContinue
Remove-Item Env:RECOMMENDATIONS_PATH -ErrorAction SilentlyContinue
Remove-Item Env:RUNTIME_DIR -ErrorAction SilentlyContinue

Write-Host "Deployment environment loaded for $Env:APP_NAME in $Env:AWS_REGION."
Write-Host "AWS credentials were not changed; verify them with: aws sts get-caller-identity"
