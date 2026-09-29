"""Pre-flight check for the hackathon AWS account.

    python scripts/check_bedrock.py                 # uses AWS_REGION / BEDROCK_MODEL_ID from env
    python scripts/check_bedrock.py --region eu-west-1

Lists usable Claude / Nova inference profiles, test-calls the configured model and
fallback with a tool definition, and checks Polly + Transcribe access.
"""
import argparse
import os
import sys

import boto3

p = argparse.ArgumentParser()
p.add_argument("--region", default=os.getenv("BEDROCK_REGION") or os.getenv("AWS_REGION") or "us-east-1")
args = p.parse_args()
region = args.region
models = [m for m in (os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
                      os.getenv("BEDROCK_FALLBACK_MODEL_ID", "us.amazon.nova-pro-v1:0")) if m]

print(f"Region: {region}\nIdentity: {boto3.client('sts').get_caller_identity()['Arn']}\n")
bedrock = boto3.client("bedrock", region_name=region)
try:
    profiles = bedrock.list_inference_profiles(maxResults=200)["inferenceProfileSummaries"]
    rel = [x["inferenceProfileId"] for x in profiles if any(k in x["inferenceProfileId"] for k in ("anthropic", "nova"))]
    print("Inference profiles (Claude/Nova):")
    for pid in sorted(rel):
        print("  ", pid)
except Exception as e:
    print("Could not list inference profiles:", e)

rt = boto3.client("bedrock-runtime", region_name=region)
tool = {"toolSpec": {"name": "get_basket", "description": "Get the basket.",
                     "inputSchema": {"json": {"type": "object", "properties": {}}}}}
ok_any = False
for m in models:
    try:
        r = rt.converse(modelId=m, messages=[{"role": "user", "content": [{"text": "Say OK."}]}],
                        toolConfig={"tools": [tool]}, inferenceConfig={"maxTokens": 20})
        text = "".join(b.get("text", "") for b in r["output"]["message"]["content"])
        print(f"\n[OK]   {m}: {text.strip()!r}")
        ok_any = True
    except Exception as e:
        print(f"\n[FAIL] {m}: {e}")

try:
    boto3.client("polly", region_name=region).synthesize_speech(Text="Hello", OutputFormat="mp3",
                                                                VoiceId=os.getenv("POLLY_VOICE_ID", "Hala"),
                                                                Engine=os.getenv("POLLY_ENGINE", "neural"))
    print("[OK]   Polly")
except Exception as e:
    print("[FAIL] Polly:", e)

try:
    import asyncio
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.config import Config
    from app.voice import Voice
    os.environ["AWS_REGION"] = region
    result = asyncio.run(Voice(Config()).transcribe(b"\x00\x00" * 16000, language="auto"))
    print(f"[OK]   Transcribe streaming auto en-US/ar-SA (silence -> {result!r})")
except Exception as e:
    print("[FAIL] Transcribe streaming:", e, "(the UI falls back to browser speech recognition)")

sys.exit(0 if ok_any else 1)
