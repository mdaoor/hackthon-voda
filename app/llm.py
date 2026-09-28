"""Amazon Bedrock Converse wrapper with automatic fallback to a second model."""
from __future__ import annotations

import logging

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

log = logging.getLogger("companion.llm")
FALLBACK_CODES = {"AccessDeniedException", "ResourceNotFoundException", "ValidationException",
                  "ModelNotReadyException", "ServiceUnavailableException", "ModelErrorException"}


class BedrockLLM:
    def __init__(self, config, client=None):
        self.config = config
        self.client = client or boto3.client(
            "bedrock-runtime", region_name=config.bedrock_region,
            config=BotoConfig(read_timeout=120, connect_timeout=10, retries={"max_attempts": 4, "mode": "adaptive"}))
        self.active_model = config.model_id

    def converse(self, system: str, messages: list, tools: list | None = None, max_tokens=None, model_id=None):
        kwargs = {
            "system": [{"text": system}],
            "messages": messages,
            "inferenceConfig": {"maxTokens": max_tokens or self.config.max_tokens, "temperature": self.config.temperature},
        }
        if tools:
            kwargs["toolConfig"] = {"tools": tools}
        model = model_id or self.active_model
        try:
            return self.client.converse(modelId=model, **kwargs)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            fb = self.config.fallback_model_id
            msg = str(e).lower()
            model_problem = code != "ValidationException" or any(
                k in msg for k in ("model identifier", "model id", "inference profile", "throughput",
                                   "isn't supported", "not supported", "access to the model"))
            if model_id is None and fb and fb != model and code in FALLBACK_CODES and model_problem:
                log.warning("Model %s failed with %s (%s); switching to fallback %s", model, code, e, fb)
                resp = self.client.converse(modelId=fb, **kwargs)
                self.active_model = fb  # stick with the working model
                return resp
            raise

    def health(self):
        try:
            r = self.converse("Reply with the single word OK.", [{"role": "user", "content": [{"text": "ping"}]}], max_tokens=5)
            text = "".join(b.get("text", "") for b in r["output"]["message"]["content"])
            return {"ok": True, "model": self.active_model, "reply": text.strip()}
        except Exception as e:
            return {"ok": False, "model": self.active_model, "error": str(e)[:300]}
