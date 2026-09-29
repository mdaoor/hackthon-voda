"""All runtime configuration comes from environment variables (see .env.example)."""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER_DIR = ROOT / "starter"
# The supplied starter's internal modules use flat imports. Keep its directory
# available for those internals while application code imports via `starter.*`.
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))


def _env(name, default=None):
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _find(data_dir: Path, *stems):
    """Return the first existing file matching any stem, with or without extension."""
    for stem in stems:
        for candidate in (data_dir / stem, *sorted(data_dir.glob(f"{stem}.*"))):
            if candidate.is_file():
                return candidate
    return None


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", ROOT / "data" / "synthetic")))
    products_path: Path | None = None
    customers_path: Path | None = None
    interactions_path: Path | None = None
    recommendation_model_path: Path | None = None
    runtime_dir: Path = field(default_factory=lambda: Path(_env("RUNTIME_DIR", ROOT / "runtime")))
    demo_date: str = field(default_factory=lambda: _env("DEMO_DATE", "2026-04-01"))
    currency: str = field(default_factory=lambda: _env("CURRENCY", "DEMO_UNITS"))

    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", _env("AWS_DEFAULT_REGION", "us-east-1")))
    bedrock_region: str | None = field(default_factory=lambda: _env("BEDROCK_REGION"))
    model_id: str = field(default_factory=lambda: _env("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"))
    fallback_model_id: str | None = field(default_factory=lambda: _env("BEDROCK_FALLBACK_MODEL_ID", "us.amazon.nova-pro-v1:0"))
    summary_model_id: str | None = field(default_factory=lambda: _env("SUMMARY_MODEL_ID"))
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.3")))
    max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "1500")))
    max_tool_steps: int = field(default_factory=lambda: int(_env("MAX_TOOL_STEPS", "10")))

    max_history_messages: int = field(default_factory=lambda: int(_env("MAX_HISTORY_MESSAGES", "40")))
    keep_recent_messages: int = field(default_factory=lambda: int(_env("KEEP_RECENT_MESSAGES", "16")))

    stt_provider: str = field(default_factory=lambda: _env("STT_PROVIDER", "auto"))  # auto|transcribe|browser
    # TRANSCRIBE_LANGUAGE remains a supported single-language compatibility
    # setting. TRANSCRIBE_LANGUAGES takes precedence when both are supplied.
    transcribe_language: str = field(default_factory=lambda: _env("TRANSCRIBE_LANGUAGE", ""))
    transcribe_languages: tuple[str, ...] = field(default_factory=tuple)
    polly_voice: str = field(default_factory=lambda: _env("POLLY_VOICE_ID", "Hala"))
    polly_engine: str = field(default_factory=lambda: _env("POLLY_ENGINE", "neural"))
    tts_provider: str = field(default_factory=lambda: _env("TTS_PROVIDER", "polly"))  # polly|browser

    show_sample_ids: bool = field(default_factory=lambda: _env("SHOW_SAMPLE_IDS", "true").lower() == "true")

    def __post_init__(self):
        configured_languages = _env("TRANSCRIBE_LANGUAGES")
        if configured_languages:
            languages = [part.strip() for part in configured_languages.split(",") if part.strip()]
        elif self.transcribe_language:
            languages = [self.transcribe_language]
        else:
            languages = ["en-US", "ar-SA"]
        self.transcribe_languages = tuple(dict.fromkeys(languages))
        self.transcribe_language = self.transcribe_language or self.transcribe_languages[0]
        d = self.data_dir
        self.products_path = Path(_env("PRODUCTS_PATH")) if _env("PRODUCTS_PATH") else _find(d, "products")
        self.customers_path = Path(_env("CUSTOMERS_PATH")) if _env("CUSTOMERS_PATH") else _find(d, "train", "customers")
        self.interactions_path = Path(_env("INTERACTIONS_PATH")) if _env("INTERACTIONS_PATH") else _find(d, "interactions")
        self.recommendation_model_path = (Path(_env("RECOMMENDATION_MODEL_PATH"))
                                          if _env("RECOMMENDATION_MODEL_PATH")
                                          else next((p for stem in ("recommendation_model", "recommender_model", "model")
                                                     for ext in (".pkl", ".pickle")
                                                     if (p := d / f"{stem}{ext}").is_file()), None))
        if not self.products_path or not self.customers_path:
            raise RuntimeError(f"products and train/customers files are required; looked in {d}")
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.bedrock_region = self.bedrock_region or self.aws_region

    @property
    def store_db(self):
        return self.runtime_dir / "simulation.sqlite"

    @property
    def memory_db(self):
        return self.runtime_dir / "companion.sqlite"
