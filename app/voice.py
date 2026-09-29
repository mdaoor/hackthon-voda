"""Voice channel.

Speech-to-text : Amazon Transcribe streaming (16 kHz, 16-bit mono PCM captured in the browser).
Text-to-speech : Amazon Polly (neural voice), mp3.
The transcript goes through exactly the same agent turn as typed text, so voice and
text share one conversation, one memory and one basket.
"""
from __future__ import annotations

import asyncio
from collections import Counter
import logging
import re
import time

import boto3

log = logging.getLogger("companion.voice")

try:
    from amazon_transcribe.client import TranscribeStreamingClient
    from amazon_transcribe.handlers import TranscriptResultStreamHandler
    from amazon_transcribe.model import TranscriptEvent
    TRANSCRIBE_AVAILABLE = True
except Exception:  # package missing -> browser speech recognition fallback
    TRANSCRIBE_AVAILABLE = False


SUPPORTED_LANGUAGES = ("en-US", "ar-SA")
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")


def text_language(text: str) -> str:
    """Classify reply text for UI/TTS routing; mixed text follows its dominant script."""
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if letters and sum(bool(ARABIC_RE.match(ch)) for ch in letters) >= len(letters) / 2:
        return "ar-SA"
    return "en-US"


def dominant_language(parts: list[tuple[str, str]], fallback: str | None = None):
    weights = Counter()
    ordered = []
    for transcript, language in parts:
        if not language:
            continue
        if language not in ordered:
            ordered.append(language)
        weights[language] += max(1, len(transcript.strip()))
    return (weights.most_common(1)[0][0] if weights else fallback), ordered


if TRANSCRIBE_AVAILABLE:
    class _Collector(TranscriptResultStreamHandler):
        def __init__(self, stream):
            super().__init__(stream)
            self.parts: list[tuple[str, str | None]] = []

        async def handle_transcript_event(self, transcript_event: TranscriptEvent):
            for result in transcript_event.transcript.results:
                if not result.is_partial and result.alternatives:
                    alternative = result.alternatives[0]
                    language = getattr(result, "language_code", None) or getattr(alternative, "language_code", None)
                    self.parts.append((alternative.transcript, language))


class Voice:
    def __init__(self, config):
        self.config = config
        self._polly = None

    @property
    def stt_mode(self):
        if self.config.stt_provider == "browser":
            return "browser"
        return "transcribe" if TRANSCRIBE_AVAILABLE else "browser"

    @property
    def languages(self):
        return tuple(self.config.transcribe_languages)

    @property
    def auto_language_available(self):
        return self.stt_mode == "transcribe" and len(self.languages) >= 2

    def validate_language(self, language: str) -> str:
        if language == "auto":
            if not self.auto_language_available:
                raise ValueError("Automatic language detection requires Amazon Transcribe and at least two configured languages.")
            return language
        if language not in self.languages or language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported voice language '{language}'. Choose one of: auto, {', '.join(self.languages)}.")
        return language

    def transcribe_parameters(self, language: str, sample_rate: int) -> dict:
        self.validate_language(language)
        base = {"media_sample_rate_hz": sample_rate, "media_encoding": "pcm"}
        if language == "auto":
            # The streaming SDK keeps language_code as a required keyword even
            # when AWS language identification is enabled.
            return {**base, "language_code": None, "identify_multiple_languages": True,
                    "language_options": list(self.languages)}
        return {**base, "language_code": language}

    async def transcribe(self, pcm: bytes, sample_rate: int = 16000, language: str = "auto") -> dict:
        if not TRANSCRIBE_AVAILABLE:
            raise RuntimeError("amazon-transcribe is not installed")
        self.validate_language(language)
        if len(pcm) < sample_rate // 5:  # < 0.1 s of audio
            return {"text": "", "language": None, "languages": []}
        started = time.perf_counter()
        client = TranscribeStreamingClient(region=self.config.aws_region)
        stream = await client.start_stream_transcription(**self.transcribe_parameters(language, sample_rate))
        handler = _Collector(stream.output_stream)
        chunk = 8192  # ~0.25 s at 16 kHz / 16-bit

        async def send():
            for i in range(0, len(pcm), chunk):
                await stream.input_stream.send_audio_event(audio_chunk=pcm[i:i + chunk])
                await asyncio.sleep(0.02)  # ~10x real time; keeps the stream well-behaved
            await stream.input_stream.end_stream()

        await asyncio.wait_for(asyncio.gather(send(), handler.handle_events()), timeout=45)
        populated = [(text.strip(), code) for text, code in handler.parts if text.strip()]
        detected, languages = dominant_language(populated, None if language == "auto" else language)
        if populated and detected and not languages:
            languages = [detected]
        log.info("voice stt provider=transcribe selected=%s detected=%s languages=%s latency_ms=%d",
                 language, detected, languages, round((time.perf_counter() - started) * 1000))
        return {"text": " ".join(text for text, _ in populated), "language": detected, "languages": languages}

    @property
    def polly(self):
        if self._polly is None:
            self._polly = boto3.client("polly", region_name=self.config.aws_region)
        return self._polly

    def speak(self, text: str, language: str | None = None) -> bytes:
        clean = speakable(text)
        selected = language or text_language(clean)
        if selected not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported voice language '{selected}'.")
        started = time.perf_counter()
        r = self.polly.synthesize_speech(Text=clean, OutputFormat="mp3", VoiceId=self.config.polly_voice,
                                         Engine=self.config.polly_engine)
        log.info("voice tts provider=polly language=%s voice=%s engine=%s latency_ms=%d",
                 selected, self.config.polly_voice, self.config.polly_engine,
                 round((time.perf_counter() - started) * 1000))
        return r["AudioStream"].read()


def speakable(text: str, limit: int = 900) -> str:
    """Strip markdown/IDs so TTS sounds natural; cut at a sentence boundary."""
    t = re.sub(r"\*\*|__|`|#+\s*", "", text or "")
    t = re.sub(r"^\s*[-*•]\s+", "", t, flags=re.M)
    t = re.sub(r"^\s*\d+[.)]\s+", "", t, flags=re.M)
    t = re.sub(r"\|", ", ", t)
    t = re.sub(r"\(?\b(I\d{3,}|ORD-[0-9a-f]{6,}|CHK-[0-9a-f]{6,})\b\)?", "", t)
    t = re.sub(r"DEMO_UNITS", "units", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > limit:
        cut = t[:limit]
        t = cut[: max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! ")) + 1] or cut
    return t or "Okay."
