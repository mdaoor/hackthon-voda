"""Voice channel.

Speech-to-text : Amazon Transcribe streaming (16 kHz, 16-bit mono PCM captured in the browser).
Text-to-speech : Amazon Polly (neural voice), mp3.
The transcript goes through exactly the same agent turn as typed text, so voice and
text share one conversation, one memory and one basket.
"""
from __future__ import annotations

import asyncio
import logging
import re

import boto3

log = logging.getLogger("companion.voice")

try:
    from amazon_transcribe.client import TranscribeStreamingClient
    from amazon_transcribe.handlers import TranscriptResultStreamHandler
    from amazon_transcribe.model import TranscriptEvent
    TRANSCRIBE_AVAILABLE = True
except Exception:  # package missing -> browser speech recognition fallback
    TRANSCRIBE_AVAILABLE = False


if TRANSCRIBE_AVAILABLE:
    class _Collector(TranscriptResultStreamHandler):
        def __init__(self, stream):
            super().__init__(stream)
            self.parts = []

        async def handle_transcript_event(self, transcript_event: TranscriptEvent):
            for result in transcript_event.transcript.results:
                if not result.is_partial and result.alternatives:
                    self.parts.append(result.alternatives[0].transcript)


class Voice:
    def __init__(self, config):
        self.config = config
        self._polly = None

    @property
    def stt_mode(self):
        if self.config.stt_provider == "browser":
            return "browser"
        return "transcribe" if TRANSCRIBE_AVAILABLE else "browser"

    async def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> str:
        if not TRANSCRIBE_AVAILABLE:
            raise RuntimeError("amazon-transcribe is not installed")
        if len(pcm) < sample_rate // 5:  # < 0.1 s of audio
            return ""
        client = TranscribeStreamingClient(region=self.config.aws_region)
        stream = await client.start_stream_transcription(
            language_code=self.config.transcribe_language, media_sample_rate_hz=sample_rate, media_encoding="pcm")
        handler = _Collector(stream.output_stream)
        chunk = 8192  # ~0.25 s at 16 kHz / 16-bit

        async def send():
            for i in range(0, len(pcm), chunk):
                await stream.input_stream.send_audio_event(audio_chunk=pcm[i:i + chunk])
                await asyncio.sleep(0.02)  # ~10x real time; keeps the stream well-behaved
            await stream.input_stream.end_stream()

        await asyncio.wait_for(asyncio.gather(send(), handler.handle_events()), timeout=45)
        return " ".join(p.strip() for p in handler.parts if p.strip())

    @property
    def polly(self):
        if self._polly is None:
            self._polly = boto3.client("polly", region_name=self.config.aws_region)
        return self._polly

    def speak(self, text: str) -> bytes:
        clean = speakable(text)
        try:
            r = self.polly.synthesize_speech(Text=clean, OutputFormat="mp3", VoiceId=self.config.polly_voice,
                                             Engine=self.config.polly_engine)
        except Exception as exc:
            log.warning("Polly %s engine failed (%s); retrying with standard", self.config.polly_engine, exc)
            r = self.polly.synthesize_speech(Text=clean, OutputFormat="mp3", VoiceId=self.config.polly_voice,
                                             Engine="standard")
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
