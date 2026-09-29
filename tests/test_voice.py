import asyncio
import io
from types import SimpleNamespace

import pytest

from app.voice import Voice, dominant_language, speakable, text_language


def config(**overrides):
    values = dict(stt_provider="transcribe", transcribe_languages=("en-US", "ar-SA"),
                  aws_region="us-east-1", polly_voice="Hala", polly_engine="neural")
    values.update(overrides)
    return SimpleNamespace(**values)


def test_transcribe_parameters_auto_and_fixed_modes():
    voice = Voice(config())
    assert voice.transcribe_parameters("auto", 16000) == {
        "media_sample_rate_hz": 16000, "media_encoding": "pcm", "language_code": None,
        "identify_multiple_languages": True, "language_options": ["en-US", "ar-SA"],
    }
    assert voice.transcribe_parameters("en-US", 8000) == {
        "media_sample_rate_hz": 8000, "media_encoding": "pcm", "language_code": "en-US",
    }
    assert voice.transcribe_parameters("ar-SA", 16000)["language_code"] == "ar-SA"


def test_invalid_and_unavailable_auto_selections():
    voice = Voice(config())
    with pytest.raises(ValueError, match="Unsupported"):
        voice.transcribe_parameters("fr-FR", 16000)
    single = Voice(config(transcribe_languages=("en-US",)))
    with pytest.raises(ValueError, match="at least two"):
        single.transcribe_parameters("auto", 16000)


def test_mixed_segments_choose_weighted_dominant_and_preserve_order():
    parts = [("hello", "en-US"), ("عايز هاتف جديد من فضلك", "ar-SA"), ("under 100", "en-US")]
    assert dominant_language(parts) == ("ar-SA", ["en-US", "ar-SA"])
    assert dominant_language([], "en-US") == ("en-US", [])


def test_short_recording_returns_empty_metadata():
    result = asyncio.run(Voice(config()).transcribe(b"\x00" * 20, language="ar-SA"))
    assert result == {"text": "", "language": None, "languages": []}


def test_transcribe_returns_mixed_metadata(monkeypatch):
    import app.voice as module

    captured = {}

    class Input:
        async def send_audio_event(self, audio_chunk):
            captured.setdefault("chunks", []).append(audio_chunk)

        async def end_stream(self):
            captured["ended"] = True

    class Client:
        def __init__(self, region):
            captured["region"] = region

        async def start_stream_transcription(self, **kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(input_stream=Input(), output_stream=object())

    class Collector:
        def __init__(self, _stream):
            self.parts = [("hello", "en-US"), ("عايز هاتف جديد", "ar-SA")]

        async def handle_events(self):
            return None

    monkeypatch.setattr(module, "TranscribeStreamingClient", Client)
    monkeypatch.setattr(module, "_Collector", Collector)
    result = asyncio.run(Voice(config()).transcribe(b"\x00" * 4000, language="auto"))
    assert result == {"text": "hello عايز هاتف جديد", "language": "ar-SA",
                      "languages": ["en-US", "ar-SA"]}
    assert captured["kwargs"]["identify_multiple_languages"] is True
    assert captured["kwargs"]["language_options"] == ["en-US", "ar-SA"]
    assert captured["ended"] is True


def test_language_detection_and_speakable_preserve_arabic():
    assert text_language("نعم، هاتف smartphone مناسب") == "ar-SA"
    assert text_language("A smartphone هاتف") == "en-US"
    assert speakable("**نعم**، هذا مناسب.") == "نعم، هذا مناسب."


def test_polly_uses_hala_neural_and_preserves_text():
    calls = []

    class Polly:
        def synthesize_speech(self, **kwargs):
            calls.append(kwargs)
            return {"AudioStream": io.BytesIO(b"mp3")}

    voice = Voice(config())
    voice._polly = Polly()
    assert voice.speak("**نعم**، smartphone مناسب", "ar-SA") == b"mp3"
    assert calls == [{"Text": "نعم، smartphone مناسب", "OutputFormat": "mp3",
                      "VoiceId": "Hala", "Engine": "neural"}]


def test_polly_failure_is_exposed_for_browser_fallback():
    class Polly:
        def synthesize_speech(self, **_kwargs):
            raise RuntimeError("provider unavailable")

    voice = Voice(config())
    voice._polly = Polly()
    with pytest.raises(RuntimeError, match="provider unavailable"):
        voice.speak("hello", "en-US")
