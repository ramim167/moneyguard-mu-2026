"""Recording API contract without speech dependencies or network access."""
import base64
import io
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from moneyguard.domain import ValidationError
from moneyguard.server import Application
import json

from moneyguard.speech import GEMINI_URL, GROQ_URL, MAX_AUDIO_BYTES, SpeechTranscriber, SpeechUnavailable, audio_clip

NO_KEYS = {"MONEYGUARD_SPEECH_MODEL_DIR": "", "MONEYGUARD_SPEECH_GROQ_API_KEY": "", "GROQ_API_KEY": "",
           "MONEYGUARD_SPEECH_GEMINI_API_KEY": "", "GEMINI_API_KEY": ""}


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.clip = b"\x1a\x45\xdf\xa3fixture"
        self.body = {"business_id": "shapla-services", "audio": base64.b64encode(self.clip).decode(),
                     "mime_type": "audio/webm", "lang": "bn", "cloud_consent": True}

    def test_checks_formats_and_byte_limit(self):
        for mime, data in {"audio/webm": self.clip, "audio/ogg": b"OggSfixture",
                           "audio/mp4": b"\x00\x00\x00\x14ftypM4A ", "audio/wav": b"RIFF\x00\x00\x00\x00WAVE"}.items():
            with self.subTest(mime=mime):
                self.assertEqual(audio_clip(base64.b64encode(data).decode(), mime), data)
        for encoded, mime in [(self.body["audio"], "audio/ogg"), ("***", "audio/webm"),
                              (self.body["audio"], []), ("", "audio/webm"),
                              (base64.b64encode(self.clip.ljust(MAX_AUDIO_BYTES + 1, b"0")).decode(), "audio/webm")]:
            with self.subTest(mime=mime), self.assertRaises(ValidationError):
                audio_clip(encoded, mime)

    def test_membership_consent_and_provider_request(self):
        with patch.dict("os.environ", {**NO_KEYS, "MONEYGUARD_SPEECH_GROQ_API_KEY": "test-key"}):
            app = Application(":memory:")
        self.addCleanup(app.store.close)
        path = "/api/speech/transcribe"
        operator = {"id": "nadia"}
        self.assertEqual(app.config(None)["speech"]["backend"], "groq")
        with patch("moneyguard.speech.open_provider_request") as fetch:
            with self.assertRaises(PermissionError):
                app.post(path, self.body, None, None)
            with self.assertRaises(PermissionError):
                app.post(path, self.body, {"id": "outsider"}, None)
            with self.assertRaises(ValidationError):
                app.post(path, {**self.body, "cloud_consent": 1}, operator, None)
            with self.assertRaises(ValidationError):
                app.post(path, {**self.body, "lang": "any"}, operator, None)
            fetch.assert_not_called()
            fetch.return_value = nullcontext(io.BytesIO(b'{"text": "  payroll question  "}'))
            self.assertEqual(app.post(path, self.body, operator, None), {"text": "payroll question"})
            request = fetch.call_args.args[0]
            self.assertEqual(request.full_url, GROQ_URL)
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            self.assertIn(b'name="language"\r\n\r\nbn\r\n', request.data)
            self.assertIn(self.clip, request.data)

    def test_missing_backend_and_explicit_local_preference(self):
        with patch.dict("os.environ", NO_KEYS):
            speech = SpeechTranscriber()
        self.assertFalse(speech.status()["available"])
        with self.assertRaises(SpeechUnavailable):
            speech.transcribe(self.body, "nadia")
        with patch.dict("os.environ", {**NO_KEYS, "MONEYGUARD_SPEECH_MODEL_DIR": "missing-model", "MONEYGUARD_SPEECH_GROQ_API_KEY": "test-key"}):
            self.assertFalse(SpeechTranscriber().status()["available"])

    def test_chat_ai_keys_enable_transcription(self):
        with patch.dict("os.environ", {**NO_KEYS, "GROQ_API_KEY": "chat-groq", "GEMINI_API_KEY": "chat-gemini"}):
            status = SpeechTranscriber().status()
        self.assertEqual((status["backend"], status["provider"]), ("groq", "Groq"))
        with patch.dict("os.environ", {**NO_KEYS, "GEMINI_API_KEY": "chat-gemini"}):
            status = SpeechTranscriber().status()
        self.assertEqual((status["backend"], status["provider"], status["formats"]), ("gemini", "Google Gemini", ["audio/wav"]))

    def test_gemini_transcribes_wav_with_consent(self):
        wav = b"RIFF" + bytes(4) + b"WAVEfmt fixture"
        body = {"business_id": "shapla-services", "audio": base64.b64encode(wav).decode(),
                "mime_type": "audio/wav", "lang": "bn", "cloud_consent": True}
        with patch.dict("os.environ", {**NO_KEYS, "GEMINI_API_KEY": "gemini-key"}):
            app = Application(":memory:")
        self.addCleanup(app.store.close)
        path, operator = "/api/speech/transcribe", {"id": "nadia"}
        with patch("moneyguard.speech.open_provider_request") as fetch:
            with self.assertRaises(ValidationError):
                app.post(path, {**body, "cloud_consent": False}, operator, None)
            with self.assertRaises(ValidationError):
                app.post(path, {**self.body, "cloud_consent": True}, operator, None)  # WebM is not sent to Gemini
            fetch.assert_not_called()
            reply = {"candidates": [{"content": {"parts": [{"text": " এই মাসে কি বেতন দিতে পারব? "}]}}]}
            fetch.return_value = nullcontext(io.BytesIO(json.dumps(reply).encode()))
            self.assertEqual(app.post(path, body, operator, None), {"text": "এই মাসে কি বেতন দিতে পারব?"})
            request = fetch.call_args.args[0]
            self.assertEqual(request.full_url, GEMINI_URL.format(model="gemini-flash-lite-latest"))
            self.assertEqual(request.headers["X-goog-api-key"], "gemini-key")
            sent = json.loads(request.data)
            inline = sent["contents"][0]["parts"][1]["inline_data"]
            self.assertEqual((inline["mime_type"], base64.b64decode(inline["data"])), ("audio/wav", wav))
            self.assertIn("Bangla", sent["contents"][0]["parts"][0]["text"])
            fetch.return_value = nullcontext(io.BytesIO(b'{"candidates": [{"finishReason": "STOP"}]}'))
            self.assertEqual(app.post(path, body, operator, None), {"text": ""})

    def test_gemini_moves_to_the_next_model_when_busy(self):
        from urllib.error import HTTPError
        wav = b"RIFF" + bytes(4) + b"WAVEfmt fixture"
        with patch.dict("os.environ", {**NO_KEYS, "GEMINI_API_KEY": "gemini-key"}):
            speech = SpeechTranscriber()
        reply = nullcontext(io.BytesIO(b'{"candidates": [{"content": {"parts": [{"text": "cash"}]}}]}'))
        busy = HTTPError(GEMINI_URL, 503, "busy", {}, io.BytesIO(b"{}"))
        with patch("moneyguard.speech.open_provider_request", side_effect=[busy, reply]) as fetch:
            self.assertEqual(speech._gemini(wav, "audio/wav", "en"), "cash")
        self.assertEqual([c.args[0].full_url.rsplit("/", 1)[1] for c in fetch.call_args_list],
                         ["gemini-flash-lite-latest:generateContent", "gemini-flash-latest:generateContent"])
        refused = HTTPError(GEMINI_URL, 400, "bad", {}, io.BytesIO(b"{}"))
        with patch("moneyguard.speech.open_provider_request", side_effect=[refused]), self.assertRaises(SpeechUnavailable):
            speech._gemini(wav, "audio/wav", "en")
