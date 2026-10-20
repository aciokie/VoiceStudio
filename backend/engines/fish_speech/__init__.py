"""fish_speech: Fish Speech S2-Pro TTS engine adapter (#1432).

Connects to a locally-running Fish Speech server (self-hosted or Docker)
via its HTTP API at ``POST /v1/tts``. Fish Speech S2-Pro is an open-source
LLM-based TTS model with strong multilingual support and reference-based
voice cloning.

Opt-in (Settings -> Engines, or ``OMNIVOICE_TTS_BACKEND=fish_speech``).
Requires a running Fish Speech server; set ``FISH_SPEECH_SERVER_URL``
(default ``http://127.0.0.1:8080``).

Licence: Apache-2.0 (model weights), BSD-3-Clause (code). Both commercial-OK.
"""
from __future__ import annotations

import logging
import os
import struct
import sys
from typing import TYPE_CHECKING

from services.tts_backend import TTSBackend

logger = logging.getLogger("omnivoice.engines.fish_speech")

if TYPE_CHECKING:
    import torch  # noqa: F401


class FishSpeechBackend(TTSBackend):
    """Fish Speech S2-Pro TTS via local HTTP server."""

    id = "fish_speech"
    display_name = "Fish Speech S2-Pro (self-hosted, Apache-2.0)"
    gpu_compat: tuple[str, ...] = ("gpu", "cpu")
    supports_cloning = True

    def __init__(self) -> None:
        super().__init__()
        self._server_url = os.environ.get("FISH_SPEECH_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
        self._sample_rate = 44100  # Fish Speech default; read from server on first call
        self._check_model = os.environ.get("FISH_SPEECH_CHECKPOINT", "checkpoints/s2-pro")

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        import httpx  # noqa: F401

        server_url = os.environ.get("FISH_SPEECH_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
        try:
            resp = httpx.get(f"{server_url}/v1/models", timeout=5.0)
            if resp.status_code == 200:
                return True, "ready"
            return False, f"Fish Speech server at {server_url} returned HTTP {resp.status_code}"
        except httpx.ConnectError:
            return False, (
                f"Fish Speech server not reachable at {server_url}. "
                "Start it with: python tools/api_server.py --listen 0.0.0.0:8080 "
                f"--checkpoint_path {cls._check_model}. "
                "Or set FISH_SPEECH_SERVER_URL to the correct address."
            )
        except Exception as e:
            return False, f"Fish Speech server check failed: {e}"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def supported_languages(self) -> list[str]:
        return ["multi"]

    def generate(self, text: str, **kw) -> "torch.Tensor":
        import httpx
        import torch

        ref_audio = kw.get("ref_audio")
        ref_text = kw.get("ref_text", "")
        reference_id = kw.get("reference_id")
        language = kw.get("language", "en")

        payload: dict = {
            "text": text,
            "format": "wav",
            "mp3_bitrate": 128,
            "normalize": True,
            "channel": "mono",
        }

        if reference_id:
            payload["reference_id"] = reference_id
        elif ref_audio:
            import base64
            import os as _os

            if _os.path.isfile(ref_audio):
                with open(ref_audio, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("ascii")
                payload["references"] = [{
                    "audio": f"data:audio/wav;base64,{audio_b64}",
                    "text": ref_text or "",
                }]

        if language and language != "auto":
            payload["language"] = language

        try:
            resp = httpx.post(
                f"{self._server_url}/v1/tts",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Fish Speech TTS failed (HTTP {e.response.status_code}): "
                f"{e.response.text[:500]}"
            ) from e
        except httpx.ConnectError:
            raise RuntimeError(
                f"Fish Speech server not reachable at {self._server_url}. "
                "Ensure the server is running."
            )

        content_type = resp.headers.get("content-type", "")
        if "audio/wav" in content_type or "audio/x-wav" in content_type or resp.content[:4] == b"RIFF":
            audio_data = resp.content
            # Parse WAV header to get sample rate
            if audio_data[:4] == b"RIFF" and len(audio_data) > 24:
                sr = struct.unpack_from("<I", audio_data, 24)[0]
                self._sample_rate = sr
                # Strip 44-byte WAV header
                audio_data = audio_data[44:]
            # Convert raw PCM to tensor
            import numpy as np
            pcm = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            return torch.from_numpy(pcm).unsqueeze(0)
        else:
            # JSON error response
            try:
                err = resp.json()
                raise RuntimeError(
                    f"Fish Speech TTS error: {err.get('detail', resp.text[:500])}"
                )
            except (ValueError, KeyError):
                raise RuntimeError(f"Fish Speech TTS: unexpected response: {resp.text[:500]}")


__all__ = ["FishSpeechBackend"]
