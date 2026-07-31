"""HTTP adapter for a local whisper.cpp server."""

from pathlib import Path

import httpx


class WhisperCppSTT:
    """Transcribe normalized WAV files through whisper.cpp's multipart API."""

    def __init__(
        self,
        base_url: str,
        *,
        inference_path: str = "/inference",
        timeout_seconds: float = 45.0,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/{inference_path.lstrip('/')}"
        self._timeout_seconds = timeout_seconds
        self.ready = False

    def load(self) -> None:
        """Mark a configured local HTTP runtime ready without loading a model."""
        self.ready = True

    async def transcribe(self, path: Path) -> str:
        """Upload one WAV file and return the server's trimmed transcript."""
        try:
            content = await _read_bytes(path)
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout_seconds)
            ) as client:
                response = await client.post(
                    self._url,
                    files={"file": (path.name, content, "audio/wav")},
                    data={"language": "tr", "response_format": "json"},
                )
                response.raise_for_status()
                payload = response.json()
        except (OSError, httpx.HTTPError, ValueError) as error:
            raise RuntimeError("whisper.cpp transcription failed") from error

        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str):
            raise RuntimeError("whisper.cpp returned an invalid response")
        return text.strip()


async def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()
