"""Validation and bounded persistence for incoming audio uploads."""

from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.errors import ApiError


SUPPORTED_CONTENT_TYPES = frozenset(
    {"audio/webm", "audio/ogg", "audio/wav", "audio/mp4", "audio/mpeg"}
)
UPLOAD_BLOCK_SIZE = 1024 * 1024


class UploadReader(Protocol):
    """The small portion of an async multipart upload used by this module."""

    async def read(self, size: int) -> bytes: ...


def validate_upload(
    content_type: str | None,
    byte_count: int,
    max_bytes: int | None = None,
) -> None:
    """Reject unsupported, empty, or oversized uploads before transcription."""
    limit = max_bytes if max_bytes is not None else Settings().audio_max_bytes
    if content_type not in SUPPORTED_CONTENT_TYPES or byte_count <= 0:
        raise ApiError("invalid_audio", "Geçerli bir ses kaydı gönderin.")
    if byte_count > limit:
        raise ApiError("audio_too_large", "Ses kaydı boyut sınırını aşıyor.")


async def save_upload(
    upload: UploadReader,
    destination: Path,
    max_bytes: int | None = None,
) -> int:
    """Stream an upload to disk without retaining data that crosses the limit."""
    limit = max_bytes if max_bytes is not None else Settings().audio_max_bytes
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(UPLOAD_BLOCK_SIZE):
                written += len(chunk)
                if written > limit:
                    raise ApiError(
                        "audio_too_large", "Ses kaydı boyut sınırını aşıyor."
                    )
                output.write(chunk)
    except ApiError:
        destination.unlink(missing_ok=True)
        raise

    if written == 0:
        destination.unlink(missing_ok=True)
        raise ApiError("invalid_audio", "Geçerli bir ses kaydı gönderin.")

    return written
