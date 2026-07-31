from pathlib import Path
import shutil
import subprocess
from subprocess import CompletedProcess
from uuid import UUID

import pytest

from app.audio.conversion import convert_to_stt_wav, probe_duration
from app.audio.storage import AudioStorage
from app.audio.validation import save_upload, validate_upload
from app.errors import ApiError


def test_rejects_oversized_audio():
    """Removing the byte-limit branch must reject a too-large upload."""
    with pytest.raises(ApiError) as error:
        validate_upload("audio/webm", byte_count=11, max_bytes=10)

    assert error.value.code == "audio_too_large"


@pytest.mark.parametrize(
    "content_type",
    ["audio/webm", "audio/ogg", "audio/wav", "audio/mp4", "audio/mpeg"],
)
def test_accepts_supported_audio_content_types(content_type):
    """Removing a browser-supported MIME type must reject that upload."""
    validate_upload(content_type, byte_count=1, max_bytes=10)


def test_rejects_empty_or_unsupported_audio():
    """Removing input checks must reject empty and unsupported uploads."""
    with pytest.raises(ApiError) as empty_error:
        validate_upload("audio/webm", byte_count=0, max_bytes=10)
    with pytest.raises(ApiError) as type_error:
        validate_upload("text/plain", byte_count=1, max_bytes=10)

    assert empty_error.value.code == "invalid_audio"
    assert type_error.value.code == "invalid_audio"


class _ChunkedUpload:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = iter(chunks)
        self.read_sizes: list[int] = []

    async def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return next(self._chunks, b"")


@pytest.mark.asyncio
async def test_streamed_upload_stops_before_writing_data_over_limit(tmp_path):
    """Dropping the streaming byte check must not leave an oversized file."""
    upload = _ChunkedUpload([b"a" * (1024 * 1024), b"b"])
    destination = tmp_path / "input.webm"

    with pytest.raises(ApiError) as error:
        await save_upload(upload, destination, max_bytes=1024 * 1024)

    assert error.value.code == "audio_too_large"
    assert upload.read_sizes == [1024 * 1024, 1024 * 1024]
    assert not destination.exists()


def test_probe_duration_rejects_too_short_audio(monkeypatch, tmp_path):
    """Removing the lower duration bound must accept a 0.2-second recording."""
    monkeypatch.setattr(
        "app.audio.conversion.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, stdout="0.2\n", stderr=""),
    )

    with pytest.raises(ApiError) as error:
        probe_duration(tmp_path / "recording.webm", max_seconds=10)

    assert error.value.code == "audio_too_short"


def test_probe_duration_rejects_too_long_audio(monkeypatch, tmp_path):
    """Removing the upper duration bound must reject audio past its limit."""
    monkeypatch.setattr(
        "app.audio.conversion.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, stdout="10.1\n", stderr=""),
    )

    with pytest.raises(ApiError) as error:
        probe_duration(tmp_path / "recording.webm", max_seconds=10)

    assert error.value.code == "audio_too_long"


def test_conversion_hides_subprocess_details_on_failure(
    monkeypatch, tmp_path, caplog
):
    """Leaking ffmpeg output or paths must remain impossible on conversion failure."""
    source = tmp_path / "private-recording.webm"
    target = tmp_path / "normalized.wav"
    monkeypatch.setattr(
        "app.audio.conversion.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(
            args[0], 1, stdout="", stderr=f"ffmpeg failed for {source}"
        ),
    )

    with pytest.raises(ApiError) as error:
        convert_to_stt_wav(source, target)

    assert error.value.code == "invalid_audio"
    assert str(source) not in error.value.message
    assert "ffmpeg" not in error.value.message.lower()
    assert str(source) not in caplog.text
    assert "ffmpeg failed for" not in caplog.text


@pytest.mark.parametrize("stdout", ["not-a-number", "nan", "inf"])
def test_probe_rejects_malformed_or_nonfinite_output_without_logging_details(
    monkeypatch, tmp_path, caplog, stdout
):
    source = tmp_path / "private-recording.webm"
    monkeypatch.setattr(
        "app.audio.conversion.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(
            args[0],
            0,
            stdout=stdout,
            stderr=f"private probe failure for {source}",
        ),
    )

    with pytest.raises(ApiError) as error:
        probe_duration(source, max_seconds=10)

    assert error.value.code == "invalid_audio"
    assert str(source) not in caplog.text
    assert "private probe failure" not in caplog.text


def test_probe_rejects_live_muxed_duration_placeholder(monkeypatch, tmp_path, caplog):
    """MediaRecorder WebM makes ffprobe print N/A and exit 0; that is not a duration."""
    source = tmp_path / "private-recording.webm"
    monkeypatch.setattr(
        "app.audio.conversion.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(
            args[0],
            0,
            stdout="N/A\n",
            stderr=f"private probe failure for {source}",
        ),
    )

    with pytest.raises(ApiError) as error:
        probe_duration(source, max_seconds=10)

    assert error.value.code == "invalid_audio"
    assert str(source) not in caplog.text


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for end-to-end container checks",
)
def test_live_muxed_webm_is_probeable_after_conversion(tmp_path):
    """Probing a live WebM upload directly must not be how duration is obtained."""
    source = tmp_path / "recording.webm"
    target = tmp_path / "normalized.wav"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:a", "libopus", "-f", "webm", "-live", "1", str(source),
        ],
        check=True,
    )

    with pytest.raises(ApiError) as error:
        probe_duration(source, max_seconds=10)
    assert error.value.code == "invalid_audio"

    convert_to_stt_wav(source, target)

    assert probe_duration(target, max_seconds=10) == pytest.approx(3.0, abs=0.1)


def test_missing_audio_tools_do_not_log_private_paths(monkeypatch, tmp_path, caplog):
    source = tmp_path / "private-recording.webm"

    def missing_tool(*args, **kwargs):
        raise FileNotFoundError(f"missing tool while reading {source}")

    monkeypatch.setattr("app.audio.conversion.subprocess.run", missing_tool)

    with pytest.raises(ApiError):
        probe_duration(source, max_seconds=10)
    with pytest.raises(ApiError):
        convert_to_stt_wav(source, tmp_path / "normalized.wav")

    assert str(source) not in caplog.text
    assert "missing tool while reading" not in caplog.text


def test_storage_isolated_by_turn_and_expires(tmp_path):
    """Dropping UUID isolation or expiry must not retain a stale final chunk."""
    storage = AudioStorage(tmp_path, retention_seconds=1)
    first = storage.chunk_path(UUID(int=1), 0)
    second = storage.chunk_path(UUID(int=2), 0)
    assert first.parent != second.parent
    first.parent.mkdir(parents=True)
    first.write_bytes(b"RIFF")

    storage.expire(now=first.stat().st_mtime + 2)

    assert not first.exists()


def test_terminal_cleanup_preserves_final_chunks(tmp_path):
    """Deleting a terminal turn's chunks must not make completed audio unavailable."""
    storage = AudioStorage(tmp_path, retention_seconds=60)
    turn_id = UUID(int=1)
    upload = storage.upload_path(turn_id, suffix=".webm")
    intermediate = storage.intermediate_path(turn_id)
    chunk = storage.chunk_path(turn_id, 0)
    for path in (upload, intermediate, chunk):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio")

    storage.cleanup_terminal(turn_id)

    assert not upload.exists()
    assert not intermediate.exists()
    assert chunk.exists()
    assert storage.chunk_url(turn_id, 0) == f"/api/audio/{turn_id}/0.wav"
