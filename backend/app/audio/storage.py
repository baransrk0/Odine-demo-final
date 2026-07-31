"""Per-turn, short-lived storage for uploaded and synthesized audio."""

from pathlib import Path
import shutil
import time
from uuid import UUID


class AudioStorage:
    """Keep raw turn inputs separate from final chunks and expire them safely."""

    def __init__(self, root: Path, retention_seconds: int) -> None:
        self.root = Path(root)
        self.retention_seconds = retention_seconds
        self.root.mkdir(parents=True, exist_ok=True)

    def turn_dir(self, turn_id: UUID) -> Path:
        """Return the directory reserved for one turn UUID."""
        return self.root / str(turn_id)

    def upload_path(self, turn_id: UUID, suffix: str = ".upload") -> Path:
        """Return the isolated path for the original browser upload."""
        return self.turn_dir(turn_id) / f"upload{suffix}"

    def intermediate_path(self, turn_id: UUID) -> Path:
        """Return the isolated STT-normalized intermediate path."""
        return self.turn_dir(turn_id) / "stt-input.wav"

    def chunk_path(self, turn_id: UUID, sequence: int) -> Path:
        """Return the path for a synthesized WAV chunk."""
        return self.turn_dir(turn_id) / "chunks" / f"{sequence}.wav"

    def chunk_url(self, turn_id: UUID, sequence: int) -> str:
        """Return the public URL corresponding to a final chunk."""
        return f"/api/audio/{turn_id}/{sequence}.wav"

    def cleanup_terminal(self, turn_id: UUID) -> None:
        """Remove a terminal turn's upload/intermediates but preserve final chunks."""
        directory = self.turn_dir(turn_id)
        if not directory.exists():
            return
        for child in directory.iterdir():
            if child.name == "chunks":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)

    def expire(self, now: float | None = None) -> None:
        """Remove final chunks that have passed their retention window."""
        timestamp = time.time() if now is None else now
        for chunk in self.root.glob("*/chunks/*.wav"):
            if chunk.stat().st_mtime + self.retention_seconds <= timestamp:
                chunk.unlink(missing_ok=True)
                chunks_dir = chunk.parent
                if not any(chunks_dir.iterdir()):
                    chunks_dir.rmdir()
                    turn_dir = chunks_dir.parent
                    if not any(turn_dir.iterdir()):
                        turn_dir.rmdir()
