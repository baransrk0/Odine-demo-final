"""Piper CLI adapter for local Turkish speech synthesis."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path


ProcessFactory = Callable[..., Awaitable[asyncio.subprocess.Process]]


class PiperTTS:
    """Synthesize one sentence at a time through the Piper command-line tool."""

    def __init__(
        self,
        *,
        binary: str,
        model_path: str,
        process_factory: ProcessFactory | None = None,
        terminate_grace_seconds: float = 1.0,
    ) -> None:
        if terminate_grace_seconds <= 0:
            raise ValueError("terminate_grace_seconds must be positive")
        self._binary = binary
        self._model_path = model_path
        self._process_factory = process_factory or asyncio.create_subprocess_exec
        self._terminate_grace_seconds = terminate_grace_seconds
        self.ready = False

    def load(self) -> None:
        """Validate the executable and model path before accepting turns."""
        binary_path = Path(self._binary)
        if not binary_path.is_file() or not binary_path.stat().st_mode & 0o111:
            raise RuntimeError("Piper executable is unavailable.")
        if not Path(self._model_path).is_file():
            raise RuntimeError("Piper model is unavailable.")
        self.ready = True

    async def synthesize(self, text: str, output_path: Path) -> None:
        """Write a Piper-generated WAV to the requested path."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        process = await self._process_factory(
            self._binary,
            "--model",
            self._model_path,
            "--output_file",
            str(output_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await process.communicate(text.encode("utf-8"))
        except BaseException:
            output_path.unlink(missing_ok=True)
            cleanup = asyncio.create_task(
                _terminate_process(process, self._terminate_grace_seconds)
            )
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
            raise

        if (
            process.returncode != 0
            or not output_path.is_file()
            or output_path.stat().st_size == 0
        ):
            output_path.unlink(missing_ok=True)
            raise RuntimeError("Piper synthesis failed")


async def _terminate_process(
    process: asyncio.subprocess.Process,
    grace_seconds: float,
) -> None:
    """Terminate and reap a child process, escalating when it does not exit."""
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
