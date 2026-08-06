"""Runtime-switchable audio output routing.

Playback can go to the browser (the backend just emits `audio_ready` and the
UI plays it) or to a local ALSA device on the Orin (e.g. USB headphones), and
the tester can switch between them live from the UI without a restart. The
orchestrator asks this controller for the current player on every chunk, so a
change takes effect on the next synthesized chunk.
"""

import logging
import re
import subprocess
from collections.abc import Callable
from typing import Literal

from app.audio.playback import LocalAudioPlayer
from app.schemas import OutputDevice

logger = logging.getLogger(__name__)

OutputMode = Literal["browser", "device"]
PlayerFactory = Callable[[str], LocalAudioPlayer]
Runner = Callable[..., "subprocess.CompletedProcess[str]"]

# card 2: Headphones [USB Headphones], device 0: USB Audio [USB Audio]
_CARD_RE = re.compile(
    r"^card (\d+): \S+ \[([^\]]+)\], device (\d+): [^\[]*\[([^\]]+)\]"
)


def list_output_devices(runner: Runner = subprocess.run) -> list[OutputDevice]:
    """Enumerate ALSA playback devices via `aplay -l`. Never raises."""
    try:
        result = runner(
            ["aplay", "-l"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        logger.warning("Could not run aplay to list output devices.")
        return []
    if result.returncode != 0:
        return []

    devices: list[OutputDevice] = []
    for line in result.stdout.splitlines():
        match = _CARD_RE.match(line.strip())
        if match is None:
            continue
        card, card_name, device, device_name = match.groups()
        value = f"plughw:{card},{device}"
        devices.append(
            OutputDevice(value=value, label=f"{card_name} — {device_name} ({value})")
        )
    return devices


class AudioOutputController:
    """Own the current output routing and hand out the matching player."""

    def __init__(
        self,
        *,
        mode: OutputMode,
        device: str,
        player_factory: PlayerFactory = LocalAudioPlayer,
    ) -> None:
        self._player_factory = player_factory
        self._mode: OutputMode = mode
        self._device = device
        self._player = self._build()

    def _build(self) -> LocalAudioPlayer | None:
        if self._mode == "device":
            return self._player_factory(self._device)
        return None

    def current_player(self) -> LocalAudioPlayer | None:
        """The player for the active route, or None when output is the browser."""
        return self._player

    def configure(
        self,
        *,
        mode: OutputMode | None = None,
        device: str | None = None,
    ) -> None:
        if device is not None and device.strip():
            self._device = device
        if mode is not None:
            self._mode = mode
        self._player = self._build()

    @property
    def mode(self) -> OutputMode:
        return self._mode

    @property
    def device(self) -> str:
        return self._device
