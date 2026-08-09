"""Runtime-switchable audio input mode (browser vs RF/I2S).

The input mode isn't auto-detected from hardware -- it's owned here. Switching
to ``rf_i2s`` starts the RF capture service (which routes the APE I2S port and
launches the arecord loop); switching back to ``browser`` stops it. Both the RF
SSE endpoints and `/api/health` read the current mode from this controller, so
a live switch is reflected everywhere without a restart.
"""

import logging
from collections.abc import Callable
from typing import Literal

from app.audio.rf_service import RFInputService

logger = logging.getLogger(__name__)

InputMode = Literal["browser", "rf_i2s"]
RFFactory = Callable[[], RFInputService]
Validate = Callable[[], None]


class InputModeController:
    """Own the active input mode and the RF capture service's lifecycle."""

    def __init__(
        self,
        *,
        mode: InputMode,
        rf_factory: RFFactory,
        validate: Validate | None = None,
    ) -> None:
        self._mode: InputMode = mode
        self._rf_factory = rf_factory
        self._validate = validate
        self._service: RFInputService | None = None

    @property
    def mode(self) -> InputMode:
        return self._mode

    def start(self) -> None:
        """Start RF capture if the initial mode is rf_i2s (called at boot)."""
        if self._mode == "rf_i2s":
            self._start_rf()

    async def set_mode(self, mode: InputMode) -> None:
        """Switch modes live. Raises (mode unchanged) if RF fails to start."""
        if mode == self._mode:
            return
        if mode == "rf_i2s":
            # Start first: if hardware routing or tool checks fail, the mode
            # stays 'browser' and the error propagates to the caller.
            self._start_rf()
        else:
            await self._stop_rf()
        self._mode = mode

    async def shutdown(self) -> None:
        await self._stop_rf()

    def _start_rf(self) -> None:
        if self._service is not None:
            return
        if self._validate is not None:
            self._validate()
        service = self._rf_factory()
        service.start()
        self._service = service

    async def _stop_rf(self) -> None:
        if self._service is None:
            return
        service = self._service
        self._service = None
        await service.stop()
