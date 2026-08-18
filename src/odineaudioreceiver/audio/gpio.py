"""Jetson.GPIO-backed "listening" indicator with a safe no-op fallback.

The rest of the application never imports ``Jetson.GPIO`` directly. This module
owns the lazy import so that a laptop, CI, or any non-Jetson host -- where the
library is absent -- degrades to a silent no-op instead of failing at import
time. Every hardware call is guarded: a GPIO glitch must never crash a voice
turn, so failures are logged once and swallowed.
"""

import logging

from odineaudioreceiver.config import Settings

logger = logging.getLogger(__name__)


class ListeningIndicator:
    """Drive one "listening" pin (and an optional "armed" pin) on the Orin.

    The listening line is HIGH while an utterance is actively being captured.
    The armed line, when configured, is HIGH while the capture loop is alive.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._gpio = None
        self._ready = False
        self._warned = False
        self._on = settings.gpio_active_high
        self._listening_pin = settings.gpio_listening_pin
        self._armed_pin = settings.gpio_armed_pin

    @property
    def enabled(self) -> bool:
        return self._settings.gpio_listening_enabled

    def setup(self) -> None:
        """Claim the configured pins. A no-op unless GPIO output is enabled."""
        if not self.enabled:
            return
        try:
            import Jetson.GPIO as GPIO  # type: ignore[import-not-found]
        except Exception:
            logger.warning(
                "Jetson.GPIO unavailable; listening pin disabled for this run."
            )
            return

        try:
            GPIO.setwarnings(False)
            GPIO.setmode(getattr(GPIO, self._settings.gpio_mode))
            low = GPIO.LOW if self._on else GPIO.HIGH
            GPIO.setup(self._listening_pin, GPIO.OUT, initial=low)
            if self._armed_pin > 0:
                GPIO.setup(self._armed_pin, GPIO.OUT, initial=low)
        except Exception:
            logger.warning("Failed to configure GPIO listening pins.")
            return

        self._gpio = GPIO
        self._ready = True

    def set_listening(self, active: bool) -> None:
        """Drive the listening pin. Safe to call even when GPIO is disabled."""
        self._write(self._listening_pin, active)

    def set_armed(self, active: bool) -> None:
        """Drive the optional armed pin, if one is configured."""
        if self._armed_pin > 0:
            self._write(self._armed_pin, active)

    def cleanup(self) -> None:
        """Release the pins on shutdown."""
        if not self._ready or self._gpio is None:
            return
        try:
            self._gpio.cleanup()
        except Exception:
            logger.warning("GPIO cleanup failed.")
        finally:
            self._ready = False
            self._gpio = None

    def _write(self, pin: int, active: bool) -> None:
        if not self._ready or self._gpio is None:
            return
        try:
            level = active if self._on else not active
            self._gpio.output(pin, self._gpio.HIGH if level else self._gpio.LOW)
        except Exception:
            # Log at most once: a stuck pin should not spam the log every frame.
            if not self._warned:
                logger.warning("GPIO write failed; listening pin may be stale.")
                self._warned = True
