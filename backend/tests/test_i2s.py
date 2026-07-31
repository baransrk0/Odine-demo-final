"""APE routing contract tests without Jetson hardware."""

from types import SimpleNamespace

from app.audio.i2s import configure_ape
from app.config import Settings


def test_configure_ape_routes_i2s_to_admaif(monkeypatch):
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("app.audio.i2s.subprocess.run", run)
    configure_ape(Settings(_env_file=None))

    assert calls == [
        ["amixer", "-c", "APE", "cset", "name=I2S2 codec master mode", "cbm-cfm"],
        ["amixer", "-c", "APE", "cset", "name=ADMAIF1 Mux", "I2S2"],
    ]
