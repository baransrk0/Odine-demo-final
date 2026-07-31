"""Behavioral tests for backend-specific deployment preflight checks."""

from pathlib import Path
import subprocess


PREFLIGHT = Path(__file__).resolve().parents[2] / "scripts" / "preflight.sh"


def _write_tool(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{body}\n")
    path.chmod(0o755)


def _external_environment(tmp_path: Path, *, piper_body: str) -> dict[str, str]:
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("df", "free"):
        _write_tool(tools, name, "exit 0")
    _write_tool(
        tools,
        "curl",
        'case "$*" in *"/health"*) exit 0;; *) exit 1;; esac',
    )
    _write_tool(
        tools,
        "ffmpeg",
        "exit 0",
    )
    _write_tool(
        tools,
        "ffprobe",
        '[[ "$(head -c 4 "${@: -1}")" == "RIFF" ]]',
    )
    _write_tool(tools, "piper", piper_body)

    model = tmp_path / "tr_TR.onnx"
    model.write_bytes(b"model")
    Path(f"{model}.json").write_text("{}")
    return {
        "PATH": f"{tools}:/usr/bin:/bin",
        "LLAMA_CPP_MODEL": "gemma",
        "LLAMA_CPP_BASE_URL": "http://127.0.0.1:8090",
        "STT_BACKEND": "whisper_cpp",
        "WHISPER_CPP_BASE_URL": "http://127.0.0.1:8080",
        "TTS_BACKEND": "piper",
        "PIPER_BINARY": str(tools / "piper"),
        "PIPER_MODEL_PATH": str(model),
    }


def _run_preflight(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(PREFLIGHT)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_external_preflight_succeeds_without_nvcc(tmp_path: Path):
    environment = _external_environment(
        tmp_path,
        piper_body=(
            'while [[ $# -gt 0 ]]; do '
            'if [[ "$1" == "--output_file" ]]; then shift; output="$1"; fi; '
            "shift; done\n"
            'printf "RIFF test" > "$output"'
        ),
    )

    result = _run_preflight(environment)

    assert result.returncode == 0, result.stderr
    assert "Preflight başarılı." in result.stdout


def test_external_preflight_rejects_missing_piper_config(tmp_path: Path):
    environment = _external_environment(tmp_path, piper_body="exit 0")
    Path(f"{environment['PIPER_MODEL_PATH']}.json").unlink()

    result = _run_preflight(environment)

    assert result.returncode != 0
    assert "Piper model yapılandırması bulunamadı." in result.stderr


def test_external_preflight_rejects_failed_piper_cold_synthesis(tmp_path: Path):
    environment = _external_environment(tmp_path, piper_body="exit 7")

    result = _run_preflight(environment)

    assert result.returncode != 0


def test_external_preflight_rejects_empty_piper_wav(tmp_path: Path):
    environment = _external_environment(tmp_path, piper_body="exit 0")

    result = _run_preflight(environment)

    assert result.returncode != 0


def test_external_preflight_rejects_invalid_piper_wav(tmp_path: Path):
    environment = _external_environment(
        tmp_path,
        piper_body=(
            'while [[ $# -gt 0 ]]; do '
            'if [[ "$1" == "--output_file" ]]; then shift; output="$1"; fi; '
            "shift; done\n"
            'printf "not audio" > "$output"'
        ),
    )

    result = _run_preflight(environment)

    assert result.returncode != 0


def test_external_preflight_accepts_valid_piper_cold_synthesis(tmp_path: Path):
    environment = _external_environment(
        tmp_path,
        piper_body=(
            'while [[ $# -gt 0 ]]; do '
            'if [[ "$1" == "--output_file" ]]; then shift; output="$1"; fi; '
            "shift; done\n"
            'printf "RIFF test" > "$output"'
        ),
    )

    result = _run_preflight(environment)

    assert result.returncode == 0, result.stderr
    assert "== Piper cold synthesis ==" in result.stdout


def test_huggingface_cpu_preflight_does_not_require_cuda_or_nvcc(tmp_path: Path):
    environment = _external_environment(
        tmp_path,
        piper_body=(
            'while [[ $# -gt 0 ]]; do '
            'if [[ "$1" == "--output_file" ]]; then shift; output="$1"; fi; '
            "shift; done\n"
            'printf "RIFF test" > "$output"'
        ),
    )
    tools = Path(environment["PATH"].split(":", 1)[0])
    _write_tool(tools, "python3", "exit 0")
    environment.update(
        {
            "STT_BACKEND": "huggingface",
            "STT_MODEL_ID": "local-stt",
            "STT_DEVICE": "cpu",
        }
    )

    result = _run_preflight(environment)

    assert result.returncode == 0, result.stderr


def test_huggingface_cuda_preflight_rejects_unavailable_cuda(tmp_path: Path):
    environment = _external_environment(
        tmp_path,
        piper_body=(
            'while [[ $# -gt 0 ]]; do '
            'if [[ "$1" == "--output_file" ]]; then shift; output="$1"; fi; '
            "shift; done\n"
            'printf "RIFF test" > "$output"'
        ),
    )
    tools = Path(environment["PATH"].split(":", 1)[0])
    _write_tool(tools, "python3", "exit 9")
    environment.update(
        {
            "STT_BACKEND": "huggingface",
            "STT_MODEL_ID": "local-stt",
            "STT_DEVICE": "cuda",
        }
    )

    result = _run_preflight(environment)

    assert result.returncode != 0
