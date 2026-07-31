#!/usr/bin/env python3
"""Run a content-safe end-to-end voice-turn smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx


CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".m4a": "audio/mp4",
}


class SmokeFailure(RuntimeError):
    """An expected smoke-contract failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one voice turn, follow SSE, and validate its audio."
    )
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("base_url")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--show-content",
        action="store_true",
        help="Print transcript and answer for explicit local diagnostics.",
    )
    return parser.parse_args()


def event_payload(lines: list[str]) -> tuple[str, dict[str, Any]] | None:
    event_name = ""
    data_lines: list[str] = []
    for line in lines:
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())
    if not event_name or not data_lines:
        return None
    try:
        payload = json.loads("\n".join(data_lines))
    except json.JSONDecodeError as error:
        raise SmokeFailure("SSE verisi geçerli JSON değil.") from error
    if not isinstance(payload, dict):
        raise SmokeFailure("SSE veri şekli geçersiz.")
    return event_name, payload


def require_number(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SmokeFailure(f"Eksik veya geçersiz metrik: {name}")
    return float(value)


def safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "upload_ms",
        "stt_ms",
        "llm_ms",
        "tts_ms",
        "total_ms",
        "first_sentence_ready_ms",
        "first_audio_started_ms",
        "audio_chunk_count",
        "llm_prompt_tokens",
        "llm_completion_tokens",
        "llm_tokens_per_second",
        "outcome",
        "timed_out",
        "configuration",
    )
    return {key: metrics.get(key) for key in allowed}


def run_smoke(args: argparse.Namespace) -> None:
    if not args.audio_file.is_file():
        raise SmokeFailure("Ses dosyası bulunamadı.")
    content_type = CONTENT_TYPES.get(args.audio_file.suffix.lower())
    if content_type is None:
        raise SmokeFailure("Desteklenmeyen ses dosyası uzantısı.")

    base_url = args.base_url.rstrip("/") + "/"
    audio_urls: dict[int, str] = {}
    transcript = ""
    answer = ""
    final_metrics: dict[str, Any] | None = None
    terminal_seen = False

    timeout = httpx.Timeout(args.timeout, connect=10.0)
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        with args.audio_file.open("rb") as audio_handle:
            response = client.post(
                urljoin(base_url, "api/turns"),
                files={
                    "audio": (
                        args.audio_file.name,
                        audio_handle,
                        content_type,
                    )
                },
            )
        if response.status_code != 202:
            raise SmokeFailure(
                f"Tur oluşturulamadı: HTTP {response.status_code}"
            )
        created = response.json()
        events_url = created.get("events_url")
        if not isinstance(events_url, str):
            raise SmokeFailure("events_url eksik.")

        with client.stream(
            "GET",
            urljoin(base_url, events_url.lstrip("/")),
            headers={"Accept": "text/event-stream"},
        ) as stream:
            stream.raise_for_status()
            frame: list[str] = []
            for line in stream.iter_lines():
                if line:
                    if not line.startswith(":"):
                        frame.append(line)
                    continue
                parsed = event_payload(frame)
                frame = []
                if parsed is None:
                    continue
                event_name, payload = parsed
                if event_name == "state":
                    print(f"stage={payload.get('stage', 'unknown')}")
                elif event_name == "audio_ready":
                    sequence = payload.get("sequence")
                    audio_url = payload.get("audio_url")
                    if not isinstance(sequence, int) or not isinstance(
                        audio_url, str
                    ):
                        raise SmokeFailure("audio_ready verisi geçersiz.")
                    audio_urls[sequence] = audio_url
                elif event_name == "failed":
                    error = payload.get("error")
                    code = error.get("code") if isinstance(error, dict) else None
                    raise SmokeFailure(f"Tur başarısız: {code or 'unknown'}")
                elif event_name == "complete":
                    transcript = payload.get("transcript", "")
                    answer = payload.get("answer", "")
                    metrics = payload.get("metrics")
                    if not isinstance(metrics, dict):
                        raise SmokeFailure("Tamamlanmış tur metriği eksik.")
                    final_metrics = metrics
                    terminal_seen = True
                    break

        if not terminal_seen or final_metrics is None:
            raise SmokeFailure("Terminal complete olayı alınamadı.")
        if not isinstance(transcript, str) or not transcript.strip():
            raise SmokeFailure("Transcript boş.")
        if not isinstance(answer, str) or not answer.strip():
            raise SmokeFailure("Yanıt boş.")
        if not audio_urls:
            raise SmokeFailure("Ses parçası üretilmedi.")

        sequences = sorted(audio_urls)
        if sequences != list(range(len(sequences))):
            raise SmokeFailure(f"Ses sıra boşluğu var: {sequences}")
        for sequence in sequences:
            audio_response = client.get(
                urljoin(base_url, audio_urls[sequence].lstrip("/"))
            )
            if audio_response.status_code != 200 or not audio_response.content:
                raise SmokeFailure(
                    f"Ses parçası indirilemedi: sequence={sequence}"
                )
            content_type = audio_response.headers.get("content-type", "")
            if not content_type.startswith("audio/"):
                raise SmokeFailure(
                    f"Ses parçası Content-Type geçersiz: sequence={sequence}"
                )

    first_sentence_ms = require_number(
        final_metrics, "first_sentence_ready_ms"
    )
    llm_ms = require_number(final_metrics, "llm_ms")
    if first_sentence_ms >= llm_ms:
        raise SmokeFailure(
            "İlk cümle LLM tamamlanmadan hazır olmadı."
        )

    print(json.dumps(safe_metrics(final_metrics), ensure_ascii=False, indent=2))
    print(f"audio_sequences={sequences}")
    if args.show_content:
        print(f"transcript={transcript}")
        print(f"answer={answer}")
    print("Smoke başarılı.")


def main() -> int:
    try:
        run_smoke(parse_args())
    except (SmokeFailure, httpx.HTTPError, OSError, ValueError) as error:
        print(f"Smoke başarısız: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
