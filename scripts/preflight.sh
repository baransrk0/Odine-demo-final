#!/usr/bin/env bash
set -euo pipefail

for required_name in LLAMA_CPP_MODEL; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "Eksik yapılandırma: ${required_name}" >&2
    exit 1
  fi
done

stt_backend="${STT_BACKEND:-huggingface}"
tts_backend="${TTS_BACKEND:-huggingface}"
llama_cpp_base_url="${LLAMA_CPP_BASE_URL:-http://127.0.0.1:8080}"
whisper_cpp_base_url="${WHISPER_CPP_BASE_URL:-http://127.0.0.1:8080}"
requires_huggingface=false
requires_cuda=false

case "$stt_backend" in
  huggingface)
    [[ -n "${STT_MODEL_ID:-}" ]] || { echo "Eksik yapılandırma: STT_MODEL_ID" >&2; exit 1; }
    requires_huggingface=true
    [[ "${STT_DEVICE:-cuda}" == cuda* ]] && requires_cuda=true
    ;;
  whisper_cpp)
    ;;
  *)
    echo "Bilinmeyen STT_BACKEND: $stt_backend" >&2
    exit 1
    ;;
esac

case "$tts_backend" in
  huggingface)
    [[ -n "${TTS_MODEL_ID:-}" ]] || { echo "Eksik yapılandırma: TTS_MODEL_ID" >&2; exit 1; }
    requires_huggingface=true
    [[ "${TTS_DEVICE:-cuda}" == cuda* ]] && requires_cuda=true
    ;;
  piper)
    [[ -x "${PIPER_BINARY:-}" ]] || { echo "Piper çalıştırılabilir dosyası bulunamadı." >&2; exit 1; }
    [[ -f "${PIPER_MODEL_PATH:-}" ]] || { echo "Piper model dosyası bulunamadı." >&2; exit 1; }
    [[ -f "${PIPER_MODEL_PATH}.json" ]] || { echo "Piper model yapılandırması bulunamadı." >&2; exit 1; }
    ;;
  *)
    echo "Bilinmeyen TTS_BACKEND: $tts_backend" >&2
    exit 1
    ;;
esac

echo "== Disk =="
df -h /

echo "== Bellek =="
free -h

echo "== Araçlar =="
command -v ffmpeg
command -v ffprobe

if [[ "$requires_huggingface" == true ]]; then
  backend_python="backend/.venv/bin/python"
  if [[ ! -x "$backend_python" ]]; then
    backend_python="$(command -v python3)"
  fi
  echo "== Hugging Face runtime =="
  "$backend_python" - "$requires_cuda" <<'PY'
import sys

import soundfile  # noqa: F401
import torch
import transformers  # noqa: F401

requires_cuda = sys.argv[1] == "true"
if requires_cuda and not torch.cuda.is_available():
    raise SystemExit("CUDA görünür değil.")
PY
fi

if [[ "$stt_backend" == "whisper_cpp" ]]; then
  echo "== whisper.cpp =="
  curl --fail --silent --show-error --max-time 2 \
    "$whisper_cpp_base_url/health" >/dev/null
fi

if [[ "$tts_backend" == "piper" ]]; then
  echo "== Piper cold synthesis =="
  piper_probe_dir="$(mktemp -d)"
  trap 'rm -rf -- "$piper_probe_dir"' EXIT
  piper_probe_wav="$piper_probe_dir/probe.wav"
  printf 'Merhaba.\n' | "$PIPER_BINARY" \
    --model "$PIPER_MODEL_PATH" \
    --output_file "$piper_probe_wav" >/dev/null
  [[ -s "$piper_probe_wav" ]] || {
    echo "Piper geçerli bir ses dosyası üretmedi." >&2
    exit 1
  }
  ffprobe -v error -select_streams a:0 \
    -show_entries stream=codec_name \
    -of default=noprint_wrappers=1:nokey=1 \
    "$piper_probe_wav" >/dev/null
fi

echo "== llama-server =="
curl --fail --silent --show-error --max-time 2 \
  "$llama_cpp_base_url/health" >/dev/null

echo "Preflight başarılı."
