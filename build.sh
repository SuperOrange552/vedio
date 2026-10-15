#!/usr/bin/env bash
# 一键重新生成视频：旁白 → 画面 → 配乐混音 → 合成 mp4
# 依赖: python3, ffmpeg, pip install sherpa-onnx numpy pillow
set -euo pipefail
cd "$(dirname "$0")"
MODELS=${MODELS:-models}
mkdir -p "$MODELS" build
if [ ! -d "$MODELS/kokoro-multi-lang-v1_1" ]; then
  curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_1.tar.bz2 | tar xj -C "$MODELS"
fi
python3 src/tts.py "$MODELS/kokoro-multi-lang-v1_1" build 60
python3 src/render.py build/timeline.json build/frames
python3 src/audio.py build/timeline.json build/voice.wav build/mix.wav
ffmpeg -y -framerate 30 -i build/frames/%05d.png -i build/mix.wav \
  -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k \
  -shortest -movflags +faststart output/草腥味.mp4
