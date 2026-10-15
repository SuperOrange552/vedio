"""用 sherpa-onnx + Kokoro v1.1-zh 离线合成旁白，输出每句 wav 与时间轴 timeline.json"""
import json, sys, os
import numpy as np, sherpa_onnx, wave
sys.path.insert(0, os.path.dirname(__file__))
from script import LINES

model_dir, out_dir = sys.argv[1], sys.argv[2]
sid = int(sys.argv[3]) if len(sys.argv) > 3 else 60
os.makedirs(out_dir, exist_ok=True)
cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=f"{model_dir}/model.onnx", voices=f"{model_dir}/voices.bin",
            tokens=f"{model_dir}/tokens.txt", data_dir=f"{model_dir}/espeak-ng-data",
            dict_dir=f"{model_dir}/dict",
            lexicon=f"{model_dir}/lexicon-us-en.txt,{model_dir}/lexicon-zh.txt"),
        num_threads=4),
    rule_fsts=f"{model_dir}/phone-zh.fst,{model_dir}/date-zh.fst,{model_dir}/number-zh.fst",
    max_num_sentences=1)
tts = sherpa_onnx.OfflineTts(cfg)
SR = None
t = 0.0
timeline = []
chunks = []
for i, (text, scene, pre, post) in enumerate(LINES):
    a = tts.generate(text, sid=sid, speed=0.82)
    SR = a.sample_rate
    s = np.array(a.samples, dtype=np.float32)
    # 去掉首尾静音
    idx = np.where(np.abs(s) > 0.01)[0]
    s = s[max(idx[0] - 200, 0): idx[-1] + 2000]
    dur = len(s) / SR
    t += pre
    timeline.append({"text": text, "scene": scene, "start": round(t, 3), "end": round(t + dur, 3)})
    chunks.append(np.zeros(int(pre * SR), np.float32)); chunks.append(s)
    chunks.append(np.zeros(int(post * SR), np.float32))
    t += dur + post
voice = np.concatenate(chunks)
voice = voice / max(np.abs(voice).max(), 1e-6) * 0.9
with wave.open(f"{out_dir}/voice.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((voice * 32767).astype(np.int16).tobytes())
json.dump({"total": round(t, 3), "lines": timeline}, open(f"{out_dir}/timeline.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(timeline, ensure_ascii=False, indent=1), t)
