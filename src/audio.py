"""合成配乐与音效，并与旁白混音 → build/mix.wav

用法: python3 src/audio.py build/timeline.json build/voice.wav build/mix.wav
"""
import json, sys, wave
import numpy as np

SR = 44100
tl = json.load(open(sys.argv[1]))
T = tl["total"]
N = int(T * SR) + SR
L = {l["scene"]: l for l in tl["lines"]}
start = {k: (0.0 if k == "ants" else v["start"] - 0.5) for k, v in L.items()}
rng = np.random.default_rng(1)

music = np.zeros(N, np.float32)
sfx = np.zeros(N, np.float32)


def note_f(name):
    names = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    return 440.0 * 2 ** ((names[name[:-1]] + 12 * (int(name[-1]) + 1) - 69) / 12)


def piano(buf, t, name, vel=0.3, dur=3.5):
    """简单的“钢琴”音色：几个谐波 + 指数衰减"""
    f = note_f(name)
    n = int(dur * SR)
    x = np.arange(n) / SR
    env = np.exp(-x * 2.2) * (1 - np.exp(-x * 400))
    tone = sum(a * np.sin(2 * np.pi * f * h * x) for h, a in ((1, 1), (2, 0.35), (3, 0.12), (4, 0.06)))
    i = int(t * SR)
    seg = (tone * env * vel).astype(np.float32)[: max(0, N - i)]
    buf[i:i + len(seg)] += seg


def pad(buf, t0, t1, names, vol=0.06, att=1.5, rel=1.5):
    n = int((t1 - t0) * SR)
    x = np.arange(n) / SR
    env = np.minimum(1, x / att) * np.minimum(1, (t1 - t0 - x) / rel).clip(0)
    sig = np.zeros(n)
    for nm in names:
        f = note_f(nm)
        for det in (-0.6, 0, 0.6):
            sig += np.sin(2 * np.pi * (f + det) * x + rng.uniform(0, 6))
    i = int(t0 * SR)
    buf[i:i + n] += (sig / (3 * len(names)) * env * vol).astype(np.float32)


def lowpass(x, a):
    """一阶低通（a 越小越闷）"""
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):   # 只用在短片段上
        acc += a * (x[i] - acc)
        y[i] = acc
    return y


def lp_fast(x, k):
    """移动平均近似低通，k 为窗口点数"""
    c = np.cumsum(np.concatenate([[0], x]))
    y = (c[k:] - c[:-k]) / k
    return np.concatenate([y, np.zeros(len(x) - len(y))])


def noise_bed(t0, t1, k, vol, att=1.0, rel=1.0, shape=None):
    n = int((t1 - t0) * SR)
    x = np.arange(n) / SR
    s = lp_fast(rng.normal(0, 1, n), k)
    s /= np.abs(s).max() + 1e-9
    env = np.minimum(1, x / att) * np.minimum(1, (t1 - t0 - x) / rel).clip(0)
    if shape is not None:
        env = env * shape(x)
    i = int(t0 * SR)
    sfx[i:i + n] += (s * env * vol).astype(np.float32)


def boom(t, vol=0.8, f0=55, dur=3.0):
    n = int(dur * SR)
    x = np.arange(n) / SR
    f = f0 * (1 + 1.5 * np.exp(-x * 6))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 1.6) * (1 - np.exp(-x * 300))
    i = int(t * SR)
    sfx[i:i + n] += (s * vol)[: max(0, N - i)].astype(np.float32)


# ---------- 1. 童年：温暖的钢琴分解和弦 + 夏日虫鸣
s1_end = start["grow"]
prog = [("C4", "G4", "E5"), ("G3", "D4", "B4"), ("A3", "E4", "C5"), ("F3", "C4", "A4")]
beat = 0.62
t = 0.4
k = 0
while t < s1_end - 0.5:
    chord = prog[(k // 4) % 4]
    piano(music, t, chord[k % 3] if k % 4 != 3 else chord[1], 0.22 if k % 4 else 0.3)
    if k % 4 == 0:
        piano(music, t, chord[0][:-1] + str(int(chord[0][-1]) - 1), 0.18, 5)
    t += beat
    k += 1
pad(music, 0.0, s1_end + 0.5, ["C4", "E4", "G4"], 0.035, 2.0, 2.0)
# 虫鸣：高频调制的窄带噪声
n = int(s1_end * SR)
x = np.arange(n) / SR
cic = np.sin(2 * np.pi * 4300 * x) * (0.5 + 0.5 * np.sin(2 * np.pi * 37 * x)) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * x)) ** 2
sfx[:n] += (cic * 0.012 * np.minimum(1, x / 2) * np.minimum(1, (s1_end - x) / 1.5)).astype(np.float32)
# 按下手指时的一声闷响
press = L["ants"]["end"] - 1.4
boom(press, 0.25, 80, 0.8)

# ---------- 2. 长大 & 压力：转小调，低音脉冲越来越密
g0, p0, p1 = start["grow"], start["pressure"], start["dust"]
pad(music, g0, p0 + 0.5, ["A3", "C4", "E4"], 0.05, 1.0, 1.0)
for i, nm in enumerate(["E5", "C5", "A4", "E4"]):
    piano(music, g0 + 0.3 + i * 0.75, nm, 0.2, 3)
pad(music, p0, p1 + 0.3, ["D3", "F3", "G#3", "B3"], 0.07, 2.0, 0.3)   # 减七和弦，不安
t = p0 + 0.3
gap = 0.9
while t < p1 - 0.2:
    boom(t, 0.35, 50, 0.5)
    t += gap
    gap = max(0.32, gap * 0.9)
noise_bed(p0, p1, 60, 0.06, 2.0, 0.2, lambda x: (x / (p1 - p0)) ** 2)

# ---------- 3. 尘埃：近乎寂静，一点高音的闪光
d0 = start["dust"]
for i, nm in enumerate(["E6", "B5", "G#6"]):
    piano(music, d0 + 0.4 + i * 0.9, nm, 0.09, 3.5)
pad(music, d0, d0 + 3.2, ["E2"], 0.05, 1.0, 1.0)

# ---------- 4. 巨浪：落下的撞击、涌起的轰鸣、拍下
w0 = start["wave"]
hit = w0 + 1.0
crash = L["wave"]["end"] + 0.4
boom(hit, 0.9, 45, 4.0)
noise_bed(hit, crash + 0.3, 25, 0.55, 2.5, 0.05, lambda x: (x / (crash - hit)) ** 1.5)
noise_bed(crash, crash + 3.0, 6, 0.75, 0.02, 2.5)
boom(crash, 1.0, 38, 4.0)

# ---------- 5. 抬头：风声 + 稀疏的童年旋律（小调）
s0, f0 = start["sky"], start["finger"]
cut = L["finger"]["end"] + 0.4
noise_bed(s0, cut, 400, 0.08, 1.5, 0.1)
pad(music, s0, cut + 0.1, ["A3", "C4", "E4"], 0.045, 2.0, 0.1)
for i, nm in enumerate(["E5", "C5", "D5", "B4", "C5", "A4"]):
    piano(music, s0 + 0.6 + i * 1.1, nm, 0.2, 3.5)
# ---------- 6. 手指降临：低频压迫越来越强，然后一切戛然而止
noise_bed(f0, cut, 40, 0.5, 4.0, 0.02, lambda x: (x / (cut - f0)) ** 2.2)
pad(music, f0, cut, ["A1", "A2"], 0.25, 4.0, 0.02)
# 黑场后的结尾音
piano(music, cut + 1.0, "A3", 0.25, 5)
piano(music, cut + 1.0, "E4", 0.15, 5)
piano(music, cut + 1.0, "C5", 0.12, 5)

# ---------- 混音
w = wave.open(sys.argv[2])
vsr = w.getframerate()
v = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
v = np.interp(np.arange(int(len(v) * SR / vsr)) / SR, np.arange(len(v)) / vsr, v).astype(np.float32)
voice = np.zeros(N, np.float32)
voice[:len(v)] = v[:N]
# 旁白出现时压低背景（ducking）
env = np.zeros(N, np.float32)
for l in tl["lines"]:
    env[int((l["start"] - 0.2) * SR):int((l["end"] + 0.3) * SR)] = 1
env = lp_fast(env, int(0.25 * SR))
bed = (music * 0.8 + sfx * 0.85) * (1 - 0.6 * env)
mix = voice * 0.95 + bed
# 尾部淡出，裁切到总长
mix = mix[:int(T * SR)]
fade = int(1.0 * SR)
mix[-fade:] *= np.linspace(1, 0, fade)
mix = np.tanh(mix * 1.1) / np.tanh(1.1)    # 软限幅
mix = mix / np.abs(mix).max() * 0.95
st = np.stack([mix, mix], 1)
with wave.open(sys.argv[3], "wb") as o:
    o.setnchannels(2); o.setsampwidth(2); o.setframerate(SR)
    o.writeframes((st * 32767).astype(np.int16).tobytes())
print("ok", len(mix) / SR)
