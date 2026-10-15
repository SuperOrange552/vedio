"""逐帧程序化绘制画面（PIL + numpy），输出 PNG 序列到 build/frames。

用法: python3 src/render.py build/timeline.json build/frames [--workers 4] [--only t1,t2,...]
"""
import json, math, os, sys, random, argparse
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
S = 2                      # 超采样倍数，绘制在 2x 画布上再缩小以抗锯齿
CW, CH = W * S, H * S
FPS = 30
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"


# ---------------------------------------------------------------- 工具函数
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ramp(t, a, b):
    """t 在 [a,b] 内从 0 平滑到 1"""
    return smooth((t - a) / (b - a)) if b > a else float(t >= a)


def lerp(a, b, k):
    return a + (b - a) * k


def mix(c1, c2, k):
    return tuple(int(lerp(c1[i], c2[i], k)) for i in range(len(c1)))


def hexc(h, a=None):
    h = h.lstrip("#")
    c = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return c + (a,) if a is not None else c


def vgrad(top, bottom, w=CW, h=CH):
    k = np.linspace(0, 1, h)[:, None, None]
    arr = np.array(top, np.float32) * (1 - k) + np.array(bottom, np.float32) * k
    return Image.fromarray(np.broadcast_to(arr, (h, w, 3)).astype(np.uint8), "RGB")


def radial(cx, cy, r, color, alpha=1.0, w=CW, h=CH, power=2.0):
    """一个径向发光层（RGBA），在 1/4 分辨率计算后放大"""
    q = 4
    ys, xs = np.mgrid[0:h // q, 0:w // q].astype(np.float32)
    d = np.sqrt((xs - cx / q) ** 2 + (ys - cy / q) ** 2) / (r / q)
    a = np.clip(1 - d, 0, 1) ** power * alpha * 255
    arr = np.zeros((h // q, w // q, 4), np.uint8)
    arr[..., :3] = color
    arr[..., 3] = a.astype(np.uint8)
    return Image.fromarray(arr, "RGBA").resize((w, h), Image.BILINEAR)


def blend(a, b, k):
    if k <= 0:
        return a
    if k >= 1:
        return b
    return Image.blend(a, b, k)


def font(size):
    return ImageFont.truetype(FONT, size)


# ---------------------------------------------------------------- 预计算素材
rng = random.Random(7)

BG_SUMMER = vgrad(hexc("#f6c978"), hexc("#fdeccb"))
SUN_GLOW = radial(CW * 0.82, CH * 0.12, CH * 0.9, (255, 236, 180), 0.85)
BG_GREY = vgrad(hexc("#4a525c"), hexc("#868d96"))
BG_VOID = vgrad(hexc("#05070d"), hexc("#141b2b"))
BG_SEA = vgrad(hexc("#0d1420"), hexc("#2a3442"))
BG_OVERCAST = vgrad(hexc("#6f7f8f"), hexc("#c9c2b0"))

# 地面土壤纹理
_soil = np.zeros((CH, CW, 3), np.float32)
_soil[:] = hexc("#7d5f3d")
_n = np.random.default_rng(3).normal(0, 1, (CH // 8, CW // 8)).astype(np.float32)
_n = np.array(Image.fromarray(((_n * 25) + 128).clip(0, 255).astype(np.uint8)).resize((CW, CH), Image.BICUBIC), np.float32)
_soil += ((_n - 128) * 0.35)[..., None]
SOIL = Image.fromarray(_soil.clip(0, 255).astype(np.uint8))

# 暗角
_ys, _xs = np.mgrid[0:H, 0:W].astype(np.float32)
_d = np.sqrt(((_xs - W / 2) / (W / 2)) ** 2 + ((_ys - H / 2) / (H / 2)) ** 2)
VIGNETTE = np.clip(1 - 0.38 * np.clip(_d - 0.45, 0, None) ** 1.6, 0, 1)[..., None].astype(np.float32)

GROUND_Y = int(CH * 0.60)

# 草叶：(x, 高度, 弯曲, 颜色, 相位, 宽度)
def make_blades(n, y0, hmin, hmax, wmin, wmax, cols, seed):
    r = random.Random(seed)
    return [(r.uniform(-0.05, 1.05) * CW, r.uniform(hmin, hmax), r.uniform(-0.35, 0.35),
             hexc(r.choice(cols)), r.uniform(0, 6.28), r.uniform(wmin, wmax), y0 + r.uniform(-8, 8) * S)
            for _ in range(n)]


BACK_BLADES = make_blades(260, GROUND_Y, 30 * S, 110 * S, 5 * S, 10 * S, ["#6f9a3c", "#86a94a", "#5f8a35"], 1)
FRONT_BLADES = make_blades(26, CH + 10, 120 * S, 300 * S, 12 * S, 24 * S, ["#4f7d2a", "#3f6b22", "#5d8c30"], 2)
# 结尾“蚂蚁视角”下巨大的草叶
GIANT_BLADES = [(x * CW, h * CH, b, hexc(c), p, w * S, CH + 20)
                for x, h, b, c, p, w in [
                    (0.04, 1.25, 0.25, "#2f4d1c", 0.0, 120), (0.13, 0.95, 0.4, "#3b5e22", 1.0, 90),
                    (0.22, 0.70, 0.5, "#46692a", 2.0, 70), (0.80, 0.80, -0.45, "#46692a", 2.5, 75),
                    (0.90, 1.10, -0.3, "#3b5e22", 3.1, 105), (0.98, 1.35, -0.2, "#2f4d1c", 4.0, 130),
                    (0.30, 0.45, 0.6, "#557a33", 1.6, 45), (0.70, 0.50, -0.6, "#557a33", 0.7, 50)]]


def draw_blade(d, x, base_y, h, bend, col, w, sway):
    pts_l, pts_r = [], []
    n = 8
    for i in range(n + 1):
        k = i / n
        off = (bend + sway) * h * k * k
        ww = w * (1 - k) * 0.5
        y = base_y - h * k
        pts_l.append((x + off - ww, y))
        pts_r.append((x + off + ww, y))
    d.polygon(pts_l + pts_r[::-1], fill=col)


def draw_blades(d, blades, t, amp=0.06):
    for x, h, b, col, ph, w, y0 in blades:
        draw_blade(d, x, y0, h, b, col, w, amp * math.sin(t * 1.3 + ph))


def draw_ant(d, x, y, size, t, phase=0.0, col=(30, 18, 12), dead=False):
    """侧视的小蚂蚁，朝右"""
    s = size
    if dead:
        d.ellipse([x - s * 0.9, y - s * 0.12, x + s * 0.9, y + s * 0.18], fill=(55, 38, 26))
        return
    bob = math.sin(t * 18 + phase) * s * 0.03
    # 腿
    for i, lx in enumerate((-0.15, 0.05, 0.25)):
        sw = math.sin(t * 18 + phase + i * 2.1) * s * 0.22
        d.line([(x + lx * s, y + bob), (x + lx * s + sw - s * 0.15, y + s * 0.45)], fill=col, width=max(2, int(s * 0.07)))
        d.line([(x + lx * s, y + bob), (x + lx * s - sw + s * 0.15, y + s * 0.45)], fill=col, width=max(2, int(s * 0.07)))
    d.ellipse([x - s * 0.95, y - s * 0.28 + bob, x - s * 0.25, y + s * 0.22 + bob], fill=col)        # 腹部
    d.ellipse([x - s * 0.3, y - s * 0.15 + bob, x + s * 0.3, y + s * 0.13 + bob], fill=col)          # 胸部
    d.ellipse([x + s * 0.28, y - s * 0.24 + bob, x + s * 0.68, y + s * 0.12 + bob], fill=col)        # 头
    d.line([(x + s * 0.6, y - s * 0.15), (x + s * 0.85, y - s * 0.5), (x + s * 1.0, y - s * 0.45)], fill=col, width=max(2, int(s * 0.05)))


SKIN = hexc("#e7b08b")
SKIN_DARK = hexc("#c98b68")
NAIL = hexc("#f4d2c0")


def _finger_shape(d, cx, tip_y, r, col, top=-CH):
    d.rectangle([cx - r, top, cx + r, tip_y - r], fill=col)
    d.ellipse([cx - r, tip_y - 2 * r, cx + r, tip_y], fill=col)


def finger_layer(cx, tip_y, width, angle=0.0, prints=False, scale_detail=1.0):
    """从画面顶部伸下来的手指（RGBA 图层），tip_y 为指尖位置"""
    lay = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    r = width / 2
    _finger_shape(d, cx, tip_y, r, SKIN_DARK)                                # 背光侧
    _finger_shape(d, cx - r * 0.14, tip_y - r * 0.06, r * 0.84, SKIN)       # 受光面
    _finger_shape(d, cx - r * 0.42, tip_y - r * 0.5, r * 0.32, (241, 196, 166))  # 高光
    # 指节褶皱
    for k in (2.6, 2.75):
        yy = tip_y - r * k * 1.4
        d.arc([cx - r * 0.7, yy - r * 0.15, cx + r * 0.5, yy + r * 0.15], 200, 340, fill=SKIN_DARK, width=int(4 * S * scale_detail))
    if prints:
        for i in range(1, 9):
            rr = r * 0.12 * i
            d.arc([cx - rr * 1.1, tip_y - r * 1.05 - rr * 0.8, cx + rr * 0.9, tip_y - r * 1.05 + rr * 0.8], 20, 160,
                  fill=SKIN_DARK + (150,), width=int(3 * S * scale_detail))
    if angle:
        lay = lay.rotate(angle, center=(cx, tip_y), resample=Image.BICUBIC)
    return lay


def finger_back_layer(cx, tip_y, width):
    """从上往下看到指甲的一侧（用于结尾从天而降的手指）"""
    lay = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    r = width / 2
    _finger_shape(d, cx, tip_y, r, SKIN_DARK)
    _finger_shape(d, cx, tip_y - r * 0.05, r * 0.86, SKIN)
    _finger_shape(d, cx - r * 0.3, tip_y - r * 0.35, r * 0.4, (240, 192, 160))
    nr = r * 0.6
    ny = tip_y - r * 0.32
    _finger_shape(d, cx, ny, nr, (226, 170, 150), top=ny - nr * 3.2)       # 甲沟
    _finger_shape(d, cx, ny - nr * 0.06, nr * 0.92, NAIL, top=ny - nr * 3.0)
    d.ellipse([cx - nr * 0.92, ny - nr * 3.3, cx + nr * 0.92, ny - nr * 2.7], fill=SKIN)   # 甲根
    d.ellipse([cx - nr * 0.55, ny - nr * 1.5, cx + nr * 0.1, ny - nr * 0.5], fill=(252, 236, 226))   # 光泽
    return lay


def person(d, x, y, h, col, look_up=0.0, hunch=0.0, child=0.0):
    """简笔人物剪影，(x,y) 为脚底，h 为身高"""
    head_r = h * (0.085 + 0.03 * child)
    neck_y = y - h + head_r * 2.2
    hip_y = y - h * 0.47
    sh_x = x + hunch * h * 0.12
    # 腿
    lw = max(2, int(h * 0.075))
    d.line([(x - h * 0.02, hip_y), (x - h * 0.07, y)], fill=col, width=lw)
    d.line([(x + h * 0.02, hip_y), (x + h * 0.07, y)], fill=col, width=lw)
    # 躯干
    d.polygon([(sh_x - h * 0.1, neck_y + head_r * 0.2), (sh_x + h * 0.1, neck_y + head_r * 0.2),
               (x + h * 0.075, hip_y + h * 0.02), (x - h * 0.075, hip_y + h * 0.02)], fill=col)
    # 手臂
    aw = max(2, int(h * 0.055))
    d.line([(sh_x - h * 0.09, neck_y + head_r * 0.4), (x - h * 0.13 + hunch * h * 0.08, hip_y + h * 0.04)], fill=col, width=aw)
    d.line([(sh_x + h * 0.09, neck_y + head_r * 0.4), (x + h * 0.13 + hunch * h * 0.08, hip_y + h * 0.04)], fill=col, width=aw)
    # 头
    hx = sh_x + hunch * h * 0.06
    hy = neck_y - head_r * 0.9 + hunch * h * 0.05 - look_up * head_r * 0.15
    d.ellipse([hx - head_r, hy - head_r, hx + head_r, hy + head_r], fill=col)
    if look_up:   # 抬头：脸部朝上，画一点下巴轮廓
        d.polygon([(hx - head_r * 0.2, hy + head_r * 0.6), (hx + head_r * 0.9, hy - head_r * 0.2 - look_up * head_r * 0.6),
                   (hx + head_r * 0.6, hy + head_r * 0.7)], fill=col)


# ---------------------------------------------------------------- 各场景
ANTS = [(-0.30 + i * 0.09, rng.uniform(0, 6.28)) for i in range(9)]   # (起始相对位置, 相位)
FINGER_X = CW * 0.52
ANT_Y = CH * 0.76


def scene_ants(t, L):
    """夏日午后，蚂蚁排队爬过；手指按下"""
    img = BG_SUMMER.copy()
    img.paste(SUN_GLOW, (0, 0), SUN_GLOW)
    d = ImageDraw.Draw(img)
    # 远处树影
    for i in range(14):
        x = (i / 13) * CW
        rr = (90 + 40 * math.sin(i * 1.7)) * S
        d.ellipse([x - rr, GROUND_Y - rr * 1.1, x + rr, GROUND_Y + rr * 0.4], fill=(168, 176, 108))
    img.paste(SOIL.crop((0, 0, CW, CH - GROUND_Y)), (0, GROUND_Y))
    d = ImageDraw.Draw(img)
    draw_blades(d, BACK_BLADES, t)

    press_down, press_up = L["press"], L["press"] + 1.6
    tip = lerp(-CH * 0.1, ANT_Y + 14 * S, ramp(t, press_down - 0.9, press_down)) if t < press_up else \
        lerp(ANT_Y + 14 * S, -CH * 0.2, ramp(t, press_up, press_up + 1.1))
    near = ramp(t, press_down - 1.3, press_down)
    # 手指投下的阴影
    if t < press_up + 1.1:
        sh = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        k = near if t < press_up else 1 - ramp(t, press_up, press_up + 1.1)
        rw = lerp(420, 150, k) * S
        sd.ellipse([FINGER_X - rw, ANT_Y - rw * 0.18 + 10 * S, FINGER_X + rw, ANT_Y + rw * 0.18 + 10 * S], fill=(40, 25, 10, int(110 * k)))
        sh = sh.filter(ImageFilter.GaussianBlur(20 * S))
        img.paste(sh, (0, 0), sh)
        d = ImageDraw.Draw(img)

    speed = 0.035  # 屏宽/秒
    for i, (x0, ph) in enumerate(ANTS):
        x = (x0 + speed * min(t, press_down)) * CW
        crushed = abs(x - FINGER_X) < 140 * S and t >= press_down
        if crushed:
            draw_ant(d, x, ANT_Y + 8 * S, 50 * S, t, ph, dead=True)
            continue
        if t > press_down:  # 惊散
            dir_ = 1 if x > FINGER_X else -1
            x += dir_ * (t - press_down) * 0.12 * CW
        draw_ant(d, x, ANT_Y, 50 * S, t if t < press_down else t * 1.8, ph)
    # 被碾后的草汁痕迹
    if t >= press_down:
        k = ramp(t, press_down, press_down + 0.5)
        lay = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lay)
        ld.ellipse([FINGER_X - 130 * S, ANT_Y - 2 * S, FINGER_X + 130 * S, ANT_Y + 26 * S], fill=(90, 120, 40, int(90 * k)))
        lay = lay.filter(ImageFilter.GaussianBlur(6 * S))
        img.paste(lay, (0, 0), lay)
    if tip > -CH * 0.05:
        twist = math.sin((t - press_down) * 9) * 4 if press_down <= t < press_up else 0
        fl = finger_layer(FINGER_X, tip, 260 * S, angle=twist)
        img.paste(fl, (0, 0), fl)
    d = ImageDraw.Draw(img)
    draw_blades(d, FRONT_BLADES, t)
    return img


WISPS = [(rng.uniform(-1, 1), rng.uniform(0, 1), rng.uniform(0.6, 1.4), rng.uniform(0, 6.28)) for _ in range(46)]


def draw_wisps(img, t, cx, cy, spread, rise, alpha, size=1.0, color=(150, 200, 90)):
    """草腥味：上升的淡绿色烟丝"""
    if alpha <= 0.01:
        return img
    lay = Image.new("RGBA", (CW // 2, CH // 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for sx, off, sp, ph in WISPS:
        k = ((t * 0.18 * sp) + off) % 1.0
        x = (cx + sx * spread * (0.3 + k) + math.sin(t * 1.2 + ph + k * 6) * 40 * S * size) / 2
        y = (cy - k * rise) / 2
        r = (10 + 50 * k) * S * size / 2
        a = int(alpha * 120 * math.sin(math.pi * k))
        d.ellipse([x - r, y - r * 0.6, x + r, y + r * 0.6], fill=color + (a,))
    lay = lay.filter(ImageFilter.GaussianBlur(14)).resize((CW, CH), Image.BILINEAR)
    img.paste(lay, (0, 0), lay)
    return img


BOKEH = [(rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(30, 120), rng.uniform(0, 6.28)) for _ in range(22)]


def scene_scent(t, L):
    """指尖特写，淡绿的气味升起"""
    img = vgrad(hexc("#f3d39a"), hexc("#d9b27a"))
    lay = Image.new("RGBA", (CW // 4, CH // 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for x, y, r, ph in BOKEH:
        xx, yy = (x * CW + math.sin(t * 0.3 + ph) * 30 * S) / 4, (y * CH) / 4
        rr = r * S / 4
        d.ellipse([xx - rr, yy - rr, xx + rr, yy + rr], fill=(255, 245, 210, 70))
    lay = lay.filter(ImageFilter.GaussianBlur(6)).resize((CW, CH), Image.BILINEAR)
    img.paste(lay, (0, 0), lay)
    # 手指从下方伸上来（把向下的手指旋转 180°）
    zoom = 1 + 0.06 * t / 5
    tip_y = CH * 0.42
    fl = finger_layer(CW / 2, CH - tip_y + CH * 0.0, 700 * S * zoom, prints=True, scale_detail=2.2).rotate(180)
    img.paste(fl, (0, 0), fl)
    d = ImageDraw.Draw(img)
    # 指尖上的一点污痕
    cx, cy = CW / 2 - 40 * S, tip_y + 150 * S
    d.ellipse([cx - 60 * S, cy - 18 * S, cx + 70 * S, cy + 22 * S], fill=(118, 112, 60))
    d.ellipse([cx - 20 * S, cy - 8 * S, cx + 30 * S, cy + 12 * S], fill=(80, 70, 40))
    img = draw_wisps(img, t, CW / 2, tip_y + 120 * S, 200 * S, 520 * S, ramp(t, 0.3, 1.8), size=1.2)
    return img


def scene_grow(t, L):
    """时间流逝：日升日落，人慢慢长大，颜色褪去"""
    dur = L["dur"]
    k = ramp(t, 0, dur)
    warm = BG_SUMMER
    img = blend(warm, BG_GREY, k)
    d = ImageDraw.Draw(img)
    # 日月轨迹
    for i in range(3):
        p = (t / dur * 3 - i)
        if 0 <= p <= 1:
            ang = math.pi * p
            x = CW * 0.5 - math.cos(ang) * CW * 0.45
            y = GROUND_Y - math.sin(ang) * CH * 0.5
            c = mix((255, 220, 140), (220, 225, 235), k)
            r = 50 * S
            d.ellipse([x - r, y - r, x + r, y + r], fill=c)
    d.rectangle([0, GROUND_Y, CW, CH], fill=mix(hexc("#7d5f3d"), hexc("#3d4249"), k))
    h = lerp(220, 520, smooth(k)) * S
    col = mix((60, 40, 25), (25, 28, 33), k)
    person(d, CW / 2, GROUND_Y + 160 * S, h, col, child=1 - k)
    d.rectangle([0, GROUND_Y + 160 * S, CW, CH], fill=mix(hexc("#6a4f33"), hexc("#30353b"), k))
    return img


BUILDINGS = []
_x = -40 * S
_r = random.Random(11)
while _x < CW:
    bw = _r.uniform(90, 220) * S
    BUILDINGS.append((_x, bw, _r.uniform(0.25, 0.7) * CH, _r.random()))
    _x += bw + _r.uniform(-10, 20) * S

WORDS = ["考试", "升学", "房租", "房贷", "加班", "KPI", "¥", "35岁", "裁员", "物价", "排名", "绩效", "贷款", "内卷", "就业", "%"]
WORD_DROPS = [(w, random.Random(i).uniform(0.08, 0.92), 0.25 + i * 0.33, random.Random(i + 50).uniform(-20, 20),
               random.Random(i + 99).choice([110, 140, 170, 200]))
              for i, w in enumerate(WORDS)]
PAPERS = [(rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.6, 1.4), rng.uniform(0, 6.28)) for _ in range(40)]


def scene_pressure(t, L):
    img = BG_GREY.copy()
    d = ImageDraw.Draw(img)
    base = int(CH * 0.86)
    for x, bw, bh, s in BUILDINGS:
        c = int(lerp(48, 70, s))
        d.rectangle([x, base - bh, x + bw, base], fill=(c, c + 4, c + 10))
        wy = base - bh + 25 * S
        while wy < base - 30 * S:
            wx = x + 18 * S
            while wx < x + bw - 30 * S:
                lit = (hash((int(x), int(wx), int(wy))) % 7) == 0
                d.rectangle([wx, wy, wx + 16 * S, wy + 22 * S], fill=(200, 190, 140) if lit else (c + 12, c + 16, c + 22))
                wx += 36 * S
            wy += 46 * S
    d.rectangle([0, base, CW, CH], fill=(36, 40, 46))
    # 漫天飘落的试卷/票据
    for x0, off, sp, ph in PAPERS:
        y = ((t * 0.12 * sp + off) % 1.2 - 0.1) * CH
        x = x0 * CW + math.sin(t * 1.5 * sp + ph) * 60 * S
        ang = math.sin(t * 2 * sp + ph) * 50
        pw, ph_ = 70 * S, 92 * S
        p = Image.new("RGBA", (int(pw) + 4, int(ph_) + 4), (0, 0, 0, 0))
        pd = ImageDraw.Draw(p)
        pd.rectangle([2, 2, pw, ph_], fill=(225, 225, 218, 210))
        for j in range(5):
            pd.line([(12 * S, 18 * S + j * 14 * S), (pw - 12 * S, 18 * S + j * 14 * S)], fill=(150, 150, 150, 210), width=2 * S)
        p = p.rotate(ang, expand=True, resample=Image.BICUBIC)
        img.paste(p, (int(x), int(y)), p)
    d = ImageDraw.Draw(img)
    # 人越来越弯
    hunch = ramp(t, 1, L["dur"])
    person(d, CW / 2, base, 300 * S * (1 - 0.12 * hunch), (18, 20, 24), hunch=hunch)
    # 砸落的字
    pile = {}
    for w, xr, t0, rot, size in WORD_DROPS:
        if t < t0:
            continue
        f = font(size * S)
        fall = (t - t0)
        y = -size * S + 0.5 * 9.8 * 260 * S * fall * fall * 0.35
        x = xr * CW
        bucket = int(x // (240 * S))
        land = base - pile.get(bucket, 0) - size * S * 1.1
        landed = y >= land
        if landed:
            y = land
            pile[bucket] = pile.get(bucket, 0) + size * S * 0.9
        tl = Image.new("RGBA", (int(size * S * 2.6), int(size * S * 1.4)), (0, 0, 0, 0))
        ImageDraw.Draw(tl).text((6, 0), w, font=f, fill=(196, 52, 46, 235))
        tl = tl.rotate(rot if landed else rot * math.sin(fall * 3), expand=True, resample=Image.BICUBIC)
        img.paste(tl, (int(x - tl.width / 2), int(y)), tl)
    # 屏幕轻微震动在 merge 阶段做
    return img


MOTES = [(rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.3, 1.0)) for _ in range(120)]


def draw_dust(img, x, y, r, alpha=1.0):
    g = radial(x, y, r * 9, (255, 230, 170), 0.9 * alpha)
    img.paste(g, (0, 0), g)
    d = ImageDraw.Draw(img)
    d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 248, 225))
    return img


def scene_dust(t, L):
    img = BG_VOID.copy()
    d = ImageDraw.Draw(img)
    for x, y, s in MOTES:   # 背景尘埃上升 → 制造“下坠”的感觉
        yy = ((y - t * 0.08 * s) % 1.0) * CH
        r = s * 3 * S
        c = int(60 + 80 * s)
        d.ellipse([x * CW - r, yy - r, x * CW + r, yy + r], fill=(c, c, int(c * 1.1)))
    x = CW / 2 + math.sin(t * 1.4) * 80 * S
    y = CH * 0.42 + math.sin(t * 0.9) * 20 * S
    return draw_dust(img, x, y, 7 * S)


WAVE_FACE = vgrad(hexc("#4f7590"), hexc("#0b1520"), h=CH)
FOAM = [(random.Random(i).random(), random.Random(i + 7).random(), random.Random(i + 13).random()) for i in range(420)]


def wave_edge(A, crest_x, t):
    """浪的上沿：返回 [(x, y)]"""
    base = CH * 0.80
    pts = []
    for i in range(0, 161):
        x = CW * i / 160
        dx = (x - crest_x) / CW
        hgt = A * math.exp(-(dx * 2.0) ** 2) + A * 0.16 * (1 + math.sin(x / CW * 9 + t * 2)) / 2
        hgt += (8 + 0.02 * A / S) * S * math.sin(x / CW * 37 + t * 5) + 5 * S * math.sin(x / CW * 83 - t * 7)
        pts.append((x, base - hgt))
    return pts


def draw_foam(d, pts, t, density=1.0, spread=1.0):
    n = len(pts)
    for k, (u, v, w) in enumerate(FOAM):
        if u > density:
            continue
        j = int(v * (n - 1))
        x, y = pts[j]
        x += (w - 0.5) * 30 * S * spread + math.sin(t * 6 + k) * 4 * S
        y += (abs(math.sin(k * 3.7)) - 0.25) * 26 * S * spread
        rr = (1.5 + 5 * w * w) * S
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(215, 228, 235, int(90 + 140 * u)))


def scene_wave(t, L):
    img = BG_SEA.copy()
    d = ImageDraw.Draw(img)
    base = CH * 0.80
    d.rectangle([0, base, CW, CH], fill=(16, 20, 26))
    px, py = CW * 0.5, base
    hit = 1.0
    # 尘埃落下
    if t < hit + 0.15:
        k = ramp(t, 0, hit)
        dy = lerp(-CH * 0.05, py - 70 * S, k * k)
        img = draw_dust(img, px + math.sin(t * 3) * 20 * S * (1 - k), dy, 7 * S)
        d = ImageDraw.Draw(img)
    # 浪
    A = lerp(0, CH * 1.05, ramp(t, hit + 0.1, hit + 3.6) ** 1.3)
    if A > 1:
        crest_x = px - CW * 0.1
        edge = wave_edge(A, crest_x, t)
        mask = Image.new("L", (CW, CH), 0)
        md = ImageDraw.Draw(mask)
        md.polygon([(0, CH)] + edge + [(CW, CH)], fill=255)
        # 卷起的浪头（向人物一侧翻卷）
        R = 0.22 * A
        cy_top = base - A * 1.28
        ccx, ccy = crest_x + R * 0.75, cy_top + R
        outer = [(ccx + R * math.cos(math.radians(a)), ccy + R * math.sin(math.radians(a))) for a in range(170, 411, 6)]
        inner = [(ccx + 0.82 * R + 0.45 * R * math.cos(math.radians(a)), ccy + R * 0.55 + 0.45 * R * math.sin(math.radians(a)))
                 for a in range(410, 169, -8)]
        if R > 10 * S:
            md.polygon(outer + inner, fill=255)
        img.paste(WAVE_FACE, (0, 0), mask)
        wl = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
        wd = ImageDraw.Draw(wl)
        draw_foam(wd, edge, t, 0.75)
        if R > 10 * S:
            draw_foam(wd, outer, t, 0.6, 0.6)
            # 浪头飞沫
            for k, (u, v, w) in enumerate(FOAM[:120]):
                life = (t * 0.7 + u) % 1.0
                x = outer[-1][0] + (v - 0.3) * R * 1.2 + life * 120 * S
                y = outer[-1][1] - R * 0.2 + life * life * 260 * S - w * R * 0.5
                rr = (1 + 3 * w) * S
                wd.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(220, 232, 240, int(200 * (1 - life))))
        img.paste(wl, (0, 0), wl)
        d = ImageDraw.Draw(img)
    person(d, px, py, 70 * S, (5, 6, 8), hunch=0.0)
    if hit <= t < hit + 0.5:   # 撞击闪光
        g = radial(px, py - 50 * S, CH * 0.6, (255, 240, 200), 1 - ramp(t, hit, hit + 0.5))
        img.paste(g, (0, 0), g)
    # 浪拍下来：整个画面被吞没
    crash = L["crash"]
    if t > crash:
        k = ramp(t, crash, crash + 0.6)
        yb = lerp(-CH * 0.3, CH * 1.15, k)
        pts = [(CW * i / 160, yb + 50 * S * math.sin(i * 0.12 + t * 4) + 18 * S * math.sin(i * 0.37 - t * 6)) for i in range(161)]
        cl = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
        cd = ImageDraw.Draw(cl)
        cd.polygon([(0, -10)] + pts + [(CW, -10)], fill=(12, 20, 30, 255))
        draw_foam(cd, pts, t, 1.0, 1.4)
        img.paste(cl, (0, 0), cl)
        img = blend(img, Image.new("RGB", (CW, CH), (0, 0, 0)), ramp(t, crash + 0.6, crash + 1.6))
    return img


def ant_view(t, L, img=None, tilt=0.0):
    """蚂蚁视角：巨大的草叶，阴天"""
    img = BG_OVERCAST.copy() if img is None else img
    g = radial(CW * 0.55, CH * 0.05, CH * 1.0, (255, 250, 230), 0.55)
    img.paste(g, (0, 0), g)
    d = ImageDraw.Draw(img)
    gy = CH * 0.74 + tilt
    d.rectangle([0, gy, CW, CH], fill=(70, 56, 40))
    for x, h, b, col, ph, w, y0 in GIANT_BLADES:
        draw_blade(d, x, y0 + tilt, h, b, col, w, 0.03 * math.sin(t * 0.8 + ph))
    return img, d, gy


def scene_sky(t, L):
    tilt = lerp(0, 60 * S, ramp(t, 0, L["dur"]))
    img, d, gy = ant_view(t, L, tilt=tilt)
    look = ramp(t, 0.6, 1.8)
    person(d, CW * 0.5, gy + 6 * S, 120 * S, (20, 16, 12), look_up=look)
    return img


def scene_finger(t, L):
    tilt = 60 * S
    darken = ramp(t, 0, L["cut"])
    img, d, gy = ant_view(t, L, tilt=tilt)
    # 巨大的阴影笼罩
    sh = Image.new("RGBA", (CW, CH), (10, 8, 6, int(170 * darken)))
    img.paste(sh, (0, 0), sh)
    d = ImageDraw.Draw(img)
    person(d, CW * 0.5, gy + 6 * S, 120 * S, (15, 12, 10), look_up=1.0)
    tip = lerp(-CH * 0.05, gy - 170 * S, ramp(t, 0.4, L["cut"]) ** 1.15)
    img = draw_wisps(img, t, CW * 0.5, tip - 40 * S, 420 * S, -(gy - tip), 0.8 * ramp(t, 1.5, 3.0) * (1 - ramp(t, L["cut"] - 0.2, L["cut"])), size=1.6)
    fl = finger_back_layer(CW * 0.5, tip, CW * 0.62)
    img.paste(fl, (0, 0), fl)
    if t >= L["cut"]:   # 黑场
        img = Image.new("RGB", (CW, CH), (0, 0, 0))
        k = ramp(t, L["cut"] + 1.0, L["cut"] + 2.0) * (1 - ramp(t, L["end"] - 1.2, L["end"] - 0.2))
        if k > 0:
            d = ImageDraw.Draw(img)
            f = font(96 * S)
            txt = "草 腥 味"
            tw = d.textlength(txt, font=f)
            c = int(225 * k)
            d.text(((CW - tw) / 2, CH * 0.42), txt, font=f, fill=(int(c * 0.82), c, int(c * 0.66)))
    return img


SCENES = {"ants": scene_ants, "scent": scene_scent, "grow": scene_grow, "pressure": scene_pressure,
          "dust": scene_dust, "wave": scene_wave, "sky": scene_sky, "finger": scene_finger}


# ---------------------------------------------------------------- 时间轴
def build_plan(tl):
    lines, total = tl["lines"], tl["total"]
    order = [l["scene"] for l in lines]
    starts = [0.0] + [lines[i]["start"] - 0.5 for i in range(1, len(lines))]
    ends = starts[1:] + [total]
    plan = []
    for sc, a, b, ln in zip(order, starts, ends, lines):
        L = {"start": a, "end": b, "dur": b - a, "line": ln}
        if sc == "ants":
            L["press"] = ln["end"] - a - 1.4
        if sc == "wave":
            L["crash"] = ln["end"] - a + 0.4
            L["end"] = b - a
        if sc == "finger":
            L["cut"] = ln["end"] - a + 0.4
            L["end"] = b - a
        plan.append((sc, L))
    return plan


XFADE = 0.5
HARD_CUTS = {"dust", "sky"}   # 这些场景硬切进入（尘埃从黑暗中出现 / 浪后的寂静）


def render_scene(sc, L, t):
    return SCENES[sc](max(t - L["start"], 0), L)


def subtitle(img, t, lines):
    for l in lines:
        if l["start"] - 0.15 <= t <= l["end"] + 0.35:
            a = ramp(t, l["start"] - 0.15, l["start"] + 0.15) * (1 - ramp(t, l["end"] + 0.05, l["end"] + 0.35))
            f = font(50)
            d = ImageDraw.Draw(img, "RGBA")
            txt = l["text"]
            tw = d.textlength(txt, font=f)
            x, y = (W - tw) / 2, H - 120
            for dx, dy in ((2, 2), (-2, 2), (2, -2), (-2, -2), (0, 3)):
                d.text((x + dx, y + dy), txt, font=f, fill=(0, 0, 0, int(160 * a)))
            d.text((x, y), txt, font=f, fill=(250, 248, 240, int(255 * a)))
    return img


def frame(i, plan, tl):
    t = i / FPS
    idx = max(k for k, (_, L) in enumerate(plan) if L["start"] <= t or k == 0)
    sc, L = plan[idx]
    img = render_scene(sc, L, t)
    # 与下一场景交叉淡化
    if idx + 1 < len(plan):
        nsc, nL = plan[idx + 1]
        if nsc not in HARD_CUTS and t > nL["start"] - XFADE:
            img = blend(img, render_scene(nsc, nL, t), ramp(t, nL["start"] - XFADE, nL["start"]))
    # 片头淡入
    img = blend(Image.new("RGB", (CW, CH)), img, ramp(t, 0, 1.2))
    img = img.reduce(S)
    # 震屏
    shake = 0.0
    if sc == "pressure":
        shake = 4 * ramp(t - L["start"], 2, L["dur"])
    if sc == "wave":
        lt = t - L["start"]
        shake = 14 * (1 - ramp(lt, 1.0, 1.6)) * (lt > 1.0) + 10 * ramp(lt, 2.0, 4.5) * (1 - ramp(lt, L["crash"] + 0.4, L["crash"] + 1.2))
    if shake > 0.3:
        r = random.Random(i)
        img = img.transform(img.size, Image.AFFINE, (1, 0, r.uniform(-shake, shake), 0, 1, r.uniform(-shake, shake)),
                            resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        img = img.crop((12, 12, W - 12, H - 12)).resize((W, H), Image.BILINEAR)
    # 暗角 + 胶片颗粒
    arr = np.asarray(img, np.float32) * VIGNETTE
    arr += np.random.default_rng(i).normal(0, 5, (H, W, 1)).astype(np.float32)
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return subtitle(img, t, tl["lines"])


_G = {}


def _init(tlp):
    tl = json.load(open(tlp))
    _G["tl"], _G["plan"] = tl, build_plan(tl)


def _work(args):
    i, out = args
    frame(i, _G["plan"], _G["tl"]).save(f"{out}/{i:05d}.png", compress_level=1)
    return i


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tl = json.load(open(a.timeline))
    n = int(math.ceil(tl["total"] * FPS))
    ids = [int(float(x) * FPS) for x in a.only.split(",")] if a.only else list(range(n))
    with Pool(a.workers, initializer=_init, initargs=(a.timeline,)) as p:
        for k, _ in enumerate(p.imap_unordered(_work, [(i, a.out) for i in ids], chunksize=4)):
            if k % 100 == 0:
                print(f"{k}/{len(ids)}", flush=True)
