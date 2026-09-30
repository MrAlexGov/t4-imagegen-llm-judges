"""Собирает compare/*.jpg из data/images/<model>/ и data/generators.json."""
import json, os, sys
D = sys.argv[1] if len(sys.argv) > 1 else "data"
from PIL import Image, ImageDraw, ImageFont
P = json.load(open(f"{D}/prompts.json"))
TIM = json.load(open(f"{D}/generators.json"))
ORDER = [("sdxl", "SDXL"), ("sd35m", "SD 3.5 Medium"), ("flux-schnell", "FLUX-schnell"), ("flux-dev", "FLUX-dev")]
M = [(m, lab, TIM[m]) for m, lab in ORDER if m in TIM]
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
S = 512
for p in P:
    c = Image.new("RGB", (len(M) * (S + 10) + 10, S + 90), "white"); d = ImageDraw.Draw(c)
    d.text((10, 10), f"{p['id']}: {p['tests']}", fill="#333", font=F)
    for j, (m, lab, t) in enumerate(M):
        x = 10 + j * (S + 10)
        c.paste(Image.open(f"{D}/images/{m}/{p['id']}.png").convert("RGB").resize((S, S)), (x, 80))
        d.text((x, 48), f"{lab} · {t['per_image_s_avg_rest']:.0f} c", fill="black", font=FB)
    c.save(f"compare/{p['id']}.jpg", quality=90)
T = 256; g = Image.new("RGB", (len(M) * (T + 10) + 10, len(P) * (T + 10) + 50), "white"); d = ImageDraw.Draw(g)
for j, (m, lab, _) in enumerate(M): d.text((10 + j * (T + 10), 12), lab, fill="black", font=F)
for i, p in enumerate(P):
    for j, (m, _, _) in enumerate(M):
        g.paste(Image.open(f"{D}/images/{m}/{p['id']}.png").convert("RGB").resize((T, T)), (10 + j * (T + 10), 50 + i * (T + 10)))
g.save("compare/00_overview.jpg", quality=85)
print("models:", [m for m, _, _ in M])
