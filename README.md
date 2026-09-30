# Four image generators on free T4 GPUs and five LLM judges: who judges whom

*Alexey Govorukhin · September 2026*

*A pilot study: SDXL, SD 3.5 Medium, FLUX.1-schnell and FLUX.1-dev on Kaggle (2×T4, free tier), rated blind by a
human and by LLM judges from the Kaggle Benchmarks quota. Code, images and every rating are open.*


**Links:** dataset — [Hugging Face](https://huggingface.co/datasets/MrAlexGov/t4-imagegen-llm-judges) · notebook — [Kaggle](https://www.kaggle.com/code/mralexgov/t4-image-generators-vs-llm-judges) · [Русская версия](README.ru.md)

## TL;DR

- **Speed on a T4** varies 42×: SDXL takes 27 s per 1024×1024 image, SD 3.5 Medium 92 s, FLUX-schnell 149 s,
  FLUX-dev 19 minutes.
- **The human** ranks FLUX-dev first on quality (9.2/10), then SD 3.5 Medium (7.9), FLUX-schnell (7.3), and SDXL far
  behind (5.7). On checkable prompt requirements the three newer models tie (~89%), SDXL gets 60%.
- **LLM judges** clearly see that SDXL is the weakest, but **none of them sees FLUX-dev's lead**: all four
  vision-capable judges put it level with SD 3.5 and FLUX-schnell, or below.
- On checklist items, **Claude Sonnet 5** and **Gemini 3.1 Flash Lite** agree with the human best (88%) — but a stub
  that answers "yes" to everything already gets 81%, so the real gain is about 7 points.
- On subjective quality, **Gemini 3 Flash** is closest to the human (ρ = 0.55); the others are at 0.28–0.42.
- **A "blind" control judge** (Qwen3-Next, which does not accept images, yet the API does not refuse) confidently
  gives any image about 8/10 and agrees with itself 94% of the time. A consistent judge is not necessarily a correct one.

## How this started

I am looking for an AI engineer role and build pet projects to close gaps I see in job postings. One of them predicts
the skills required for an IT job title ([it-job-title-to-skills](https://github.com/MrAlexGov/it-job-title-to-skills)).
It needed a GPU, so I set up the free Kaggle tier: 2×T4, about 30 hours a week, jobs pushed through the API
(`kaggle kernels push`). The quota page had a second, unexpected line — **AI: $10 per day** for
[Kaggle Benchmarks](https://www.kaggle.com/benchmarks), i.e. access to LLMs (Claude, Gemini, GPT, DeepSeek, Qwen) for
evaluating models. These models cannot generate images, but they can look at them and judge. Hence two questions:
what can free T4s really do for image generation, and can we trust an LLM to grade the result?

## Generators

| Model | Setup on T4 | Steps | Time / image | Load | Peak VRAM | License |
|---|---|---|---|---|---|---|
| SDXL base 1.0 | fp16 | 30 | **27 s** | 77 s | 10.5 GB | CreativeML Open RAIL++-M |
| SD 3.5 Medium | fp16, T5 in NF4, CPU offload | 40 | 92 s | 123 s | **7.7 GB** | Stability Community (free under $1M revenue) |
| FLUX.1-schnell | NF4 + bf16, CPU offload | 4 | 149 s | 284 s | 11.3 GB | Apache-2.0 |
| FLUX.1-dev | NF4 + bf16, CPU offload | 28 | 1145 s | 311 s | 11.8 GB | non-commercial |

Lessons learned the hard way:
- FLUX does not fit a T4 in fp16; in NF4 on one card it OOMs in attention, because the T4 has no native bf16 and no
  memory-efficient SDPA kernel for bf16. `enable_model_cpu_offload()` plus VAE tiling fixed it.
- The FLUX bottleneck is the denoising step itself, not offloading: ~30 s per step for schnell and ~40 s for dev
  (longer T5 context). Splitting components across the two GPUs would save only ~3% for dev.
- FLUX and SD 3.5 are gated on Hugging Face: accept the license and pass a read-only token to Kaggle (I kept it in a
  private Kaggle dataset).

The 10 prompts probe typical weaknesses: photorealism, **text rendering** (an "OPEN 24/7" neon sign, an
"AI AGENT / PLAN / ACT / CHECK" infographic), a pianist's hands, composition ("a red cube on top of a blue sphere,
a green cone to the left"), a landscape, a flat vector illustration, a product shot, anime and food. One seed (42)
for everything.

![Neon sign](compare/02_text_sign.jpg)
![Infographic](compare/10_infographic.jpg)

## How images were rated

Each prompt has a checklist of 4–5 verifiable items (e.g. "the text reads exactly "OPEN 24/7" with no extra
characters", "the green cone is to the left of the cube and sphere") plus two 1–10 scores: quality/aesthetics and
absence of artifacts.

- **The human** (the author, Alexey Govorukhin) rated all 40 images blind on a dedicated page: the model is hidden,
  file names are anonymized, the order is shuffled.
- **LLM judges** from the Kaggle Benchmarks quota got the same image (768 px), the prompt and the checklist, and
  answered in structured JSON. Every image was rated twice (temperature 0 and 1), in random order. Of the 8 available
  models, 4 accept images: Claude Sonnet 5, Gemini 3 Flash, Gemini 3.1 Flash Lite, GPT-5.4 nano. Qwen3-Next-80B was
  added as a **control**: it cannot see the image, but it does not refuse either.

## Results

### Generators through the eyes of the human and the judges

Share of checklist items passed / mean quality score.

| | SDXL | SD 3.5 Medium | FLUX-schnell | FLUX-dev |
|---|---|---|---|---|
| **Human** | 60% / 5.7 | 89% / 7.9 | 89% / 7.3 | **89% / 9.2** |
| Claude Sonnet 5 | 65% / 7.3 | **84%** / 7.7 | 81% / 7.8 | 78% / **7.8** |
| Gemini 3 Flash | 60% / 6.1 | **80%** / 7.1 | 78% / 7.0 | 75% / **7.4** |
| Gemini 3.1 Flash Lite | 65% / 7.3 | 77% / 8.1 | **84%** / 8.2 | 82% / **8.3** |
| GPT-5.4 nano | 68% / 7.5 | **88% / 8.3** | 86% / 8.2 | 76% / 7.9 |
| Qwen3-Next (blind) | 70% / 8.1 | 72% / 8.0 | 70% / 7.9 | 71% / 8.1 |

The human puts FLUX-dev 1.3–1.9 points above the rest on quality. For the judges, the gap between the best and worst
of the three newer models is at most 0.4 points, and on the checklist three of four judges rank FLUX-dev last among them.

### How well the judges agree with the human

95% bootstrap intervals over 40 images in parentheses.

| Judge | Checklist agreement | ρ (items passed) | ρ (quality) | ρ (artifacts) | Self-agreement |
|---|---|---|---|---|---|
| Claude Sonnet 5 | **0.88** (0.83–0.92) | **0.70** (0.46–0.87) | 0.39 (0.04–0.66) | 0.26 (−0.07–0.55) | 0.94 |
| Gemini 3.1 Flash Lite | **0.88** (0.83–0.92) | 0.58 (0.30–0.80) | 0.42 (0.10–0.67) | 0.41 (0.08–0.66) | 0.98 |
| Gemini 3 Flash | 0.85 (0.79–0.90) | 0.57 (0.28–0.80) | **0.55** (0.29–0.74) | **0.55** (0.27–0.76) | 0.93 |
| GPT-5.4 nano | 0.84 (0.79–0.88) | 0.44 (0.11–0.69) | 0.28 (−0.05–0.55) | 0.22 (−0.12–0.53) | 0.90 |
| Qwen3-Next (blind) | 0.67 (0.57–0.76) | 0.08 (−0.32–0.40) | 0.11 (−0.22–0.42) | 0.04 (−0.30–0.37) | 0.94 |
| *"Always yes" stub* | *0.81* | — | — | — | — |

Hardest items for the generators (per the human): "wind blowing her jacket" — 0 of 4 models, "no extra illegible
text" in the infographic — 1 of 4, exact sign text, correct fingers and piano keys — 2 of 4.

## Takeaways

1. **On a free T4 the best trade-off is SD 3.5 Medium**: 1.6× faster than FLUX-schnell, the lowest memory, and,
   together with FLUX-dev, the only model that wrote "OPEN 24/7" correctly. FLUX-dev gives the most beautiful images,
   but costs 19 minutes per image and has a non-commercial license. SDXL is for when you need speed, not accuracy.
2. **LLM judges are fine for checking facts, not taste.** They pass the checklist close to the human, yet none of them
   noticed FLUX-dev's aesthetic lead. For "which model looks better", an LLM judge does not replace a human yet.
3. **Always report a baseline.** 88% agreement with the human sounds good until you see 81% for a stub.
4. **Test your judge with a control.** A model that cannot see the image produces plausible, consistent and useless
   ratings. High self-agreement guarantees nothing.
5. **Judges differ in strictness** (Gemini 3 Flash gives artifacts ~5–6, GPT-5.4 nano ~8–9), so compare the ranking
   within one judge, not raw scores across judges.

## Limitations

One human rater, 10 prompts, one seed per prompt, 4-bit quantization for FLUX and SD 3.5 (T5) — this is a pilot, not
a model ranking. Intervals are wide. Judges saw 768 px images. Kaggle Benchmarks models are preview versions served
through Kaggle's proxy as of the experiment date (September 2026); some requests got 429 "heavy load" and were retried
with back-off, and one of 400 judge ratings (from the Qwen control) was never obtained. IBM Granite 4 was not used as
a judge: it rejected the image request as non-multimodal (the error text mentioned a different model name — possibly a
routing detail of the proxy; we did not investigate).

## Reproduce

Code: `kernel/` (generation on Kaggle), `judge/` (prompts, checklists, LLM judges), `human/index.html` (blind rating
page), `analysis.py` (all tables and intervals), `make_compare.py` (comparison grids).
Data: 40 images, prompts, checklists, every judge answer and the human ratings — dataset on Hugging Face.

### Running

```bash
pip install numpy scipy pillow huggingface_hub
huggingface-cli download MrAlexGov/t4-imagegen-llm-judges --repo-type dataset --local-dir data
python analysis.py            # all tables, bootstrap CIs -> data/results.json
python make_compare.py        # comparison grids -> compare/
```
Generation (`kernel/`) runs as Kaggle GPU scripts; `gen_q.py` is a template (`__MODEL__`, `__PROMPTS__`) and expects a
read-only HF token in an attached private dataset (`hf_token.txt`). Judges (`judge/judge.py`) need Kaggle Benchmarks
proxy credentials: `kaggle benchmarks init`, then export the variables from the generated env file.
