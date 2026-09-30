"""Замер генерации изображений на Kaggle 2xT4: время загрузки, время на картинку, пик памяти. MODEL=sdxl|flux-schnell|flux-dev|sd35m"""
import json, os, time, glob, torch
MODEL = "sdxl"
prompts = [{"id": "01_portrait", "prompt": "Close-up portrait photo of an elderly fisherman with a weathered face and grey beard, soft morning light, shallow depth of field, 85mm lens, highly detailed skin texture", "tests": "фотореализм, кожа, свет"}, {"id": "02_text_sign", "prompt": "A neon sign on a brick wall that says \"OPEN 24/7\" in bright pink letters, rainy night street, reflections on wet asphalt", "tests": "рендер текста"}, {"id": "03_hands", "prompt": "A pianist's hands playing a grand piano keyboard, top-down view, dramatic lighting, realistic fingers", "tests": "руки и пальцы"}, {"id": "04_composition", "prompt": "A red cube on top of a blue sphere, a green cone to the left of them, on a white table, studio lighting, minimalist", "tests": "следование композиции"}, {"id": "05_landscape", "prompt": "Wide landscape of snowy mountains at sunset, a frozen lake in the foreground, pine trees, golden hour, ultra detailed", "tests": "пейзаж, детализация"}, {"id": "06_illustration", "prompt": "Flat vector illustration of a cozy home office: a laptop, a cat sleeping on the desk, a plant, warm pastel colors, clean lines", "tests": "иллюстрация, стиль"}, {"id": "07_product", "prompt": "Product photo of a matte black wireless headphones on a marble surface, soft shadows, commercial advertising style", "tests": "предметная съёмка"}, {"id": "08_anime", "prompt": "Anime style girl with short silver hair standing on a rooftop at night, city lights below, wind blowing her jacket, detailed background", "tests": "аниме-стиль"}, {"id": "09_food", "prompt": "Top-down photo of a bowl of ramen with a soft-boiled egg, green onions, chashu pork, steam rising, dark wooden table", "tests": "еда, текстуры"}, {"id": "10_infographic", "prompt": "A simple infographic poster titled \"AI AGENT\" with three labeled icons: \"PLAN\", \"ACT\", \"CHECK\", connected by arrows, clean corporate design", "tests": "текст + схема"}]
os.makedirs(f"out/{MODEL}", exist_ok=True)
t0 = time.time()
from diffusers import StableDiffusionXLPipeline
if MODEL == "sdxl":
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    kw = dict(num_inference_steps=30, guidance_scale=7.0, height=1024, width=1024)
load = time.time() - t0
print("gpu:", torch.cuda.get_device_name(0), "load s:", round(load, 1), flush=True)
times = []
for p in prompts:
    torch.cuda.synchronize(); t = time.time()
    img = pipe(p["prompt"], generator=torch.Generator("cuda").manual_seed(42), **kw).images[0]
    torch.cuda.synchronize(); dt = time.time() - t; times.append(dt)
    img.save(f"out/{MODEL}/{p['id']}.png"); print(p["id"], round(dt, 1), "s", flush=True)
res = {"model": MODEL, "settings": {k: v for k, v in kw.items()}, "load_s": round(load, 1),
       "per_image_s_first": round(times[0], 1), "per_image_s_avg_rest": round(sum(times[1:]) / len(times[1:]), 1),
       "peak_mem_gb": [round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(torch.cuda.device_count())]}
json.dump(res, open(f"out/{MODEL}/timing.json", "w"), indent=2); print(json.dumps(res))
