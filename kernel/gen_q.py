"""FLUX.1-schnell / FLUX.1-dev / SD3.5-medium на Kaggle 2xT4: 4-bit NF4 (bitsandbytes), замер времени и памяти."""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers>=0.33", "transformers", "accelerate", "bitsandbytes", "sentencepiece", "protobuf"], check=True)
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import json, time, glob, torch, numpy as np
from huggingface_hub import login
login(token=open(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)[0]).read().strip())
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from diffusers.quantizers import PipelineQuantizationConfig
MODEL = "__MODEL__"
PROMPTS = __PROMPTS__
DT = torch.bfloat16 if MODEL.startswith("flux") else torch.float16   # FLUX в fp16 даёт NaN/чёрные картинки
t0 = time.time()
q = lambda comps: PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": DT}, components_to_quantize=comps)
if MODEL == "flux-schnell":
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=DT, quantization_config=q(["transformer", "text_encoder_2"]))
    kw = dict(num_inference_steps=4, guidance_scale=0.0, height=1024, width=1024, max_sequence_length=256)
elif MODEL == "flux-dev":
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=DT, quantization_config=q(["transformer", "text_encoder_2"]))
    kw = dict(num_inference_steps=28, guidance_scale=3.5, height=1024, width=1024, max_sequence_length=512)
else:
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=DT, quantization_config=q(["text_encoder_3"]))
    kw = dict(num_inference_steps=40, guidance_scale=4.5, height=1024, width=1024)
# текстовые энкодеры и трансформер по очереди на GPU: одна T4 не держит всё сразу вместе с активациями внимания
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
load = time.time() - t0
print("load s", round(load, 1), "mem GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
os.makedirs(f"out/{MODEL}", exist_ok=True); times = []; black = []
for p in PROMPTS:
    torch.cuda.synchronize(); t = time.time()
    img = pipe(p["prompt"], generator=torch.Generator("cuda").manual_seed(42), **kw).images[0]
    torch.cuda.synchronize(); dt = time.time() - t; times.append(dt)
    if np.asarray(img).max() < 10: black.append(p["id"])
    img.save(f"out/{MODEL}/{p['id']}.png"); print(p["id"], round(dt, 1), "s", flush=True)
res = {"model": MODEL, "quant": "NF4 4-bit (bitsandbytes) + model CPU offload + VAE tiling", "dtype": str(DT), "settings": kw, "load_s": round(load, 1),
       "per_image_s_first": round(times[0], 1), "per_image_s_avg_rest": round(sum(times[1:]) / len(times[1:]), 1),
       "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "black_images": black}
json.dump(res, open(f"out/{MODEL}/timing.json", "w"), indent=2); print(json.dumps(res))
