"""LLM-судьи оценивают картинки генераторов вслепую по чек-листам (Kaggle Benchmarks model proxy).
Кэширует ответы в judgments.jsonl; можно перезапускать — уже оценённое пропускается."""
import dataclasses, json, os, random, sys, time, concurrent.futures as cf
import kaggle_benchmarks as kbench
from kaggle_benchmarks.content_types import images
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
PROMPTS = {p["id"]: p for p in json.load(open(f"{ROOT}/kernel/prompts.json"))}
RUBRIC = json.load(open(f"{ROOT}/judge/rubric.json"))
GENS = [g for g in ["sdxl", "sd35m", "flux-schnell", "flux-dev"] if os.path.exists(f"{ROOT}/res_{g}/out/{g}/timing.json")]
JUDGES = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "anthropic/claude-sonnet-5@default", "google/gemini-3-flash-preview", "google/gemini-3.1-flash-lite-preview",
    "openai/gpt-5.4-nano-2026-03-17", "qwen/qwen3-next-80b-a3b-instruct"]  # qwen — контроль: картинки не видит
REPEATS = int(os.environ.get("REPEATS", 2))
OUT = f"{ROOT}/judge/judgments.jsonl"


@dataclasses.dataclass
class Check:
    criterion: str
    passed: bool
    comment: str


@dataclasses.dataclass
class Verdict:
    checks: list[Check]
    image_quality: int      # 1–10: техническое качество, детализация, эстетика
    artifacts: int          # 1–10: 10 = артефактов нет (руки, геометрия, мусорный текст, «плывущие» детали)
    summary: str


TEMPLATE = """Ты строгий эксперт по оценке изображений, сгенерированных нейросетью.
Промпт, по которому сгенерировано изображение:
«{prompt}»

Проверь изображение по каждому критерию и для каждого реши, выполнен ли он (passed: true/false), с коротким комментарием.
Будь строг: если текст написан с ошибкой или критерий выполнен частично — это false.
Критерии:
{criteria}

Также оцени по шкале 1–10:
- image_quality — техническое качество и эстетика;
- artifacts — отсутствие артефактов (10 = артефактов нет, 1 = сильные артефакты).
В summary — одно предложение по-русски."""


def small(path):  # 768px JPEG: дешевле по токенам, детали для оценки сохраняются
    out = f"/tmp/claude-1000/judge_img/{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)}.jpg"
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Image.open(path).convert("RGB").resize((768, 768)).save(out, quality=90)
    return out


def load_done():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            if "error" not in r: done.add((r["judge"], r["generator"], r["prompt_id"], r["repeat"]))
    return done


def judge_one(job):
    judge, gen, pid, rep = job
    crit = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(RUBRIC[pid]))
    img = images.from_path(small(f"{ROOT}/res_{gen}/out/{gen}/{pid}.png"))
    err = None
    for attempt in range(6):
        try:
            t = time.time()
            with kbench.chats.new(f"{judge}-{gen}-{pid}-{rep}"):
                v = kbench.llms[judge].prompt(TEMPLATE.format(prompt=PROMPTS[pid]["prompt"], criteria=crit),
                                              schema=Verdict, image=img, temperature=0 if rep == 0 else 1, seed=rep)
            d = dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
            return {"judge": judge, "generator": gen, "prompt_id": pid, "repeat": rep, "sec": round(time.time() - t, 1), **d}
        except Exception as e:
            err = repr(e)[:300]; time.sleep(min(60, 5 * 2 ** attempt))
    return {"judge": judge, "generator": gen, "prompt_id": pid, "repeat": rep, "error": err}


if __name__ == "__main__":
    done = load_done()
    jobs = [(j, g, p, r) for j in JUDGES for r in range(REPEATS) for g in GENS for p in PROMPTS if (j, g, p, r) not in done]
    random.Random(0).shuffle(jobs)  # случайный порядок: судья не видит картинки одного генератора подряд
    print(f"генераторы {GENS}, судьи {len(JUDGES)}, заданий {len(jobs)}", flush=True)
    with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", 4))) as ex, open(OUT, "a") as f:
        for n, r in enumerate(ex.map(judge_one, jobs), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if n % 20 == 0 or "error" in r: print(n, r["judge"], r.get("error", "ok")[:120], flush=True)
