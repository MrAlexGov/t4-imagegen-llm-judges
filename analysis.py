"""Итоговый анализ: генераторы глазами человека и LLM-судей, согласие судей с человеком.
Входы (папка данных, по умолчанию data/ — скачайте датасет с HF): human_ratings.jsonl, judge_ratings.jsonl, generators.json, rubric.json.
Выход: results.json + таблицы в stdout. Доверительные интервалы — бутстреп по картинкам (2000 выборок)."""
import collections, glob, json
import numpy as np
from scipy.stats import spearmanr

GENS = ["sdxl", "sd35m", "flux-schnell", "flux-dev"]
NAMES = {"sdxl": "SDXL", "sd35m": "SD 3.5 Medium", "flux-schnell": "FLUX.1-schnell", "flux-dev": "FLUX.1-dev"}
rng = np.random.default_rng(0)

import sys
D = sys.argv[1] if len(sys.argv) > 1 else "data"
H = {}
for l in open(f"{D}/human_ratings.jsonl"):
    r = json.loads(l); H[(r["generator"], r["prompt_id"])] = r
J = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(f"{D}/judge_ratings.jsonl"):
    r = json.loads(l); J[r["judge"]][(r["generator"], r["prompt_id"])].append(r)
timing = json.load(open(f"{D}/generators.json"))
KEYS = sorted(H)

def gen_table(score):  # score(key) -> (pass, quality, artifacts) | None
    out = {}
    for g in GENS:
        v = [s for k in KEYS if k[0] == g and (s := score(k)) is not None]
        out[g] = {"pass": round(float(np.mean([x[0] for x in v])), 3), "quality": round(float(np.mean([x[1] for x in v])), 2),
                  "artifacts": round(float(np.mean([x[2] for x in v])), 2), "n": len(v)}
    return out

res = {"n_images": len(KEYS), "timing": {g: {k: timing[g].get(k, "fp16, без квантования" if k == "quant" else None) for k in ("per_image_s_avg_rest", "load_s", "peak_mem_gb", "quant", "settings")} for g in GENS}}
res["human"] = gen_table(lambda k: (np.mean(H[k]["checks"]), H[k]["quality"], H[k]["artifacts"]))
res["judges"] = {}
for j, d in J.items():
    avg = lambda k: None if not d.get(k) else (np.mean([np.mean([c["passed"] for c in r["checks"]]) for r in d[k]]),
                                                np.mean([r["image_quality"] for r in d[k]]), np.mean([r["artifacts"] for r in d[k]]))
    ranking = gen_table(avg)
    # согласие с человеком
    items = [k for k in KEYS if d.get(k)]
    def stats(idx):
        acc, hp, jp, hq, jq, ha, ja = [], [], [], [], [], [], []
        for k in idx:
            h = H[k]
            for r in d[k]:
                c = [x["passed"] for x in r["checks"]]
                if len(c) == len(h["checks"]): acc += [a == b for a, b in zip(c, h["checks"])]
            a = avg(k); hp.append(np.mean(h["checks"])); jp.append(a[0]); hq.append(h["quality"]); jq.append(a[1]); ha.append(h["artifacts"]); ja.append(a[2])
        return np.mean(acc), spearmanr(hp, jp)[0], spearmanr(hq, jq)[0], spearmanr(ha, ja)[0]
    point = stats(items)
    boot = np.array([stats([items[i] for i in rng.integers(0, len(items), len(items))]) for _ in range(2000)])
    ci = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    selfc = [a["passed"] == b["passed"] for k in items if len(d[k]) >= 2 and len(d[k][0]["checks"]) == len(d[k][1]["checks"])
             for a, b in zip(d[k][0]["checks"], d[k][1]["checks"])]
    names = ["check_accuracy", "rho_pass", "rho_quality", "rho_artifacts"]
    res["judges"][j] = {"by_generator": ranking, "self_consistency": round(float(np.mean(selfc)), 3), "n_images": len(items),
                        **{n: {"value": round(float(point[i]), 3), "ci95": [round(float(ci[0][i]), 3), round(float(ci[1][i]), 3)]} for i, n in enumerate(names)}}
res["baseline_always_yes"] = round(float(np.mean([c for k in KEYS for c in H[k]["checks"]])), 3)
# самые трудные пункты чек-листа для генераторов (по человеку)
rub = json.load(open(f"{D}/rubric.json"))
hard = collections.defaultdict(list)
for (g, p), h in H.items():
    for crit, ok in zip(rub[p], h["checks"]): hard[(p, crit)].append(ok)
res["hardest_criteria_human"] = [{"prompt": p, "criterion": c, "pass_rate": round(float(np.mean(v)), 2)} for (p, c), v in sorted(hard.items(), key=lambda kv: np.mean(kv[1]))[:8]]
json.dump(res, open(f"{D}/results.json", "w"), ensure_ascii=False, indent=2)

print(f"{len(KEYS)} картинок; базовая линия «всегда да» = {res['baseline_always_yes']}\n")
print("Человек:", {NAMES[g]: (v["pass"], v["quality"], v["artifacts"]) for g, v in res["human"].items()})
for j, v in res["judges"].items():
    print(f"\n{j}  self={v['self_consistency']}")
    print("  по генераторам:", {NAMES[g]: (x["pass"], x["quality"], x["artifacts"]) for g, x in v["by_generator"].items()})
    print("  vs человек:", {n: (v[n]["value"], v[n]["ci95"]) for n in ["check_accuracy", "rho_pass", "rho_quality", "rho_artifacts"]})
print("\nСамые трудные пункты:", [(x["prompt"], x["criterion"][:50], x["pass_rate"]) for x in res["hardest_criteria_human"]])
