"""
实验 2：上下文系统辨识 —— 给的历史越多，模型对没见过的世界预测得越准吗？
=========================================================================
对每个评测世界，选目标年 T，只给模型 T 之前的 k 年历史（k = 0,1,2,4,8,12），
再给第 T 年的年初状态、行动、天气，让模型续写结果。

指标：
  harvest_err   收成预测的 |log(预测/真实)|，越小越好（0.1 ≈ 差 10%）
  within10      收成误差在 10% 以内的比例
  food_acc      粮食涨/跌/持平方向的准确率
  drought_brier 在看到天气之前，模型给出的 P(干旱) 的 Brier 分数（越小越好）
  parse         输出能被解析的比例

参照线：
  persist   收成 = 上一年收成；粮食方向 = 上一年方向
  observer  理想贝叶斯观察者（知道所有规则族，见 observer.py）

两个评测集：
  id   训练见过的规则组合、新种子
  ood  训练中从未出现的规则组合

用法：
  python eval_icl.py --ckpt out/civlm26m/best.pt
  python eval_icl.py --ckpt_dir out/civlm26m/ckpts      # 实验 1：每个检查点都评一遍，画学习曲线
  python eval_icl.py --baselines_only                    # 不需要模型，只算两条参照线
"""
import argparse
import glob
import json
import math
import os
import random
import re
import time
from collections import defaultdict

import numpy as np

from chronicle import FOOD_DIR_WORDS, Renderer
from engine import simulate
from observer import Observer

DIR_RE = re.compile(r"\b(" + "|".join(FOOD_DIR_WORDS) + r")\b")
INT_RE = re.compile(r"\d+")
ERR_CAP = 1.0


def parse_outcome(text):
    m = INT_RE.search(text)
    if not m:
        return None, None
    harvest = int(m.group())
    d = DIR_RE.search(text, m.end())
    return harvest, (FOOD_DIR_WORDS[d.group(1)] if d else None)


def logerr(pred, true):
    if pred is None:
        return ERR_CAP
    return min(ERR_CAP, abs(math.log(max(1, pred) / max(1, true))))


def build_cases(worlds, years, ks, styles, targets_per_world):
    """预先构造全部评测样本（与模型无关），参照线也在这里算好。"""
    cases = []
    for w in worlds:
        ep = simulate(w["seed"], w["rules"], years)
        ob = Observer(ep)
        rng = random.Random(w["seed"])
        Ts = rng.sample(range(max(ks) + 1, years - 1), targets_per_world)
        renders = {st: Renderer(ep, st) for st in styles}
        segs = {st: [renders[st].year(y) for y in ep["years"]] for st in styles}
        for T in Ts:
            o = ep["years"][T]["out"]
            canals = ep["years"][T + 1]["state"].canals
            for k in ks:
                window = list(range(T - k, T))
                obs = ob.predict(window, T)
                last = ep["years"][T - 1]["out"] if k > 0 else None
                base = {"seed": w["seed"], "rules": w["rules"], "T": T, "k": k, "canals": canals,
                        "true_harvest": o["harvest"], "true_dir": o["food_dir"],
                        "true_drought": int(o["weather"] == "drought"),
                        "obs_harvest": obs["harvest"], "obs_dir": obs["food_dir"], "obs_pdrought": obs["p_drought"],
                        "obs_p_true_irr": obs["p_irrigation"][w["rules"]["irrigation"]],
                        "persist_harvest": last["harvest"] if last else None,
                        "persist_dir": last["food_dir"] if last else "up"}
                for st in styles:
                    r = renders[st]
                    hist = "".join("".join(segs[st][i]) for i in window)
                    cases.append(dict(base, style=st,
                                      header=r.header(),
                                      hist=hist,
                                      pre_weather="".join(segs[st][T][:2]),
                                      weather_seg=segs[st][T][2],
                                      weather_cands=r.weather_candidates()))
    return cases


# ---------------- 模型部分 ----------------
class LMScorer:
    def __init__(self, ckpt, device=None):
        import torch
        from model import load_checkpoint
        from tokenizer import BPETokenizer
        from utils import amp_setup, get_device
        self.torch = torch
        self.device = get_device(device)
        self.model = load_checkpoint(ckpt, self.device).eval()
        self.block = self.model.cfg.block_size
        data_dir = os.environ.get("CIVLM_DATA", "data")
        self.tok = BPETokenizer(os.path.join(data_dir, "tokenizer.json"))
        self.amp, _ = amp_setup(self.device)

    def _ids(self, header, body, reserve):
        h, b = self.tok.encode(header), self.tok.encode(body)
        room = self.block - reserve - len(h)
        if len(b) > room:                       # 太长时从左边截断历史，保留世界标题
            b = b[-room:]
        return h + b

    def run(self, case, max_new=48):
        torch = self.torch
        F = torch.nn.functional
        with torch.no_grad(), self.amp:
            # (a) 天气之前：P(干旱)
            ids = self._ids(case["header"], case["hist"] + case["pre_weather"], reserve=40)
            x = torch.tensor([ids], device=self.device)
            logits, _, past = self.model(x)
            last_lp = F.log_softmax(logits[0, -1].float(), -1)
            cls_lp = {}
            for wcls, texts in case["weather_cands"].items():
                lps = []
                for t in texts:
                    c = self.tok.encode(t)
                    lp = last_lp[c[0]].item()
                    if len(c) > 1:
                        lg, _, _ = self.model(torch.tensor([c[:-1]], device=self.device), past_kv=past,
                                              start_pos=len(ids))
                        lsm = F.log_softmax(lg[0].float(), -1)
                        lp += lsm[torch.arange(len(c) - 1), torch.tensor(c[1:], device=self.device)].sum().item()
                    lps.append(lp)
                m = max(lps)
                cls_lp[wcls] = m + math.log(sum(math.exp(v - m) for v in lps))
            m = max(cls_lp.values())
            z = sum(math.exp(v - m) for v in cls_lp.values())
            p_drought = math.exp(cls_lp["drought"] - m) / z
            # (b) 天气之后：续写结果
            ids = self._ids(case["header"], case["hist"] + case["pre_weather"] + case["weather_seg"], reserve=max_new)
            x = torch.tensor([ids], device=self.device)
            out = self.model.generate(x, max_new, temperature=0, eos_id=self.tok.eot_id)
            gen = self.tok.decode(out[0, len(ids):].tolist())
        gen = gen.split("\n\n")[0]
        h, d = parse_outcome(gen)
        return {"lm_harvest": h, "lm_dir": d, "lm_pdrought": p_drought, "lm_text": gen[:200]}


# ---------------- 汇总 ----------------
def summarize(cases, with_lm):
    groups = defaultdict(list)
    for c in cases:
        groups[(c["split"], c["style"], c["k"])].append(c)
    rows = []
    for (split, style, k), cs in sorted(groups.items()):
        r = {"split": split, "style": style, "k": k, "n": len(cs)}
        irr = [c for c in cs if c["canals"] > 0]
        for who in (["lm"] if with_lm else []) + ["obs", "persist"]:
            hk, dk = f"{who}_harvest", f"{who}_dir"
            if who == "persist" and k == 0:
                r.update({f"{who}_harvest_err": None, f"{who}_within10": None, f"{who}_food_acc": None,
                          f"{who}_harvest_err_irrigated": None})
                continue
            errs = [logerr(c[hk], c["true_harvest"]) for c in cs]
            r[f"{who}_harvest_err"] = float(np.mean(errs))
            r[f"{who}_within10"] = float(np.mean([e < math.log(1.1) for e in errs]))
            r[f"{who}_food_acc"] = float(np.mean([c[dk] == c["true_dir"] for c in cs]))
            r[f"{who}_harvest_err_irrigated"] = float(np.mean([logerr(c[hk], c["true_harvest"]) for c in irr])) if irr else None
        for who in (["lm"] if with_lm else []) + ["obs"]:
            r[f"{who}_drought_brier"] = float(np.mean([(c[f"{who}_pdrought"] - c["true_drought"]) ** 2 for c in cs]))
        if with_lm:
            r["lm_parse"] = float(np.mean([c["lm_harvest"] is not None for c in cs]))
        r["obs_p_true_irr"] = float(np.mean([c["obs_p_true_irr"] for c in cs]))
        rows.append(r)
    return rows


def fmt(v, w=6):
    return f"{v:{w}.3f}" if isinstance(v, float) else f"{'-':>{w}}"


def print_table(rows, with_lm):
    print("\n收成对数误差（越小越好）| 粮食方向准确率 | 干旱 Brier")
    head = f"{'split':5s} {'style':10s} {'k':>3s} | " + ("  LM   " if with_lm else "") + \
           " obs    persist | " + ("  LM   " if with_lm else "") + " obs    persist | " + \
           ("  LM   " if with_lm else "") + " obs   " + (" | parse" if with_lm else "")
    print(head)
    print("-" * len(head))
    for r in rows:
        line = f"{r['split']:5s} {r['style']:10s} {r['k']:3d} | "
        line += (fmt(r.get("lm_harvest_err")) + " " if with_lm else "") + fmt(r["obs_harvest_err"]) + " " + fmt(r["persist_harvest_err"]) + " | "
        line += (fmt(r.get("lm_food_acc")) + " " if with_lm else "") + fmt(r["obs_food_acc"]) + " " + fmt(r["persist_food_acc"]) + " | "
        line += (fmt(r.get("lm_drought_brier")) + " " if with_lm else "") + fmt(r["obs_drought_brier"])
        if with_lm:
            line += " | " + fmt(r["lm_parse"], 5)
        print(line)


def plot(rows, path, title, with_lm):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    splits = sorted({r["split"] for r in rows})
    styles = sorted({r["style"] for r in rows})
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4.2), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        rs0 = [r for r in rows if r["split"] == sp and r["style"] == styles[0]]
        ks = [r["k"] for r in rs0]
        ax.plot(ks, [r["obs_harvest_err"] for r in rs0], "k--", label="ideal observer")
        pk = [(r["k"], r["persist_harvest_err"]) for r in rs0 if r["persist_harvest_err"] is not None]
        ax.plot([a for a, _ in pk], [b for _, b in pk], ":", color="gray", label="persistence")
        if with_lm:
            for st in styles:
                rs = [r for r in rows if r["split"] == sp and r["style"] == st]
                ax.plot([r["k"] for r in rs], [r["lm_harvest_err"] for r in rs], "o-", label=f"LM ({st})")
        ax.set_title(f"{title} · {sp}")
        ax.set_xlabel("years of history in context (k)")
        ax.set_ylabel("harvest |log error|")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"图 → {path}")


def evaluate_ckpt(ckpt, cases, args):
    scorer = LMScorer(ckpt, args.device)
    t0 = time.time()
    out = []
    for i, c in enumerate(cases):
        out.append(dict(c, **scorer.run(c)))
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(cases)}，剩余约 {el / (i + 1) * (len(cases) - i - 1) / 60:.1f} 分钟", end="\r")
    print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ckpt_dir", default=None, help="评测目录下所有检查点（实验 1）")
    ap.add_argument("--baselines_only", action="store_true")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--splits", nargs="+", default=["id", "ood"])
    ap.add_argument("--styles", nargs="+", default=["structured", "narrative"])
    ap.add_argument("--ks", nargs="+", type=int, default=[0, 1, 2, 4, 8, 12])
    ap.add_argument("--n_worlds", type=int, default=60, help="每个评测集用多少个世界")
    ap.add_argument("--targets", type=int, default=2, help="每个世界评几个目标年")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    os.environ["CIVLM_DATA"] = args.data_dir
    os.makedirs(args.out, exist_ok=True)

    ew = json.load(open(os.path.join(args.data_dir, "eval_worlds.json")))
    cases = []
    for sp in args.splits:
        cs = build_cases(ew[sp][:args.n_worlds], ew["years"], args.ks, args.styles, args.targets)
        for c in cs:
            c["split"] = sp
        cases += cs
    print(f"评测样本 {len(cases)} 个（{len(args.splits)} 个评测集 × {args.n_worlds} 世界 × {args.targets} 目标年 × "
          f"{len(args.ks)} 种 k × {len(args.styles)} 种文体）")

    if args.baselines_only:
        rows = summarize(cases, False)
        print_table(rows, False)
        plot(rows, os.path.join(args.out, "baselines.png"), "baselines", False)
        json.dump(rows, open(os.path.join(args.out, "baselines.json"), "w"), indent=1)
        return

    if args.ckpt_dir:
        ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt")))
        curve = []
        for ck in ckpts:
            print(f"== {ck}")
            res = evaluate_ckpt(ck, cases, args)
            rows = summarize(res, True)
            step = int(re.findall(r"\d+", os.path.basename(ck))[-1])
            for r in rows:
                r["step"] = step
            curve += rows
        json.dump(curve, open(os.path.join(args.out, "learning_curve.json"), "w"), indent=1)
        plot_curve(curve, os.path.join(args.out, "learning_curve.png"))
        return

    res = evaluate_ckpt(args.ckpt, cases, args)
    rows = summarize(res, True)
    print_table(rows, True)
    name = os.path.splitext(os.path.basename(args.ckpt))[0]
    tag = os.path.basename(os.path.dirname(args.ckpt)) + "_" + name
    json.dump(rows, open(os.path.join(args.out, f"{tag}.json"), "w"), indent=1)
    with open(os.path.join(args.out, f"{tag}_samples.jsonl"), "w", encoding="utf-8") as f:
        for c in res:
            f.write(json.dumps({k: v for k, v in c.items() if k not in ("hist", "weather_cands", "header")},
                               ensure_ascii=False) + "\n")
    plot(rows, os.path.join(args.out, f"{tag}.png"), tag, True)


def plot_curve(curve, path):
    """实验 1：训练进程中，格式能力（parse）与世界规律能力（收成误差、粮食方向）分别何时出现。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for sp in sorted({r["split"] for r in curve}):
        for k in sorted({r["k"] for r in curve}):
            if k not in (0, 4, 12):
                continue
            rs = sorted([r for r in curve if r["split"] == sp and r["k"] == k and r["style"] == "structured"],
                        key=lambda r: r["step"])
            if not rs:
                continue
            st = [r["step"] for r in rs]
            axes[0].plot(st, [r["lm_parse"] for r in rs], label=f"{sp} k={k}")
            axes[1].plot(st, [r["lm_harvest_err"] for r in rs], label=f"{sp} k={k}")
            axes[2].plot(st, [r["lm_food_acc"] for r in rs], label=f"{sp} k={k}")
    for ax, t in zip(axes, ["format: parse rate", "harvest |log error|", "food direction accuracy"]):
        ax.set_title(t)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"图 → {path}")


if __name__ == "__main__":
    main()
