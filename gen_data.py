"""
生成训练数据
============
1. 用 CivEngine 模拟 N 个文明（规则组合只来自 TRAIN_COMBOS，两种组合完全留给测试）
2. 用 CivChronicle 写成文本：叙事 / 结构化按比例混合，部分文档附反事实分支
3. 训练字节级 BPE 分词器（数字按单个数位切分），编码成 train.bin / val.bin
4. 写出评测世界清单 eval_worlds.json：
     id   训练见过的规则组合、但没见过的种子
     ood  训练从未出现的规则组合

用法：
  python gen_data.py                       # 默认 2 万个文明，约 5000 万 token
  python gen_data.py --n_worlds 2000       # 快速试跑
  python gen_data.py --narrative_frac 0    # 消融：只用结构化文本
"""
import argparse
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np

from chronicle import render_episode
from engine import ACTIONS, ALL_COMBOS, HELDOUT_COMBOS, TRAIN_COMBOS, fork, simulate
from tokenizer import train_bpe

VAL_SEED0, EVAL_SEED0 = 10_000_000, 20_000_000


def make_doc(job):
    seed, years, narrative_frac, cf_frac = job
    r = random.Random(seed * 2654435761 % (2 ** 31))
    rules = r.choice(TRAIN_COMBOS)
    ep = simulate(seed, rules, years)
    style = "narrative" if r.random() < narrative_frac else "structured"
    cf = None
    if r.random() < cf_frac:
        idx = r.randrange(5, years - 6)
        alt = r.choice([a for a in ACTIONS if a != ep["years"][idx]["out"]["act"]])
        cf = (idx, alt, fork(ep, idx, alt, n_years=5))
    return render_episode(ep, style, cf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--n_worlds", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=200, help="每个评测集（id/ood）的世界数")
    ap.add_argument("--years", type=int, default=40)
    ap.add_argument("--narrative_frac", type=float, default=0.5)
    ap.add_argument("--cf_frac", type=float, default=0.1, help="附带反事实分支的文档比例")
    ap.add_argument("--vocab_size", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    jobs = [(s, args.years, args.narrative_frac, args.cf_frac) for s in range(args.n_worlds)]
    vjobs = [(VAL_SEED0 + s, args.years, args.narrative_frac, args.cf_frac) for s in range(args.n_val)]
    with Pool(args.workers) as pool:
        train_docs = pool.map(make_doc, jobs, chunksize=64)
        val_docs = pool.map(make_doc, vjobs, chunksize=64)
    print(f"生成 {len(train_docs)} 篇训练文档、{len(val_docs)} 篇验证文档，用时 {time.time() - t0:.0f}s")
    print("示例：\n" + train_docs[0][:600] + "\n...")

    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    sample = random.Random(0).sample(train_docs, min(3000, len(train_docs)))
    tok = train_bpe(iter(sample), args.vocab_size, tok_path)
    print(f"分词器词表 {tok.vocab_size}")

    def write_bin(docs, path):
        n = 0
        with open(path, "wb") as f:
            for i in range(0, len(docs), 1000):
                ids = []
                for e in tok.encode_batch(docs[i:i + 1000]):
                    ids += e + [tok.eot_id]
                arr = np.array(ids, dtype=np.uint16)
                arr.tofile(f)
                n += len(arr)
        return n

    ntr = write_bin(train_docs, os.path.join(args.out_dir, "train.bin"))
    nva = write_bin(val_docs, os.path.join(args.out_dir, "val.bin"))

    er = random.Random(1)
    eval_worlds = {
        "id": [{"seed": EVAL_SEED0 + i, "rules": er.choice(TRAIN_COMBOS)} for i in range(args.n_eval)],
        "ood": [{"seed": EVAL_SEED0 + 100000 + i, "rules": HELDOUT_COMBOS[i % len(HELDOUT_COMBOS)]}
                for i in range(args.n_eval)],
        "years": args.years,
    }
    json.dump(eval_worlds, open(os.path.join(args.out_dir, "eval_worlds.json"), "w"), indent=1)
    meta = {"train_tokens": ntr, "val_tokens": nva, "n_worlds": args.n_worlds, "years": args.years,
            "narrative_frac": args.narrative_frac, "cf_frac": args.cf_frac, "vocab_size": tok.vocab_size,
            "train_combos": TRAIN_COMBOS, "heldout_combos": HELDOUT_COMBOS,
            "tokens_per_doc": ntr / len(train_docs)}
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=1)
    print(f"train {ntr / 1e6:.1f}M tokens，val {nva / 1e6:.2f}M tokens，平均每篇 {ntr / len(train_docs):.0f} tokens")
    print(f"完成，用时 {time.time() - t0:.0f}s → {args.out_dir}/")


if __name__ == "__main__":
    main()
