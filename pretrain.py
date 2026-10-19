"""
从零预训练 CivLM（改自 energy_llm_v2/pretrain.py）
==================================================
只做下一个 token 预测，不告诉模型任何世界规律。

默认 8 层 × 512 维 ≈ 2600 万参数；上下文 2048（叙事文体 12 年历史约 1600 token）。
batch 16 × 2048 = 每步 3.3 万 token。2080Ti 预计每步 0.6~1 秒（估算，以屏幕显示为准）。

每 --ckpt_every 步另存一个检查点到 out/<name>/ckpts/，用于实验 1（能力随训练形成的曲线）：
  python eval_icl.py --ckpt_dir out/civlm26m/ckpts

100M 配置示例（先确认 26M 跑通、曲线有意义再上）：
  python pretrain.py --name civlm100m --n_layer 12 --n_head 12 --n_embd 768 --batch_size 8 --grad_accum 2
"""
import argparse
import os
import time

import numpy as np
import torch

from chronicle import Renderer
from engine import HELDOUT_COMBOS, simulate
from model import MiniLM, ModelConfig, load_checkpoint, save_checkpoint
from tokenizer import BPETokenizer
from utils import amp_setup, cosine_lr, get_device, print_gpu_mem, set_seed


def sample_prompts():
    """取一个没见过的世界，给 3 年历史，让模型续写第 4 年的结果。"""
    ep = simulate(99_999_999, HELDOUT_COMBOS[0], 6)
    out = []
    for style in ("structured", "narrative"):
        r = Renderer(ep, style)
        segs = [r.year(y) for y in ep["years"]]
        p = r.header() + "".join("".join(s) for s in segs[:3]) + "".join(segs[3][:3])
        out.append((p, "".join(segs[3][3:]).strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--name", default="civlm26m")
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=512)
    ap.add_argument("--block_size", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = os.path.join("out", args.name)
    ck_dir = os.path.join(out_dir, "ckpts")
    os.makedirs(ck_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best.pt")

    tok = BPETokenizer(os.path.join(args.data_dir, "tokenizer.json"))
    train = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    if len(val) < args.block_size + 2:
        val = train
    tok_per_step = args.batch_size * args.grad_accum * args.block_size
    print(f"训练集 {len(train) / 1e6:.1f}M tokens；计划训练 {args.max_steps * tok_per_step / 1e6:.0f}M tokens"
          f"（约 {args.max_steps * tok_per_step / len(train):.1f} 轮）")

    def get_batch(d):
        ix = np.random.randint(0, len(d) - args.block_size - 1, size=args.batch_size)
        x = np.stack([d[i:i + args.block_size] for i in ix]).astype(np.int64)
        y = np.stack([d[i + 1:i + 1 + args.block_size] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(device, non_blocking=True), torch.from_numpy(y).to(device, non_blocking=True)

    if args.resume and os.path.exists(best_path):
        model = load_checkpoint(best_path, device)
        print(f"从 {best_path} 继续训练")
    else:
        cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                          n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
        model = MiniLM(cfg).to(device)
    print(f"参数量 {model.num_params() / 1e6:.1f}M，设备 {device}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    amp_ctx, scaler = amp_setup(device)
    prompts = sample_prompts()

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for _ in range(20):
            x, y = get_batch(val)
            with amp_ctx:
                losses.append(model(x, y)[1].item())
        for p, truth in prompts:
            ids = torch.tensor([tok.encode(p)], device=device)
            with amp_ctx:
                out = model.generate(ids, 50, temperature=0, eos_id=tok.eot_id)
            gen = tok.decode(out[0, ids.size(1):].tolist()).split("\n\n")[0].strip()
            print("  预测 →", gen.replace("\n", " | ")[:150])
            print("  真实 →", truth.replace("\n", " | ")[:150])
        model.train()
        return float(np.mean(losses))

    log = open(os.path.join(out_dir, "pretrain_log.csv"), "a" if args.resume else "w", encoding="utf-8")
    if not args.resume:
        log.write("step,tokens,train_loss,val_loss,lr\n")
    best, t0 = float("inf"), time.time()
    model.train()
    try:
        for step in range(args.max_steps + 1):
            lr = cosine_lr(step, args.max_steps, args.lr, args.min_lr, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            if step % args.eval_interval == 0 or step == args.max_steps:
                vl = evaluate()
                print(f"[step {step}] val_loss {vl:.4f}  已用 {(time.time() - t0) / 60:.1f} 分钟")
                log.write(f"{step},{step * tok_per_step},,{vl:.4f},{lr:.6f}\n")
                log.flush()
                if vl < best:
                    best = vl
                    save_checkpoint(model, best_path, {"step": step, "val_loss": vl})
            if step % args.ckpt_every == 0 or step == args.max_steps:
                save_checkpoint(model, os.path.join(ck_dir, f"step_{step:06d}.pt"),
                                {"step": step, "tokens": step * tok_per_step})
            if step == args.max_steps:
                break
            for _ in range(args.grad_accum):
                x, y = get_batch(train)
                with amp_ctx:
                    loss = model(x, y)[1] / args.grad_accum
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if step % 25 == 0:
                el = time.time() - t0
                eta = el / max(step, 1) * (args.max_steps - step) / 60
                print(f"step {step:6d} | loss {loss.item() * args.grad_accum:.4f} | lr {lr:.2e} | 剩余约 {eta:.0f} 分钟", end="\r")
                log.write(f"{step},{step * tok_per_step},{loss.item() * args.grad_accum:.4f},,{lr:.6f}\n")
    except KeyboardInterrupt:
        print("\n手动中断。")
    log.close()
    print(f"\n预训练结束，最佳 val_loss {best:.4f} → {best_path}")
    print_gpu_mem()


if __name__ == "__main__":
    main()
