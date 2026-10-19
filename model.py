"""
LLaMA 风格的迷你语言模型（约 2600 万参数）
==========================================
与 v1 的 GPT-2 结构相比，采用了现代大模型（LLaMA / Qwen）的标准组件：
  - RMSNorm      代替 LayerNorm（更简单，没有均值和偏置）
  - RoPE         旋转位置编码，代替可学习的位置嵌入
  - SwiGLU       门控前馈网络，代替 GELU-MLP
  - KV Cache     生成时缓存历史的 K/V，避免重复计算
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 6400
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def precompute_rope(head_dim, max_len, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)          # (T, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: (B, nh, T, hd)。把向量两两一组看成复数，按位置旋转一个角度。"""
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh = cfg.n_head
        self.hd = cfg.n_embd // cfg.n_head
        self.wq = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, past=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past is not None:                       # 拼上缓存的历史 K/V
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        # 无缓存（训练/首次前向）时用因果掩码；有缓存时新 token 可以看到全部历史
        y = F.scaled_dot_product_attention(q, k, v, is_causal=past is None and T > 1,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)   # 门控
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)   # 值
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, past=None):
        h, kv = self.attn(self.norm1(x), cos, sin, past)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, kv


class MiniLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight     # 输入输出嵌入共享
        cos, sin = precompute_rope(cfg.n_embd // cfg.n_head, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("wo.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, past_kv=None, start_pos=0):
        B, T = idx.shape
        assert start_pos + T <= self.cfg.block_size, "超出最大上下文长度"
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        x = self.drop(self.tok_emb(idx))
        new_kv = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, cos, sin, past_kv[i] if past_kv is not None else None)
            new_kv.append(kv)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss, new_kv

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None, top_p=None,
                 eos_id=None, repetition_penalty=1.0):
        """
        idx: (B, T) 等长提示。temperature=0 为贪心解码。
        返回 (B, T + 新生成长度)；遇到 eos 的序列之后用 eos 填充。
        """
        B = idx.size(0)
        finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
        max_new_tokens = min(max_new_tokens, self.cfg.block_size - idx.size(1))
        logits, _, past = self(idx)                         # 预填充（prefill）
        logits = logits[:, -1, :].float()
        out = idx
        for step in range(max_new_tokens):
            if repetition_penalty != 1.0:                   # 抑制重复（小模型常见问题）
                score = torch.gather(logits, 1, out)
                score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
                logits = logits.scatter(1, out, score)
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                if top_p and top_p < 1.0:                    # nucleus 采样
                    sp, si = torch.sort(probs, descending=True)
                    cum = sp.cumsum(-1)
                    sp[(cum - sp) > top_p] = 0
                    probs = torch.zeros_like(probs).scatter(1, si, sp)
                    probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1)
            if eos_id is not None:
                nxt = torch.where(finished.unsqueeze(1), torch.full_like(nxt, eos_id), nxt)
                finished |= nxt.squeeze(1) == eos_id
            out = torch.cat([out, nxt], dim=1)
            if (eos_id is not None and finished.all()) or step == max_new_tokens - 1:
                break
            logits, _, past = self(nxt, past_kv=past, start_pos=out.size(1) - 1)
            logits = logits[:, -1, :].float()
        return out


def save_checkpoint(model, path, extra=None):
    torch.save({"model": model.state_dict(), "config": asdict(model.cfg), "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    m = MiniLM(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["model"])
    return m.to(device)
