"""
CivLM 游乐场：和语言模型一起推演文明
====================================
python play.py            → 浏览器打开 http://127.0.0.1:7880

每一年：
  1. 你选择一个行动
  2. 模型先猜天气（P(干旱)），再看到真实天气后预测这一年的结果
  3. 模拟器揭晓真实结果，和模型、理想观察者的预测并排对比
你还可以：回到过去改写某一年（反事实）、让模型自己做梦续写几年、猜这个世界的隐藏规则。
"""
import argparse
import json
import math
import os
import random
import re
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F

from chronicle import FOOD_DIR_WORDS, Renderer
from engine import ACTIONS, ALL_COMBOS, HELDOUT_COMBOS, RULES, make_world, step
from model import load_checkpoint
from observer import Observer
from tokenizer import BPETokenizer
from utils import amp_setup, get_device

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = {}
LOCK = threading.Lock()


class LM:
    def __init__(self, ckpt, data_dir, device):
        self.device = get_device(device)
        self.model = load_checkpoint(ckpt, self.device).eval()
        self.block = self.model.cfg.block_size
        self.tok = BPETokenizer(os.path.join(data_dir, "tokenizer.json"))
        self.amp, _ = amp_setup(self.device)
        print(f"模型 {ckpt}：{self.model.num_params() / 1e6:.1f}M 参数，设备 {self.device}")

    def ids(self, header, body, reserve):
        h, b = self.tok.encode(header), self.tok.encode(body)
        room = self.block - reserve - len(h)
        truncated = len(b) > room
        if truncated:
            b = b[-room:]
        return h + b, truncated

    @torch.no_grad()
    def weather_probs(self, header, body, cands):
        ids, _ = self.ids(header, body, 40)
        with self.amp:
            logits, _, past = self.model(torch.tensor([ids], device=self.device))
            last = F.log_softmax(logits[0, -1].float(), -1)
            cls = {}
            for w, texts in cands.items():
                lps = []
                for t in texts:
                    c = self.tok.encode(t)
                    lp = last[c[0]].item()
                    if len(c) > 1:
                        lg, _, _ = self.model(torch.tensor([c[:-1]], device=self.device), past_kv=past, start_pos=len(ids))
                        ls = F.log_softmax(lg[0].float(), -1)
                        lp += ls[torch.arange(len(c) - 1), torch.tensor(c[1:], device=self.device)].sum().item()
                    lps.append(lp)
                m = max(lps)
                cls[w] = m + math.log(sum(math.exp(v - m) for v in lps))
        m = max(cls.values())
        z = sum(math.exp(v - m) for v in cls.values())
        return {w: math.exp(v - m) / z for w, v in cls.items()}

    @torch.no_grad()
    def generate(self, header, body, max_new, temperature=0.0):
        ids, trunc = self.ids(header, body, max_new)
        with self.amp:
            out = self.model.generate(torch.tensor([ids], device=self.device), max_new,
                                      temperature=temperature, top_k=40 if temperature else None,
                                      eos_id=self.tok.eot_id)
        return self.tok.decode(out[0, len(ids):].tolist()), trunc, len(ids)


def parse_outcome(text):
    text = text.split("\n\n")[0]
    r = {"harvest": None, "food": None, "food_dir": None, "pop": None, "trade_in": None, "text": text.strip()}
    m = re.search(r"\d+", text)
    if m:
        r["harvest"] = int(m.group())
    m = re.search(r"food (\d+) (up|down|same)", text)
    if m:
        r["food"], r["food_dir"] = int(m.group(1)), m.group(2)
    else:
        m = re.search(r"\b(rose|climbed|fell|sank|held)(?: at| to)? (\d+)", text)
        if m:
            r["food_dir"], r["food"] = FOOD_DIR_WORDS[m.group(1)], int(m.group(2))
    m = re.search(r"pop (\d+)", text) or re.search(r"(?:to|at) (\d+)\.?\s*$", text)
    if m:
        r["pop"] = int(m.group(1))
    m = re.search(r"trade \+(\d+)", text) or re.search(r"(\d+) measures of grain for the timber|left (\d+) measures", text)
    if m:
        r["trade_in"] = int(next(g for g in m.groups() if g))
    return r


def st_dict(s):
    return {"year": s.year, "pop": s.pop, "food": s.food, "wood": s.wood, "farms": s.farms, "canals": s.canals,
            "cap": s.cap, "tech": s.tech, "mood": s.mood}


class Session:
    def __init__(self, seed, rules, style, years=200):
        self.seed, self.style = seed, style
        self.P, self.s0, self.weather, self.noise = make_world(seed, rules, years)
        self.years = []          # [{"state","out"}]
        self.segs = []           # 每年的 4 段文本
        self.turns = []          # 每年的预测记录
        self.r = Renderer({"seed": seed}, style)
        self.header = self.r.header()
        self.branch_from = None
        self.cur = self.s0

    @property
    def state(self):
        return self.cur

    def history_text(self):
        return "".join("".join(s) for s in self.segs)

    def public(self):
        return {"seed": self.seed, "style": self.style, "header": self.header, "state": st_dict(self.state),
                "turns": self.turns, "branch_from": self.branch_from}


def do_turn(lm, S: Session, act):
    i = len(S.years)
    s = S.state
    s2, o = step(S.P, s, act, S.weather[i], S.noise[i])
    rec = {"state": s, "out": o}
    segs = S.r.year(rec)
    hist = S.history_text()
    # 1) 模型在看到天气之前：P(天气)
    pw = lm.weather_probs(S.header, hist + segs[0] + segs[1], S.r.weather_candidates())
    # 2) 看到天气后：续写结果
    gen, trunc, ntok = lm.generate(S.header, hist + segs[0] + segs[1] + segs[2], 64)
    pred = parse_outcome(gen)
    # 3) 理想观察者
    ob = Observer({"years": S.years + [rec]})
    window = list(range(max(0, i - 12), i))
    obs = ob.predict(window, i)
    S.years.append(rec)
    S.segs.append(segs)
    S.cur = s2
    turn = {
        "year": s.year, "act": act, "ok": o["ok"], "weather": o["weather"],
        "state_before": st_dict(s),
        "text": {"state": segs[0].strip(), "act": segs[1].strip(), "weather": segs[2].strip(), "outcome": segs[3].strip()},
        "truth": {"harvest": o["harvest"], "food": o["food"], "food_dir": o["food_dir"], "pop": o["pop"],
                  "trade_in": o["trade_in"], "deaths": o["deaths"], "left": o["left"]},
        "lm": dict(pred, p_weather=pw, truncated=trunc, ctx_tokens=ntok),
        "obs": {"harvest": round(obs["harvest"]), "food_dir": obs["food_dir"], "p_drought": obs["p_drought"],
                "p_irrigation": obs["p_irrigation"]},
    }
    S.turns.append(turn)
    return turn


def dream(lm, S: Session, n_years, temperature):
    """让模型从当前历史出发，自己写接下来几年（行动、天气、结果全由模型生成）。"""
    hist = S.history_text()
    seg0 = S.r.year({"state": S.state, "out": {"act": "rest", "ok": True, "weather": "normal", "harvest": 0,
                                               "trade_in": 0, "feast_cost": 0, "tech_up": False, "deaths": 0,
                                               "left": 0, "crowded": False, "food": 0, "pop": 0,
                                               "food_dir": "same", "pop_dir": "same"}})[0]
    per_year = 70 if S.style == "structured" else 150
    text, _, _ = lm.generate(S.header, hist + seg0, min(per_year * n_years, 1000), temperature)
    full = seg0 + text
    parts = [p.strip() for p in full.split("\n\n") if p.strip()]
    return parts[:n_years]


class H(BaseHTTPRequestHandler):
    lm = None

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if "json" in ctype or "html" in ctype else ""))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "play.html"), "rb") as f:
                return self._send(200, f.read(), "text/html")
        if self.path == "/api/meta":
            return self._send(200, {"actions": ACTIONS, "rules": RULES, "heldout": HELDOUT_COMBOS})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with LOCK:
                return self._send(200, self.route(req))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def route(self, req):
        p = self.path
        if p == "/api/new":
            seed = int(req.get("seed") or random.randint(1, 10 ** 7))
            mode = req.get("rules", "random")
            if mode == "random":
                rules = random.Random(seed).choice(ALL_COMBOS)
            elif mode == "heldout":
                rules = random.Random(seed).choice(HELDOUT_COMBOS)
            else:
                rules = mode
            sid = uuid.uuid4().hex[:8]
            SESS[sid] = Session(seed, rules, req.get("style", "structured"))
            return dict(SESS[sid].public(), sid=sid)
        S = SESS[req["sid"]]
        if p == "/api/turn":
            t = do_turn(self.lm, S, req["act"])
            return {"turn": t, "state": st_dict(S.state)}
        if p == "/api/rewind":
            y = int(req["year"])          # 回到第 y 年年初
            k = y - 1
            if 0 <= k < len(S.years):
                S.cur = S.years[k]["state"]
            S.years, S.segs, S.turns = S.years[:k], S.segs[:k], S.turns[:k]
            S.branch_from = y
            return S.public()
        if p == "/api/dream":
            return {"years": dream(self.lm, S, int(req.get("n", 5)), float(req.get("temperature", 0.8)))}
        if p == "/api/reveal":
            in_train = S.P.rules not in HELDOUT_COMBOS
            return {"rules": S.P.rules, "y0": round(S.P.y0, 1), "trade_rate": round(S.P.trade_rate, 2),
                    "seen_in_training": in_train}
        raise ValueError(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/civlm26m/best.pt")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--port", type=int, default=7880)
    ap.add_argument("--no_browser", action="store_true")
    args = ap.parse_args()
    os.chdir(HERE)
    H.lm = LM(args.ckpt, args.data_dir, args.device)
    url = f"http://127.0.0.1:{args.port}"
    print(f"打开 {url}  （Ctrl+C 退出）")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
