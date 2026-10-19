"""
理想观察者（Ideal Bayesian Observer）—— 评测的上限参照线
=========================================================
它知道所有规则族（3 种灌溉 × 2 种气候 × 2 种贸易）和抖动范围，
只用上下文窗口里那 k 年的观测（收成、贸易所得、天气）做贝叶斯推断，然后预测下一年。

说明：为了计算土壤盐碱化，它可以看到窗口之前的水渠修建记录（比语言模型多一点信息），
因此它是"偏强"的上限。语言模型越接近它，说明从上下文推断隐藏规律的能力越强。

三个隐藏规律相互独立，分别推断：
  收成  ← (灌溉规则, 基础产量抖动)
  贸易  ← (贸易规则, 汇率抖动)
  天气  ← 气候规则
"""
import copy
import math

import numpy as np

from engine import (BASE_YIELD, IRR_GAIN, P_BOUNTIFUL, P_DROUGHT, RULES, TRADE_RATE, YIELD_NOISE, Params,
                    harvest_value, next_soil, step)

Y0_GRID = np.linspace(0.85, 1.15, 31) * BASE_YIELD
TR_GRID = np.linspace(0.85, 1.15, 31)
SIGMA_LOG = YIELD_NOISE   # 收成有约 10% 的随机波动（观察者知道波动幅度，但不知道具体值）
SIGMA_TRADE = 0.02        # 贸易所得没有随机波动，只有取整误差


def soil_series(years, irrigation):
    """按水渠历史算出每年年初（收成时）的土壤肥力。"""
    soil, out = 1.0, []
    for y in years:
        out.append(soil)
        st, o = y["state"], y["out"]
        canals = st.canals + (1 if (o["act"] == "canal" and o["ok"]) else 0)
        soil = next_soil(soil, canals, irrigation)
    return out


def post_action(st, o):
    s = copy.copy(st)
    if o["ok"] and o["act"] in ("farm", "canal", "house"):
        attr = {"farm": "farms", "canal": "canals", "house": "houses"}[o["act"]]
        setattr(s, attr, getattr(s, attr) + 1)
    if o["ok"] and o["act"] == "research" and o["tech_up"]:
        s.tech += 1
    return s


class Observer:
    def __init__(self, ep):
        self.ep = ep
        self.years = ep["years"]
        self.soils = {irr: soil_series(self.years, irr) for irr in RULES["irrigation"]}

    def _harvest(self, i, irr, y0):
        st, o = self.years[i]["state"], self.years[i]["out"]
        s = post_action(st, o)
        return harvest_value(s.farms, s.pop, s.canals, s.tech, self.soils[irr][i], o["weather"], y0, IRR_GAIN[irr])

    def posterior(self, window):
        """window: 可见年份的下标列表。返回各隐藏变量的后验。"""
        # 收成：灌溉 × y0
        hyps, logw = [], []
        for irr in RULES["irrigation"]:
            for y0 in Y0_GRID:
                lw = 0.0
                for i in window:
                    pred = max(1, self._harvest(i, irr, y0))
                    obs = max(1, self.years[i]["out"]["harvest"])
                    lw += -0.5 * ((math.log(obs) - math.log(pred)) / SIGMA_LOG) ** 2
                hyps.append((irr, y0))
                logw.append(lw)
        w = np.exp(np.array(logw) - max(logw))
        w /= w.sum()
        # 贸易
        th, tl = [], []
        for tr in RULES["trade"]:
            for j in TR_GRID:
                rate = TRADE_RATE[tr] * j
                lw = 0.0
                for i in window:
                    o = self.years[i]["out"]
                    if o["trade_in"]:
                        lw += -0.5 * ((math.log(o["trade_in"]) - math.log(round(30 * rate))) / SIGMA_TRADE) ** 2
                th.append((tr, rate))
                tl.append(lw)
        tw = np.exp(np.array(tl) - max(tl))
        tw /= tw.sum()
        # 气候
        cl = {}
        for c in RULES["climate"]:
            pd, pb = P_DROUGHT[c], P_BOUNTIFUL
            lw = 0.0
            for i in window:
                wx = self.years[i]["out"]["weather"]
                lw += math.log(pd if wx == "drought" else (pb if wx == "bountiful" else 1 - pd - pb))
            cl[c] = lw
        m = max(cl.values())
        z = sum(math.exp(v - m) for v in cl.values())
        cpost = {c: math.exp(v - m) / z for c, v in cl.items()}
        return hyps, w, th, tw, cpost

    def predict(self, window, t):
        """预测第 t 年（下标）的收成、粮食方向和干旱概率。"""
        hyps, w, th, tw, cpost = self.posterior(window)
        st, o = self.years[t]["state"], self.years[t]["out"]
        rate = float(np.dot(tw, [r for _, r in th]))
        logs, dirs = [], {"up": 0.0, "down": 0.0, "same": 0.0}
        top = np.argsort(-w)[:60]                 # 只算权重最大的若干假设，足够精确
        tot = w[top].sum()
        for k in top:
            irr, y0 = hyps[k]
            P = Params(irrigation=irr, climate="wet", trade="rich", y0=y0, trade_rate=rate)
            s = copy.copy(st)
            s.soil = self.soils[irr][t]
            _, oo = step(P, s, o["act"], o["weather"])
            logs.append((math.log(max(1, oo["harvest"])), w[k] / tot))
            dirs[oo["food_dir"]] += w[k] / tot
        logs.sort()
        acc, med = 0.0, logs[-1][0]
        for v, ww in logs:                         # 加权中位数（最小化对数误差）
            acc += ww
            if acc >= 0.5:
                med = v
                break
        p_drought = sum(cpost[c] * P_DROUGHT[c] for c in cpost)
        return {"harvest": math.exp(med), "food_dir": max(dirs, key=dirs.get), "p_drought": p_drought,
                "p_irrigation": {irr: float(sum(w[i] for i, h in enumerate(hyps) if h[0] == irr))
                                 for irr in RULES["irrigation"]}}
