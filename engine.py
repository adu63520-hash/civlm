"""
CivEngine v0：确定性的文明模拟器
================================
- 1 个文明，每年：玩家/策略选择一个行动 → 外部天气 → 结算收成、贸易、人口、情绪。
- 每个世界有 3 个【隐藏规则】，语言模型永远看不到，只能从历史中推断：
    irrigation  灌溉   boost    每条水渠 +30% 产量
                       salinize 同样 +30%，但土壤逐年盐碱化，产量随后下降
                       weak     每条水渠只 +5%
    climate     气候   wet      干旱概率 5%
                       dry      干旱概率 35%
    trade       贸易   rich     30 木材换约 66 粮食
                       poor     30 木材换约 21 粮食
  另有连续的隐藏参数（单块农田基础产量、贸易汇率的 ±15% 抖动），防止模型只背下 12 种世界。
- 完全可复现：同一 (seed, rules) 永远生成同一段历史。
- 天气序列事先由种子抽好（外部事件 E_t），因此可以从任意一年复制世界做反事实分叉，
  两个分支面对完全相同的天气。
"""
import copy
import itertools
import math
import random
from dataclasses import asdict, dataclass

RULES = {
    "irrigation": ["boost", "salinize", "weak"],
    "climate": ["wet", "dry"],
    "trade": ["rich", "poor"],
}
ALL_COMBOS = [dict(zip(RULES, v)) for v in itertools.product(*RULES.values())]
# 训练中完全不出现的规则组合（每个规则取值单独都见过，只是组合没见过 = 组合泛化）
HELDOUT_COMBOS = [
    {"irrigation": "salinize", "climate": "dry", "trade": "rich"},
    {"irrigation": "boost", "climate": "wet", "trade": "poor"},
]
TRAIN_COMBOS = [c for c in ALL_COMBOS if c not in HELDOUT_COMBOS]

IRR_GAIN = {"boost": 0.30, "salinize": 0.30, "weak": 0.05}
P_DROUGHT = {"wet": 0.05, "dry": 0.35}
P_BOUNTIFUL = 0.12
TRADE_RATE = {"rich": 2.2, "poor": 0.7}
BASE_YIELD = 45.0
SALT_PER_CANAL = 0.06      # 盐碱化：每条水渠每年使土壤肥力下降 0.06
SOIL_MIN = 0.35
SOIL_RECOVER = 0.05
YIELD_NOISE = 0.10         # 每年产量的随机波动（外部因素，分叉时保持一致）

ACTIONS = ["farm", "canal", "forest", "trade", "research", "feast", "house", "rest"]
WEATHERS = ["normal", "drought", "bountiful"]
COST = {"farm": 25, "canal": 35, "house": 30, "trade": 30}
MAX_CANALS, MAX_HOUSES, MAX_TECH = 3, 10, 4
PEOPLE_PER_FARM = 25
HOUSE_CAP = 50


@dataclass
class Params:               # 隐藏的世界参数
    irrigation: str
    climate: str
    trade: str
    y0: float               # 单块农田基础产量
    trade_rate: float       # 每 1 木材换多少粮食

    @property
    def rules(self):
        return {"irrigation": self.irrigation, "climate": self.climate, "trade": self.trade}


@dataclass
class State:
    year: int
    pop: int
    food: int
    wood: int
    farms: int
    canals: int
    houses: int
    tech: int
    rp: int                  # 研究进度（不显示）
    mood: int
    soil: float              # 土壤肥力（隐藏）

    @property
    def cap(self):
        return HOUSE_CAP * self.houses


def weather_mult(weather, canals):
    if weather == "drought":
        return min(1.0, 0.5 + 0.15 * canals)      # 水渠缓解干旱（所有世界相同的公开规律）
    return 1.25 if weather == "bountiful" else 1.0


def harvest_value(farms, pop, canals, tech, soil, weather, y0, irr_gain, noise=1.0):
    """收成公式。模拟器和理想观察者共用，保证一致。"""
    worked = min(farms, pop / PEOPLE_PER_FARM)       # 劳动力不足时农田会闲置
    return round(worked * y0 * (1 + irr_gain * canals) * soil * weather_mult(weather, canals)
                 * (1 + 0.08 * tech) * noise)


def next_soil(soil, canals, irrigation):
    if irrigation == "salinize" and canals > 0:
        return max(SOIL_MIN, soil - SALT_PER_CANAL * canals)
    return min(1.0, soil + SOIL_RECOVER)


def step(P: Params, s: State, act: str, weather: str, noise: float = 1.0):
    """执行一年。返回 (新状态, 结果记录)。不修改输入。"""
    s = copy.copy(s)
    o = {"act": act, "ok": True, "weather": weather, "trade_in": 0, "feast_cost": 0,
         "tech_up": False, "deaths": 0, "left": 0, "crowded": False}
    food0, pop0 = s.food, s.pop

    # 1. 行动
    if act in ("farm", "canal", "house"):
        limit = {"farm": 99, "canal": MAX_CANALS, "house": MAX_HOUSES}[act]
        attr = {"farm": "farms", "canal": "canals", "house": "houses"}[act]
        if s.wood >= COST[act] and getattr(s, attr) < limit:
            s.wood -= COST[act]
            setattr(s, attr, getattr(s, attr) + 1)
        else:
            o["ok"] = False
    elif act == "forest":
        s.wood += round(45 * (1 + 0.1 * s.tech))
    elif act == "trade":
        if s.wood >= COST["trade"]:
            s.wood -= COST["trade"]
            o["trade_in"] = round(COST["trade"] * P.trade_rate)
        else:
            o["ok"] = False
    elif act == "research":
        s.rp += 1
        if s.rp >= 2 + s.tech and s.tech < MAX_TECH:
            s.tech += 1
            s.rp = 0
            o["tech_up"] = True
    elif act == "feast":
        cost = round(0.15 * s.pop)
        if s.food >= cost:
            o["feast_cost"] = cost
            s.mood += 12
        else:
            o["ok"] = False
    elif act == "rest":
        s.mood += 4

    # 2. 收成（使用行动之后的农田/水渠数）
    o["harvest"] = harvest_value(s.farms, s.pop, s.canals, s.tech, s.soil, weather, P.y0, IRR_GAIN[P.irrigation], noise)
    s.soil = next_soil(s.soil, s.canals, P.irrigation)

    # 3. 粮食与人口
    food = s.food + o["harvest"] + o["trade_in"] - o["feast_cost"] - s.pop
    if food < 0:
        o["deaths"] = math.ceil(-food / 2)
        s.pop = max(5, s.pop - o["deaths"])
        food = 0
        s.mood -= 12
    else:
        food -= round(0.08 * food)               # 储存损耗
        g = 0.07 if food >= 0.5 * s.pop else 0.02
        s.pop += round(s.pop * g)
    s.food = food
    if s.pop > s.cap:
        s.pop = s.cap
        s.mood -= 5
        o["crowded"] = True

    # 4. 情绪、迁出、木材
    s.mood += round((60 - s.mood) * 0.15)
    s.mood = max(0, min(100, s.mood))
    if s.mood < 25:
        o["left"] = round(0.06 * s.pop)
        s.pop -= o["left"]
    s.wood += 8
    s.year += 1

    o["food_dir"] = "up" if s.food > food0 else ("down" if s.food < food0 else "same")
    o["pop_dir"] = "up" if s.pop > pop0 else ("down" if s.pop < pop0 else "same")
    o["food"], o["pop"] = s.food, s.pop
    return s, o


class Policy:
    """带随机性的启发式策略：每个文明有自己的"性格"（行动偏好），再根据处境调整。"""

    def __init__(self, rng):
        self.base = {a: rng.gammavariate(1.0, 1.0) + 0.15 for a in ACTIONS}
        self.base["canal"] += 0.6          # 保证灌溉经常出现，否则灌溉规则无从观察
        self.base["trade"] += 0.3

    def act(self, s: State, rng):
        w = dict(self.base)
        if s.food < 1.2 * s.pop:
            w["farm"] *= 3
            w["trade"] *= 2
        if s.wood < 30:
            w["forest"] *= 4
        if s.pop > 0.9 * s.cap:
            w["house"] *= 4
        if s.mood < 40:
            w["feast"] *= 3
            w["rest"] *= 2
        if s.canals >= MAX_CANALS:
            w["canal"] = 0.02
        if s.tech >= MAX_TECH:
            w["research"] = 0.05
        acts, ws = zip(*w.items())
        return rng.choices(acts, ws)[0]


def make_world(seed, rules=None, years=40):
    rng = random.Random(seed)
    rules = rules or rng.choice(ALL_COMBOS)
    P = Params(**rules, y0=BASE_YIELD * rng.uniform(0.85, 1.15),
               trade_rate=TRADE_RATE[rules["trade"]] * rng.uniform(0.85, 1.15))
    s0 = State(year=1, pop=rng.randint(80, 140), food=rng.randint(200, 400), wood=rng.randint(40, 120),
               farms=3, canals=0, houses=3, tech=0, rp=0, mood=60, soil=1.0)
    wr = random.Random(seed * 7919 + 17)           # 天气：独立的随机流，分叉时保持一致
    weather = []
    for _ in range(years + 20):
        u = wr.random()
        weather.append("drought" if u < P_DROUGHT[rules["climate"]] else ("bountiful" if u > 1 - P_BOUNTIFUL else "normal"))
    nr = random.Random(seed * 7919 + 29)
    noise = [min(1.25, max(0.75, nr.gauss(1.0, YIELD_NOISE))) for _ in range(years + 20)]
    return P, s0, weather, noise


def simulate(seed, rules=None, years=40):
    """生成一个文明的完整历史。years[i] = {"state": 年初状态, "out": 本年结果}"""
    P, s, weather, noise = make_world(seed, rules, years)
    prng = random.Random(seed * 104729 + 3)
    pol = Policy(prng)
    hist = []
    for i in range(years):
        a = pol.act(s, prng)
        s2, o = step(P, s, a, weather[i], noise[i])
        hist.append({"state": s, "out": o})
        s = s2
    return {"seed": seed, "params": P, "weather": weather, "noise": noise, "years": hist, "final": s}


def fork(ep, year_idx, alt_action, n_years=5, policy_seed=None):
    """反事实：从第 year_idx 年年初复制世界，改为执行 alt_action，之后用同样的天气继续 n_years 年。"""
    P, weather, noise = ep["params"], ep["weather"], ep["noise"]
    s = ep["years"][year_idx]["state"]
    prng = random.Random(policy_seed if policy_seed is not None else ep["seed"] * 31 + year_idx)
    pol = Policy(random.Random(ep["seed"] * 104729 + 3))     # 同一个文明性格
    hist = []
    for j in range(n_years):
        a = alt_action if j == 0 else pol.act(s, prng)
        s2, o = step(P, s, a, weather[year_idx + j], noise[year_idx + j])
        hist.append({"state": s, "out": o})
        s = s2
    return hist


if __name__ == "__main__":
    from collections import Counter
    for combo in ALL_COMBOS[:3]:
        ep = simulate(1, combo, 40)
        print(combo, "y0=%.1f" % ep["params"].y0)
        for y in ep["years"][:12]:
            st, o = y["state"], y["out"]
            print(f"  Y{st.year:2d} pop {st.pop:3d} food {st.food:4d} wood {st.wood:3d} farms {st.farms} canals {st.canals} "
                  f"soil {st.soil:.2f} | {o['act']:8s}{'' if o['ok'] else '(x)'} {o['weather']:9s} harvest {o['harvest']}")
    acts = Counter()
    pops = []
    for sd in range(300):
        ep = simulate(sd, None, 40)
        acts.update(y["out"]["act"] for y in ep["years"])
        pops.append(ep["final"].pop)
    print("action freq", acts.most_common())
    print("final pop min/median/max", min(pops), sorted(pops)[150], max(pops))
