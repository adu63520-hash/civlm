"""
CivChronicle v0：把模拟器的历史写成文本
=======================================
两种文体，信息量完全相同，便于做"语言起了什么作用"的消融实验：
  structured  紧凑的 DSL 行
  narrative   英文叙事，每类事件有多种说法（防止模型只记模板）

每一年分成 4 段：state（年初状态）/ act（行动）/ weather（天气）/ outcome（结果）。
评测时可以在任意一段之后截断，让模型续写。
"""
import random

from engine import ACTIONS, WEATHERS

NAMES = ["Arvel", "Brannoch", "Celedor", "Dunmar", "Eskara", "Fennick", "Galdur", "Hollin", "Iskarn", "Jorvel",
         "Kestrel", "Lunmoor", "Marrow", "Nethis", "Orlan", "Pellam", "Quarry", "Rennick", "Sorrow", "Tamsin",
         "Ulvar", "Vessa", "Wyndham", "Yarrow", "Zennor"]
# 地貌描述只是"风味"，与隐藏规则无关：模型不能靠看名字猜规则
FLAVORS = ["a wide valley", "a stretch of low hills", "a plain beside a slow river", "the edge of an old forest",
           "a high plateau", "a shallow basin", "a windy coast", "a land of scattered lakes"]


def pl(n, word, plural=None):
    return f"{n} {word if n == 1 else (plural or word + 's')}"


# ---------------- structured ----------------
def s_header(ep):
    r = random.Random(ep["seed"])
    return f"<world> {r.choice(NAMES)}{ep['seed'] % 1000} | {r.choice(FLAVORS)}\n"


def s_state(st):
    return (f"Y{st.year} pop {st.pop} food {st.food} wood {st.wood} farms {st.farms} canals {st.canals} "
            f"cap {st.cap} tech {st.tech} mood {st.mood}\n")


def s_act(o):
    return f"act {o['act']}{'' if o['ok'] else ' failed'}\n"


def s_weather(o):
    return f"weather {o['weather']}\n"


def s_outcome(o):
    parts = [f"harvest {o['harvest']}"]
    if o["trade_in"]:
        parts.append(f"trade +{o['trade_in']}")
    if o["feast_cost"]:
        parts.append(f"feast -{o['feast_cost']}")
    if o["tech_up"]:
        parts.append("techup")
    if o["deaths"]:
        parts.append(f"starved {o['deaths']}")
    if o["left"]:
        parts.append(f"left {o['left']}")
    if o["crowded"]:
        parts.append("crowded")
    parts.append(f"food {o['food']} {o['food_dir']}")
    parts.append(f"pop {o['pop']} {o['pop_dir']}")
    return " ".join(parts) + "\n\n"


# ---------------- narrative ----------------
N_HEADER = [
    "This is the chronicle of {name}, a settlement in {flavor}.\n\n",
    "The people of {name} made their home in {flavor}. Here is their history.\n\n",
    "A chronicle of {name}, founded in {flavor}.\n\n",
]
N_STATE = [
    "Year {y}. {pop} people lived in {name}, with {food} measures of grain and {wood} loads of timber in store. "
    "They worked {farms} and {canals}, had room for {cap} people, knowledge level {tech}, and their mood stood at {mood}.",
    "In year {y}, {name} numbered {pop} souls. The granaries held {food} measures and the woodpiles {wood} loads; "
    "there were {farms}, {canals}, housing for {cap}, knowledge level {tech} and mood {mood}.",
    "Year {y} began with {pop} people, {food} grain, {wood} timber, {farms}, {canals}, space for {cap}, "
    "knowledge level {tech} and mood {mood}.",
]
N_ACT = {
    "farm": ["The council ordered new fields cleared for a farm.", "The elders decided to open another farm."],
    "canal": ["The council ordered a new canal dug.", "The elders set the people to digging an irrigation canal."],
    "forest": ["Woodcutters were sent into the forest.", "The council sent crews to fell timber."],
    "trade": ["A caravan of timber was sent to the traders.", "The elders offered timber to passing merchants."],
    "research": ["The scholars were given leave to study.", "The council invested in learning."],
    "feast": ["The council declared a great feast.", "The elders held a festival for the people."],
    "house": ["The council ordered new houses built.", "Builders raised new homes."],
    "rest": ["The people were allowed a quiet year.", "The council asked little of the people this year."],
}
N_FAIL = ["But there was not enough to carry it out.", "But the plan came to nothing."]
N_WEATHER = {
    "normal": ["The seasons passed without trouble.", "The rains came as usual.", "It was an ordinary year for weather."],
    "drought": ["A drought parched the fields.", "The rains failed and the land dried out.", "Drought struck the land."],
    "bountiful": ["The rains were generous.", "Gentle rains blessed the fields.", "The weather was kind to the farmers."],
}
N_HARVEST = ["The harvest brought in {h} measures of grain.", "The fields yielded {h} measures.",
             "{h} measures of grain were gathered."]
N_TRADE = ["The traders gave {g} measures of grain for the timber.", "In exchange, merchants left {g} measures of grain."]
N_FEAST = ["The feast consumed {c} measures.", "Feasting used up {c} measures of grain."]
N_TECH = ["The scholars reached a new level of knowledge.", "Learning advanced to a new level."]
N_DEATH = ["Hunger killed {d} people.", "{d} people died of hunger."]
N_LEFT = ["{l} people left in discontent.", "Unhappy, {l} people abandoned the settlement."]
N_CROWD = ["Crowding soured the mood.", "There was not enough room for everyone."]
# 粮食方向词与人口方向词不重叠，方便评测时解析
N_FOOD = {"up": ["The stores rose to {f} measures.", "The granaries climbed to {f} measures."],
          "down": ["The stores fell to {f} measures.", "The granaries sank to {f} measures."],
          "same": ["The stores held at {f} measures."]}
N_POP = {"up": ["The population grew to {p}.", "Their number increased to {p}."],
         "down": ["The population shrank to {p}.", "Their number dropped to {p}."],
         "same": ["The population stayed at {p}."]}
FOOD_DIR_WORDS = {"rose": "up", "climbed": "up", "fell": "down", "sank": "down", "held": "same",
                  "up": "up", "down": "down", "same": "same"}


def n_header(ep, rng):
    r = random.Random(ep["seed"])
    return rng.choice(N_HEADER).format(name=r.choice(NAMES), flavor=r.choice(FLAVORS))


def n_state(st, rng, name):
    return rng.choice(N_STATE).format(y=st.year, pop=st.pop, name=name, food=st.food, wood=st.wood,
                                      farms=pl(st.farms, "farm"), canals=pl(st.canals, "canal"),
                                      cap=st.cap, tech=st.tech, mood=st.mood)


def n_act(o, rng):
    t = " " + rng.choice(N_ACT[o["act"]])
    if not o["ok"]:
        t += " " + rng.choice(N_FAIL)
    return t


def n_weather(o, rng):
    return " " + rng.choice(N_WEATHER[o["weather"]])


def n_outcome(o, rng):
    p = [rng.choice(N_HARVEST).format(h=o["harvest"])]
    if o["trade_in"]:
        p.append(rng.choice(N_TRADE).format(g=o["trade_in"]))
    if o["feast_cost"]:
        p.append(rng.choice(N_FEAST).format(c=o["feast_cost"]))
    if o["tech_up"]:
        p.append(rng.choice(N_TECH))
    if o["deaths"]:
        p.append(rng.choice(N_DEATH).format(d=o["deaths"]))
    if o["left"]:
        p.append(rng.choice(N_LEFT).format(l=o["left"]))
    if o["crowded"]:
        p.append(rng.choice(N_CROWD))
    p.append(rng.choice(N_FOOD[o["food_dir"]]).format(f=o["food"]))
    p.append(rng.choice(N_POP[o["pop_dir"]]).format(p=o["pop"]))
    return " " + " ".join(p) + "\n\n"


# ---------------- 统一接口 ----------------
class Renderer:
    def __init__(self, ep, style, rng_seed=None):
        self.ep, self.style = ep, style
        self.rng = random.Random(rng_seed if rng_seed is not None else ep["seed"] * 13 + 5)
        self.name = random.Random(ep["seed"]).choice(NAMES)

    def header(self):
        return s_header(self.ep) if self.style == "structured" else n_header(self.ep, self.rng)

    def year(self, y):
        """返回 [state, act, weather, outcome] 四段文本，直接拼接即为该年全文。"""
        st, o = y["state"], y["out"]
        if self.style == "structured":
            return [s_state(st), s_act(o), s_weather(o), s_outcome(o)]
        return [n_state(st, self.rng, self.name), n_act(o, self.rng), n_weather(o, self.rng), n_outcome(o, self.rng)]

    def weather_candidates(self, act_seg_end=True):
        """天气的所有可能说法，按类别分组（用于评测时计算 P(天气类别)）。"""
        if self.style == "structured":
            return {w: [f"weather {w}\n"] for w in WEATHERS}
        return {w: [" " + t for t in N_WEATHER[w]] for w in WEATHERS}

    def whatif(self, year, act):
        if self.style == "structured":
            return f"whatif Y{year} act {act}\n"
        return f"What if: in year {year} the council had chosen differently. " \
               f"{N_ACT[act][0]}\n\n"


def render_episode(ep, style, cf=None):
    """完整文档。cf = (year_idx, alt_action, branch_years) 时，在主历史后附一段反事实分支。"""
    r = Renderer(ep, style)
    out = [r.header()]
    for y in ep["years"]:
        out += r.year(y)
    if cf is not None:
        idx, alt, branch = cf
        out.append(r.whatif(ep["years"][idx]["state"].year, alt))
        for y in branch:
            out += r.year(y)
    return "".join(out)


if __name__ == "__main__":
    from engine import simulate
    ep = simulate(3, None, 4)
    print(render_episode(ep, "structured"))
    print(render_episode(ep, "narrative"))
