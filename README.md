# CivLM：一个从零训练的小模型，能从历史里看出陌生世界的规律吗？

> Can a small language model infer the laws of an unseen world from its history?

我们写了一个小型文明模拟器。每个世界有 3 条隐藏规律：灌溉、气候、贸易。用它生成 3 万个虚构文明的历史书（约 1 亿 token），从零训练了一个 2600 万参数的语言模型。模型从未被告知任何规律，只做下一个词预测。

📖 **文章：[它没学过物理，但它发现了盐碱地](docs/ARTICLE.md)**

![游戏界面](docs/images/game_ui.png)

## 主要发现

在训练中**从未出现过的规律组合**上测试：

| 发现 | 数字 |
| --- | --- |
| 只看 1 年历史，收成预测误差就降到接近理想上限 | 22% → 11%（理想观察者 11%） |
| 看 12 年历史后，干旱概率估计与贝叶斯理想观察者几乎重合 | 干燥世界 29%（观察者 31%），湿润世界 8%（观察者 8%） |
| 看过一次贸易后，下一次贸易所得预测完全准确 | 第一次猜 23，实际 60；第二次猜 60 |
| 反事实：修渠带来的收益差 | 增产世界 +2285（真实 +2299），盐碱化世界 +559（真实 +656） |
| 有趣的毛病：收成预测的个位数 | 66% 是 8 |

![收成误差随历史长度变化](docs/images/chart_harvest_vs_history.png)
![干旱预报](docs/images/chart_drought_forecast.png)

## 快速开始

### 1. 只想玩（需要训练好的权重）

```bat
pip install torch numpy tokenizers
```

从本仓库的 **Releases** 页面下载 `civlm26m.pt`（约 53MB），放到 `out/civlm26m/best.pt`，然后：

```bat
python play.py        REM 浏览器打开 http://127.0.0.1:7880
```

Windows 下也可以直接双击 `start_play.bat`。有 GPU 时用 GPU，没有就用 CPU，每年约 2～3 秒。

### 2. 自己训练

**用 Colab（推荐，A100 约几十分钟）**：上传 `colab_train.ipynb`，运行时选 A100，然后「全部运行」。代码已内嵌在笔记本里。

**本地（单卡）**：

```bat
pip install torch numpy tokenizers matplotlib
python gen_data.py                       REM 生成 3 万个文明的历史（默认 2 万，可用 --n_worlds 30000）
python pretrain.py                       REM 2600 万参数，5000 步
python eval_icl.py --ckpt out/civlm26m/best.pt                        REM 实验 2：上下文长度 vs 误差
python eval_icl.py --ckpt_dir out/civlm26m/ckpts --ks 0 4 12 --styles structured   REM 实验 1：学习曲线
```

或双击 `run_all.bat`。

## 文件

| 文件 | 作用 |
| --- | --- |
| `engine.py` | 确定性文明模拟器：8 种行动、3 种天气、3 条隐藏规律、可从任意一年分叉做反事实 |
| `chronicle.py` | 把历史写成文本：结构化 DSL 和英文叙事两种文体，信息量相同 |
| `gen_data.py` | 生成训练数据，按规律组合划分训练集和测试集，训练 BPE 分词器 |
| `observer.py` | 理想贝叶斯观察者：知道全部规律族，作为评测上限 |
| `model.py` | LLaMA 结构小模型（RMSNorm、RoPE、SwiGLU、KV Cache） |
| `pretrain.py` | 从零预训练，定期保存检查点 |
| `eval_icl.py` | 实验 1（学习曲线）和实验 2（上下文系统辨识） |
| `play.py` + `play.html` | 交互游戏：每年选行动，看模型、观察者和真实结果的对比；可改写历史、让模型做梦 |
| `colab_train.ipynb` | Colab 一键训练 |
| `results/` | 我们这次训练的评测结果（JSON） |
| `data/` | 分词器、评测世界清单（训练数据可用 `gen_data.py` 重新生成，完全可复现） |

## 隐藏规律

| 规律 | 取值 | 效果 |
| --- | --- | --- |
| 灌溉 | boost / salinize / weak | 每条水渠 +30% / +30% 但土壤逐年盐碱化 / 只 +5% |
| 气候 | wet / dry | 干旱概率 5% / 35% |
| 贸易 | rich / poor | 30 木材约换 66 / 21 粮食 |

训练中完全不出现的组合：`salinize + dry + rich`、`boost + wet + poor`。

## 局限

- 收成预测只需 1 年历史就能接近上限，任务偏简单；需要长历史的只有天气推断。
- 理想观察者只用收成记录，不会从人口、存粮等状态里读线索，所以在零历史时输给了模型。
- 这是一个虚构的合成世界，结论不代表模型能理解真实历史。
