# 基于LSTM模型的用户意图识别

构建一套基于 LSTM 的用户意图识别系统，核心目标是将用户自然语言文本精准分类到预设意图类别（如问候、了解病情、播放音乐等）。

系统需完成文本预处理（分词、构建词汇表、文本数字化）、模型构建与训练、意图预测三大核心任务。数据层面需生成多类别意图样本并划分训练 / 验证 / 测试集，保证数据多样性；

模型层面采用双向 LSTM 结合注意力机制，提升文本特征提取能力；

功能上需支持单条 / 批量文本意图预测，输出 Top-K 意图及置信度，并通过训练曲线、混淆矩阵评估模型性能。系统需具备实用性，支持模型保存 / 加载、交互式测试，同时通过早停机制防止过拟合，最终实现用户输入文本的意图快速、准确识别，满足对话系统的核心意图判断需求。

## 什么是 LSTM

**LSTM（Long Short-Term Memory，长短期记忆网络）** 是一种特殊的循环神经网络（RNN），专门用来处理**序列数据**（如文本、语音、时间序列），核心是解决普通 RNN 的「长距离依赖」和「梯度消失」问题。

### 为什么需要它

普通 RNN 逐字处理序列时，信息要一步步往后传。序列一长，早期的信息就会被「冲淡」（梯度消失），导致模型记不住前文。LSTM 通过引入「门控机制」让网络**有选择地记住或遗忘**信息。

### 核心思想：细胞状态 + 三个门

LSTM 在每个时间步维护一条「细胞状态」(cell state)，像一条传送带贯穿整个序列，信息可以几乎无损地长距离流动。三个「门」（本质是 0~1 的开关，由 sigmoid 控制）决定如何更新它：

| 门 | 作用 | 通俗理解 |
| --- | --- | --- |
| **遗忘门** (forget) | 决定丢弃旧记忆中的哪些 | 「这部分前文可以忘了」 |
| **输入门** (input) | 决定写入哪些新信息 | 「这个新词值得记住」 |
| **输出门** (output) | 决定本步输出什么 | 「当前该露出哪部分记忆」 |

这样它既能保留**长期**信息（传送带），又能关注**短期**变化，故名「长短期记忆」。

### 在本项目中的角色

对照下方架构图：文本经 `Embedding` 变成词向量序列后，**BiLSTM（双向 LSTM）** 从左到右、右到左各扫一遍，让每个词同时获得「上文 + 下文」语境（例如判断 `play some jazz` 属于 `play_music`，需要结合整句）；`Attention` 再给关键词更高权重，最后 `Linear` + softmax 输出各意图的概率。一句话：**LSTM = RNN + 门控记忆机制**，能在长序列中「记住该记的、忘掉该忘的」，非常适合根据整句话理解用户意图的文本分类任务。

## 架构

```
text → tokenize → vocab → padded ids → Embedding → BiLSTM → Attention → Linear → intent probabilities
```

代码包位于 `intent/` 目录下：

| 模块 | 职责 |
| --- | --- |
| `config.py` | 共享的超参数（hyperparameters）与产物（artifact）路径 |
| `data.py` | 基于模板（template）的合成意图样本 + 分层（stratified）划分 train/val/test |
| `snips.py` | 下载 / 缓存 / 解析 SNIPS 2017 真实意图数据集，复用同一套划分逻辑 |
| `preprocess.py` | 分词器（whitespace + CJK fallback）、词汇表（vocabulary）、数字化（numericalization） |
| `model.py` | Bidirectional LSTM + additive attention 分类器 |
| `train.py` | 训练循环（training loop）、early stopping、模型保存 / 加载、训练曲线 |
| `predict.py` | 单条 / 批量 Top-K 预测、注意力可解释输出与交互式 CLI |
| `evaluate.py` | 在 test split 上输出 confusion matrix 与 classification report |
| `demo.py` | 端到端流水线演示：逐步打印 分词 → 词表 ID → padding → 预测 → 注意力 |

意图类别（intent classes）：`greeting`、`goodbye`、`ask_illness`、`play_music`、`set_alarm`、`weather`。

## 安装

需要 Python 3.10–3.12。

本项目推荐使用 **Poetry** 管理依赖和运行程序。Poetry 会统一管理虚拟环境、锁定依赖版本（`poetry.lock`）并提供一致的命令入口，避免“本地能跑、线上不行”的环境差异。

如果尚未安装 Poetry，可参考官方文档：[https://python-poetry.org/docs/](https://python-poetry.org/docs/)

在项目根目录执行：

```bash
cd intent-recognition-demo
poetry install
```

## 使用

请在仓库根目录下运行所有命令，以保证 `intent` 包可被正确导入。

训练（生成数据，启用 early stopping 训练，并把模型与训练曲线图保存到 `artifacts/`）：

```bash
poetry run python -m intent.train
```

端到端流水线演示（**推荐第一步**，加载已保存的模型，逐步打印每个阶段的中间结果，最适合理解整条 pipeline）：

```bash
poetry run python -m intent.demo
# 或指定自己的句子
poetry run python -m intent.demo "play some jazz" "is the flu serious"
```

输出会依次展示：原始文本 → 分词结果 → 每个 token 对应的词表 ID（并标记 OOV 未登录词）→ padding 后的定长向量 → Top-K 意图及置信度 → 各 token 获得的**注意力权重**（直观看到模型"看"了哪些词）。

交互式预测（加载已保存的模型，并从 stdin 读取文本）：

```bash
poetry run python -m intent.predict
```

```
> what are the symptoms of the flu
  Top-K intents:
    ask_illness    ███████████████████· 93.3%
    weather        ·················· ·  2.1%
    greeting       ·················· ·  2.0%
  Attention (which words drove the decision):
    what           ████·············  18%
    are            ██···············   9%
    symptoms       ████████████████·  78%
    flu            ███████████████··  73%
```

注意力条会高亮模型在做判断时关注度最高的词（这里是 `symptoms` 与 `flu`），这正是"注意力机制"在做的事情。

以代码方式进行单条 / 批量预测（`explain()` 还能拿到 tokens 与注意力权重）：

```python
from intent.predict import IntentPredictor

predictor = IntentPredictor.from_saved()
print(predictor.predict("play some jazz for me"))          # 单条文本的 Top-K
print(predictor.predict_batch(["hello there", "set an alarm for noon"], k=2))

exp = predictor.explain("play some jazz for me")            # 可解释结果
print(exp.top_k)                                            # [(intent, 置信度), ...]
print(list(zip(exp.tokens, exp.attention)))                 # 每个 token 的注意力权重
```

在 test split 上评估（打印 classification report，并把 confusion matrix 保存到 `artifacts/`）：

```bash
poetry run python -m intent.evaluate
```

输出文件写入 `artifacts/`：
`model.pt`、`training_curves.png`、`confusion_matrix.png`。

## 使用真实数据集：SNIPS

除了内置的合成数据，本项目还支持 **SNIPS 2017 NLU benchmark**——一个被广泛使用的语音助手意图识别开源数据集，含 7 个真实意图：`AddToPlaylist`、`BookRestaurant`、`GetWeather`、`PlayMusic`、`RateBook`、`SearchCreativeWork`、`SearchScreeningEvent`。

只需给 `train` / `evaluate` 加上 `--dataset snips` 即可（首次运行会自动从 GitHub 下载并缓存到 `data/snips/`，无需额外安装依赖；后续从缓存读取）：

```bash
poetry run python -m intent.train --dataset snips
poetry run python -m intent.evaluate --dataset snips
```

`evaluate` 的 `--dataset` 必须与训练时一致，才能复现相同的 test split。预测/演示无需改动（直接加载已保存的模型）：

```bash
poetry run python -m intent.demo "play some jazz on spotify" "what is the weather in paris tomorrow"
```

与合成数据不同，SNIPS 是真实、含噪声的语料，准确率更贴近实际（约 87%），并能在 confusion matrix 中看到真实的易混淆意图（如 `RateBook` 与 `SearchCreativeWork`），演示更有说服力。`config.py` 的 `samples_per_intent`（默认 120）控制每个意图从 SNIPS 中抽取多少样本——调大可用更多数据、获得更高准确率，但训练更慢。

## 关于数据集的说明

训练数据是**合成的、基于模板（template）生成的**（见 `data.py`），因此该示例可在 CPU 上离线运行、几秒内完成，且无需任何下载。由于语句表达有限、各意图之间界限清晰，模型在 test split 上能达到很高的准确率（约 98%）——这对教学演示是预期之中的，**并不**代表它在真实、含噪声的用户输入上的表现。若要用于真实应用，请把 `data.py` 中的生成器替换为你自己的标注数据（一组 `(text, intent)` 对），其余 pipeline 保持不变即可。

想直观感受"真实数据更难"？把 `config.py` 中的 `noise_prob` 调高（例如 `0.2`），数据生成时会按概率注入口语化填充词（`um`、`please` 等）和轻微拼写错误，重新训练后即可看到准确率随之下降——这更贴近真实用户输入。默认 `noise_prob=0.0`，保持干净、可复现的演示效果。
