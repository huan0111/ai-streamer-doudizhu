# AI 主播欢乐斗地主（Windows 微信小程序）

给 AI 主播使用的实时斗地主行动与口播事件服务。

它在 Windows 电脑版微信的《腾讯欢乐斗地主》**小程序**里识别牌局并实际出牌，同时把“这手决定做什么、为什么、可以怎么说、最后是否执行成功”持续写成事件。其他 AI 主播项目只需要消费这些事件，就能在自己的主播端叠加**人设、语音、弹幕互动和直播节奏**，不需要改动视觉识别或出牌策略代码。

> 非官方项目，与腾讯、微信及《欢乐斗地主》运营方无关。仅供技术学习和研究，请遵守游戏规则、服务条款及适用法律。

## 推荐接入方式（先读这里）

**本项目已经是一套完整的斗地主执行系统，不只是策略库或口播生成器。** 以下能力可以直接复用，不需要其他项目重新实现：

- 游戏窗口与牌面识别、合法牌型枚举、本地正式推荐和 DeepSeek 有限改选。
- 分批选牌、桌面点击、出牌回读验证、失败重试、取消托管和结算识别。
- 单局限额、持续开局、人工接管、未知页面停手、充值及视频返豆保护。

**AI 主播只需要做表达层：** 消费程序产生的口播素材和执行事件，把中性文本改写成自己角色会说的话，再交给自己的语音模型或 TTS 播放。人设、语音、弹幕和直播编排都由主播项目负责，游戏识别和出牌执行由本项目负责。

推荐接入流程：

1. 配置 API 并启动 `mini_bot.py`，让它完成识别、决策、点击和验证。
2. 在主播进程中消费 `outputs/decision_events.jsonl` 或 `outputs/streamer_latest.json`。
3. 将 `utterance` 作为事实素材，用主播人设改写，但保留牌型、目标、动作和执行状态。
4. 用主播自己的 TTS 播放改写后的文本；需要确认实际出牌结果时继续消费 `execution_result`，不要把 `decided` 当成已经打出。

> [!IMPORTANT]
> **不要在主播进程里重复做窗口尺寸拦截。** `mini_bot.py` 只在它自己的进程里设置 DPI 感知，不会把这个设置传给主播进程。最稳妥的方式是直接启动 `mini_bot.py` 并消费它的事件，使用它记录的 `client_size`、`image_size` 和 `capture_method`，不要再用另一套读数拒绝启动。
>
> 如果确实必须自行读取窗口尺寸，必须在该进程的任何 Win32 窗口或截图调用之前设置 `SetProcessDpiAwareness(2)`，并用物理像素检查客户区。否则在 125% 缩放下，真实的 `1017×627` 可能被读成 `813×501`，从而被错误地当成“窗口太小”。这不是游戏窗口需要放大，而是读取方式错了。

接入示例（`persona_rewrite` 和 `tts` 由主播项目提供）：

```python
from streamer_events import follow_events

for event in follow_events():
    if not event.get('speak'):
        continue
    text = persona_rewrite(event['utterance'], event)
    tts.speak(text)
```

**不需要主播另写斗地主识别、选牌、点击或验证逻辑。** 如果只需要口播，也不要为了获得文本而绕过游戏端单独请求一套新的决策；直接复用这里的事件，才能保持“看到的牌、决定出的牌、实际点击和口播”一致。

人工接管始终是独立的：按 `Ctrl + Alt + P` 暂停或恢复自动操作，暂停期间用户可以完全手动打牌，自动点击会被拒绝。

## 这个项目为什么这样设计

如果每一步都把整局信息交给 DeepSeek 自由判断，实际直播中会遇到三个问题：

- **慢**：每次请求都要等网络和大模型响应，容易错过回合计时。
- **不稳定**：可能选出不合法或不合理的牌，前后决策也不一致。
- **口播漂移**：说的牌和实际打出的牌可能不一致，还会夹带不必要的人设表达。

所以项目拆成三层：

| 层 | 主要文件 | 负责什么 |
| --- | --- | --- |
| 视觉与执行 | `mini_perceive.py`、`mini_bot.py` | 截图、认手牌、认桌面出牌、认按钮、选牌、点击、回读验证 |
| 本地规则与正式推荐 | `vendor/dsh-doudizhu/src/rules.js`、`bot.js` | 快速枚举合法牌型，由 `chooseAction` 给出稳定的本地正式推荐 |
| DeepSeek 有限改选与中性素材 | `mini_bot.py`、`model_decision.py` | 只从“正式推荐 + 按与推荐分差排序的两个备选”中改选，并生成与动作一致的中性口播素材 |

**本地引擎不是为了取代 AI 主播，而是为了替主播兜底速度、合法性和基本牌技。** DeepSeek 仍然参与改选和表达，实机运行需要配置自己的 API。

本地提示词要求优先采用正式推荐，只有明显更差时才允许改选并说明原因。API 异常时的本地推荐只是执行兜底，不把本项目包装成无 API 的纯本地机器人。候选固定为“正式推荐＋最多两个按真实评分排序的备选”，不为了凑三手补动作。正式推荐是合法非 `pass` 时，不会把“过”放进 DeepSeek 候选；正式推荐为“过”时才保留“过”及其他合法备选；没有任何合法出牌时只有“过”。

## 视觉识别与 OCR 回退顺序

项目不是只有一套文字识别。不同画面区域使用不同方法，不能用“有没有调用 OCR”判断实际耗时。

| 场景 | 识别与回退顺序 | 日志/备注 |
| --- | --- | --- |
| 叫地主、不叫按钮 | 先匹配 `templates/mini_ui_buttons/` 中的两个按钮模板；任一命中即返回，否则 RapidOCR | 模板命中时 `source=template`，回退时为 `rapidocr`；决策日志的 `point_source` 也保留真实来源 |
| “提示”按钮 | 直接 RapidOCR；找不到再用已知相对位置兜底 | 不使用叫牌按钮模板 |
| 两位数徽章 | 直接 RapidOCR；仅接受 10–20 | 读数异常时按记账规则复核 |
| 个位数徽章 | RapidOCR 无数字结果时，才使用 0–9 模板 | 最高分 ≥0.72 且与次高分差 ≥0.04，个位数还需连续两次一致 |
| 取消托管 | 在局部区域做模板匹配，超过 0.7 才点击 | 没有 OCR 回退，模板缺失则停手 |
| 大厅、弹窗、结算文字 | `ocr_words.ps1` 调用 Windows.Media.Ocr | 用于整页文字、按钮定位及结果交叉确认；失败时未知页面不点击 |
| 手牌和桌面牌点 | `mini_skin`、`table_skin` 的形状与模板匹配，普通牌颜色无关，大小王保留颜色证据 | 不是 RapidOCR；不可靠时先重新截图，再进入保护兜底 |
| 时钟、地主标记、胜负等 | 颜色探针与相对位置 | 不属于文字识别 |

**性能差异必须关注。** 当前实测按钮模板命中约 0.03 秒，按钮 RapidOCR 在 2–4 秒之间，单个徽章约 0.2–0.4 秒；Windows 场景 OCR 还会启动 PowerShell 子进程。RapidOCR 引擎只延迟初始化一次，不是每次重新创建。

为了限制重复开销，叫牌按钮采用两类安全去重：

- **像素完全相同的画面**在 1 秒内复用识别结果；画面有变化立即重新识别，不拿旧结果猜测新阶段。
- 同一阶段、同一动作达到两次点击上限后，在再次读牌和 OCR 前直接 `bid_click_deduped`；阶段、手牌或动作不同则重新识别。

主播或接入方应先检查 `point_source` 和按钮项中的 `source`，不要把历史日志中的 `rapidocr_button` 直接理解为“每次都走了 OCR”。不要绕过这些保护另写一套窗口识别或取消点击去重。模板连续未命中导致倒计时超时时，先保存现场并检查皮肤、DPI、模板覆盖和 OCR 耗时，而不是直接关闭安全保护。

## 给 AI 主播的接口

核心接口是 `streamer_events.py` 产生的两条数据流：

| 输出 | 用途 |
| --- | --- |
| `outputs/decision_events.jsonl` | 持续追加的完整事件流：决策、口播素材、执行回执、结算和控制事件 |
| `outputs/streamer_latest.json` | 最新一条可播报素材，原子替换，适合轮询 |

**游戏主程序当前只提供上述文件事件接口。** `mini_bot.py` 没有 `--event-stdout` 参数，也没有 `DOUDIZHU_EVENT` stdout 协议。主播项目用子进程启动游戏端后，应通过 `follow_events()` 或自行读取上述文件消费事件，不要自行假设主程序会按行输出事件；`streamer_events.py` 的终端输出只是现有文件事件的监听展示。

游戏端只生成**中性口播素材**，不内置固定人设。主播端可以在素材外层增加自己的角色设定和 TTS，但改写时应保持动作、牌型、目标和执行状态一致。

### 监听新口播事件

另开一个终端：

```powershell
.venv\Scripts\python.exe streamer_events.py
```

默认只输出新产生的、`speak: true` 的事件，避免启动时重播历史口播。

调试时可以：

```powershell
# 包含诊断和非口播事件
.venv\Scripts\python.exe streamer_events.py --include-non-speech

# 从文件开头读取一次后退出；不要用它做实时播报，避免重复播报
.venv\Scripts\python.exe streamer_events.py --from-start --once --include-non-speech
```

也可以在主播进程中直接调用：

```python
from streamer_events import follow_events

for event in follow_events():
    if event['event_kind'] == 'decision':
        # 在这里接入主播人设、语音合成和直播编排
        print(event['utterance'])
```

### 事件里有什么

典型决策事件：

```json
{
  "schema_version": 2,
  "event_kind": "decision",
  "phase": "follow",
  "action": "play",
  "chosen_ranks": ["2", "2"],
  "utterance": "用对2压住上家的对8，拿回出牌权。",
  "execution": "decided",
  "speak": true
}
```

主播端常用字段：

| 字段 | 含义 |
| --- | --- |
| `session_id` / `sequence` | 本次运行及递增事件序号 |
| `event_id` / `action_id` | 事件与动作标识，可用于去重 |
| `event_kind` | `decision` 或 `execution_result` |
| `phase` | `lobby`、`bid`、`lead`、`follow`、`result`、`control` 等 |
| `action` / `chosen` | 动作类型和本地执行标识 |
| `chosen_ranks` | 最终选择打出的牌 |
| `utterance` | 可交给主播端二次包装的中性素材 |
| `execution` | `decided`、`submitted`、`confirmed`、`rejected`、`failed`、`finished` 等状态 |
| `speak` | 是否建议进入口播队列 |
| `result_for` | 执行回执对应的原动作 ID |

**决策已生成不等于出牌已成功。** 需要执行结果时，应继续消费 `execution_result`，不要只凭 `decision` 事件宣称已经打出。

## 已实现的游戏能力

- 大厅选场、准备房开始、叫地主、抢地主、加倍与自动出牌。
- 手牌和桌面牌型识别，领出与跟牌决策。
- 普通牌使用与颜色无关的形状匹配，只有大小王保留颜色证据辅助区分。
- 叫牌、领出、跟牌共用手牌二次识别：首次点数不可靠或读取失败时，等待 0.3 秒重新截图再读一次，第二次仍不可靠才进入保护兜底。
- 领出和跟牌评估会惩罚拆残炸弹；三带一在最低带牌拆炸时提供不拆炸的替代候选，正式动作按具体卡牌精确匹配。
- 每批最多选择 3 张，批后回读并累计选中状态，只补剩余牌；残余选择不明时停止本手。
- 两位数徽章 OCR；个位数徽章使用真实截图生成的 `0–9` 模板，并要求连续两次一致。
- 识别不可靠时优先 OCR 定位游戏“提示”，失败后使用已知位置兜底。
- `pass_only` 自动过牌前额外复核两帧，降低动画或颜色波动造成的误过。
- 决策、口播素材与执行回执关联，方便主播端和开发者复盘。
- 人工接管、未知页面停手、充值和视频返豆停手等保护。

程序只能看到当前窗口中的公开信息，不读取游戏内存，也不知道其他玩家的暗牌。

## 环境要求

- Windows 10/11。
- 电脑版微信中的《腾讯欢乐斗地主》**小程序**，窗口标题为“腾讯欢乐斗地主”。
- Python 3.13（当前验证版本 3.13.16）。
- Node.js（当前验证版本 24.14.0）；只使用 Node 内置模块，不需要 `npm install`。
- 自己的 OpenAI 兼容 API；默认配置示例为 DeepSeek。

不支持手机直接运行，也不是 QQ 游戏大厅版本的控制程序。

## 安装

在项目目录执行 PowerShell 命令：

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints.txt
```

主要 Python 依赖：OpenCV、NumPy、Pillow、PyDirectInput、RapidOCR。`constraints.txt` 额外固定间接依赖 `onnxruntime==1.31.1`，避免 OCR 运行时版本漂移；安装时必须同时带上 `-c constraints.txt`。窗口操作同时使用 Python `ctypes` 和 Windows API。

## 配置 API

复制公开示例，生成本机配置：

```powershell
Copy-Item api_config.example.json api_config.json
Copy-Item .env.example .env
```

`api_config.json` 默认只包含公开接口和模型名：

```json
{
  "base_url": "https://api.deepseek.com/v1",
  "model": "deepseek-chat",
  "api_key_env": "DOUDIZHU_API_KEY",
  "timeout_seconds": 8
}
```

在本机 `.env` 中填入自己的密钥：

```dotenv
DOUDIZHU_API_KEY=replace_with_your_key
```

也可以直接设置同名环境变量。`.env` 和 `api_config.json` 已加入 `.gitignore`，不要提交、截图公开或写入 Issue。

DeepSeek 每手可能进行多次 API 请求，会产生网络延迟和 API 费用，具体取决于所选服务商及模型。API 异常时只有本地推荐负责执行兜底，不等于完整的离线 AI 主播；没有网络和 API 时，DeepSeek 改选及相应素材生成能力不可用。

## 自检

```powershell
.venv\Scripts\python.exe mini_selftest.py
```

当前为 **260 项自检**，2026-11-23 验证为 0 失败。测试覆盖规则、决策候选、识别、模板、点击保护、选牌回读、截图尺寸回退、浮窗位置、结算退出、主播事件及异常兜底，不需要打开游戏，也不需要 API。完整环境、上游规则测试命令及实测范围见 [验证环境与实测记录](docs/verification.md)；版本变化见 [更新记录](CHANGELOG.md)。

`outputs/` 是个人运行输出目录，已被 Git 忽略；公开自检使用的 50 张历史图片独立放在 `samples/selftest/`。公开样本中的玩家昵称、账号 ID 和头像均已脱敏。首次运行测试时会自动创建 `outputs/`。

## 桌面分辨率、窗口位置与 DPI

> [!WARNING]
> **本项目不是在任意电脑、分辨率或 Windows 缩放比例下都能即插即用。** 坐标会按客户区缩放，但卡牌字形、颜色探针、OCR 和模板仍依赖实际渲染效果。尚未验证的环境必须先自检、再由人工盯着试玩一局；不要直接无人值守或连续开局。离线自检通过只代表代码和历史样本通过测试，**不代表你的电脑画面已经兼容**。

**不要求和开发者的桌面分辨率、显示器尺寸或窗口摆放位置一致。** 当前实现不是按整个桌面找固定屏幕坐标：

1. 按窗口标题“腾讯欢乐斗地主”和窗口类名 `Chrome_WidgetWin_0` 找到小程序窗口。
2. 通过 Windows `GetClientRect` / `ClientToScreen` 获取客户区实际大小及其左上角屏幕位置。
3. 手牌与模板识别以 `933×580` 为基准使用统一比例，不分别拉伸宽、高；牌是否抬起根据相对牌面位置判断，按钮模板在相应局部区域搜索。
4. 最终屏幕坐标根据识别到的实际位置映射回客户区。窗口移动后会重新读取左上角；点击前还会复核客户区尺寸。

因此，桌面是 1080p、2K 或 4K 本身不是决定因素，真正相关的是**游戏客户区渲染出来的像素大小、DPI 和皮肤**：

- 客户区小于 `933×580` 会停止点击，避免在裁切画面上乱点；这不是鼓励强行拉伸窗口，画面变形或识别异常时仍应停手。
- 已实机验证的环境之一为 Windows 显示缩放 125%、客户区 `933×620`；历史自检图也覆盖 `933×580`。其他尺寸、缩放比例、微信版本和皮肤不保证兼容。
- 更大的窗口和非等比例缩放有坐标换算，但字形、颜色探针及模板仍可能受影响，不能直接视为已经兼容。
- 主程序会调用 `SetProcessDpiAwareness(2)`。自行编写诊断脚本也要设置同样的 DPI 感知，否则在当前环境下可能把物理 `1001×620` 错看成虚拟化的 `800×496`。
- 多显示器、不同显示器使用不同 DPI 的场景尚未验证；这类环境首次试玩建议加 `--no-overlay`，排除浮窗干扰。
- 运行中改变客户区尺寸会触发 `window_resized` 并放弃该次点击；调整后先保持窗口稳定，再重新开始试玩。
- 截图与客户区尺寸不一致时会记录 `capture_size_mismatch`、重新捕获并尝试 ImageGrab；连续失败会停手，不会用黑图继续识别。日志分别记录客户区、窗口外框、截图、DPI 和捕获方式。
- 浮窗会跟随窗口几何变化重新放置；屏幕没有足够空间时会隐藏，不覆盖游戏客户区。

## 准备游戏窗口与首次试玩

1. 先运行 `mini_selftest.py`，确认 **260 项全部通过**；不要把这项离线结果当成当前电脑的实机验收。
2. 在电脑版微信中打开《腾讯欢乐斗地主》小程序，保持窗口可见、稳定且不最小化，不要用其他窗口遮挡，运行中不要调整尺寸。
3. 确认客户区不小于 `933×580`，第一次只运行 `--hands 1`，并人工观察叫牌、选牌和点击位置；发现异常立即按 `Ctrl + Alt + P` 暂停。
4. 启动后检查 `outputs/mini_bot.jsonl` 第一条 `client_size` 日志：`client_size` 与 `image_size` 应一致，并应有合理的 `window_rect`、`dpi` 和 `capture_method`。
5. 若反复出现 `client_too_small`、`capture_size_mismatch`、`capture_failed` 或 `window_resized`，不要继续自动出牌，更不要删除保护或自动拉伸窗口；先核对 DPI、客户区和截图。
6. 不同 Windows 缩放、微信版本或游戏皮肤可能需要重新校准模板与坐标。`outputs/` 中的日志和现场图可能包含玩家昵称、账号 ID 或头像，排查后不要公开上传。

## 启动一局

```powershell
.venv\Scripts\python.exe mini_bot.py --seconds 900 --method message --interval 0.2 --hands 1
```

达到一局后程序停在结算页，不会自动连续开局。主播端可以同时运行上面的 `streamer_events.py` 监听命令。

常用参数：

| 参数 | 含义 |
| --- | --- |
| `--seconds 900` | 最长运行 900 秒 |
| `--hands 1` | 只打一局；`0` 表示不限制局数，不建议试玩时使用 |
| `--method message` | 窗口消息点击；另有 `physical` |
| `--interval 0.2` | 主循环间隔，单位秒 |
| `--no-overlay` | 关闭游戏端自带的中性口播浮窗 |
| `--trace-selection` | 保存选牌排查信息 |

**人工接管：** 按 `Ctrl + Alt + P` 暂停或恢复。暂停期间所有自动点击都会被拒绝。

### 持续开局（完成单局验证后使用）

确认当前电脑能够稳定完成单局后，可以把 `--hands` 设为 `0`，程序会在结算后继续下一局；`--seconds` 仍作为总运行时限：

```powershell
.venv\Scripts\python.exe mini_bot.py --seconds 3600 --method message --interval 0.2 --hands 0
```

只有 `--hands 0` 表示不限制局数，**`--seconds 0` 不表示无限运行**，不要用后者实现持续开局。持续运行请像示例一样同时设置正数总时限和 `--hands 0`。

持续开局不等于无人值守。首次部署、窗口或皮肤变化、识别异常时，应先用 `--hands 1` 重新验证。主播端持续消费事件即可，不需要自行实现开局循环。

## 安全边界

- 不认识的页面不点击，只记录并口播提示。
- 购买、充值页面停下等待用户。
- “免费视频返豆”一律不点。
- 窗口变小、光标未到位或前台校验失败时放弃点击。
- 出牌后检查手牌数和状态；失败只进行有限重试，不无限重复提交。
- 请始终在可人工接管的电脑前运行，不要无人值守。

### 公开发布与隐私检查

不要上传真实对局截图、运行日志、昵称、UID、房间号、API 密钥或其他个人信息。新增 `samples/` 图片前必须脱敏并人工检查；`.gitignore` 只能降低误提交概率，不能替代发布前检查。不要将本机 `outputs/`、`.env` 或 `api_config.json` 强制加入版本控制。

## 常见坑与排查

| 现象 | 常见原因与处理 |
| --- | --- |
| 一直找不到游戏窗口 | 确认使用的是电脑版微信**小程序**，不是手机投屏或 QQ 游戏大厅；窗口标题需为“腾讯欢乐斗地主”，类名为 `Chrome_WidgetWin_0`，并保持可见。多个同类窗口尚未做选择逻辑，建议只保留一个。 |
| 诊断脚本看到 `800×496`，以为画面被裁切 | 这是 DPI 虚拟化尺寸，不是当前实机物理尺寸。诊断脚本与主程序一样设置 `SetProcessDpiAwareness(2)`，再看客户区大小。 |
| 日志出现 `window_resized`、`cursor_mismatch`、`click_blocked` | 分别是窗口尺寸变化、鼠标位置被抢占、点击保护生效。保持窗口稳定并减少人工鼠标干扰；不要通过删除保护来“修复”。 |
| 卡牌或按钮突然大量认错 | 先检查微信/游戏更新、皮肤、DPI 和客户区大小是否变化，再重新校准模板。不要先改 `bot.js`，识别错误和策略错误是两件事。 |
| 偶发一次桌面目标不可靠并改用提示 | 这是保护性兜底，不代表整局失败。若频繁发生，再检查 `outputs/mini_bot.jsonl` 中的 `target_unreliable`、`target_retry` 和现场截图。 |
| 提示 `node` 不存在或 Node 接口失败 | 安装 Node.js 并确认 `node` 在 PATH 中。本地引擎已随仓库打包，不需要 `npm install`，也不需要另开 DSH 服务。 |
| DeepSeek 超时或配置报错 | 检查本机 `api_config.json`、`.env` 中的变量名、密钥和网络。完整 AI 改选与素材生成需要 API；本地推荐只是异常时的执行兜底。 |
| 主播端重复播旧口播 | 实时监听不要使用 `--from-start`。默认监听只接收新事件；`--from-start --once` 仅用于调试历史事件。 |
| 主播端没有声音 | 本仓库只提供行动和中性素材，不自带 TTS。确认主播进程正在消费 `speak: true` 的事件，并已接入自己的语音与播放链路。 |
| 自检或运行写不出文件 | 检查项目目录写权限及 `outputs/` 是否可创建。公开测试图片在 `samples/selftest/`，不要依赖本机被忽略的 `outputs/` 历史文件。 |

## 目录说明

```text
mini_bot.py                 牌局主循环、有限改选和执行验证
mini_perceive.py            视觉识别与模板匹配
mini_selftest.py            离线自检
model_decision.py           OpenAI 兼容 API 请求与解析
streamer_events.py          给 AI 主播消费的文件事件接口
rules_cli.mjs               合法牌型枚举接口
bot_cli.mjs                 本地引擎推荐与候选接口
capture_window.py           Windows 窗口捕获
ocr_words.ps1               场景 OCR 辅助脚本
requirements.txt            直接 Python 依赖
constraints.txt             固定 onnxruntime 等间接依赖
templates/                  手牌、桌面牌、按钮及徽章模板
samples/selftest/           已脱敏的公开自检图片
vendor/dsh-doudizhu/        本地斗地主规则、决策引擎及许可证
docs/verification.md        验证环境、实测范围和测试命令
CHANGELOG.md                公开版本更新记录
outputs/                    本机日志、事件和截图，Git 忽略
```

主要排查文件：

- `outputs/mini_bot.jsonl`：识别、点击、回读和验证日志。
- `outputs/decision_events.jsonl`：给主播端的决策与执行事件。
- `outputs/streamer_latest.json`：最新可播报素材。

## 已知限制

- 目前只支持 Windows 和特定微信小程序画面，不是通用斗地主 AI。
- 桌面多张牌仍可能偶发点数或张数误识别；不可靠时会退回游戏提示。
- 徽章模板不匹配时不会硬猜，会继续使用出牌扣减记账。
- AI 改选受候选集合约束，但提示词约束不是形式化策略证明。
- 实机表现依赖窗口尺寸、皮肤、动画、鼠标状态和 API 延迟。
- 项目提供行动和中性素材，不包含完整的人设系统、TTS、弹幕系统或直播平台接入；这些由使用它的 AI 主播项目实现。

## 主项目许可证

本项目以 **MIT License** 发布，允许他人使用、修改、商用和再发布，但需保留版权及许可证声明。完整条款见根目录 `LICENSE`。项目按“现状”提供，不作任何担保。

第三方组件仍分别遵循其原有许可证，不能只依据主项目的 MIT 许可证忽略上游声明。

## 第三方规则与许可证

本地规则和电脑玩家来自 [`dingchenhui0618-arch/dsh-doudizhu`](https://github.com/dingchenhui0618-arch/dsh-doudizhu)，对应包版本 `0.2.5`，当前固定在提交 `ed23ffca3b9b50da3c3b4bfd256bb0194ca72545`，许可证为 MIT。`vendor/dsh-doudizhu/LICENSE` 和 `THIRD_PARTY_NOTICES.md` 一并保留。

**完整本地引擎已经直接打包在 `vendor/dsh-doudizhu/` 中，包括实际参与推荐的 `src/bot.js`，不需要用户再去上游下载。** 这样可以直接使用我们确认过的规则行为，避免上游版本变化造成差异。其中：

- `src/rules.js` 用于合法规则与候选枚举；本项目确认四带二的两张单牌可以同点数，并在三带一的最低带牌拆炸时补充不拆炸的替代带牌。
- `src/bot.js` 用于本地正式推荐；领出和跟牌评估对每组被拆残的炸弹增加 2 手成本，完整使用炸弹不罚，合法四带两对仍保留。
- `src/game.js` 提供阶段常量。
- 不运行上游 DSH 服务，不使用上游联网牌局。

上游规则还包含 Aevi 家庭斗地主的 MIT 授权声明，详见 `vendor/dsh-doudizhu/THIRD_PARTY_NOTICES.md`。第三方来源登记见 `vendor/SOURCES.json`。
