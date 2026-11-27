# AI 主播接入文档

本项目已经包含游戏识别、策略、桌面点击、执行验证、人工接管和持续开局。AI 主播只需要把原有口播素材改写成角色会说的话，再交给自己的语音模型或 TTS，不需要重新实现一套斗地主逻辑。

## 推荐接入流程

1. 配置 API 并启动 `mini_bot.py`，由它完成识别、决策、点击和验证。
2. 消费 `outputs/decision_events.jsonl` 或 `outputs/streamer_latest.json`。
3. 将 `utterance` 作为事实素材，用人设改写，但保留牌型、目标、动作和执行状态。
4. 用主播自己的 TTS 播放；需要确认实际出牌结果时继续消费 `execution_result`，不要把 `decided` 当成已经打出。

游戏端只生成中性口播素材，不内置固定人设、TTS、弹幕系统或直播平台接入。

## 文件事件接口

| 输出 | 用途 |
| --- | --- |
| `outputs/decision_events.jsonl` | 持续追加的完整事件流：决策、口播素材、执行回执、结算和控制事件 |
| `outputs/streamer_latest.json` | 最新一条可播报素材，原子替换，适合轮询 |

**游戏主程序当前只提供上述文件事件接口。** `mini_bot.py` 没有 `--event-stdout` 参数，也没有 `DOUDIZHU_EVENT` stdout 协议。通过子进程启动后，应使用 `follow_events()` 或自行读取文件，不要假设主程序会按行向 stdout 输出事件；`streamer_events.py` 的终端输出只是文件事件的监听展示。

## 监听事件

另开终端：

```powershell
.venv\Scripts\python.exe streamer_events.py
```

默认只输出新产生的、`speak: true` 的事件，避免启动时重播历史口播。

调试命令：

```powershell
# 包含诊断和非口播事件
.venv\Scripts\python.exe streamer_events.py --include-non-speech

# 从文件开头读取一次后退出；不要用于实时播报
.venv\Scripts\python.exe streamer_events.py --from-start --once --include-non-speech
```

主播进程直接调用：

```python
from streamer_events import follow_events

for event in follow_events():
    if not event.get('speak'):
        continue
    text = persona_rewrite(event['utterance'], event)
    tts.speak(text)
```

## 事件格式

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

常用字段：

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

**决策已生成不等于出牌已成功。** 需要执行结果时继续消费 `execution_result`，不要只凭 `decision` 宣称已经打出。

## 启动参数

单局试玩：

```powershell
.venv\Scripts\python.exe mini_bot.py --seconds 900 --method message --interval 0.2 --hands 1
```

达到一局后记录 `hand_limit_reached` 并停在结算页，不自动连续开局。

| 参数 | 含义 |
| --- | --- |
| `--seconds 900` | 最长运行 900 秒 |
| `--hands 1` | 只打一局；`0` 表示不限制局数，不建议试玩时使用 |
| `--method message` | 窗口消息点击；另有 `physical` |
| `--interval 0.2` | 主循环间隔，单位秒 |
| `--no-overlay` | 关闭游戏端中性口播浮窗 |
| `--trace-selection` | 保存选牌排查信息 |

按 `Ctrl + Alt + P` 暂停或恢复自动操作，暂停期间所有自动点击都会被拒绝，用户可以完全手动打牌。

## 持续开局

完成当前电脑的单局验证后：

```powershell
.venv\Scripts\python.exe mini_bot.py --seconds 3600 --method message --interval 0.2 --hands 0
```

只有 `--hands 0` 表示不限制局数，**`--seconds 0` 不表示无限运行**。持续运行请同时设置正数总时限和 `--hands 0`。

持续开局不等于无人值守。首次部署、窗口或皮肤变化、识别异常时，先用 `--hands 1` 重新验证。主播端持续消费文件事件即可，不需要另写开局循环。

## API 与口播边界

DeepSeek 每手可能进行多次 API 请求，会产生网络延迟和费用。API 异常时只有本地推荐负责执行兜底，不把本项目包装成无 API 的纯本地机器人。没有网络和 API 时，DeepSeek 改选及相应素材生成能力不可用。
