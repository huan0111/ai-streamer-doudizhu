# 排查指南

## 常见问题

| 现象 | 常见原因与处理 |
| --- | --- |
| 一直找不到游戏窗口 | 确认使用电脑版微信**小程序**，不是手机投屏或 QQ 游戏大厅；窗口标题需为“腾讯欢乐斗地主”，类名为 `Chrome_WidgetWin_0`，并保持可见。多个同类窗口尚未做选择逻辑，建议只保留一个。 |
| 诊断脚本看到 `800×496`，以为画面被裁切 | 这是 DPI 虚拟化尺寸，不是当前实机物理尺寸。诊断脚本与主程序一样设置 `SetProcessDpiAwareness(2)`，再看客户区大小。 |
| 日志出现 `window_resized`、`cursor_mismatch`、`click_blocked` | 分别是窗口尺寸变化、鼠标位置被抢占、点击保护生效。保持窗口稳定并减少人工鼠标干扰，不要删除保护来“修复”。 |
| 卡牌或按钮突然大量认错 | 先检查微信/游戏更新、皮肤、DPI 和客户区大小是否变化，再重新校准模板。不要先改 `bot.js`，识别错误和策略错误是两件事。 |
| 偶发一次桌面目标不可靠并改用提示 | 这是保护性兜底，不代表整局失败。若频繁发生，检查 `outputs/mini_bot.jsonl` 中的 `target_unreliable`、`target_retry` 和现场截图。 |
| 提示 `node` 不存在或 Node 接口失败 | 安装 Node.js 并确认 `node` 在 PATH 中。本地引擎已随仓库打包，不需要 `npm install`，也不需要另开 DSH 服务。 |
| DeepSeek 超时或配置报错 | 检查本机 `api_config.json`、`.env` 中的变量名、密钥和网络。完整 AI 改选与素材生成需要 API；本地推荐只是异常时的执行兜底。 |
| 主播端重复播旧口播 | 实时监听不要使用 `--from-start`。默认监听只接收新事件；`--from-start --once` 仅用于调试历史事件。 |
| 主播端没有声音 | 本仓库只提供行动和中性素材，不自带 TTS。确认主播进程正在消费 `speak: true` 的事件，并已接入自己的语音与播放链路。 |
| 自检或运行写不出文件 | 检查项目目录写权限及 `outputs/` 是否可创建。公开测试图片在 `samples/selftest/`，不要依赖本机被忽略的 `outputs/` 历史文件。 |

## 日志与现场

- `outputs/mini_bot.jsonl`：几何、识别、点击、回读和验证日志。
- `outputs/decision_events.jsonl`：给主播端的决策与执行事件。
- `outputs/streamer_latest.json`：最新可播报素材。
- `outputs/hand_unreliable_*.png`：两次仍不可靠的手牌现场。
- `outputs/follow_target_invalid_*.png`：无法通过规则校验的桌面目标现场。
- `outputs/selection_unstable_*.png`：选牌槽位回读不稳定现场。
- `outputs/lead_rejected_*.png` / `follow_rejected_*.png`：提交失败现场。

`outputs/` 是个人运行输出目录，已被 Git 忽略。日志和现场图可能包含昵称、UID、房间号、头像或其他个人信息，排查后不要公开上传。

## 排查顺序

1. 看 `client_size` 的几何和捕获方式，不先调整窗口。
2. 看完整手牌、槽位、未知位置和失败现场。
3. 区分点数失败、槽位失败、目标失败、点击失败和回读失败。
4. 看 `selection_click` 的实际屏幕点与牌面位置。
5. 看 `recommended_source`、正式动作和 DeepSeek 候选集。
6. 最后判断是否为策略问题，不把识别错误直接当成引擎牌技问题。

## 项目目录

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
docs/                       接入、识别、DPI、排查和验证文档
CHANGELOG.md                公开版本更新记录
outputs/                    本机日志、事件和截图，Git 忽略
```

测试命令和实测范围见 [验证记录](verification.md)。
