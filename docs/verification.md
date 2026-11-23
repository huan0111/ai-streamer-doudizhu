# 验证环境与实测记录

本文记录公开版本的实际验证环境，不代表其他电脑、微信版本或皮肤也自动兼容。

## 环境

| 项目 | 当前验证环境 |
| --- | --- |
| 日期 | 2026-11-23 |
| Windows | Windows 11，系统版本 10.0.22621 |
| CPU | Intel Core i5-10210U 1.60GHz |
| Python | 3.13.16 |
| Node.js | 24.14.0 |
| RapidOCR | rapidocr_onnxruntime 1.2.3 |
| ONNX Runtime | onnxruntime 1.31.1，由 `constraints.txt` 固定 |
| OpenCV | 当前安装元数据 opencv-python 4.14.0.94 |
| NumPy | 2.5.4 |
| Pillow | 12.1.1 |
| PyDirectInput | 1.0.4 |
| 游戏形态 | Windows 电脑版微信小程序《腾讯欢乐斗地主》 |
| 微信/游戏版本 | 本次未记录，不能据此声称兼容其他版本 |
| 画面 | 欢乐经典新手场；其他皮肤未验证 |
| Windows 缩放 / DPI | 125% / DPI 120 |
| 客户区 | 实机覆盖 933×620 和 1017×627；历史自检图另覆盖 933×580 |
| 多显示器 | 尚未验证；首次试玩建议使用 `--no-overlay` |

## 识别路径与耗时

同一 CPU 上的实际抽样：

- 叫牌按钮模板命中约 0.03 秒。
- 按钮 RapidOCR 实测约 2–4 秒。
- 单个徽章约 0.2–0.4 秒。
- Windows 场景 OCR 需要启动 PowerShell 子进程，未将其耗时伪装成已测量数据。

RapidOCR 引擎只延迟初始化一次。主循环还会周期读取徽章，因此需要结合 `point_source`、按钮项 `source` 和实际耗时判断瓶颈。

## 实机覆盖

已进行多局真实牌局，包含胜利和失败。实际覆盖过：

- 长牌分批选择、跨批纠偏、普通对子、大小王、连对、三带一、四带二和合法四带两对。
- 跟牌与领出、系统托管取消、结算识别和单局退出。
- 没有将某局胜负当成识别正确率或策略胜率的统计证明。

这些记录用于说明功能曾被真实执行，不代表其他环境可以直接无人值守运行。

## 测试命令

```powershell
.venv\Scripts\python.exe mini_selftest.py
node --test vendor/dsh-doudizhu/test/bot.test.mjs vendor/dsh-doudizhu/test/rules.test.mjs
```

当前结果：

- `mini_selftest.py`：260 项，0 失败。
- 上游 bot/rules 测试：37/37 通过。

**离线自检通过不等于当前电脑实机兼容。** 新环境仍应先运行自检，再人工监督完成一局。
