# CyberTone

把一段视频/音频**整体加速到 2x**，但**不把每个词压短**。

常见做法（`atempo` / 相位声码器 / WSOLA 全片变速）只保住了音高，词还是被等比压短了。CyberTone 换个切法：

> 先切句，再分配时间。**停顿先花，发声段后付**。每句在给定窗口内用克隆音色重新合成，
> 音色是学来的，节奏是按预算算出来的。

时间从停顿里省，不从音节里省。

---

## 先看能不能做到（时长守恒）

把音轨切成发声 `S` 和停顿 `V`，`T = S + V`，目标 `T/2`：

```
需要的发声速率 b = S / (T/2 − 停顿预算)
```

| 素材 `S/T` | 达到 2x 所需 `b` | 结论 |
|---|---|---|
| 50% | 1.00 | 纯删停顿，零失真 |
| 70% | 1.53 | 舒适区 |
| 80% | 1.90 | 超出自然语速，要取舍 |
| 90% | 1.80+ | 这个素材不适合 2x |

`b > r_max`（默认 1.75）时工具会**明确报 feasible=no 和缺口秒数**，而不是悄悄把参数拉爆。
“每个词物理长度完全不变 + 总时长减半”在密集语音上不成立；本工具保证的是**停顿被压干**、
**发声段只跑必要的那点倍速**，而不是全片硬压 2.0x。

---

## 实测（本仓库的 `Introduction.mp3`，3 分 57 秒英文旁白）

```
total        237.14s
target       118.57s
speech       163.60s (69.0%)
pause budget  15.38s
needed rate b 1.534x      feasible: yes
```

用 `mock` 后端（分句原声重排，不换音色）跑完整链路后的客观结果：

| | 原片 | CyberTone 输出 | 全片 atempo=2 基线 |
|---|---|---|---|
| 时长 | 237.1s | 118.6s | 118.6s |
| 发声占比 | 60.9% | **83.2%** | 64.9% |
| 最长静音 | 9.56s | **1.27s** | 4.77s |
| 响度 / 峰值 | −18.0 LUFS / +0.28 dBFS | −16.2 LUFS / −1.0 dBFS | −19.1 LUFS / −0.2 dBFS |

输出里发声帧占比从 61% 提到 83%，说明**词的绝对时长基本没动，被吃掉的是停顿**——正是想要的方向。
换成 `cosyvoice` 后端后，同样的时间轴、同样的窗口，只是发声段由克隆音色重新合成。

---

## 安装

```bash
uv sync                 # 运行时依赖
uv sync --extra asr     # 加 faster-whisper，拿到文字与词级时间戳
```

`ffmpeg` 不用装：默认用 `imageio-ffmpeg` 自带的静态二进制（macOS 上也不污染系统）。
想改用系统 ffmpeg，`brew install ffmpeg` 后会自动优先使用。

> macOS 提示：所有产物和缓存都落在仓库内（`work/`、`.uv-cache/`、`.hf/`）。
> 若 `~/.cache/uv` 不可写，用 `UV_CACHE_DIR=$PWD/.uv-cache uv sync`。

## 用法

```bash
# 1. 先量一下：这一步决定后面值不值得做
uv run cybertone inspect Introduction.mp3 --asr

# 2. 全链路（mock 后端，不需要 GPU）
uv run cybertone run Introduction.mp3 --asr

# 3. 生成 A/B 对照：全片硬压 2x 的版本
uv run cybertone baseline Introduction.mp3

# 4. 抽一段干净参考音，给克隆用
uv run cybertone ref Introduction.mp3 -o refs/ref.wav
```

产物：

```
work/
├── triggers.jsonl      # 每句一条记录：src 区间、文本、词数
├── transcript.json     # ASR 缓存（词级时间戳）
├── plan/plan.json      # 每句的速率窗口 win_min/win_max 与停顿预算
├── stems/0000.wav …    # 每句一条 stem，可单句返工
├── stems.json          # stem 时长/速率/质检字段
└── out/
    ├── master.wav      # 拼好的音轨
    ├── cues.json       # 新时间轴上的字幕 cue
    ├── out.srt         # 字幕
    └── Introduction.2x.mp3
```

分阶段跑（每步都落盘，可单独重跑）：

```bash
uv run cybertone triggers Introduction.mp3 --asr
uv run cybertone plan     Introduction.mp3
uv run cybertone synth    Introduction.mp3 --backend mock
uv run cybertone stitch   Introduction.mp3
uv run cybertone video    input.mp4 work/out/master.wav -o work/out/out.mp4
```

## 配置

复制 `cybertone.example.toml` 为 `cybertone.toml` 用 `--config` 传入；任何字段也能用环境变量
`CYBERTONE_<NAME>` 覆盖（`CYBERTONE_TTS_URL=http://gpu-box:8000`）。

关键旋钮：

| 字段 | 默认 | 含义 |
|---|---|---|
| `r_max` | 1.75 | 语速上限，**必须用你自己声音 AB 测出来**，别信默认值 |
| `merge_gap` | 0.60 | 短于这个的停顿留在句子内部，不再切一刀 |
| `max_sentence` | 12.0 | 单句上限，超了强制切分 |
| `gap_scale` / `gap_max` | 0.40 / 0.22 | 停顿保留比例与上限，段落处单独用 `paragraph_gap` |
| `stem_lufs` | −20 | 逐句响度归一（仅 TTS 后端），压掉句间音量抖动 |
| `loudness_lufs` / `peak_ceiling_db` | −16 / −1 | 母带目标与限幅天花板 |

## 架构

| 模块 | 职责 |
|---|---|
| `audio.py` | 解码（soundfile → ffmpeg 兜底）、WSOLA 变速、BS.1770 响度、前瞻限幅、房间底噪 |
| `vad.py` | 能量 VAD + 迟滞；带“噪声底不可高于 −38 dBFS”的兜底，密集素材不会漏检 |
| `asr.py` | 可选 faster-whisper，出词级时间戳；缺失时不阻塞流程 |
| `triggers.py` | VAD 跨度 + 标点切句 + 长句强切 → 句级触发节点 |
| `plan.py` | 停顿预算 → 全局速率 `b` → 每句 `r` 与 `[win_min, win_max]` |
| `synth.py` | 后端抽象（mock / cosyvoice）、闭环二分落窗、缓存、质检 |
| `stitch.py` | 按预算重排停顿、铺底噪、逐句淡入淡出、响度归一 |
| `mux.py` | 探测容器、视频 `setpts`、MP3/MP4 封装、A/B 基线、SRT |

## TTS 后端（学音色）

`mock` 后端用原声分句重排，是为了让你**先验证时间轴**，它不换音色。

真正的克隆走 `services/cosyvoice`（CosyVoice2 零样本，原生 `speed`）：

```bash
docker compose --profile gpu up -d tts          # 需要 Linux + NVIDIA
uv run cybertone ref Introduction.mp3 -o refs/ref.wav
uv run cybertone run Introduction.mp3 --asr --backend cosyvoice \
    --config cybertone.toml        # tts_ref = "refs/ref.wav"
```

客户端对每句做**闭环落窗**：以 `r` 二分调用（默认 4 次），测回实际时长，落在
`[win_min, win_max]` 就停；`r_max` 仍塞不下的句子做 ≤8% 残余拉伸并标 `degraded`。
缓存键含文本、参考音色、speed、seed，改一句只重跑一句。

> Apple silicon 没有 CUDA，CosyVoice2 在 CPU 上慢到不可用。Mac 上要么用 `mock`
> 验证流程，要么把 `CYBERTONE_TTS_URL` 指向一台 Linux 机器。

## 测试

```bash
uv run pytest -q      # 19 passed
uv run ruff check src tests
```

覆盖：预算求解的可行/不可行分支、窗口单调性、停顿重分配守恒、WSOLA 时长误差（<1%）、
BS.1770 参考值（−20 dBFS 正弦 ≈ −23 LUFS）、限幅器不靠衰减、VAD 的合并/强切/不重叠。

## 已知限制

- `mock` 后端不做音色迁移，听感等同“分句 + 停顿压缩”，可作为质量下限对照。
- `cosyvoice` 服务镜像需要 GPU 且**未在本机跑通**（无 CUDA）；镜像内含模型下载步骤，首次启动较慢。
- 逐句合成天然有韵律重置：句间 F0 断层目前只能靠统一参考音 + 固定 seed 缓解，
  尚未实现跨句韵律延续（批量段落合成是下一步）。
- 视频侧走匀速 `setpts`，2x 下唇形本就不同步；若素材包含屏幕录制，建议改用 EDL 模式。

## 路线图

- [x] 触发节点 / 预算求解 / 闭环落窗 / 缝合 / 封装 / 字幕
- [x] mock 后端全链路 + A/B 基线
- [ ] 段落级批量合成，减少句间韵律重置
- [ ] 音素级可压缩性权重（拖音多压、辅音锁死 1.0）
- [ ] `wer` / `spk_cos` 自动质检闸门
- [ ] EDL 模式：音视频同一张剪辑单，替代匀速 2x
