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

用 `retime` 后端（分句原声重排，不换音色）跑完整链路后的客观结果：

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

# 2. 全链路（retime 后端，不需要模型，只验证时间轴）
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
| `clean_tts` | true | TTS 后处理链：高通 70 Hz → 谱减降噪 → 裁静音 |
| `denoise_strength` | 0.9 | 谱减比例，0 关闭；1.0 最狠，过高会有“水声” |
| `room_tone_db` | −58 | 铺底噪上限，实际会自动压到 stem 噪声底以下 6 dB；≤ −90 关闭 |
| `placement` | flow | `flow` 尽量吃满停顿；`anchor` 把每句钉在 `src/speed` 上保 A/V 对齐 |
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

三种后端，协议一致，换一行配置即可：

| backend | 跑的机器 | 做什么 | 音色 |
|---|---|---|---|
| `retime` | 任何机器 | 分句后把**原声**重新定时，不重新合成 | 原声（**会继承原片语速**，只作时间轴脚手架） |
| `f5tts` | **Mac / MPS**、CPU、CUDA | F5-TTS 零样本克隆，`speed` 原生可控 | 学来的 |
| `cosyvoice` | Linux + NVIDIA | CosyVoice2 零样本克隆 | 学来的 |

`retime` 只是为了让你先验证切句、预算、字幕和封装——它听起来就是“原声被加速”，
不要拿它当成品。真正的听感收益来自后两种：**STT 出文本 → TTS 用克隆音色重新发声**。

### Mac 上跑 F5（走 MPS，不需要 NVIDIA）

```bash
cd services/f5tts && uv sync                 # torch + f5-tts，约 2.5 GB
uv run uvicorn app:app --port 8000           # 首次请求会下载模型 ~1.3 GB

# 另一个终端，仓库根目录
uv run cybertone ref Introduction.mp3 -o refs/ref.wav --dur 8
CYBERTONE_TTS_URL=http://127.0.0.1:8000 \
  uv run cybertone run clip.wav --asr --backend f5tts --config cybertone.toml
```

M4 Pro 实测：单句 RTF 3–8（`nfe_step=16`），60 秒片段 6 句共 8 分钟，`degraded` 为 0。
两个 macOS 专属的坑已经在 `services/f5tts/compat.py` 里绕开：torch 2.14 的
`torchaudio.load` 会去 dlopen 不存在的 ffmpeg 4.x dylib；F5 的参考音预处理又会去调
`ffmpeg/ffprobe`。

### Linux + GPU（CosyVoice2）

```bash
docker compose --profile gpu up -d tts
CYBERTONE_TTS_URL=http://127.0.0.1:8000 uv run cybertone run in.mp4 --asr --backend cosyvoice
```

客户端对每句做**闭环落窗**：二分 `speed`，测回实际时长，落在 `[win_min, win_max]` 就停；
再用**割线一步**用实测时长反推需要的 speed 重渲一次（TTS 时长对 speed 近似线性）；
仍超窗才补 ≤3% 残余拉伸并标 `degraded`。缓存键含文本、参考音色、speed、seed。

## 底噪怎么处理

克隆出来的声音发闷、带嘶声，来源有三层，必须分开处理：

1. **参考音把噪声“教”给了模型** —— F5/CosyVoice 都会模仿 prompt 的通道特征，参考音有嘶声，
   每句都有嘶声。参考音先过 70 Hz 高通 + 降噪再喂（`make_reference` 默认已做）。
2. **vocoder 自身的高频嘶声** —— 逐句处理而不是整轨，让每句噪声特性一致，拼接处不“呼吸”。
3. **铺底噪铺过头** —— 底噪床用来避免数字静音，一旦高过 stem 自身噪声就全程可闻。
   现在床电平自动取 `min(配置值, stem 噪声底 − 6 dB)`，永远不会盖过素材。

降噪用的是 **noisereduce 的 profile 模式**：从该句自己的停顿里取 0.5 s 噪声轮廓，
做平稳谱减。比盲估计（`stationary=False`）多 3–4 dB 信噪比，而且只吃 1.4 dB 语音
（盲估计要吃 4.4 dB）。stem 没有可用静音时自动退回非平稳模式并降低强度。

实测（60 秒片段，6 句）：

| | 母带噪声底 | SNR（p95/p10 帧） | LUFS |
|---|---|---|---|
| 修复前（床 −42，无降噪） | −39.2 dBFS | 29.9 dB | −16.2 |
| 只压床 + 谱减 | −54.6 dBFS | 38 dB | −16.2 |
| **profile 降噪 + 自动床** | **−55.0 dBFS** | **46.0 dB** | −16.5 |

关键点是**语音电平几乎不动**（逐句 RMS 变化 −0.1 ~ +0.2 dB），降的全是噪声。
顺序也有讲究：**先降噪，再归一响度**，反过来会把降噪成果拉回去。

还想更狠就 `denoise_strength = 1.0`，或者把床设成 `room_tone_db = -120` 走数字静音。
素材噪声更脏（风扇、键盘、混响）时，把 `denoise` 换成 DeepFilterNet，接口一致。

## 测试

```bash
uv run pytest -q      # 26 passed
uv run ruff check src tests
```

覆盖：预算求解的可行/不可行分支、窗口单调性与守恒、`flow`/`anchor` 两种落点、
WSOLA 时长误差（<1%）、BS.1770 参考值（−20 dBFS 正弦 ≈ −23 LUFS）、限幅器不靠衰减、
降噪的透明性与降噪量、高通滤除隆隆声、VAD 的合并/强切/不重叠。

## 已知限制

- `retime` 后端不做音色迁移，听感等同“原声 + 停顿压缩”，只作时间轴脚手架。
- **STT 的 WER 就是最终错误率**：`medium` 会把“Home shot”听成“Homeshot”、把领域词念错，
  TTS 会一字不落地念出来。走 TTS 后端时用 `asr_model = "large-v3"`，或者人工校对
  `triggers.jsonl` 里的 `text`。
- `cosyvoice` 镜像需要 GPU，**未在本机验证**；Mac 上走 `f5tts`（已验证，M4 Pro / MPS）。
- 逐句合成天然有韵律重置：句间 F0 断层目前只能靠统一参考音 + 固定 seed 缓解，
  尚未实现跨句韵律延续（批量段落合成是下一步）。
- 视频侧走匀速 `setpts`，2x 下唇形本就不同步；`placement=flow` 会相对视频漂移，
  需要严格对齐就换 `anchor`（代价是停顿只能减半）。

## 路线图

- [x] 触发节点 / 预算求解 / 闭环落窗 / 缝合 / 封装 / 字幕
- [x] `retime` 后端全链路 + A/B 基线
- [x] Mac 原生 F5-TTS（MPS）零样本克隆，实测 60s→30s、degraded 0
- [x] 底噪治理：参考音降噪 + 逐句降噪 + 底噪床电平
- [ ] 整片批量渲染（34 句 / 约 70 分钟）与人工校对回路
- [ ] 段落级批量合成，减少句间韵律重置
- [ ] 音素级可压缩性权重（拖音多压、辅音锁死 1.0）
- [ ] `wer` / `spk_cos` 自动质检闸门
- [ ] EDL 模式：音视频同一张剪辑单，替代匀速 2x
