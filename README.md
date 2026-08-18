# MiniMax H3 ComfyUI 智能一体化（GH）
<a href="https://github.com/goohai/Goohai-MiniMax-H3_Integration" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/GitHub-MiniMax--H3%20Integration-1E90FF?style=flat-square" alt="GitHub" /></a>
<a href="https://github.com/goohai/Goohaitools-comfyui" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/GitHub-Goohaitools--comfyui-1E90FF?style=flat-square" alt="Goohaitools-comfyui" /></a>
<a href="https://space.bilibili.com/1194488958" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/B%E7%AB%99-%E5%AD%A4%E6%B5%B7FOTO-FF69B4?style=flat-square&logo=bilibili" alt="B站 孤海FOTO" /></a>

## 🎬 视频教程

### 👉 <a href="https://space.bilibili.com/1194488958" target="_blank" rel="noopener noreferrer">点击前往 B 站主页观看全部教程</a>

<a href="https://space.bilibili.com/1194488958" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/B%E7%AB%99-%E8%A7%86%E9%A2%91%E6%95%99%E7%A8%8B-FF69B4?style=for-the-badge&logo=bilibili" alt="B站视频教程" /></a>

---

## 🧩 RunningHub 工作流

### 👉 <a href="https://www.runninghub.cn/user-center/1959241033525714946/webapp?inviteCode=rh-v1557" target="_blank" rel="noopener noreferrer">点击访问我的 RunningHub 个人主页</a>

<a href="https://vibex.runninghub.cn/p/app-5b97e951de8d47aab61ce3e9731f50a9/?inviteCode=rh-v1557" target="_blank" rel="noopener noreferrer"><img src="https://www.runninghub.cn/assets/images/logo.png" alt="RunningHub" height="28" align="middle" /> <span style="display:inline-block;margin-left:8px;font-size:15px;font-weight:600;color:#1E90FF;vertical-align:middle;">RH Vibex主页 · 在线运行AI应用</span></a>

---

MiniMax H3 的独立 ComfyUI 一体化集成节点，支持文本生视频、首尾帧、全能参考和音视频混合条件构建。节点会自动识别素材组合、生成官方媒体标签，自动优化提示词，支持本地模型优化与在线API优化。

An independent ComfyUI integration node for MiniMax H3 conditioning.

The diffusion model remains an external input in the downstream sampler. This package bundles the <a href="https://github.com/T8mars/comfyui-minimax-h3-audio-T8" target="_blank" rel="noopener noreferrer" style="color:#1E90FF">T8</a> conditioning implementation.

## 功能概览

根据上传的素材智能判定任务类型，负责收集图片、视频、音频和提示词，加载 H3 所需的 CLIP/VAE，并输出供后续采样器使用的 AV（Audio-Video）混合潜空间条件，支持原声音轨输出。

扩散模型本身仍由下游采样器执行。本仓库内置了 <a href="https://github.com/T8mars/comfyui-minimax-h3-audio-T8" target="_blank" rel="noopener noreferrer" style="color:#1E90FF">T8</a> 风格的条件构建逻辑。

## 节点

### `MiniMax-H3 智能一体化(GH)`

主节点，负责：

- 加载 MiniMax H3 CLIP、视频 VAE 和音频 VAE；
- 根据画幅、百万像素和参考素材自动计算输出尺寸；
- 读取首帧、尾帧、参考图片、参考视频和参考音频；
- 根据素材组合生成 T2VA、I2VA、L2VA、FL2VA、Ref2VA 或 Hybrid 条件；
- 所有输出汇集成一条管道，交给配套适配器分流，界面更简洁。

前端支持拖放/上传图片、视频和音频；首尾帧模式与全能参考模式可以分别保存提示词和素材状态，切换模式不会丢失另一模式的内容。

### `MiniMax-H3 适配器(GH)`

接收主节点的 `MiniMax` 输出，并拆分为下游工作流需要的端口：

- 正面条件（`positive`）；
- AV 混合潜空间（`av_latent`，同时包含视频和音频潜空间）；
- 视频 VAE、音频 VAE；
- 模型序号（自动切换大模型类型）；
- 混合音频、条件提示词、媒体映射 JSON 和执行报告。

适配器还会输出“原始音频”（内部端口名为 `mux_audio`），可连接到 `VHS_VideoCombine.audio`。

适配器输出的“原始音频”可直接连 `VHS_VideoCombine` 的音频输入，保证原始音质不变。

## 工作模式

### 首尾帧 / 文生视频

前端模式名为 `text_keyframes`，根据上传情况自动解析任务类型：

| 输入 | 解析任务 | 标签格式 |
| --- | --- | --- |
| 无首尾帧 | T2VA | 无图片标签 |
| 只有首帧 | I2VA | `<Picture 1>` |
| 只有尾帧 | L2VA | `<Picture 1>` |
| 首帧和尾帧 | FL2VA | `Picture 1`、`Picture 2`（不带尖括号） |
| 首帧或尾帧 + `hybrid_audio` | Hybrid | `<Picture 1>`，音频作为参考条件 |

首帧和尾帧素材点击后会自动将官方格式标签插入提示词。切换任务类型、上传或删除素材时，已有媒体标签也会自动统一为当前任务所需的带尖括号或不带尖括号格式。标签编号是否仍对应现有素材由“严格提示词标签”选项控制。

首尾帧模式中的 `hybrid_audio` 是专用参考音频。上传后默认将音频模式切换为“锁定源音频”，移除后恢复为“自动生成”；用户可以在高级选项中手动覆盖。

### 全能参考

前端模式名为 `all_reference`，用于 Ref2VA/Hybrid 参考素材工作流，支持：

- 最多 9 张参考图片；
- 最多 3 个参考视频；
- 最多 3 个独立参考音频；
- 参考视频自带音轨（可在视频卡片右上角静音）。

全能参考模式会按视频音轨和独立音频的顺序生成音频序号，可在高级选项中选择驱动音频。参考媒体标签使用官方尖括号格式，例如 `<Picture 1>`、`<Video 1>`、`<Audio 2>`。

## 提示词标签

单击已上传素材会插入对应标签；双击视频会插入该视频音轨的音频标签。视频卡片右上角静音后，该视频音频不会传入参考条件。

标签格式会根据当前任务类型自动统一：

- I2VA、L2VA、Hybrid、Ref2VA：`<Picture 1>`、`<Video 1>`、`<Audio 1>`；
- FL2VA：`Picture 1`、`Picture 2`。

当“严格提示词标签”开启时，引用不存在的素材编号会停止执行并报错提示：

> 当前提示词引用了不存在的素材标签，请引用正确的标签后重试

关闭后不会因编号不存在而中断，但仍会进行标签格式规范化。

## 高级音频模式

- **自动生成（`native`）**：不使用驱动音频初始化音频 latent，由模型生成目标音频；
- **仅参考音频（`reference_only`）**：音频只作为参考条件，模型生成目标音频；
- **原声输出（`lock_source`）**：将驱动音频写入音频 latent，保持源音频，同时单独输出一份同时长源音质音频；
- **重混源音频（`remix_source`）**：以驱动音频为基础重绘，重绘程度由音频重绘强度控制。

没有可用驱动音频时，节点会自动回退为“自动生成”，避免将空音频传入后端。

## 推荐连接

```text
MiniMax-H3 智能一体化(GH)
        ↓ MiniMax
MiniMax-H3 适配器(GH)
        ├─ positive → BasicGuider / 条件输入
        ├─ av_latent → 采样器 latent_image 或 H3 音频潜空间控制节点
        ├─ video_vae / audio_vae → H3 AV 解码节点
        └─ 原始音频（mux_audio）→ VHS_VideoCombine.audio
        └─ 模型序号（models munber）→ int整数，全能参考时输出1，非全能参考输出0
```

## 新增功能

### 提示词输入与富文本显示

- 增加可选的上游提示词输入；连接后自动以上游文本为准，并禁用节点内部提示词编辑框，断开后恢复编辑；
- 首尾帧模式和全能参考模式分别持久化提示词、优化前原文及编辑状态，刷新浏览器或分享工作流后仍可恢复；
- 提示词框会在图片、视频和音频标签旁显示对应的图片缩略图、视频首帧或音频图标，素材上传、替换、删除及标签编号变化时同步更新；
- 支持 `picture 1`、`image1`、`图像1`、`视频1`、`音频1` 等中英文素材标签及对应尖括号格式；
- H3 标签使用代码块样式显示，支持 `<Subject 1>`、`[Shot 1]`、`[场景1]`、`<d>`、`[Chinese]` 等标签；
- `subject_definitions:`、`summary:`、`retention_analysis:`、`detailed_description:`、`overall_soundscape:` 等官方段落标题会以独立颜色显示，对话正文也会高亮，后台仍保存原始纯文本；

### H3 提示词优化

- 提示词框内置提示词优化按钮，可读取当前提示词及参考素材，按照 MiniMax H3 官方结构生成视频提示词；
- 支持 RunningHub（推荐）、OpenAI Chat Completions、OpenAI Responses、Gemini GenerateContent 及兼容协议的自定义 API；
- RunningHub 模式可直接选择云端模型，输入企业API Key即可使用；参考图片最多上传前 8 张，参考视频会按目标时长处理后作为低帧率视觉参考上传；
- 支持本地 Transformers 视觉语言模型，以及带视觉投影模型的 GGUF 模型；本地模型统一从 `ComfyUI/models/llm` 目录读取；
- GGUF 主模型会自动匹配名称最接近的 `mmproj` 视觉模型；存在多个候选版本时可选择使用的版本，并会显示依赖环境检测结果；
- 支持中文或英文输出、视觉素材读取开关，以及运行工作流前自动优化；自动优化最长等待 200 秒，失败或超时后使用原始提示词继续执行；
- 相同提示词、素材和配置会复用上一次优化结果，避免重复请求；支持取消正在进行的优化、恢复优化前原文，并在优化完成后播放提示音；
- 本地优化前会释放 ComfyUI 已加载的其他模型，优化完成后立即卸载本地语言模型，减少显存占用；工作流执行期间禁用本地模型优化，在线 API 优化不受限制。

### 音频裁剪与时间对齐

- 音频素材增加波形裁剪界面，只保存选区起止时间，不修改或复制原始音频文件；
- 支持拖动起止边界、整体移动选区、滚轮缩放时间轴、中键平移视图、选区播放和实时播放位置线；
- 可一键将选区同步到视频目标时长；再次打开裁剪界面时会恢复上一次保存的选区；
- 工作流执行时按保存的起止时间读取音频，超出目标时长自动裁剪，不足目标时长补静音；节点中的时长显示和播放范围同步使用裁剪后的选区；
- Hybrid 模式中的参考音频、写入 latent 的目标音频及原声输出音频使用同一条 H3 对齐时间线，避免参考条件与视频时长不一致；
- 解码输出可自动识别开头短促异常波形后紧接静音的情况，仅在符合特征时静音异常片段并对后续音频淡入，减少 H3 偶发的开头破音。

### 独立采样与解码节点

#### `MiniMax-H3 双时钟-T8(GH)`

基于 T8 双时钟设计实现的独立 H3 音视频采样配置节点，不依赖安装 T8 节点包。主要解决0.32.0以上版本comfyui口型对不上或报错问题。

#### `MiniMax-H3解码-T8(GH)`

独立的 H3 音视频潜空间解码节点，不依赖安装 T8 节点包。主要解决性能开销与开头破音的问题。


## 限制与注意事项

- 输出宽高会自动限制到 MiniMax H3 的 32 像素对齐要求；
- 默认输出时长为 2–30 秒，并按 H3 帧数网格对齐；
- 全能参考模式最多 9 张图片、3 个视频和 3 个独立音频；
- 参考视频会按目标时长截取或用最后一帧补齐；
- 图片、音频和视频处理优先使用 ComfyUI 环境及 `requirements.txt` 中安装的依赖，不要求用户另外安装系统级 FFmpeg；
- `node --check`、Python 编译检查和单元测试只能验证代码层面，不能替代实际模型采样和浏览器交互测试。

## 安装

将本节点克隆到custom_nodes：

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/goohai/Goohai-MiniMax-H3_Integration.git
```

然后使用 ComfyUI 自己的 Python 环境安装依赖：

```bash
python -m pip install -r ComfyUI/custom_nodes/Goohai-MiniMax-H3_Integration/requirements.txt
```

秋叶整合包等 Windows 便携版可在 ComfyUI 根目录执行：

```powershell
..\python\python.exe -m pip install -r .\custom_nodes\Goohai-MiniMax-H3_Integration\requirements.txt
```

`requirements.txt` 包含本地 Transformers 视觉模型和视频预处理所需的依赖。`imageio-ffmpeg` 会提供可随 Python 环境使用的 FFmpeg，通常不需要用户另外安装系统版 FFmpeg。

GGUF 本地提示词优化属于可选功能，还需要安装支持当前 Python、操作系统及 CUDA 环境的 `llama-cpp-python`。预编译依赖可从以下地址下载：

- <a href="https://github.com/JamePeng/llama-cpp-python/releases" target="_blank" rel="noopener noreferrer">llama-cpp-python 预编译 CUDA Wheels</a>

选择 `.whl` 文件时，需要同时匹配：

- 操作系统及架构，例如 Windows 64 位对应 `win_amd64`；
- ComfyUI 使用的 Python 版本，例如 Python 3.13 对应 `cp313`；
- 当前 PyTorch 使用的 CUDA 版本，例如 CUDA 13.0 对应 `cu130`。

节点会在选择 GGUF 模型时自动检测以上环境，并在配置页面优先显示匹配的 wheel 文件名和下载链接。用户可直接按节点给出的链接安装；如果没有找到完全匹配的预编译版本，再前往发布页面手动选择。不使用 GGUF 时无需安装该依赖。

下载后请使用 ComfyUI 自己的 Python 环境安装 wheel，例如：

```powershell
..\python\python.exe -m pip install "下载的llama_cpp_python文件.whl"
```

完成后重启 ComfyUI，在节点菜单中搜索 `MiniMax-H3 智能一体化(GH)`，输出端拉出连线可自动连接 `MiniMax-H3 适配器(GH)`。

## License

GPL-3.0-or-later. See `LICENSE`.
