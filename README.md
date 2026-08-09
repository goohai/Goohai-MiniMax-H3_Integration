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

MiniMax H3 的独立 ComfyUI 一体化集成节点，支持文本生视频、首尾帧、全能参考和音视频混合条件构建。节点会自动识别素材组合、生成官方媒体标签，并把 AV（Audio-Video）混合潜空间交给下游采样/解码节点。

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
```

适配器的 AV 混合潜空间不能直接当作纯视频 latent 使用，应连接到支持 MiniMax H3 AV latent 的采样和解码节点。

## 限制与注意事项

- 输出宽高会自动限制到 MiniMax H3 的 32 像素对齐要求；
- 默认输出时长为 2–15 秒，并按 H3 帧数网格对齐；
- 全能参考模式最多 9 张图片、3 个视频和 3 个独立音频；
- 参考视频会按目标时长截取或用最后一帧补齐；
- 当前节点只使用 ComfyUI 内置的图片、音频和视频读取路径，不需要额外 Python 依赖；
- `node --check`、Python 编译检查和单元测试只能验证代码层面，不能替代实际模型采样和浏览器交互测试。

## 安装

将本目录放入：

```text
ComfyUI/custom_nodes/Goohai-MiniMax-H3_Integration/
```

重启 ComfyUI 后，在节点菜单中搜索 `MiniMax-H3 智能一体化(GH)` 或 `MiniMax-H3 适配器(GH)`。

## License

GPL-3.0-or-later. See `LICENSE`.
