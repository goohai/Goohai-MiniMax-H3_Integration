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

主节点默认画幅为“自适应”，会优先根据当前模式下第一个有效视觉素材的宽高比计算目标画布；没有视觉素材时使用 16:9。用户也可以手动选择常用固定比例和目标百万像素。

素材区还支持：

- 将本地图片、视频或音频直接拖入对应上传区域；
- 在素材上传区域或已有素材卡片上粘贴剪贴板图片，不会同时触发 ComfyUI 的全局“粘贴节点”操作；
- 视频和音频卡片内直接播放预览，视频静音、音频裁剪等按钮仅在鼠标悬停素材时显示；
- 素材、提示词、模式、音频裁剪范围、静音状态、驱动音频和优化器配置均随工作流保存。

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
- API Key会随节点状态和工作流保存，便于本地刷新后继续使用；分享或发布工作流前请先在配置页面清空Key并保存；
- 支持本地 Transformers 视觉语言模型，以及带视觉投影模型的 GGUF 模型；本地模型统一从 `ComfyUI/models/llm` 目录读取；
- GGUF 主模型会自动选中名称最接近的 `mmproj` 视觉模型，也可在独立的视觉模型下拉框中手动更换；主模型后的刷新按钮会同时刷新两份模型列表，并会显示依赖环境检测结果；
- 支持中文或英文输出、视觉素材读取开关，以及运行工作流前自动优化；自动优化最长等待 200 秒，失败或超时后使用原始提示词继续执行；
- 相同提示词、素材和配置会复用上一次优化结果，避免重复请求；支持取消正在进行的优化、恢复优化前原文，并在优化完成后播放提示音；
- 优化完成提示音使用节点包内置的 `web/audio/done.mp3`，仅在本地模型或在线 API 真正完成一次新优化时播放；在原文与缓存结果之间切换不会重复播放；
- 优化请求会记录发起时所在的工作模式。即使优化期间切换到另一模式，完成结果仍写回最初发起优化的提示词框，不会覆盖另一个模式的内容；
- 点击“恢复优化前”时会保存当前手动编辑后的优化结果，再次切回优化结果时仍可恢复这些修改，无需对输入过程进行持续实时保存；
- 本地优化前会释放 ComfyUI 已加载的其他模型，优化完成后立即卸载本地语言模型，减少显存占用；工作流执行期间禁用本地模型优化，在线 API 优化不受限制。

#### 在线 API 与 RunningHub

- RunningHub 分为“RunningHub 国内版（推荐）”和“RunningHub 海外版”，分别调用各自的 API 域名与 AI 应用；两个平台的 API Key、已选模型及模型列表相互独立，并随工作流分别保存，切换平台后会恢复该平台上次选择的模型；
- 国内版与海外版均支持手动刷新云端模型列表，并提供模型关键字实时搜索；只有点击刷新按钮时才访问 RunningHub 模型页面，不会因打开设置窗口而自动联网或卡顿；
- 国内版 AI 应用 ID 为 `2089252473927196673`，海外版 AI 应用 ID 为 `2090668262521675778`；参考图片、视频、系统提示词、用户提示词和最大输出 Token 会按两个应用各自的节点映射传入；
- 在线 API 还支持 OpenAI Chat Completions、OpenAI Responses、Google Gemini、OpenRouter、阿里云百炼、SiliconFlow 及自定义兼容接口；
- SiliconFlow 的 Qwen3、DeepSeek、GLM 等推理模型会自动关闭思考模式，避免输出额度全部消耗在 `reasoning_content` 后出现“后台成功并扣费，但最终提示词为空”的情况；如果第三方接口仍只返回推理内容，节点会显示明确的空输出原因；
- RunningHub 云端任务由节点持续查询到最终成功或失败；返回 TXT 下载地址时会自动读取文件正文作为优化结果，不需要用户手动下载；
- 国内版和海外版 API Key 不能混用。节点会根据当前平台强制选取对应 Key，避免切换平台后误用另一区域的密钥。

#### 最大输出 Tokens

- 配置页面提供统一的“最大输出 Tokens”选项，默认 `4096`，最小 `512`，最大 `8192`，并随工作流保存；
- 此数值统一作用于 Transformers 本地模型、GGUF 本地模型、OpenAI 兼容接口、OpenAI Responses、Gemini、SiliconFlow，以及 RunningHub 国内版和海外版公开的 `max_Token` 节点；
- 最大 Token 仅代表允许生成的输出上限，不会强制模型写满该长度，也不会直接裁剪输入内容；输入提示词、视觉 Token 与最大输出仍需共同满足所用模型的上下文窗口限制。

#### 本地视觉模型管理

- 本地模型下拉列表支持点击后显示搜索框，输入任意局部关键词即可实时筛选模型；模型名称过长时列表会按文件名扩展显示宽度；
- GGUF 主模型和 `mmproj` 视觉模型分别显示在两个下拉列表中。选择主模型时会自动推荐名称匹配度最高的 `mmproj`，用户仍可手动覆盖，刷新按钮会统一更新两份列表；
- 本地模型只扫描 `ComfyUI/models/llm`，主模型列表自动过滤 `mmproj` 文件，并只显示节点能够识别的视觉模型；
- 优化开始前会先卸载工作流残留模型并清理缓存，为本地语言模型释放显存；优化结束或失败后再次卸载本地模型，不常驻显存和内存。

### 素材排序与标签联动

- 首尾帧模式下，长按首帧或尾帧素材约 0.35 秒后，可拖到另一张图的位置交换顺序；
- 全能参考模式始终保持“视频 → 图像 → 音频”的素材类型顺序，同类型素材之间可长按自由排序；拖动生效时素材中央会显示移动图标，并高亮可放置目标；
- 排序、添加、删除或替换素材后，提示词中的 `<Picture N>`、`<Video N>`、`<Audio N>` 编号及缩略图会按新顺序同步更新；视频增删造成音频序号变化时，对应音频标签也会自动重排；
- 已选择驱动音频时，素材排序会按文件身份继续绑定同一个音频，只更新它的新序号；驱动音频设为“无”时，任何排序都不会自动选择音频。

### 富文本标签与素材预览

- 提示词框可识别 `picture 1`、`picture1`、`image 1`、`图像1`、`图1`、`图 1`、`视频1`、`音频1` 及对应的尖括号格式；有对应素材时显示 26 像素缩略图或音频图标，没有素材时仍保留代码块标签样式；
- 支持 `(S1)` 至 `(S20)`、`<Subject 1>`、`[Shot 1]`、`[场景1]`、`<d>`、`[Chinese]` 等官方标签；系统提示词会约束人物简写统一使用带括号的 `(S1)`、`(S2)` 格式；
- 图像标签使用低饱和暗青蓝色底，视频标签使用低饱和暗紫色底，音频及其他标签使用深灰色底，便于快速区分素材类型且不过度抢眼；
- 中文台词在英文提示词中保留原文，并使用 `<d>[Chinese]中文台词</d>` 格式；台词正文和官方段落标题分别高亮显示；
- 富文本编辑器支持自动换行、独立滚动、文本框内撤销和粘贴焦点保护，可兼容旧版 ComfyUI、Nodes 2.0、RunningHub 及其他云端环境，避免输入快捷键误触发工作流全局操作。

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
- 全能参考素材尺寸可选择匹配目标视频总像素、1.2倍、1.5倍、2倍或最大值，并同时作用于参考图片和参考视频帧；图片最高约400万像素，视频最低约30万、最高约160万像素，所有模式均只缩小不放大，并居中裁去不足32倍数的边缘，不拉伸也不填色；
- 默认输出时长为 2–30 秒，并按 H3 帧数网格对齐；
- 全能参考模式最多 9 张图片、3 个视频和 3 个独立音频；
- 超出目标时长的参考视频会被截取，较短的参考视频保持自身时长，不再重复最后一帧补齐；
- 条件编码完成后会优先通过ComfyUI模型管理接口卸载文本编码器并清理显存缓存，为后续采样释放空间；旧版ComfyUI缺少对应接口时会自动使用兼容释放路径；
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
