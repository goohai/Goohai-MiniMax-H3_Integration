import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE = "MiniMaxH3IntegrationGH";
const WIDTH = 500;
const PANEL_WIDTH = 476;
const INITIAL_NODE_HEIGHT = Math.round(WIDTH * 2 * 0.9);
const MAX_RESTORED_NODE_HEIGHT = INITIAL_NODE_HEIGHT * 1.5;
const MIN_NODE_HEIGHT = 0;
const ASPECTS = {
    adaptive: null,
    "16:9": 16 / 9, "9:16": 9 / 16, "3:2": 3 / 2, "2:3": 2 / 3, "4:3": 4 / 3,
    "3:4": 3 / 4, "1:1": 1, "21:9": 21 / 9,
};
const imageSlots = ["first_frame", "last_frame", ...Array.from({ length: 9 }, (_, i) => `ref_image_${i + 1}`)];
const videoSlots = Array.from({ length: 3 }, (_, i) => `ref_video_${i + 1}`);
const audioSlots = ["hybrid_audio", ...Array.from({ length: 3 }, (_, i) => `ref_audio_${i + 1}`)];
const mediaSlots = [...imageSlots, ...videoSlots, ...audioSlots];
const typeOrder = { video: 0, image: 1, audio: 2 };

const DOM_TRANSLATIONS = {
    "Audio": "音频",
    "First and last frames": "首尾帧",
    "Reference audio": "参考音频",
    "+ Add media": "＋ 添加素材",
    "T2VA · Text-to-video": "T2VA · 文生视频",
    "Text-to-video": "文生视频",
    "First/last frames / Text-to-video": "首尾帧 / 文生视频",
    "All-purpose reference": "全能参考",
    "Prompt:\nClick an uploaded asset to insert its tag, e.g. <picture 1>, <video 1>, or <audio 2>;\nDouble-click a video to insert its audio tag; mute a video at the top-right to exclude its audio from references": "提示词：\n单击已上传的素材自动添加标签，如：<picture 1>、<video 1>、<audio 2>；\n双击视频自动添加视频中的音频标签；视频右上角静音后音频将不传入参考",
    "Advanced options": "高级选项",
    "Audio mode": "音频模式",
    "Audio denoise strength": "音频去噪强度",
    "Source audio as reference": "源音频作为参考",
    "Strict prompt tags": "严格提示词标签",
    "Reference image size": "参考图尺寸",
    "Reference video policy": "参考视频策略",
    "Force FPS": "强制帧率",
    "match": "匹配",
    "max": "最大值",
    "First frame": "首帧",
    "Last frame": "尾帧",
    "First frame empty means text-to-video": "首帧留空为文生视频",
    "Last frame empty means text-to-video": "尾帧留空为文生视频",
    "Reference audio": "参考音频",
    "Optional": "可选",
    "Add media": "添加素材",
    "Reference media": "参考素材",
    "Supports up to 3 videos, 9 images, and 3 audios · Video MP4/MOV (2-15 seconds) · Audio MP3/WAV (2-15 seconds)": "支持视频x3，图片x9，音频x3 · 视频MP4/MOV(2-15秒) · 音频MP3/WAV(2-15秒)",
    "Up to 9 images": "最多9张图像",
    "Up to 3 videos": "最多3个视频",
    "Up to 3 audios": "最多3个音频",
};

Object.assign(DOM_TRANSLATIONS, {
    "Audio redraw strength": "\u97f3\u9891\u91cd\u7ed8\u5f3a\u5ea6",
    "native": "\u81ea\u52a8\u751f\u6210",
    "reference_only": "\u4ec5\u53c2\u8003\u97f3\u9891",
    "lock_source": "\u539f\u58f0\u8f93\u51fa",
    "remix_source": "\u91cd\u6df7\u6e90\u97f3\u9891",
    "Mute video": "\u4e0d\u53c2\u8003\u97f3\u9891",
    "Unmute video": "\u53c2\u8003\u97f3\u9891",
    "Drive audio ordinal": "\u9a71\u52a8\u97f3\u9891\u5e8f\u53f7",
    "Drive audio": "\u9a71\u52a8\u97f3\u9891",
    "None": "\u65e0",
});

function currentLocale() {
    const candidates = [
        app?.ui?.settings?.getSettingValue?.("Comfy.Locale"),
        app?.ui?.settings?.getSettingValue?.("Comfy.Language"),
        document.documentElement?.lang,
        document.body?.dataset?.locale,
        document.body?.dataset?.language,
        localStorage.getItem("Comfy.Settings.Language"),
        localStorage.getItem("Comfy.Settings.language"),
        localStorage.getItem("language"),
        localStorage.getItem("locale"),
        navigator.language,
    ];
    return candidates.find(value => typeof value === "string" && value.trim())?.toLowerCase() || "en";
}

function isChineseLocale() { return /^(zh|cn|中文)/i.test(currentLocale()); }
function t(text) { return isChineseLocale() ? (DOM_TRANSLATIONS[text] || text) : text; }
function aspectInternalValue(value) {
    const aliases = {
        "自适应": "adaptive",
        "16:9 (Widescreen)": "16:9",
        "9:16 (Portrait)": "9:16",
        "21:9 (Ultrawide)": "21:9",
    };
    return aliases[value] || value;
}
function aspectDisplayValue(value) {
    const internal = aspectInternalValue(value);
    if (isChineseLocale() && internal === "adaptive") return "自适应";
    return internal;
}

const ASPECT_INTERNAL_VALUES = [
    "adaptive", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "1:1", "21:9",
];
const ASPECT_DISPLAY_VALUES = {
    adaptive: "自适应",
    "16:9": "16:9",
    "9:16": "9:16",
    "3:2": "3:2",
    "2:3": "2:3",
    "4:3": "4:3",
    "3:4": "3:4",
    "1:1": "1:1",
    "21:9": "21:9",
};
function aspectOptionLabel(value) {
    const internal = aspectInternalValue(value);
    return isChineseLocale() ? (ASPECT_DISPLAY_VALUES[internal] || internal) : internal;
}
function normalizeAspectWidget(node) {
    const control = widget(node, "aspect");
    if (!control) return;
    const normalized = aspectInternalValue(control.value);
    if (!ASPECT_INTERNAL_VALUES.includes(normalized)) return;
    if (control.value !== normalized) control.value = normalized;
    control._ghH3AspectInternalValue = normalized;
}
function installAspectDisplay(node) {
    const control = widget(node, "aspect");
    if (!control || control._ghH3AspectDisplayWrapped || typeof control.draw !== "function") return;
    const originalDraw = control.draw;
    control.draw = function(ctx, nodeRef, width, y, height) {
        const savedValue = this.value;
        const savedValues = this.options?.values;
        const internalValues = ASPECT_INTERNAL_VALUES.slice();
        try {
            if (this.options && Array.isArray(savedValues)) {
                this.options.values = internalValues.map(aspectOptionLabel);
                this.value = aspectOptionLabel(savedValue);
            }
            return originalDraw.call(this, ctx, nodeRef, width, y, height);
        } finally {
            this.value = savedValue;
            if (this.options && Array.isArray(savedValues)) this.options.values = savedValues;
        }
    };
    control._ghH3AspectDisplayWrapped = true;
}
function drawAspectDisplayOverlay(node, ctx) {
    if (!isChineseLocale()) return;
    const control = widget(node, "aspect");
    if (!control || control.value !== "adaptive" || control === node.__ghH3ActiveWidget) return;
    const y = Number(control.y);
    const height = Number(control.height) || 30;
    const width = Number(node.size?.[0]) || 0;
    if (!Number.isFinite(y) || width <= 80) return;
    const left = 34;
    const right = width - 34;
    ctx.save();
    ctx.fillStyle = node.bgcolor || "#252a34";
    ctx.fillRect(left, y + 2, Math.max(0, right - left), Math.max(0, height - 4));
    ctx.fillStyle = "#e4e8ed";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = `${Math.max(12, Math.round(height * 0.55))}px Arial,sans-serif`;
    ctx.fillText("自适应", (left + right) / 2, y + height / 2);
    ctx.restore();
}

function round32(v) { return Math.max(32, Math.floor(v / 32 + 0.5) * 32); }
function canvas(aspect, mp, adaptiveRatio = 16 / 9) {
    const area = Math.max(0.2, Number(mp) || 0.5) * 1024 * 1024;
    const ratio = ASPECTS[aspectInternalValue(aspect)] ?? adaptiveRatio;
    const width = round32(Math.sqrt(area * ratio));
    const height = ratio === 1 ? width : round32(Math.sqrt(area / ratio));
    return `${width}x${height}`;
}
function widget(node, name) { return node.widgets?.find(w => w.name === name); }
function setWidget(node, name, value) {
    const w = widget(node, name); if (!w) return;
    // Zero is a valid value for integer/float controls such as the optional
    // primary-audio ordinal. Only actual empty values should become (none).
    w.value = value === null || value === undefined || value === "" ? "(none)" : value;
    w.callback?.call(w, w.value);
}
function setPromptWidget(node, value) {
    const w = widget(node, "prompt"); if (!w) return;
    w.value = value ?? "";
    w.callback?.call(w, w.value);
}
function cleanPrompt(value) { return value && value !== "(none)" ? value : ""; }
function setMediaWidget(node, name, value) {
    const w = widget(node, name); if (!w) return;
    // Media slots are optional STRING inputs. Keep unused slots truly empty;
    // the old Combo sentinel "(none)" is not a valid uploaded filename.
    w.value = value === null || value === undefined || value === "" || value === "(none)" ? "" : value;
    w.callback?.call(w, w.value);
}
function normalizeIntWidget(node, name, fallback, minimum, maximum) {
    const w = widget(node, name); if (!w) return fallback;
    const parsed = Number(w.value);
    const value = Number.isFinite(parsed)
        ? Math.max(minimum, Math.min(maximum, Math.trunc(parsed)))
        : fallback;
    if (w.value !== value) {
        w.value = value;
        w.callback?.call(w, value);
    }
    return value;
}
function sanitizeHiddenInputs(node) {
    normalizeIntWidget(node, "drive_audio_ordinal", 1, 0, 6);
    const task = widget(node, "task_type");
    if (task && !["auto", "t2va", "i2va", "l2va", "fl2va", "ref2va", "hybrid"].includes(task.value)) {
        task.value = "auto";
    }
}
function hideWidget(w) {
    if (!w) return;
    w.hidden = true; w.options = w.options || {}; w.options.hidden = true;
    w.computeSize = () => [0, -4]; w.serialize = true;
}
function make(tag, css = {}, text = "") {
    const el = document.createElement(tag); Object.assign(el.style, css);
    if (text) el.textContent = text; return el;
}
function kindOf(file) {
    if (file.type?.startsWith("image/") || /\.(png|jpe?g|webp|bmp|gif)$/i.test(file.name)) return "image";
    if (file.type?.startsWith("video/") || /\.(mp4|mov|webm|mkv|avi)$/i.test(file.name)) return "video";
    if (file.type?.startsWith("audio/") || /\.(mp3|wav|flac|m4a|ogg|aac)$/i.test(file.name)) return "audio";
    return null;
}
function fileUrl(name) { return name && name !== "(none)" ? `/view?filename=${encodeURIComponent(name)}&type=input&subfolder=` : ""; }
async function uploadFile(file) {
    const body = new FormData(); body.append("image", file, file.name); body.append("type", "input");
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
    return (await response.json()).name;
}

function createPanel(node) {
    const root = make("div", { position: "relative", width: `${PANEL_WIDTH}px`, maxWidth: "100%", boxSizing: "border-box", color: "#d7e3ef", fontFamily: "Arial,sans-serif", fontSize: "12px", userSelect: "none", padding: "3px 0 2px", overflow: "visible" });
    Object.assign(root.style, {
        display: "grid",
        gridTemplateRows: "auto auto auto minmax(0, 1fr)",
        alignItems: "stretch",
        height: "100%",
        minHeight: "0",
        overflow: "visible",
    });
    const style = make("style");
    style.textContent = `
      .ghh3-modes{display:grid;grid-template-columns:1fr 1fr;border:1px solid #2d4255;border-radius:9px;overflow:hidden;margin:3px 0 6px}
      .ghh3-mode{height:30px;border:0;background:#14202c;color:#d9e5ef;font-size:13px;cursor:pointer}.ghh3-mode.active{background:#0aa4d6;color:#06131b;font-weight:600}
      .ghh3-box{border:1px solid #334a5d;border-radius:8px;padding:7px;margin:0 0 6px;background:#111c27}.ghh3-title{font-size:12px;color:#edf5fb;margin-bottom:3px}.ghh3-hint{font-size:10px;color:#8697a7;line-height:1.3}
      .ghh3-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:5px}.ghh3-drop{aspect-ratio:1;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;color:#08b4ed;cursor:pointer;border:1px dashed #2c5368;border-radius:6px;background:#101b26;padding:4px;box-sizing:border-box}.ghh3-drop:hover{border-color:#0aa4d6;background:#142633}.ghh3-reference-empty{grid-column:1/-1;width:100%;aspect-ratio:5.2/1;align-items:flex-start;justify-content:center;text-align:left;padding:18px 24px}.ghh3-reference-empty .ghh3-drop-icon{font-size:14px;margin:0 7px 0 0}.ghh3-reference-empty .ghh3-drop-title{font-size:12px}.ghh3-reference-empty .ghh3-drop-subtitle{font-size:9px;margin-top:8px}.ghh3-drop-icon{font-size:14px;line-height:1.2;margin:0;color:#08b4ed;font-family:Arial,sans-serif}.ghh3-drop-title{font-size:10px;line-height:1.2;color:#d9e8f2}.ghh3-drop-title-row{display:flex;align-items:center;justify-content:center;gap:5px;line-height:1.2}.ghh3-optional{color:#416d86;font-size:.82em;line-height:1.2;position:relative;top:-1px}.ghh3-audio-drop .ghh3-optional{top:-2px}.ghh3-drop-subtitle{font-size:8px;line-height:1.25;color:#8697a7;margin-top:3px}.ghh3-limit{grid-column:1/-1;color:#d47d8b;font-size:9px;padding:2px 3px 0;text-align:left}
      .ghh3-keygrid{display:grid;grid-template-columns:1fr 1fr;gap:5px}.ghh3-keygrid .ghh3-drop{aspect-ratio:16/9}.ghh3-keygrid .ghh3-audio-card,.ghh3-keygrid .ghh3-audio-drop{grid-column:1/-1;width:100%;height:34px;aspect-ratio:auto;margin-top:3px}
      .ghh3-card{min-width:0;aspect-ratio:1;border:1px solid #30485c;border-radius:6px;background:#1a2938;overflow:hidden;position:relative;cursor:pointer}.ghh3-card img,.ghh3-card video{display:block;width:100%;height:100%;object-fit:cover;background:#071018}.ghh3-card:hover img,.ghh3-card:hover video{object-fit:contain}.ghh3-card-name{position:absolute;left:0;right:0;bottom:0;padding:2px 15px 2px 3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#fff;background:rgba(10,20,30,.6);font-size:7px;line-height:1.15}.ghh3-remove{position:absolute;right:1px;bottom:0;border:0;background:transparent;color:#d3e0ea;cursor:pointer;font-size:11px;z-index:3}.ghh3-media-controls{position:absolute;left:3px;right:3px;bottom:12px;z-index:4;height:14px;display:flex;align-items:center;color:rgba(255,255,255,.6);font:8px/1 Arial,sans-serif;pointer-events:none}.ghh3-media-toggle{width:14px;height:14px;padding:0;border:0;background:transparent;cursor:pointer;opacity:.6;display:flex;align-items:center;justify-content:center;pointer-events:auto}.ghh3-card:not(.ghh3-audio-card) .ghh3-media-toggle{background:rgba(0,0,0,.65);border-radius:50%}.ghh3-media-toggle svg{display:block;width:10px;height:10px;overflow:visible}.ghh3-media-time{margin-left:auto}.ghh3-audio-drop{grid-column:1/-1;width:100%;height:34px;min-height:34px;aspect-ratio:auto;margin-top:0;font-size:9px}.ghh3-audio-card{grid-column:1/-1;width:100%;height:34px;aspect-ratio:auto;margin-top:0}.ghh3-prompt{display:block;width:100%;height:100%;min-height:0;resize:none;overflow:auto;box-sizing:border-box;border:0;border-radius:6px;background:#1d2731;color:#e1e9ef;padding:7px 7px calc(7px + 14 * 1.4em);font:12px/1.4 Arial,sans-serif;outline:none;user-select:text;scrollbar-width:thin;scrollbar-color:#1f3540 transparent}.ghh3-prompt::placeholder{color:#6f7d89;opacity:1}.ghh3-prompt::-webkit-scrollbar{width:5px}.ghh3-prompt::-webkit-scrollbar-track{background:transparent}.ghh3-prompt::-webkit-scrollbar-thumb{background:#1f3540;border-radius:3px}.ghh3-prompt::-webkit-scrollbar-thumb:hover{background:#294955}.ghh3-advanced{position:absolute;left:0;right:0;top:auto;bottom:0;z-index:50;display:flex;flex-direction:column-reverse;height:auto;min-height:0;margin:0;padding:0 0 2px;box-sizing:border-box;user-select:none;overflow:visible;background:var(--ghh3-node-bg,#1d2731)!important;border:0;border-radius:0;box-shadow:none}.ghh3-advanced>summary{background:var(--ghh3-node-bg,#1d2731)!important;padding-left:16px;padding-right:16px}.ghh3-advanced .ghh3-advanced-body{background:var(--ghh3-node-bg,#1d2731)!important;padding-left:16px;padding-right:16px}.ghh3-advanced[open]{background:var(--ghh3-node-bg,#1d2731)!important;border:0;border-radius:0;box-sizing:border-box;box-shadow:none}.ghh3-size{color:#0db5e8;font-size:12px;padding:2px 0 4px}
      .ghh3-size{display:flex;justify-content:space-between;align-items:center;color:#0db5e8;font-size:12px;padding:2px 3px 5px}.ghh3-task{white-space:nowrap}.ghh3-dimensions{white-space:nowrap;text-align:right}.ghh3-advanced-row{display:grid;grid-template-columns:minmax(0,1fr) 220px;align-items:center;gap:8px;min-height:30px}.ghh3-advanced-row>label{text-align:left;color:#aebdca}.ghh3-control{width:220px;justify-self:end;box-sizing:border-box;background:#182633;color:#dbe8f1;border:1px solid #354b5d;border-radius:4px;padding:5px}.ghh3-number{width:220px;height:30px;display:grid;grid-template-columns:26px minmax(0,1fr) 26px;align-items:stretch;justify-self:end}.ghh3-number button{border:1px solid #354b5d;background:#182633;color:#c7d8e4;font-size:10px;padding:0;cursor:pointer}.ghh3-number button:first-child{border-radius:4px 0 0 4px}.ghh3-number button:last-child{border-radius:0 4px 4px 0}.ghh3-number input{width:100%;min-width:0;border:1px solid #354b5d;border-left:0;border-right:0;border-radius:0;background:#182633;color:#dbe8f1;padding:5px;box-sizing:border-box}.ghh3-number input::-webkit-inner-spin-button,.ghh3-number input::-webkit-outer-spin-button{appearance:none;margin:0}.ghh3-toggle{position:relative;display:inline-flex;width:38px;height:22px;justify-self:end;cursor:pointer}.ghh3-toggle input{opacity:0;width:0;height:0}.ghh3-toggle span{position:absolute;inset:0;border-radius:12px;background:#39434d;border:1px solid #52616d;transition:.15s}.ghh3-toggle span:before{content:"";position:absolute;width:16px;height:16px;left:2px;top:2px;border-radius:50%;background:#c3cbd1;transition:.15s}.ghh3-toggle input:checked+span{background:#0aa4d6;border-color:#0aa4d6}.ghh3-toggle input:checked+span:before{transform:translateX(16px);background:#fff}
      .ghh3-drop-title-row .ghh3-drop-icon{display:inline-flex;align-items:center;justify-content:center;height:1.2em;font-size:10px;line-height:1;margin:0}
      .ghh3-audio-drop .ghh3-drop-icon{font-size:14px;line-height:1;height:1.2em}
      .ghh3-drop-title-row .ghh3-optional{position:static;display:inline-flex;align-items:center;height:1.2em;line-height:1.2}
      .ghh3-keygrid .ghh3-drop:not(.ghh3-audio-drop) .ghh3-drop-icon,.ghh3-keygrid .ghh3-drop:not(.ghh3-audio-drop) .ghh3-optional{transform:translateY(-1px)}
    `;
    root.appendChild(style);
    const size = make("div"); size.className = "ghh3-size";
    const taskStatus = make("span", {}, t("T2VA · Text-to-video")); taskStatus.className = "ghh3-task";
    const dimensions = make("span", {}, canvas(widget(node, "aspect")?.value, widget(node, "megapixels")?.value)); dimensions.className = "ghh3-dimensions";
    size.append(taskStatus, dimensions);
    const modes = make("div"); modes.className = "ghh3-modes";
    const modeText = make("button", {}, t("First/last frames / Text-to-video")); modeText.className = "ghh3-mode";
    const modeRef = make("button", {}, t("All-purpose reference")); modeRef.className = "ghh3-mode"; modes.append(modeText, modeRef);
    root.append(size, modes);

    let layoutLock = false;
    let userHeight = INITIAL_NODE_HEIGHT;
    const stateKey = "gh_h3_state";
    const savedState = (() => {
        try { return JSON.parse(node.properties?.[stateKey] || "{}"); } catch { return {}; }
    })();
    node.properties = node.properties || {};
    const stateWidget = widget(node, "gh_state_json");
    if (stateWidget?.value) {
        try { Object.assign(savedState, JSON.parse(stateWidget.value)); } catch {}
    }
    const serializedState = () => {
        promptByMode[state.mode] = prompt.value;
        return JSON.stringify({
            mode: state.mode,
            media: [...media.entries()],
            prompt: prompt.value,
            prompts: { ...promptByMode },
            height: userHeight,
            advanced: !!advanced?.open,
        });
    };
    const persistState = () => {
        const value = serializedState();
        node.properties[stateKey] = value;
        if (stateWidget) stateWidget.value = value;
    };
    const domWidget = node.addDOMWidget("gh_h3_panel", "gh_h3_panel", root, { serialize: false, hideOnZoom: false });
    domWidget.options = domWidget.options || {}; domWidget.options.serialize = false;
    domWidget.options.getMinHeight = () => MIN_NODE_HEIGHT;
    domWidget.options.getHeight = () => "100%";
    domWidget.options.minNodeSize = [WIDTH, 0];
    const baseComputeSize = node.computeSize.bind(node);
    node.computeSize = function(out) {
        const measured = baseComputeSize(out);
        measured[0] = WIDTH;
        return measured;
    };
    const promptWidget = widget(node, "prompt"); hideWidget(widget(node, "main_mode")); hideWidget(promptWidget);
    for (const name of [
        ...mediaSlots, "task_type", "audio_mode", "audio_denoise_strength",
        "drive_audio_ordinal",
        "ref_image_size", "strict_prompt_tags", "gh_state_json",
    ]) hideWidget(widget(node, name));
    // Old graphs may contain the literal string "(none)" in this hidden Int
    // input. Clean it before ComfyUI serializes/submits the prompt.
    sanitizeHiddenInputs(node);
    const prompt = make("textarea", { }); prompt.className = "ghh3-prompt"; prompt.value = cleanPrompt(promptWidget?.value); prompt.placeholder = t("Prompt:\nClick an uploaded asset to insert its tag, e.g. <picture 1>, <video 1>, or <audio 2>;\nDouble-click a video to insert its audio tag; mute a video at the top-right to exclude its audio from references");
    const legacyPrompt = cleanPrompt(savedState.prompt) || prompt.value;
    const promptByMode = {
        text_keyframes: savedState.prompts?.text_keyframes ?? legacyPrompt,
        all_reference: cleanPrompt(savedState.prompts?.all_reference),
    };
    let wheelBoundaryDirection = 0;
    let wheelBoundaryUntil = 0;
    let wheelBoundaryReleased = false;
    const advanced = make("details"); advanced.className = "ghh3-advanced";
    const advancedSummary = make("summary", { cursor: "pointer", color: "#a9bac8", padding: "7px 16px 8px", minHeight: "20px", lineHeight: "20px", boxSizing: "border-box" }, t("Advanced options"));
    advanced.appendChild(advancedSummary);
    const advancedBody = make("div", { display: "grid", gap: "5px", padding: "5px 16px 8px" }); advancedBody.className = "ghh3-advanced-body";
    advanced.appendChild(advancedBody);
function nodeColorToCss(value) {
        if (typeof value === "string" && value.trim()) return value.trim();
        if (Array.isArray(value) && value.length >= 3) {
            const values = value.slice(0, 4).map(Number);
            if (values.slice(0, 3).every(Number.isFinite)) {
                const normalized = values.slice(0, 3).every(v => v >= 0 && v <= 1);
                const rgb = values.slice(0, 3).map(v => Math.max(0, Math.min(255, Math.round(normalized ? v * 255 : v))));
                const alpha = Number.isFinite(values[3]) ? Math.max(0, Math.min(1, values[3])) : 1;
                return `rgba(${rgb.join(",")},${alpha})`;
            }
        }
        if (Number.isFinite(value)) {
            const hex = Math.max(0, Math.min(0xffffff, Math.trunc(value))).toString(16).padStart(6, "0");
            return `#${hex}`;
        }
        return "";
    }
    function advancedBackgroundColor() {
        const color = nodeColorToCss(node.bgcolor)
            || nodeColorToCss(node.color)
            || nodeColorToCss(node.properties?.bgcolor)
            || nodeColorToCss(node.properties?.color)
            || nodeColorToCss(node._bgcolor)
            || nodeColorToCss(node._color)
            || "#1d2731";
        return color;
    }
    function syncAdvancedBackground(force = false) {
        const color = advancedBackgroundColor();
        if (!force && color === root.dataset.ghh3NodeBg) return;
        root.dataset.ghh3NodeBg = color;
        root.style.setProperty("--ghh3-node-bg", color);
        advanced.style.setProperty("background-color", color, "important");
        advancedSummary.style.setProperty("background-color", color, "important");
        advancedBody.style.setProperty("background-color", color, "important");
    }
    syncAdvancedBackground(true);
    const advancedLabels = new Map();
    const advancedRows = new Map();
    const addAdvanced = (name, label, control) => { const row = make("div"); row.className = "ghh3-advanced-row"; const labelEl = make("label", {}, label); labelEl.dataset.ghh3Translation = label; row.appendChild(labelEl); row.appendChild(control); advancedBody.appendChild(row); advancedLabels.set(name, labelEl); advancedRows.set(name, row); };
    const localizedSelects = [];
    const select = (name, values) => {
        const s = document.createElement("select"); s.className = "ghh3-control";
        const syncOptions = () => {
            const current = s.value || widget(node, name)?.value || values[0];
            s.replaceChildren(...values.map(v => new Option(t(v), v)));
            s.value = values.includes(current) ? current : values[0];
        };
        syncOptions();
        s.onchange = () => setWidget(node, name, s.value);
        s._ghH3SyncOptions = syncOptions;
        localizedSelects.push(s);
        return s;
    };
    const number = (name, step, min, max) => { const wrap = make("div"); wrap.className = "ghh3-number"; const left = make("button", {}, "◀"); const n = document.createElement("input"); n.type = "number"; n.step = step; n.min = min; n.max = max; const raw = Number(widget(node, name)?.value); n.value = Number.isFinite(raw) ? raw : min; const right = make("button", {}, "▶"); const update = value => { const numeric = Number(value); const next = Math.max(Number(min), Math.min(Number(max), Number.isFinite(numeric) ? numeric : Number(min))); n.value = Number.isInteger(next) ? next : next.toFixed(2).replace(/0+$/, "").replace(/\.$/, ""); setWidget(node, name, next); }; left.onclick = () => update(Number(n.value || min) - Number(step)); right.onclick = () => update(Number(n.value || min) + Number(step)); n.onchange = () => update(n.value); wrap.append(left, n, right); return wrap; };
    const check = (name) => { const label = make("label"); label.className = "ghh3-toggle"; const c = document.createElement("input"); c.type = "checkbox"; c.checked = !!widget(node, name)?.value; const track = make("span"); c.onchange = () => setWidget(node, name, c.checked); label.append(c, track); return label; };
    addAdvanced("audio_mode", t("Audio mode"), select("audio_mode", ["native", "lock_source", "reference_only", "remix_source"]));
    const audioStrengthControl = number("audio_denoise_strength", "0.05", "0", "1");
    addAdvanced("audio_denoise_strength", t("Audio redraw strength"), audioStrengthControl);
    const driveAudioOrdinalControl = document.createElement("select");
    driveAudioOrdinalControl.className = "ghh3-control";
    addAdvanced("drive_audio_ordinal", t("Drive audio"), driveAudioOrdinalControl);
    addAdvanced("strict_prompt_tags", t("Strict prompt tags"), check("strict_prompt_tags"));
    addAdvanced("ref_image_size", t("Reference image size"), select("ref_image_size", ["match", "max"]));
    const audioModeWidget = widget(node, "audio_mode");
    const audioModeByMode = {};
    const audioModeAutoByMode = {};
    let audioStrengthManual = false;
    const driveAudioEntries = () => {
        const videos = [...media.entries()]
            .filter(([slot, entry]) => entry?.kind === "video" && !entry.muted)
            .sort((a, b) => a[0].localeCompare(b[0]));
        const audios = [...media.entries()]
            .filter(([slot, entry]) => entry?.kind === "audio" && slot !== "hybrid_audio")
            .sort((a, b) => a[0].localeCompare(b[0]));
        return [
            ...videos.map(([slot, entry], index) => ({ ordinal: index + 1, name: entry.name, slot })),
            ...audios.map(([slot, entry], index) => ({ ordinal: videos.length + index + 1, name: entry.name, slot })),
        ];
    };
    const allReferenceAudioCount = () => driveAudioEntries().length;
    const hasModeAudio = () => state.mode === "text_keyframes"
        ? media.has("hybrid_audio")
        : allReferenceAudioCount() > 0 && Number(widget(node, "drive_audio_ordinal")?.value || 0) > 0;
    const syncAudioModeDefault = () => {
        const mode = state.mode;
        if (audioModeAutoByMode[mode] !== false) {
            audioModeByMode[mode] = hasModeAudio() ? "lock_source" : "native";
        }
        setWidget(node, "audio_mode", audioModeByMode[mode] || "native");
        const control = advancedRows.get("audio_mode")?.querySelector("select");
        if (control) control.value = widget(node, "audio_mode")?.value;
        if (!audioStrengthManual) {
            const modeValue = widget(node, "audio_mode")?.value || "native";
            const strength = modeValue === "lock_source" ? 0 : 1;
            setWidget(node, "audio_denoise_strength", strength);
            const input = audioStrengthControl.querySelector("input");
            if (input) input.value = strength;
        }
    };
    const syncAudioModeAfterMediaChange = (slot) => {
        if (state.mode !== "text_keyframes" || slot !== "hybrid_audio") return;
        // Reference-audio add/remove starts a fresh automatic decision; the
        // user can still override it manually afterwards.
        audioModeAutoByMode[state.mode] = true;
        syncAudioModeDefault();
    };
    const updateAdvancedVisibility = () => {
        if (state.mode === "all_reference") {
            const count = allReferenceAudioCount();
            const current = Number(widget(node, "drive_audio_ordinal")?.value || 0);
            const entries = driveAudioEntries();
            const next = count === 0 ? 0 : entries.some(entry => entry.ordinal === current) ? current : 1;
            setWidget(node, "drive_audio_ordinal", next);
            driveAudioOrdinalControl.replaceChildren(
                new Option(t("None"), "0"),
                ...entries.slice(0, 6).map(entry => new Option(`<audio ${entry.ordinal}>${entry.name}`, String(entry.ordinal))),
            );
            driveAudioOrdinalControl.value = String(next);
        }
        const visible = state.mode === "text_keyframes"
            ? ["audio_mode", "audio_denoise_strength", "strict_prompt_tags"]
            : ["audio_mode", "audio_denoise_strength", "ref_image_size", "strict_prompt_tags", "drive_audio_ordinal"];
        for (const [name, row] of advancedRows) {
            const isVisible = visible.includes(name);
            row.hidden = !isVisible;
            row.style.display = isVisible ? "grid" : "none";
        }
        syncAudioModeDefault();
    };
    advancedRows.get("audio_mode")?.querySelector("select")?.addEventListener("change", event => {
        audioModeByMode[state.mode] = event.target.value;
        audioModeAutoByMode[state.mode] = false;
        audioStrengthManual = false;
        syncAudioModeDefault();
    });
    audioStrengthControl.querySelector("input")?.addEventListener("input", () => { audioStrengthManual = true; });
    driveAudioOrdinalControl.addEventListener("change", () => {
        if (state.mode === "all_reference" && Number(driveAudioOrdinalControl.value || 0) === 0) {
            audioModeAutoByMode[state.mode] = true;
            audioModeByMode[state.mode] = "native";
            setWidget(node, "audio_mode", "native");
            const control = advancedRows.get("audio_mode")?.querySelector("select");
            if (control) control.value = "native";
            return;
        }
        if (audioModeAutoByMode[state.mode] !== false) syncAudioModeDefault();
    });
    function applyLocale() {
        taskStatus.textContent = `${resolvedTaskType()} · ${taskLabel(resolvedTaskType())}`;
        modeText.textContent = t("First/last frames / Text-to-video");
        modeRef.textContent = t("All-purpose reference");
        prompt.placeholder = t("Prompt:\nClick an uploaded asset to insert its tag, e.g. <picture 1>, <video 1>, or <audio 2>;\nDouble-click a video to insert its audio tag; mute a video at the top-right to exclude its audio from references");
        advancedSummary.textContent = t("Advanced options");
        for (const label of advancedLabels.values()) label.textContent = t(label.dataset.ghh3Translation);
        localizedSelects.forEach(control => control._ghH3SyncOptions?.());
        updateAdvancedVisibility();
        render();
    }
    // Read locale at node initialization. A storage event can still refresh
    // the UI after a real settings change, but no periodic polling is needed.
    window.addEventListener("storage", applyLocale);
    advanced.addEventListener("toggle", () => { persistState(); });
    root.appendChild(prompt); root.appendChild(advanced);
    prompt.addEventListener("input", () => { promptByMode[state.mode] = prompt.value; setPromptWidget(node, prompt.value); persistState(); });
    prompt.addEventListener("pointerdown", e => { if (e.button !== 1) e.stopPropagation(); });
    const promptCanConsumeWheel = event => {
        const inPrompt = event.composedPath?.().includes(prompt) || event.target === prompt;
        if (!inPrompt || prompt.scrollHeight <= prompt.clientHeight) return false;
        const atTop = prompt.scrollTop <= 0;
        const atBottom = prompt.scrollTop + prompt.clientHeight >= prompt.scrollHeight - 1;
        return (event.deltaY < 0 && !atTop) || (event.deltaY > 0 && !atBottom);
    };
    const capturePromptWheel = event => {
        if (promptCanConsumeWheel(event)) {
            wheelBoundaryDirection = 0;
            wheelBoundaryUntil = 0;
            wheelBoundaryReleased = false;
            event.preventDefault();
            event.stopImmediatePropagation();
            prompt.scrollTop += event.deltaY;
            return;
        }
        const inPrompt = event.composedPath?.().includes(prompt) || event.target === prompt;
        if (!inPrompt || prompt.scrollHeight <= prompt.clientHeight || !event.deltaY) return;
        const direction = event.deltaY < 0 ? -1 : 1;
        const now = performance.now();
        if (wheelBoundaryDirection !== direction) {
            wheelBoundaryDirection = direction;
            wheelBoundaryUntil = now + 1000;
            wheelBoundaryReleased = false;
        }
        if (!wheelBoundaryReleased && now < wheelBoundaryUntil) {
            event.preventDefault();
            event.stopImmediatePropagation();
            return;
        }
        if (!wheelBoundaryReleased) wheelBoundaryReleased = true;
    };
    window.addEventListener("wheel", capturePromptWheel, { capture: true, passive: false });
    root.addEventListener("wheel", event => {
        // Forward the native delta unchanged when the prompt cannot consume
        // the event, including when it is at the top or bottom edge.
        if (promptCanConsumeWheel(event)) return;
        if (event.defaultPrevented) return;
        const canvas = app.canvas;
        if (canvas?.processMouseWheel) {
            event.preventDefault();
            canvas.processMouseWheel(event);
        }
    }, { capture: true, passive: false });
    let middlePan = null;
    const moveMiddlePan = event => {
        if (!middlePan) return;
        const canvas = app.canvas;
        const scale = Number(canvas?.ds?.scale) || 1;
        const dx = (event.clientX - middlePan.x) / scale;
        const dy = (event.clientY - middlePan.y) / scale;
        middlePan.x = event.clientX;
        middlePan.y = event.clientY;
        if (canvas?.ds?.offset) {
            canvas.ds.offset[0] += dx;
            canvas.ds.offset[1] += dy;
            canvas.setDirty?.(true, true);
            node.setDirtyCanvas(true, true);
        }
        event.preventDefault();
        event.stopImmediatePropagation();
    };
    const stopMiddlePan = event => {
        if (!middlePan || (event.type !== "blur" && event.button !== 1)) return;
        middlePan = null;
        document.body.style.removeProperty("cursor");
        window.removeEventListener("pointermove", moveMiddlePan, true);
        window.removeEventListener("pointerup", stopMiddlePan, true);
        window.removeEventListener("blur", stopMiddlePan, true);
        event.preventDefault?.();
        event.stopImmediatePropagation?.();
    };
    prompt.addEventListener("pointerdown", event => {
        if (event.button !== 1) return;
        middlePan = { x: event.clientX, y: event.clientY };
        document.body.style.cursor = "grabbing";
        window.addEventListener("pointermove", moveMiddlePan, true);
        window.addEventListener("pointerup", stopMiddlePan, true);
        window.addEventListener("blur", stopMiddlePan, true);
        event.preventDefault();
        event.stopImmediatePropagation();
    }, true);
    function syncLayout(height = userHeight, writeNode = false) {
        if (layoutLock) return;
        layoutLock = true;
        try {
            const numericHeight = Number(height);
            const nextHeight = Number.isFinite(numericHeight)
                ? Math.max(0, Math.ceil(numericHeight))
                : userHeight;
            userHeight = nextHeight;
            if (writeNode && (node.size?.[0] !== WIDTH || node.size?.[1] !== nextHeight)) {
                node.setSize([WIDTH, nextHeight]);
            }
            root.style.height = "100%";
            prompt.style.height = "100%";
            node.setDirtyCanvas(true, true);
        } finally {
            layoutLock = false;
        }
    }
    const previousOnResize = node.onResize;
    node.onResize = function(...args) {
        const nextHeight = Number(this.size?.[1]);
        if (this.size?.[0] !== WIDTH) this.size[0] = WIDTH;
        previousOnResize?.apply(this, args);
        if (layoutLock || !Number.isFinite(nextHeight)) return;
        syncLayout(nextHeight, false);
        persistState();
    };

    const mode = widget(node, "main_mode"), aspect = widget(node, "aspect"), mp = widget(node, "megapixels");
    const durationWidget = widget(node, "duration_seconds");
    if (durationWidget) {
        durationWidget.options = durationWidget.options || {};
        durationWidget.options.precision = 0;
        durationWidget.options.step = 1;
        try {
            Object.defineProperty(durationWidget.options, "step2", {
                value: 1, writable: true, configurable: true,
            });
        } catch {
            durationWidget.options.step2 = 1;
        }
        durationWidget.options.round = 1;
        const oldDurationCallback = durationWidget.callback;
        durationWidget.callback = function(value) {
            oldDurationCallback?.call(this, value);
            this.value = Math.max(2, Math.min(15, Math.round(Number(value) || 2)));
            node.setDirtyCanvas(true, true);
        };
    }
    const state = { mode: savedState.mode || mode?.value || "text_keyframes" }; const media = new Map(savedState.media || []);
    let adaptiveRatio = 16 / 9;
    function firstVisualEntry() {
        const slots = state.mode === "text_keyframes"
            ? ["first_frame", "last_frame"]
            : [...videoSlots, ...imageSlots.slice(2)];
        return slots.map(slot => media.get(slot)).find(entry =>
            entry && (entry.kind === "image" || entry.kind === "video")
        ) || null;
    }
    function refreshAdaptiveRatio() {
        if (aspectInternalValue(aspect?.value) !== "adaptive") {
            adaptiveRatio = 16 / 9;
            refreshSize?.();
            return;
        }
        const entry = firstVisualEntry();
        if (!entry) {
            adaptiveRatio = 16 / 9;
            refreshSize?.();
            return;
        }
        const url = fileUrl(entry.name);
        if (entry.kind === "image") {
            const image = new Image();
            image.onload = () => { if (image.naturalWidth && image.naturalHeight) { adaptiveRatio = image.naturalWidth / image.naturalHeight; refreshSize?.(); } };
            image.src = url;
        } else {
            const video = document.createElement("video");
            video.onloadedmetadata = () => { if (video.videoWidth && video.videoHeight) { adaptiveRatio = video.videoWidth / video.videoHeight; refreshSize?.(); } };
            video.src = url;
        }
    }
    prompt.value = cleanPrompt(promptByMode[state.mode]);
    setPromptWidget(node, prompt.value);
    let uploadNotice = "";
    function syncMediaWidgets() {
        for (const name of mediaSlots) setMediaWidget(node, name, media.get(name)?.name || "");
    }
    if (stateWidget) {
        stateWidget.serialize = true;
        stateWidget.options = stateWidget.options || {};
        stateWidget.options.serialize = true;
        stateWidget.serializeValue = () => {
            syncMediaWidgets();
            persistState();
            return stateWidget.value;
        };
    }
    for (const name of mediaSlots) {
        const value = widget(node, name)?.value;
        if (!media.has(name) && value && value !== "(none)") media.set(name, { name: value, kind: kindOf({ name: value, type: "" }) });
    }
    syncMediaWidgets();
    function nextSlot(kind) { const slots = kind === "image" ? imageSlots.slice(2) : kind === "video" ? videoSlots : audioSlots.slice(1); return slots.find(s => !media.has(s)); }
    function resolvedTaskType() {
        const first = media.has("first_frame");
        const last = media.has("last_frame");
        const refs = [...media.entries()].some(([slot]) => !["first_frame", "last_frame", "hybrid_audio"].includes(slot));
        const hybridAudio = media.has("hybrid_audio");
        if (state.mode === "all_reference") return refs ? "Ref2VA" : "T2VA";
        // Keyframe mode ignores all-reference slots. Old serialized graphs
        // may still contain ref_* entries after switching modes, so they must
        // not turn a plain I2VA/FL2VA/L2VA state into Hybrid in the UI.
        if (state.mode === "text_keyframes") {
            if (hybridAudio && (first || last)) return "Hybrid";
            if (first && last) return "FL2VA";
            if (first) return "I2VA";
            if (last) return "L2VA";
            return "T2VA";
        }
        if (refs) return (first || last) ? "Hybrid" : "Ref2VA";
        if (hybridAudio && (first || last)) return "Hybrid";
        if (first && last) return "FL2VA";
        if (first) return "I2VA";
        if (last) return "L2VA";
        return "T2VA";
    }
    const taskLabels = { I2VA: "First frame", FL2VA: "First and last frames", L2VA: "Last frame", T2VA: "Text-to-video", Ref2VA: "All-purpose reference" };
    function taskLabel(task) {
        if (task !== "Hybrid") return t(taskLabels[task] || task);
        if (state.mode === "text_keyframes") {
            const first = media.has("first_frame");
            const last = media.has("last_frame");
            if (first && last) return t("First and last frames") + " + " + t("Audio");
            if (first) return t("First frame") + " + " + t("Audio");
            if (last) return t("Last frame") + " + " + t("Audio");
        }
        return t("Reference media") + " + " + t("Audio");
    }
    function repairLegacyReferenceTags() {
        if (state.mode !== "all_reference" || !prompt.value) return;
        const activeCounts = { image: 0, video: 0, audio: 0 };
        for (const [slot, entry] of media) {
            if (["first_frame", "last_frame", "hybrid_audio"].includes(slot)) continue;
            if (entry?.kind in activeCounts) activeCounts[entry.kind] += 1;
        }
        let value = prompt.value;
        for (const [kind, label] of [["image", "Picture"], ["video", "Video"], ["audio", "Audio"]]) {
            if (activeCounts[kind] !== 1) continue;
            const one = new RegExp(`<${label}\\s+1>`, "i");
            if (one.test(value)) continue;
            value = value.replace(new RegExp(`<${label}\\s+\\d+>`, "i"), `<${label} 1>`);
        }
        if (value !== prompt.value) {
            prompt.value = value;
            promptByMode[state.mode] = value;
            setPromptWidget(node, value);
            persistState();
        }
    }
    function refreshTaskType() { const task = resolvedTaskType(); taskStatus.textContent = `${task} · ${taskLabel(task)}`; dimensions.textContent = canvas(aspectInternalValue(aspect?.value), mp?.value, adaptiveRatio); setWidget(node, "task_type", "auto"); }
    function normalizePromptTagFormat() {
        const task = resolvedTaskType();
        const bare = task === "FL2VA";
        const normalized = (prompt.value || "").replace(
            /<\s*(Picture|Image|Video|Audio)\s*#?\s*(\d+)\s*>|(?<![\w<])(Picture|Image|Video|Audio)\s*#?\s*(\d+)\b(?!\s*>)/gi,
            (_match, bracketType, bracketOrdinal, bareType, bareOrdinal) => {
                const type = (bracketType || bareType).toLowerCase();
                const ordinal = bracketOrdinal || bareOrdinal;
                const officialType = type === "image" ? "Picture" : type[0].toUpperCase() + type.slice(1);
                return bare ? `${officialType} ${ordinal}` : `<${officialType} ${ordinal}>`;
            },
        );
        if (normalized !== prompt.value) {
            prompt.value = normalized;
            promptByMode[state.mode] = normalized;
            setPromptWidget(node, normalized);
            persistState();
        }
    }
    function labelFor(slot, kind) {
        if (slot === "first_frame") return t("First frame");
        if (slot === "last_frame") return t("Last frame");
        if (kind === "audio" && slot !== "hybrid_audio") return `audio ${audioOrdinalFor(slot)}`;
        const prefix = kind === "video" ? "video" : kind === "image" ? "picture" : "audio";
        // Frame uploads are kept in state when switching modes, but they are
        // not reference media in Ref2VA. Exclude them from reference ordinals
        // so a single reference picture is always <picture 1>.
        const list = [...media.entries()].filter(([s, e]) => {
            if (e.kind !== kind) return false;
            if (state.mode === "all_reference" && (s === "first_frame" || s === "last_frame")) return false;
            if (state.mode === "all_reference" && s === "hybrid_audio") return false;
            return true;
        }).sort((a,b) => a[0].localeCompare(b[0]));
        return `${prefix} ${list.findIndex(([s]) => s === slot) + 1}`;
    }
    function audioOrdinalFor(slot) {
        const videos = [...media.entries()].filter(([s, e]) => e?.kind === "video" && !e.muted).sort((a, b) => a[0].localeCompare(b[0]));
        const videoIndex = videos.findIndex(([s]) => s === slot);
        if (videoIndex >= 0) return videoIndex + 1;
        const audios = [...media.entries()].filter(([s, e]) => e?.kind === "audio" && s !== "hybrid_audio").sort((a, b) => a[0].localeCompare(b[0]));
        const audioIndex = audios.findIndex(([s]) => s === slot);
        return audioIndex >= 0 ? videos.length + audioIndex + 1 : 0;
    }
    function insertTag(kind, slot) {
        const raw = labelFor(slot, kind);
        const task = resolvedTaskType();
        let tag;
        if (slot === "first_frame" || slot === "last_frame") {
            const ordinal = task === "FL2VA"
                ? (slot === "first_frame" ? 1 : 2)
                : 1;
            tag = task === "FL2VA" ? `Picture ${ordinal}` : `<Picture ${ordinal}>`;
        } else {
            tag = raw.startsWith("picture ") ? `<Picture ${raw.slice(8)}>`
                : raw.startsWith("video ") ? `<Video ${raw.slice(6)}>`
                : raw.startsWith("audio ") ? `<Audio ${raw.slice(6)}>`
                : `<${raw}>`;
        }
        const a = document.activeElement === prompt ? prompt.selectionStart : prompt.value.length;
        const b = document.activeElement === prompt ? prompt.selectionEnd : a;
        prompt.setRangeText(tag, a, b, "end");
        promptByMode[state.mode] = prompt.value;
        setPromptWidget(node, prompt.value);
        persistState();
    }
    function insertVideoAudioTag(slot) {
        const ordinal = audioOrdinalFor(slot);
        if (media.get(slot)?.muted || !ordinal) return;
        const a = document.activeElement === prompt ? prompt.selectionStart : prompt.value.length;
        const b = document.activeElement === prompt ? prompt.selectionEnd : a;
        prompt.setRangeText(`<Audio ${ordinal}>`, a, b, "end"); promptByMode[state.mode] = prompt.value; setPromptWidget(node, prompt.value); persistState();
    }
    function ensureReferenceTags() {
        if (state.mode !== "all_reference" || !prompt.value) return;
        // If the prompt already references one media type, make sure newly
        // uploaded media of the other types is not silently omitted from H3.
        // Clicking a card still inserts at the caret; this only fills missing
        // tags after an upload when the prompt is already using media tags.
        if (!/<\s*(Picture|Image|Video|Audio)\s*\d+\s*>/i.test(prompt.value)) return;
        let value = prompt.value;
        for (const kind of ["image", "video", "audio"]) {
            const entries = [...media.entries()]
                .filter(([slot, entry]) => entry?.kind === kind && !["first_frame", "last_frame", "hybrid_audio"].includes(slot))
                .sort((a, b) => a[0].localeCompare(b[0]));
            entries.forEach(([slot], index) => {
                const label = kind === "image" ? "Picture" : kind[0].toUpperCase() + kind.slice(1);
                const tag = `<${label} ${index + 1}>`;
                if (!new RegExp(`<\\s*${label}\\s+${index + 1}\\s*>`, "i").test(value)) value += ` ${tag}`;
            });
        }
        if (value !== prompt.value) {
            prompt.value = value;
            promptByMode[state.mode] = value;
            setPromptWidget(node, value);
            persistState();
        }
    }
    let hoverPasteSlot = null;
    let activeMedia = null;
    const formatMediaTime = value => String(Math.max(0, Math.floor(Number(value) || 0))).padStart(2, "0");
    const stopActiveMedia = () => { if (!activeMedia) return; activeMedia.media.pause(); activeMedia.media.currentTime = 0; activeMedia.setPlaying(false); activeMedia = null; };
    const mediaIcon = playing => playing ? '<svg viewBox="0 0 12 12"><path d="M3 2v8M9 2v8" fill="none" stroke="rgba(255,255,255,.6)" stroke-width="1.5"/></svg>' : '<svg viewBox="0 0 12 12"><path d="M2.5 1.5 10 6l-7.5 4.5Z" fill="rgba(255,255,255,.6)" stroke="rgba(255,255,255,.6)" stroke-width="1"/></svg>';
    function card(slot, entry, square = true) {
        const item = make("div"); item.className = "ghh3-card"; if (!square) item.style.aspectRatio = "16/9"; if (slot === "hybrid_audio") { item.classList.add("ghh3-audio-card"); item.style.aspectRatio = "auto"; }
        item.dataset.ghh3DropSlot = slot;
        if (entry.kind === "image") { const img = make("img"); img.src = fileUrl(entry.name); item.appendChild(img); }
        else if (entry.kind === "video") { const video = make("video"); video.src = fileUrl(entry.name); video.muted = !!entry.muted; video.preload = "metadata"; item.appendChild(video); const play = make("button", {}, "▶"); play.className = "ghh3-play"; play.onclick = e => { e.stopPropagation(); video.muted = !!entry.muted; video.paused ? video.play() : video.pause(); }; item.appendChild(play); }
        else {
            item.appendChild(make("div", { height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#8ea3b4", fontSize: "22px" }, "♫"));
            const audio = new Audio(fileUrl(entry.name));
            const controls = make("div"); controls.className = "ghh3-media-controls";
            const toggle = make("button"); toggle.className = "ghh3-media-toggle";
            const time = make("span", {}, "00 s"); time.className = "ghh3-media-time";
            const setPlaying = playing => { toggle.innerHTML = mediaIcon(playing); };
            const refresh = () => { const remaining = audio.paused ? audio.duration : Math.max(0, audio.duration - audio.currentTime); time.textContent = `${formatMediaTime(remaining)} s`; };
            const start = () => { if (activeMedia && activeMedia.media !== audio) stopActiveMedia(); activeMedia = { media: audio, setPlaying }; audio.play().catch(() => {}); setPlaying(true); };
            setPlaying(false);
            toggle.onclick = e => { e.stopPropagation(); audio.paused ? start() : stopActiveMedia(); };
            audio.addEventListener("loadedmetadata", refresh); audio.addEventListener("timeupdate", refresh); audio.addEventListener("ended", () => { refresh(); setPlaying(false); if (activeMedia?.media === audio) activeMedia = null; });
            controls.append(toggle, time); item.appendChild(controls);
        }
        if (entry.kind === "video") {
            const sound = make("button", {}, entry.muted ? String.fromCodePoint(0x1F507) : String.fromCodePoint(0x1F50A));
            sound.className = "ghh3-sound";
            sound.style.cssText = "position:absolute;right:2px;top:2px;z-index:4;border:0;background:rgba(0,0,0,.62);color:#fff;border-radius:50%;width:14px;height:14px;padding:0;cursor:pointer;font-size:9px;line-height:14px";
            sound.title = t(entry.muted ? "Unmute video" : "Mute video");
            sound.onclick = e => { e.stopPropagation(); entry.muted = !entry.muted; persistState(); render(); };
            item.appendChild(sound);
            const video = item.querySelector("video");
            const oldPlay = item.querySelector(".ghh3-play");
            if (oldPlay) oldPlay.style.display = "none";
            const controls = make("div"); controls.className = "ghh3-media-controls";
            const toggle = make("button"); toggle.className = "ghh3-media-toggle";
            const time = make("span", {}, "00 s"); time.className = "ghh3-media-time";
            const setPlaying = playing => { toggle.innerHTML = mediaIcon(playing); };
            const refresh = () => { const remaining = video.paused ? video.duration : Math.max(0, video.duration - video.currentTime); time.textContent = `${formatMediaTime(remaining)} s`; };
            const start = () => { if (activeMedia && activeMedia.media !== video) stopActiveMedia(); activeMedia = { media: video, setPlaying }; video.muted = !!entry.muted; video.play().catch(() => {}); setPlaying(true); };
            setPlaying(false);
            toggle.onclick = e => { e.stopPropagation(); video.paused ? start() : stopActiveMedia(); };
            video.addEventListener("loadedmetadata", refresh); video.addEventListener("timeupdate", refresh); video.addEventListener("ended", () => { refresh(); setPlaying(false); if (activeMedia?.media === video) activeMedia = null; });
            controls.append(toggle, time); item.appendChild(controls);
        }
        item.appendChild(make("div", {}, `${labelFor(slot, entry.kind)}: ${entry.name}`)).className = "ghh3-card-name";
        const remove = make("button", {}, "×"); remove.className = "ghh3-remove"; remove.onclick = e => { e.stopPropagation(); media.delete(slot); setMediaWidget(node, slot, ""); syncAudioModeAfterMediaChange(slot); persistState(); render(); }; item.appendChild(remove);
        let clickTimer;
        item.onpointerdown = e => { if (e.target.closest("button")) return; e.preventDefault(); e.stopPropagation(); };
        item.onclick = e => { if (e.target.closest("button")) return; e.stopPropagation(); clearTimeout(clickTimer); clickTimer = setTimeout(() => insertTag(entry.kind, slot), 220); };
        item.ondblclick = e => { if (e.target.closest("button")) return; e.stopPropagation(); clearTimeout(clickTimer); if (entry.kind === "video") insertVideoAudioTag(slot); };
        item.onpointerenter = () => { hoverPasteSlot = slot; };
        item.onpointerleave = () => { if (hoverPasteSlot === slot) hoverPasteSlot = null; };
        item.ondragover = e => { e.preventDefault(); e.stopPropagation(); };
        item.ondrop = e => { e.preventDefault(); e.stopPropagation(); accept(e.dataTransfer.files, slot); };
        return item;
    }
    function choose(slot) { const input = document.createElement("input"); input.type = "file"; input.multiple = !slot; input.accept = slot === "first_frame" || slot === "last_frame" ? "image/*" : "image/*,video/*,audio/*"; input.onchange = () => accept(input.files, slot); input.click(); }
    function addDrop(slot = null, referenceEmpty = false) {
        let title = t("+ Add media");
        let subtitle = "";
        let icon = "📤︎";
        if (slot === "first_frame") { title = t("First frame"); subtitle = t("First frame empty means text-to-video"); }
        else if (slot === "last_frame") { title = t("Last frame"); subtitle = t("Last frame empty means text-to-video"); }
        else if (slot === "hybrid_audio") { title = t("Reference audio"); subtitle = t("Optional"); icon = "♫"; }
        else if (referenceEmpty) { title = t("Reference media"); subtitle = t("Supports up to 3 videos, 9 images, and 3 audios · Video MP4/MOV (2-15 seconds) · Audio MP3/WAV (2-15 seconds)"); }
        const d = make("div"); d.className = "ghh3-drop"; if (slot === "hybrid_audio") d.classList.add("ghh3-audio-drop"); if (referenceEmpty) d.classList.add("ghh3-reference-empty");
        d.dataset.ghh3DropSlot = slot || "";
        if (slot === "first_frame" || slot === "last_frame") {
            const row = make("div"); row.className = "ghh3-drop-title-row";
            row.appendChild(make("span", {}, icon)).className = "ghh3-drop-icon";
            row.appendChild(make("span", {}, title)).className = "ghh3-drop-title";
            row.appendChild(make("span", {}, `· ${t("Optional")}`)).className = "ghh3-optional";
            d.appendChild(row);
            d.appendChild(make("div", {}, subtitle)).className = "ghh3-drop-subtitle";
        } else if (slot === "hybrid_audio") {
            const row = make("div"); row.className = "ghh3-drop-title-row";
            row.appendChild(make("span", {}, icon)).className = "ghh3-drop-icon";
            row.appendChild(make("span", {}, title)).className = "ghh3-drop-title";
            row.appendChild(make("span", {}, `· ${t("Optional")}`)).className = "ghh3-optional";
            d.appendChild(row);
        } else {
            const row = make("div"); row.className = "ghh3-drop-title-row";
            row.appendChild(make("span", {}, icon)).className = "ghh3-drop-icon";
            row.appendChild(make("span", {}, title)).className = "ghh3-drop-title";
            d.appendChild(row);
            if (subtitle) d.appendChild(make("div", {}, subtitle)).className = "ghh3-drop-subtitle";
        }
        d.onclick = () => choose(slot); d.onpointerenter = () => { hoverPasteSlot = slot; }; d.onpointerleave = () => { if (hoverPasteSlot === slot) hoverPasteSlot = null; }; d.ondragover = e => { e.preventDefault(); e.stopPropagation(); }; d.ondrop = e => { e.preventDefault(); e.stopPropagation(); accept(e.dataTransfer.files, slot); }; return d;
    }
    function render() {
        normalizePromptTagFormat();
        updateAdvancedVisibility();
        root.querySelector(".ghh3-dynamic")?.remove(); const box = make("div"); box.className = "ghh3-box ghh3-dynamic";
        if (state.mode === "text_keyframes") {
            const grid = make("div"); grid.className = "ghh3-keygrid";
            for (const s of ["first_frame", "last_frame"]) grid.appendChild(media.has(s) ? card(s, media.get(s), false) : addDrop(s));
            grid.appendChild(media.has("hybrid_audio") ? card("hybrid_audio", media.get("hybrid_audio"), false) : addDrop("hybrid_audio"));
            box.appendChild(grid);
        } else {
            const grid = make("div"); grid.className = "ghh3-grid"; const entries = [...media.entries()].filter(([slot]) => !["first_frame", "last_frame", "hybrid_audio"].includes(slot)).sort((a,b) => typeOrder[a[1].kind] - typeOrder[b[1].kind]);
            if (!entries.length) grid.appendChild(addDrop(null, true));
            else { entries.forEach(([s,e]) => grid.appendChild(card(s,e))); grid.appendChild(addDrop()); }
            if (uploadNotice) grid.appendChild(make("div", {}, uploadNotice)).className = "ghh3-limit";
            box.appendChild(grid);
        }
    root.insertBefore(box, prompt); repairLegacyReferenceTags(); ensureReferenceTags(); refreshTaskType(); refreshAdaptiveRatio(); syncLayout();
    }
    function limitText(kind) {
        return kind === "image" ? t("Up to 9 images") : kind === "video" ? t("Up to 3 videos") : t("Up to 3 audios");
    }
    function filesFromPasteEvent(event) {
        const files = [];
        const items = event.clipboardData?.items || [];
        for (const item of items) {
            if (item.kind !== "file") continue;
            const file = item.getAsFile();
            if (!file) continue;
            const kind = kindOf(file);
            if (kind === "image" || kind === "video") files.push(file);
        }
        return files;
    }
    const onPaste = event => {
        if (!hoverPasteSlot) return;
        const files = filesFromPasteEvent(event);
        if (!files.length) return;
        event.preventDefault();
        accept(files, hoverPasteSlot);
    };
    document.addEventListener("paste", onPaste, true);
    async function accept(files, preferredSlot) {
        uploadNotice = "";
        for (const file of files || []) {
            const kind = kindOf(file); if (!kind) continue;
            const slot = preferredSlot || nextSlot(kind);
            if (!slot) { uploadNotice = limitText(kind); continue; }
            const existing = media.get(slot);
            if (existing && existing.kind !== kind) continue;
            try {
                const name = await uploadFile(file); media.set(slot, { name, kind }); setMediaWidget(node, slot, name); syncAudioModeAfterMediaChange(slot);
                ensureReferenceTags();
                render();
            } catch (e) { console.error("[MiniMax H3 Integration]", e); }
            preferredSlot = null;
        }
        persistState();
        render();
    }
    const captureMaterialDrop = event => {
        const target = event.target instanceof Element ? event.target : null;
        const materialArea = target?.closest?.(".ghh3-card, .ghh3-drop, .ghh3-dynamic");
        if (!materialArea || !root.contains(materialArea)) return;
        if (!event.dataTransfer?.types?.includes?.("Files")) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        if (event.type !== "drop") return;
        const slotElement = target.closest("[data-ghh3-drop-slot]");
        const preferredSlot = slotElement?.dataset?.ghh3DropSlot || null;
        accept(event.dataTransfer.files, preferredSlot);
    };
    window.addEventListener("dragenter", captureMaterialDrop, true);
    window.addEventListener("dragover", captureMaterialDrop, true);
    window.addEventListener("drop", captureMaterialDrop, true);
    function switchMode(nextMode) {
        promptByMode[state.mode] = prompt.value;
        state.mode = nextMode;
        prompt.value = cleanPrompt(promptByMode[state.mode]);
        setPromptWidget(node, prompt.value);
        setWidget(node, "main_mode", state.mode);
        persistState();
        modeText.classList.toggle("active", state.mode === "text_keyframes");
        modeRef.classList.toggle("active", state.mode === "all_reference");
        updateAdvancedVisibility();
        render();
    }
    modeText.onclick = () => switchMode("text_keyframes");
    modeRef.onclick = () => switchMode("all_reference");
    function refreshSize() {
        dimensions.textContent = canvas(aspectInternalValue(aspect?.value), mp?.value, adaptiveRatio);
        node.setDirtyCanvas(true, true);
    }
    normalizeAspectWidget(node);
    installAspectDisplay(node);
    aspect && (aspect.callback = ((old) => function(v) {
        const internalValue = ASPECT_INTERNAL_VALUES.includes(v) ? v : aspectInternalValue(v);
        old?.call(this, internalValue);
        this.value = internalValue;
        this._ghH3AspectInternalValue = internalValue;
        refreshSize();
        refreshAdaptiveRatio();
    })(aspect.callback));
    mp && (mp.callback = ((old) => function(v) { old?.call(this,v); refreshSize(); })(mp.callback));
    setWidget(node, "main_mode", state.mode);
    modeText.classList.toggle("active", state.mode === "text_keyframes");
    modeRef.classList.toggle("active", state.mode === "all_reference");
    updateAdvancedVisibility();
    render();
    advanced.open = false;
    for (const hook of ["onAdded", "onConfigure", "onGraphConfigured"]) {
        const old = node[hook];
        node[hook] = function(...args) {
            const configuredState = hook === "onConfigure" ? args[0]?.properties?.[stateKey] : null;
            const result = old?.apply(this, args);
            sanitizeHiddenInputs(this);
            for (const n of [
                "main_mode", "prompt", "task_type", ...mediaSlots,
                "audio_mode", "audio_denoise_strength",
                "drive_audio_ordinal", "ref_image_size", "strict_prompt_tags",
                "gh_state_json",
            ]) hideWidget(widget(node,n));
            normalizeIntWidget(node, "drive_audio_ordinal", 1, 0, 6);
            if (hook !== "onAdded") requestAnimationFrame(() => {
                let restored = {};
                try { restored = JSON.parse(configuredState || widget(node, "gh_state_json")?.value || node.properties?.[stateKey] || "{}"); } catch {}
                state.mode = restored.mode || widget(node, "main_mode")?.value || "text_keyframes";
                const restoredLegacyPrompt = cleanPrompt(restored.prompt) || cleanPrompt(widget(node, "prompt")?.value);
                promptByMode.text_keyframes = cleanPrompt(restored.prompts?.text_keyframes) || restoredLegacyPrompt;
                promptByMode.all_reference = cleanPrompt(restored.prompts?.all_reference);
                prompt.value = cleanPrompt(promptByMode[state.mode]);
                setPromptWidget(node, prompt.value);
                setWidget(node, "main_mode", state.mode);
                media.clear();
                for (const [slot, entry] of restored.media || []) media.set(slot, entry);
                for (const name of mediaSlots) {
                    const value = widget(node, name)?.value;
                    if (!media.has(name) && value && value !== "(none)") media.set(name, { name: value, kind: kindOf({ name: value, type: "" }) });
                    if (value === "(none)") setMediaWidget(node, name, "");
                }
                syncMediaWidgets();
                const restoredHasContent = Boolean(
                    restored.prompt
                    || restored.prompts?.text_keyframes
                    || restored.prompts?.all_reference
                    || (Array.isArray(restored.media) && restored.media.length),
                );
                if (restoredHasContent && Number.isFinite(Number(restored.height)) && Number(restored.height) <= MAX_RESTORED_NODE_HEIGHT) {
                    userHeight = Number(restored.height);
                }
                modeText.classList.toggle("active", state.mode === "text_keyframes");
                modeRef.classList.toggle("active", state.mode === "all_reference");
                updateAdvancedVisibility();
                render();
                syncLayout(userHeight, true);
            });
            return result;
        };
    }
    const oldDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function(...args) {
        syncAdvancedBackground();
        const result = oldDrawForeground?.apply(this, args);
        drawAspectDisplayOverlay(this, args[0]);
        return result;
    };
    const oldPropertyChanged = node.onPropertyChanged;
    node.onPropertyChanged = function(...args) {
        const result = oldPropertyChanged?.apply(this, args);
        syncAdvancedBackground(true);
        return result;
    };
    // Background synchronization is driven by node draw/property changes.
    // A per-frame style write can cause Nodes 2.0 to remeasure this DOM widget
    // continuously, so it must not participate in the animation loop.
    const oldSerialize = node.onSerialize;
    node.onSerialize = function(...args) {
        sanitizeHiddenInputs(this);
        syncMediaWidgets();
        persistState();
        const result = oldSerialize?.apply(this, args);
        const serializedNode = args[0];
        if (serializedNode) {
            serializedNode.properties = serializedNode.properties || {};
            serializedNode.properties[stateKey] = node.properties[stateKey];
        }
        return result;
    };
    const oldRemoved = node.onRemoved;
    node.onRemoved = function(...args) {
        window.removeEventListener("dragenter", captureMaterialDrop, true);
        window.removeEventListener("dragover", captureMaterialDrop, true);
        window.removeEventListener("drop", captureMaterialDrop, true);
        window.removeEventListener("wheel", capturePromptWheel, true);
        if (middlePan) stopMiddlePan({ type: "blur", preventDefault() {}, stopImmediatePropagation() {} });
        document.removeEventListener("paste", onPaste, true);
        window.removeEventListener("storage", applyLocale);
        return oldRemoved?.apply(this, args);
    };
    requestAnimationFrame(() => {
        const savedHasContent = Boolean(
            savedState.prompt
            || savedState.prompts?.text_keyframes
            || savedState.prompts?.all_reference
            || (Array.isArray(savedState.media) && savedState.media.length),
        );
        if (savedHasContent && Number.isFinite(Number(savedState.height)) && Number(savedState.height) <= MAX_RESTORED_NODE_HEIGHT) {
            userHeight = Math.max(0, Number(savedState.height));
        }
        advanced.open = !!savedState.advanced;
        if (node.size?.[0] !== WIDTH || node.size?.[1] !== userHeight) node.setSize([WIDTH, userHeight]);
        syncLayout(userHeight, true);
    });
}

app.registerExtension({ name: "goohai.minimax_h3_integration", async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "MiniMaxH3IntegrationAdapterGH") {
        const previousAdapter = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            previousAdapter?.apply(this, arguments);
            this.size = [320, Math.max(120, this.size?.[1] || 120)];
        };
        return;
    }
    if (nodeData.name !== NODE) return;
    const previous = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() { previous?.apply(this, arguments); if (!this._ghH3PanelReady) { this._ghH3PanelReady = true; createPanel(this); } };
} });
