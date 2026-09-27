"""Qwen Image 2.1 Studio — 一体化桌面启动器 + 操作界面。

双击 启动.bat 后：
  1. 自动拉起 ComfyUI（用实例自带 python，日志写入 comfyui.log）
  2. 打开本地 Gradio 界面（用 Edge 的 --app 模式呈现为独立窗口）
  3. 上传图片即走图生图，不传图自动走文生图
"""
from __future__ import annotations

import io
import json
import os
import random
import subprocess
import threading
import time
import uuid
import warnings
import webbrowser

# 必须在 import gradio 之前：本地回环不走系统代理。
# 部分代理软件（如 127.0.0.1:7892 这类本地代理）会把 127.0.0.1 也一并劫持，
# 导致 Gradio 启动自检请求被代理返回 502，程序刚启动就崩溃。
for _key in ("NO_PROXY", "no_proxy"):
    _hosts = [item.strip() for item in os.environ.get(_key, "").split(",") if item.strip()]
    for _host in ("127.0.0.1", "localhost"):
        if _host not in _hosts:
            _hosts.append(_host)
    os.environ[_key] = ",".join(_hosts)

import gradio as gr
from PIL import Image

from comfy_client import (
    ASPECT_RATIO_LABELS,
    LABEL_TO_RATIO,
    ComfyClient,
    ComfyError,
    build_graph,
    compute_resolution,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
UI_PORT = int(os.environ.get("QWEN_UI_PORT", "7860"))

DEFAULT_CONFIG = {
    # 留空 = 使用本程序所在目录，这样整个文件夹拷到任何位置都能直接运行
    "instance_dir": "",
    "port": 8188,
    "unet_name": "qwen_image_2_1_bf16.safetensors",
    "clip_name": "qwen3vl_8b_int8_convrot.safetensors",
    "vae_name": "qwen_image_2.1_vae_bf16.safetensors",
    # 采样器与调度器固定在配置里，界面上不暴露
    "sampler_name": "euler",
    "scheduler": "beta",
    # 常开的 LoRA：按关键词匹配文件名，界面上不出现
    "always_loras": [
        {"keyword": "viggle-turbo", "strength": 1.0},
    ],
    # 按模式自动启用的 LoRA，两者互斥：文生图用旧的，图生图用新的
    "mode_loras": {
        "t2i": {"keyword": "NSFW Qwen Lora", "strength": 0.5},
        "i2i": {"keyword": "NSFW Image Edit", "strength": 0.8},
    },
    # 界面上唯一保留的开关
    "style_lora_keyword": "CG画风",
    "style_lora_strength": 0.8,
}

CUSTOM_CSS = """/* =====================================================================
   Qwen Image 2.1 Studio — 紧凑布局 + 双主题（暗黑 / 明亮）
   —— 设计目标：精确满高、无留白；控件卡片化；暗/亮两套均清晰可读。
   ===================================================================== */

/* ---------- 基础 ---------- */
:root { --q-gap: 10px; --q-pad: 8px; --q-radius: 10px; }

html, body { height: 100% !important; overflow: hidden !important; }
gradio-app { height: 100% !important; display: block !important; }

.gradio-container {
    --layout-gap: var(--q-gap) !important;
    height: 100vh !important; max-height: 100vh !important;
    display: flex !important; flex-direction: column !important; overflow: hidden !important;
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", system-ui, sans-serif !important;
}
body .gradio-container main { max-width: 100% !important; padding: var(--q-pad) !important; }
.gradio-container footer { display: none !important; }
.gradio-container .block { padding: 0 !important; border: none !important; background: transparent !important; }
.gradio-container .form { gap: var(--q-gap) !important; }
.gradio-container .wrap { gap: var(--q-gap) !important; }

/* ---------- 满高布局：逐层撑满 ---------- */
.gradio-container main { flex: 1 1 auto !important; display: flex !important;
                         flex-direction: column !important; min-height: 0 !important; overflow: hidden !important; }
.gradio-container main > .wrap {
    flex: 1 1 auto !important; min-height: 0 !important; height: auto !important;
    display: flex !important; flex-direction: column !important; overflow: hidden !important;
}
.gradio-container main > .wrap > #component-0 {
    flex: 1 1 auto !important; min-height: 0 !important; height: auto !important;
    display: flex !important; flex-direction: column !important; overflow: hidden !important;
}
#q-main-row { flex: 1 1 auto !important; min-height: 0 !important; overflow: hidden !important; }
#q-left-col { display: flex !important; flex-direction: column !important; overflow: hidden !important; }
#q-right-col { display: flex !important; flex-direction: column !important;
               min-height: 0 !important; overflow: hidden !important; }

#q-output-image { height: calc(100vh - 176px) !important; min-height: 160px !important;
                  display: flex !important; flex-direction: column !important; overflow: hidden !important; }
#q-output-image > .image-container,
#q-output-image .image-container,
#q-output-image [data-testid="image"],
#q-output-image > div {
    flex: 1 1 auto !important; min-height: 0 !important; max-height: 100% !important; overflow: hidden !important;
}
#q-output-image img { object-fit: contain !important; width: 100% !important; height: 100% !important; }
#q-output-info { flex: 0 0 auto !important; }

/* ---------- 参数行：步数 / CFG / 降噪 / 种子 四张卡片等宽一行 ---------- */
#q-param-row { flex-wrap: nowrap !important; gap: var(--q-gap) !important; }
#q-param-row > .block,
#q-param-row .block { flex: 1 1 0 !important; min-width: 0 !important; }
#q-param-row input[type=number] { text-align: right !important; }
/* 去掉数字输入框的上下箭头 */
#q-param-row input[type=number]::-webkit-outer-spin-button,
#q-param-row input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none !important; margin: 0 !important; }
#q-param-row input[type=number] { -moz-appearance: textfield !important; }

/* 种子卡片：仿「左标签 / 右数值」布局（同滑块卡片），无拖动条，下方留白 */
#q-param-row label.container:has(input[type="number"]) {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    min-height: 100% !important;
    gap: 8px !important;
    padding: 8px 10px !important;
    box-sizing: border-box !important;
}
#q-param-row label.container > span[data-testid="block-info"] {
    flex: 0 0 auto !important;
    color: var(--q-fg) !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
    line-height: 1.2 !important;
    padding: 0 !important;
}
#q-param-row label.container input[type="number"] {
    flex: 0 0 auto !important;
    width: 76px !important;
    min-width: 0 !important;
    max-width: 90px !important;
    text-align: right !important;
    background: var(--q-input-bg) !important;
    color: var(--q-fg) !important;
    border: 1px solid var(--q-border) !important;
    border-radius: 7px !important;
    padding: 2px 8px !important;
    height: 24px !important;
    line-height: 1.2 !important;
    font-size: .82rem !important;
    font-variant-numeric: tabular-nums !important;
}

/* ---------- 右下角实时进度条 ---------- */
#q-output-info { background: transparent !important; border: none !important; padding: 0 !important; }
.q-prog {
    background: var(--q-panel) !important;
    border: 1px solid var(--q-border) !important;
    border-radius: var(--q-radius) !important;
    padding: 9px 12px !important;
    box-shadow: var(--q-shadow-soft);
}
.q-prog-head { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }
.q-prog-icon { font-size: 1rem; color: var(--q-muted); line-height: 1; }
.q-prog-label { font-size: .82rem; font-weight: 700; color: var(--q-fg); flex: 1 1 auto; }
.q-prog-pct { font-size: .8rem; font-weight: 700; color: var(--q-accent); font-variant-numeric: tabular-nums; }
.q-prog-track {
    height: 7px; border-radius: 999px; background: var(--q-panel-3) !important; overflow: hidden;
}
.q-prog-fill {
    height: 100%; border-radius: 999px; width: 0;
    background: linear-gradient(90deg, var(--q-accent), var(--q-accent-2));
    transition: width .45s cubic-bezier(.22,.61,.36,1);
}
.q-prog-detail { font-size: .72rem; color: var(--q-muted) !important; margin-top: 6px; min-height: 1em; }
/* 状态造型 */
.q-prog-running .q-prog-icon { color: var(--q-accent); animation: q-spin 1.1s linear infinite; }
.q-prog-done .q-prog-icon { color: #4ade80; }
.q-prog-done .q-prog-fill { background: linear-gradient(90deg, #22c55e, #4ade80); }
.q-prog-error .q-prog-icon { color: #f87171; }
.q-prog-error .q-prog-fill { background: linear-gradient(90deg, #ef4444, #f87171); }
@keyframes q-spin { from { transform: rotate(0); } to { transform: rotate(360deg); } }

/* ---------- 标题区 ---------- */
#app-title { text-align: center; margin: 0 !important; line-height: 1.2 !important; }
#app-title h1 { font-size: 1.12rem !important; margin: 0 !important; font-weight: 700 !important;
                letter-spacing: .03em; }
#app-sub { text-align: center; font-size: 0.74rem !important; margin: 0 !important; }
#app-sub p, #app-sub .prose { margin: 0 !important; }

/* ---------- 主题切换按钮 ---------- */
#q-topbar { position: absolute; top: 8px; right: 10px; z-index: 60; }
#q-theme-btn {
    cursor: pointer; border: 1px solid var(--q-border); border-radius: 999px;
    background: var(--q-panel-2); color: var(--q-fg);
    font-size: .78rem; font-weight: 600; line-height: 1; padding: 5px 11px;
    transition: background .15s, transform .08s, border-color .15s, box-shadow .15s;
    box-shadow: var(--q-shadow-soft);
}
#q-theme-btn:hover { background: var(--q-panel-3); border-color: var(--q-accent); }
#q-theme-btn:active { transform: scale(.94); }

/* =====================================================================
   暗黑主题（默认）
   ===================================================================== */
.theme-dark {
    --q-bg:        #141418;
    --q-bg-2:      #191920;
    --q-panel:     #1e1e26;
    --q-panel-2:   #26262f;
    --q-panel-3:   #2f2f3a;
    --q-border:    #33333e;
    --q-border-2:  #3f3f4b;
    --q-fg:        #ececf1;
    --q-fg-2:      #c2c2cc;
    --q-muted:     #8b8b98;
    --q-accent:    #6d8cff;
    --q-accent-2:  #8aa0ff;
    --q-accent-soft: rgba(109,140,255,.14);
    --q-input-bg:  #202028;
    --q-shadow:    0 2px 10px rgba(0,0,0,.35);
    --q-shadow-soft: 0 1px 3px rgba(0,0,0,.3);
}
.theme-dark { background: var(--q-bg) !important; }
.theme-dark body, body.theme-dark { background: var(--q-bg) !important; color: var(--q-fg) !important; }
.theme-dark gradio-app { background: var(--q-bg) !important; }
.theme-dark .gradio-container,
.theme-dark .gradio-container main { background: var(--q-bg) !important; }

/* =====================================================================
   明亮主题
   ===================================================================== */
.theme-light {
    --q-bg:        #f4f5f8;
    --q-bg-2:      #ffffff;
    --q-panel:     #ffffff;
    --q-panel-2:   #f0f1f5;
    --q-panel-3:   #e6e8ee;
    --q-border:    #dcdfe6;
    --q-border-2:  #c9cdd6;
    --q-fg:        #1c2230;
    --q-fg-2:      #3a4252;
    --q-muted:     #6b7280;
    --q-accent:    #3b6ef5;
    --q-accent-2:  #5b83f7;
    --q-accent-soft: rgba(59,110,245,.12);
    --q-input-bg:  #ffffff;
    --q-shadow:    0 2px 12px rgba(20,30,60,.08);
    --q-shadow-soft: 0 1px 3px rgba(20,30,60,.08);
}
.theme-light { background: var(--q-bg) !important; }
.theme-light body, body.theme-light { background: var(--q-bg) !important; color: var(--q-fg) !important; }
.theme-light gradio-app { background: var(--q-bg) !important; }
.theme-light .gradio-container,
.theme-light .gradio-container main { background: var(--q-bg) !important; }

/* =====================================================================
   通用控件样式（两主题共用变量，自动切换）
   —— 彻底覆盖 Gradio 所有表单控件，杜绝"黑字黑底"
   ===================================================================== */
/* ★★ 系统性根治：直接改写 Gradio 的标签/文字 CSS 变量（比逐个选择器覆盖更彻底） */
.gradio-container,
.gradio-container .block,
.gradio-container .container {
    --block-label-text-color: var(--q-fg) !important;
    --block-info-text-color: var(--q-muted) !important;
    --block-title-text-color: var(--q-fg) !important;
    --body-text-color: var(--q-fg) !important;
    --body-text-color-subdued: var(--q-muted) !important;
    --input-text-color: var(--q-fg) !important;
    --input-placeholder-color: var(--q-muted) !important;
    --input-background-fill: var(--q-input-bg) !important;
    --block-background-fill: var(--q-panel) !important;
    --block-border-color: var(--q-border) !important;
    --border-color-primary: var(--q-border) !important;
    --panel-background-fill: var(--q-panel) !important;
    --background-fill-primary: var(--q-panel) !important;
    --background-fill-secondary: var(--q-bg-2) !important;
    --neutral-500: var(--q-muted) !important;
    --neutral-600: var(--q-fg-2) !important;
    --neutral-700: var(--q-fg) !important;
    --neutral-800: var(--q-fg) !important;
    --neutral-900: var(--q-fg) !important;
}
.gradio-container { color: var(--q-fg) !important; }

/* 所有文字默认前景色 */
.gradio-container label,
.gradio-container label span,
.gradio-container label span[class*="svelte"],
.gradio-container .block-title,
.gradio-container .label-wrap span,
.gradio-container span,
.gradio-container p,
.gradio-container .prose,
.gradio-container .prose *,
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container .info,
.gradio-container b, .gradio-container strong { color: var(--q-fg) !important; }

/* Gradio 的 label 内文字常有浅蓝/浅灰底色文字色，逐类强制覆盖（选择器要够具体） */
.gradio-container label > span,
.gradio-container label span.svelte-g2oxp3,
.gradio-container span.svelte-g2oxp3,
.gradio-container .head span,
.gradio-container .head label,
.gradio-container .head label span,
.gradio-container .head .has-info,
.gradio-container [class*="svelte-g2oxp3"] { color: var(--q-fg) !important; opacity: 1 !important; }

/* ★ 关键：Gradio 给标签 span 加了浅蓝/浅灰背景（原本配深色字）。
   这里让所有带背景的 label span 背景透明，文字用主题色，避免"白字浅底"。
   —— 但保留真正语义上的 chip/badge 背景色为 accent-soft。 */
.gradio-container span[class*="svelte-g2oxp3"],
.gradio-container label span[class*="svelte-"],
.gradio-container .head span,
.gradio-container label > span,
.gradio-container label[class*="svelte"],
.gradio-container label.svelte-j0zqjt,
.gradio-container .float,
.gradio-container label.float,
.gradio-container [class*="float"] {
    background: transparent !important;
}
/* 浮动标签（float label）常见的浅蓝底一并清掉 */
.theme-dark .gradio-container label[class*="svelte"],
.theme-light .gradio-container label[class*="svelte"] { background: transparent !important; }
/* 表单容器 form 的浅灰底 — 改成透明或面板色 */
.gradio-container .form.svelte-1vd8eap { background: transparent !important; }

/* 次要说明文字 */
.gradio-container span[data-testid="block-info"],
.gradio-container .info,
.gradio-container #app-sub,
.gradio-container small { color: var(--q-muted) !important; }

/* ★ 下拉框（Dropdown）的标签文字被 Gradio 默认渲染成 muted 灰，与其他标签不一致
   —— 强制统一为正常前景色（.svelte-1hfxrpf 是 Dropdown 的容器类） */
.gradio-container .dropdown label,
.gradio-container .dropdown label span,
.gradio-container .dropdown span[class*="svelte"],
.gradio-container .block.dropdown span,
.gradio-container label.dropdown-label,
.gradio-container .dropdown-container label,
.gradio-container .dropdown-container label span,
.gradio-container [class*="dropdown"] label,
.gradio-container [class*="dropdown"] label span,
.gradio-container .container.svelte-1hfxrpf span,
.gradio-container .container.svelte-1hfxrpf span.svelte-g2oxp3,
.gradio-container div.svelte-1hfxrpf span { color: var(--q-fg) !important; opacity: 1 !important; }
/* 通用兜底：Dropdown 组件整体把标签色变量改掉 */
.gradio-container .container.svelte-1hfxrpf { --block-label-text-color: var(--q-fg) !important; }

/* 输入框 / 文本域 / 数字框 */
.gradio-container textarea,
.gradio-container input,
.gradio-container select {
    background: var(--q-input-bg) !important;
    color: var(--q-fg) !important;
    border: 1px solid var(--q-border) !important;
    border-radius: 8px !important;
    transition: border-color .15s, box-shadow .15s;
    font-size: .82rem !important;
}
.gradio-container textarea:focus,
.gradio-container input:focus,
.gradio-container select:focus {
    border-color: var(--q-accent) !important;
    box-shadow: 0 0 0 2px var(--q-accent-soft) !important;
    outline: none !important;
}
.gradio-container textarea::placeholder,
.gradio-container input::placeholder { color: var(--q-muted) !important; }

/* 包裹层（Gradio 的 .wrap / .secondary-wrap / .input-container） */
.gradio-container .wrap,
.gradio-container .wrap-inner,
.gradio-container .secondary-wrap,
.gradio-container .input-container,
.gradio-container .container {
    background: transparent !important;
    border-color: var(--q-border) !important;
}

/* 下拉框（Dropdown）*/
.gradio-container .wrap-inner,
.gradio-container .secondary-wrap { background: var(--q-input-bg) !important; border-radius: 8px !important; }
.gradio-container .wrap-inner input,
.gradio-container .secondary-wrap input { border: none !important; background: transparent !important; }
.gradio-container ul.options,
.gradio-container .options { background: var(--q-panel) !important; border: 1px solid var(--q-border) !important;
                             border-radius: 8px !important; box-shadow: var(--q-shadow) !important; }
.gradio-container ul.options li,
.gradio-container .options li { color: var(--q-fg) !important; }
.gradio-container ul.options li:hover,
.gradio-container ul.options .selected { background: var(--q-accent-soft) !important; color: var(--q-fg) !important; }
/* 多选标签 chip */
.gradio-container .token, .gradio-container .token * {
    background: var(--q-accent-soft) !important; color: var(--q-fg) !important; border: none !important;
}

/* 滑块 */
.gradio-container input[type="range"] { background: transparent !important; border: none !important;
                                        padding: 0 !important; }
.gradio-container input[type="range"]::-webkit-slider-runnable-track {
    background: var(--q-panel-3) !important; height: 4px; border-radius: 2px; }
.gradio-container input[type="range"]::-webkit-slider-thumb {
    background: var(--q-accent) !important; border: none !important;
    box-shadow: 0 0 0 3px var(--q-accent-soft) !important; }

/* 组件卡片容器（参数区每一个 block 做成卡片） */
.gradio-container .block.padded:not(:has(.block)),
.gradio-container .form .block:not(:has(.block)):not(.form) {
    background: var(--q-panel) !important;
    border: 1px solid var(--q-border) !important;
    border-radius: var(--q-radius) !important;
    padding: 8px 10px !important;
    box-shadow: var(--q-shadow-soft);
}
/* Row/Column 容器: 不画边框/背景/阴影, 避免与子卡片边缘线条重叠 */
.gradio-container .block.form,
.gradio-container .block:has(.block) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}
/* 图片/信息框也卡片化 */
#q-output-image .image-container { border: 1px solid var(--q-border) !important;
                                   border-radius: var(--q-radius) !important;
                                   background: var(--q-panel) !important; }

/* 按钮 */
.gradio-container button {
    border-radius: 8px !important;
    font-size: .82rem !important;
    transition: background .15s, border-color .15s, transform .08s, box-shadow .15s;
}
.gradio-container button:active { transform: scale(.985); }
.gradio-container button.secondary {
    background: var(--q-panel-2) !important; color: var(--q-fg) !important;
    border: 1px solid var(--q-border) !important;
}
.gradio-container button.secondary:hover { background: var(--q-panel-3) !important; border-color: var(--q-border-2) !important; }
/* 主按钮：醒目渐变 */
.gradio-container button.primary {
    background: linear-gradient(135deg, var(--q-accent), var(--q-accent-2)) !important;
    border: none !important; color: #fff !important; font-weight: 700 !important;
    letter-spacing: .02em; box-shadow: 0 3px 12px var(--q-accent-soft) !important;
}
.gradio-container button.primary:hover { filter: brightness(1.08); box-shadow: 0 4px 16px var(--q-accent-soft) !important; }

/* 选项卡 / 折叠 */
.gradio-container .tabs, .gradio-container .tab-nav { background: transparent !important; }
.gradio-container .tab-nav button { color: var(--q-muted) !important; border: none !important; }
.gradio-container .tab-nav button.selected { color: var(--q-fg) !important; border-bottom: 2px solid var(--q-accent) !important; }

/* 单选框 / 复选框 */
.gradio-container input[type="radio"], .gradio-container input[type="checkbox"] { accent-color: var(--q-accent); }

/* 滚动条美化 */
.gradio-container ::-webkit-scrollbar { width: 9px; height: 9px; }
.gradio-container ::-webkit-scrollbar-track { background: transparent; }
.gradio-container ::-webkit-scrollbar-thumb { background: var(--q-panel-3); border-radius: 6px; }
.gradio-container ::-webkit-scrollbar-thumb:hover { background: var(--q-border-2); }

/* ---------- 上传放置区（参考图片）：做成与其它控件一致的卡片 ---------- */
.gradio-container .block.svelte-1svsvh2 button.svelte-edrmkl,
.gradio-container button[aria-label*="upload"],
.gradio-container button[aria-dropeffect] {
    background: var(--q-input-bg) !important;
    border: 1px dashed var(--q-border-2) !important;
    border-radius: var(--q-radius) !important;
    color: var(--q-fg-2) !important;
    transition: border-color .15s, background .15s;
}
.gradio-container button[aria-dropeffect]:hover {
    border-color: var(--q-accent) !important;
    background: var(--q-accent-soft) !important;
}
.gradio-container .wrap.svelte-12ioyct { color: var(--q-muted) !important; font-size: .78rem !important; }
.gradio-container .wrap.svelte-12ioyct .icon-wrap,
.gradio-container .wrap.svelte-12ioyct svg { color: var(--q-accent) !important; opacity: .55; }
.gradio-container .wrap.svelte-12ioyct .or { color: var(--q-muted) !important; opacity: .7; }

/* 悬浮标签（float label）样式：小字号、贴左上、与卡片协调 */
.gradio-container label.float,
.gradio-container label.svelte-j0zqjt.float {
    background: transparent !important;
    color: var(--q-fg) !important;
    font-size: .78rem !important;
    font-weight: 600 !important;
}
.gradio-container label.svelte-j0zqjt.float svg { color: var(--q-accent) !important; opacity: .8; }

/* 结果图圆角 */
.gradio-container [data-testid="image"] { border-radius: var(--q-radius) !important; overflow: hidden !important; }

/* ---------- Gallery（参考图片上传区）：背景跟随主题，避免浅底+浅字 ---------- */
.gradio-container .gallery,
.gradio-container .gallery * ,
.gradio-container [data-testid="gallery"],
.gradio-container .image-container,
.gradio-container .upload-container,
.gradio-container .file-preview {
    background: var(--q-panel) !important;
    color: var(--q-fg) !important;
    border-color: var(--q-border) !important;
}
.gradio-container .gallery .wrap,
.gradio-container .gallery .grid-wrap,
.gradio-container .gallery .thumbnails { background: var(--q-panel) !important; }
.gradio-container .gallery .empty,
.gradio-container .gallery .icon-button,
.gradio-container .gallery svg { color: var(--q-fg-2) !important; fill: var(--q-fg-2) !important; }
/* Gradio 上传区常见浅蓝/浅灰底 — 强制改成主题面板色 */
.gradio-container .upload-container,
.gradio-container [data-testid="block-label"] + div,
.gradio-container .gallery.gallery-container {
    background: var(--q-panel) !important;
}
/* 通用兜底：任何浅色系背景的容器在有主题时都用面板色 */
.theme-dark .gradio-container .gallery,
.theme-dark .gradio-container [class*="upload"],
.theme-dark .gradio-container [class*="file"] { background: var(--q-panel) !important; }
.theme-light .gradio-container .gallery,
.theme-light .gradio-container [class*="upload"],
.theme-light .gradio-container [class*="file"] { background: var(--q-panel) !important; }
"""

# 主题切换注入脚本（放进 gr.Blocks 的 js 参数）
THEME_JS = """
() => {
    const KEY = 'qwen-studio-theme';
    const apply = (mode) => {
        const root = document.documentElement;
        root.classList.remove('theme-dark', 'theme-light');
        root.classList.add(mode === 'light' ? 'theme-light' : 'theme-dark');
        const btn = document.getElementById('q-theme-btn');
        if (btn) btn.textContent = mode === 'light' ? '🌙  暗黑' : '☀️  明亮';
        try { localStorage.setItem(KEY, mode); } catch (e) {}
    };
    const init = () => {
        if (document.getElementById('q-theme-btn')) return;
        let mode = 'dark';
        try { mode = localStorage.getItem(KEY) || 'dark'; } catch (e) {}
        apply(mode);
        const bar = document.createElement('div');
        bar.id = 'q-topbar';
        const btn = document.createElement('button');
        btn.id = 'q-theme-btn';
        btn.type = 'button';
        btn.title = '切换主题';
        btn.onclick = () => {
            let cur = 'dark';
            try { cur = localStorage.getItem(KEY) || 'dark'; } catch (e) {}
            apply(cur === 'light' ? 'dark' : 'light');
        };
        bar.appendChild(btn);
        document.body.appendChild(bar);
        apply(mode);
    };
    if (document.readyState === 'complete') init();
    else window.addEventListener('load', init);
    // Gradio 是异步挂载，兜底再跑几次
    setTimeout(init, 400);
    setTimeout(init, 1200);
}
"""


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

def load_config() -> dict:
    config = dict(DEFAULT_CONFIG)
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
                config.update(json.load(handle))
        except Exception as error:
            print(f"[warn] 读取 config.json 失败，使用默认配置：{error}")
    else:
        save_config(config)
    return config


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


CONFIG = load_config()
# instance_dir 留空表示用程序自身所在目录。整个文件夹拷到别的盘或目录后，
# 只要 ComfyUI 和 python 跟 app.py 在一起，就无需改任何配置。
CONFIG["instance_dir"] = str(CONFIG.get("instance_dir") or "").strip() or APP_DIR
COMFY_BASE = f"http://127.0.0.1:{CONFIG['port']}"
COMFY_DIR = os.path.join(CONFIG["instance_dir"], "ComfyUI")
CLIENT = ComfyClient(COMFY_BASE)


# --------------------------------------------------------------------------
# 启动器
# --------------------------------------------------------------------------

class Launcher:
    def __init__(self, config: dict):
        self.config = config
        self.process: subprocess.Popen | None = None
        self.started = False
        self.log_path = os.path.join(APP_DIR, "comfyui.log")

    @property
    def python(self) -> str:
        return os.path.join(self.config["instance_dir"], "python", "python.exe")

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        if not os.path.isfile(self.python):
            raise ComfyError(f"找不到实例 python：{self.python}\n请检查 config.json 里的 instance_dir。")

        log = open(self.log_path, "ab", buffering=0)
        log.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 ComfyUI =====\n".encode("utf-8"))
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            [self.python, "main.py", "--port", str(self.config["port"]), "--listen", "127.0.0.1"],
            cwd=COMFY_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )

    def tail_log(self, lines: int = 25) -> str:
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as handle:
                return "".join(handle.readlines()[-lines:])
        except Exception:
            return "(暂无日志)"


LAUNCHER = Launcher(CONFIG)
CURRENT = {"prompt_id": None}


# --------------------------------------------------------------------------
# LoRA 扫描
# --------------------------------------------------------------------------

def scan_loras() -> list[str]:
    root = os.path.join(COMFY_DIR, "models", "loras")
    found: list[str] = []
    if os.path.isdir(root):
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith((".safetensors", ".ckpt", ".pt", ".sft")):
                    relative = os.path.relpath(os.path.join(dirpath, filename), root)
                    found.append(relative.replace("\\", "/"))
    if not found:
        found = CLIENT.list_loras()
    return sorted(found, key=str.lower)


LORAS = scan_loras()


def find_lora(keyword: str) -> str | None:
    """按关键词在已扫描到的 LoRA 文件名里匹配。"""
    if not keyword:
        return None
    for name in LORAS:
        if keyword.lower() in name.lower():
            return name
    return None


def resolve_lora(entry: dict | None) -> dict | None:
    """把配置里的 {keyword, strength} 解析成真实的 LoRA 文件名。"""
    if not entry:
        return None
    matched = find_lora(str(entry.get("keyword", "")))
    if not matched:
        return None
    return {"name": matched, "strength": float(entry.get("strength", 1.0))}


ALWAYS_LORAS: list[dict] = []
for _entry in CONFIG.get("always_loras") or []:
    _resolved = resolve_lora(_entry)
    if _resolved:
        ALWAYS_LORAS.append(_resolved)

MODE_LORAS: dict[str, dict] = {}
for _mode in ("t2i", "i2i"):
    _entry = (CONFIG.get("mode_loras") or {}).get(_mode)
    _resolved = resolve_lora(_entry)
    if _resolved:
        MODE_LORAS[_mode] = _resolved
    elif _entry:
        print(f"[lora] 未找到 {_mode} 对应的 LoRA（关键词 {_entry.get('keyword')!r}），该模式将不加载。")

STYLE_LORA = find_lora(str(CONFIG.get("style_lora_keyword", "")))
STYLE_STRENGTH = float(CONFIG.get("style_lora_strength", 0.8))

for _lora in ALWAYS_LORAS:
    print(f"[lora] 常开：{_lora['name']}（{_lora['strength']}）")
for _mode, _lora in MODE_LORAS.items():
    print(f"[lora] {_mode} 自动启用：{_lora['name']}（{_lora['strength']}）")
if STYLE_LORA:
    print(f"[lora] 界面开关：{STYLE_LORA}（{STYLE_STRENGTH}）")
else:
    print(f"[lora] 未找到画风 LoRA（关键词 {CONFIG.get('style_lora_keyword')!r}），该开关将被忽略。")


# --------------------------------------------------------------------------
# 界面回调
# --------------------------------------------------------------------------

def wait_until_ready(timeout: float = 300.0):
    """生成器：启动 ComfyUI 并轮询到就绪，逐步产出状态文本。"""
    if CLIENT.is_up():
        yield "✅ ComfyUI 已就绪"
        return
    try:
        LAUNCHER.start()
    except ComfyError as error:
        yield f"❌ {error}"
        return
    started = time.time()
    while time.time() - started < timeout:
        if CLIENT.is_up():
            yield f"✅ ComfyUI 已就绪（用时 {time.time() - started:.0f} 秒）"
            return
        yield f"⏳ 正在启动 ComfyUI… {time.time() - started:.0f} 秒（首次启动需加载模型，请稍候）"
        time.sleep(1.5)
    yield "❌ ComfyUI 启动超时，请查看 comfyui.log 末尾的报错。"


def on_load():
    yield from wait_until_ready()


def update_summary(files, aspect_label, megapixels):
    """把「当前模式」和「输出尺寸」合并成一行，省一行高度。"""
    count = len(files) if files else 0
    mode = f"**图生图**（{count} 张参考图）" if count else "**文生图**（未上传参考图）"
    ratio = LABEL_TO_RATIO.get(aspect_label, aspect_label)
    width, height = compute_resolution(ratio, megapixels, 32)
    return f"当前模式：{mode} ｜ 输出尺寸：**{width} × {height}**"


# --------------------------------------------------------------------------
# 进度条（右下角实时显示）
# --------------------------------------------------------------------------

_PROGRESS = {"pct": 0, "label": "待命", "detail": "", "state": "idle"}


def render_progress(state: str | None = None, pct: float | None = None,
                    label: str | None = None, detail: str | None = None) -> str:
    """渲染右下角进度条 HTML。state: idle/running/done/error。"""
    if state is not None:
        _PROGRESS["state"] = state
    if pct is not None:
        _PROGRESS["pct"] = max(0.0, min(100.0, float(pct)))
    if label is not None:
        _PROGRESS["label"] = label
    if detail is not None:
        _PROGRESS["detail"] = detail

    st = _PROGRESS["state"]
    pct_v = _PROGRESS["pct"]
    icon = {"idle": "●", "running": "◐", "done": "✓", "error": "✕"}.get(st, "●")
    return (
        f'<div class="q-prog q-prog-{st}">'
        f'  <div class="q-prog-head">'
        f'    <span class="q-prog-icon">{icon}</span>'
        f'    <span class="q-prog-label">{_PROGRESS["label"]}</span>'
        f'    <span class="q-prog-pct">{pct_v:.0f}%</span>'
        f'  </div>'
        f'  <div class="q-prog-track"><div class="q-prog-fill" style="width:{pct_v:.1f}%"></div></div>'
        f'  <div class="q-prog-detail">{_PROGRESS["detail"]}</div>'
        f'</div>'
    )


def _watch_comfy_progress(prompt_id: str, client_id: str, total_steps: int,
                          stop_event: threading.Event) -> None:
    """后台线程：连 ComfyUI websocket 抓真实步进进度，写入 _PROGRESS。"""
    try:
        import websocket  # websocket-client
    except Exception:
        return
    ws_url = CLIENT.base_url.replace("http://", "ws://").replace("https://", "wss://")
    try:
        ws = websocket.create_connection(f"{ws_url}/ws?clientId={client_id}", timeout=5)
    except Exception:
        return
    try:
        while not stop_event.is_set():
            try:
                ws.settimeout(1.0)
                raw = ws.recv()
            except Exception:
                continue
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type")
            data = msg.get("data", {}) or {}
            if data.get("prompt_id") and data["prompt_id"] != prompt_id:
                continue
            if mtype == "progress":
                val = float(data.get("value", 0))
                mx = float(data.get("max", total_steps) or total_steps)
                if mx > 0:
                    _PROGRESS["pct"] = min(99.0, val / mx * 100.0)
                    _PROGRESS["detail"] = f"采样中 {int(val)} / {int(mx)} 步"
            elif mtype == "executing":
                node = data.get("node")
                if node:
                    _PROGRESS["detail"] = "正在执行工作流…"
            elif mtype == "progress_state":
                nodes = data.get("nodes", [])
                if nodes:
                    _PROGRESS["detail"] = "生成中…"
    finally:
        try:
            ws.close()
        except Exception:
            pass


def on_generate(files, prompt_text, negative_text, aspect_label, megapixels,
                steps, cfg, denoise, seed, randomize, style_on):
    """生成主流程：必要时启动 ComfyUI → 上传图片 → 提交 → 轮询 → 取图。"""
    if not CLIENT.is_up():
        yield None, render_progress("running", 2, "正在启动 ComfyUI", "首次生成需要加载模型，请稍候…")
        try:
            LAUNCHER.start()
        except ComfyError as error:
            yield None, render_progress("error", 0, "启动失败", str(error))
            return
        started = time.time()
        while not CLIENT.is_up() and time.time() - started < 300:
            el = time.time() - started
            yield None, render_progress("running", min(20, el / 300 * 20),
                                        "正在启动 ComfyUI",
                                        f"已等待 {el:.0f} 秒…")
            time.sleep(1.5)
        if not CLIENT.is_up():
            yield None, render_progress("error", 0, "ComfyUI 启动超时",
                                        "请查看 comfyui.log 末尾的报错。")
            return

    stop_event = threading.Event()
    watcher = None
    try:
        # 参考图上传
        image_names: list[str] = []
        if files:
            total = len(files)
            for idx, item in enumerate(files, 1):
                # Gradio 5.x 的 Gallery value 可能是 (filepath, caption) 元组，
                # 也可能是纯 filepath 字符串，这里统一取出路径。
                if isinstance(item, (tuple, list)):
                    path = item[0]
                else:
                    path = item
                if not path:
                    continue
                yield None, render_progress("running", 5 + idx / max(total, 1) * 10,
                                            "正在上传参考图",
                                            f"{idx}/{total}：{os.path.basename(path)}")
                image_names.append(CLIENT.upload_image(path))

        mode = "i2i" if image_names else "t2i"
        ratio = LABEL_TO_RATIO.get(aspect_label, aspect_label)
        width, height = compute_resolution(ratio, megapixels, 32)

        actual_seed = random.randint(0, 2 ** 63 - 1) if randomize else int(seed)
        # 加速常开；NSFW 按模式二选一（t2i 用旧的、i2i 用新的）；CG 由界面开关决定
        selected_loras = [dict(item) for item in ALWAYS_LORAS]
        mode_lora = MODE_LORAS.get(mode)
        if mode_lora:
            selected_loras.append(dict(mode_lora))
        if style_on and STYLE_LORA:
            selected_loras.append({"name": STYLE_LORA, "strength": STYLE_STRENGTH})

        graph = build_graph(
            mode=mode,
            unet_name=CONFIG["unet_name"],
            clip_name=CONFIG["clip_name"],
            vae_name=CONFIG["vae_name"],
            loras=selected_loras,
            prompt=prompt_text or "",
            negative_prompt=negative_text or "",
            ref_resolution=1024,
            width=width,
            height=height,
            steps=int(steps),
            cfg=float(cfg),
            sampler_name=CONFIG["sampler_name"],
            scheduler=CONFIG["scheduler"],
            denoise=float(denoise),
            seed=actual_seed,
            images=image_names,
            filename_prefix=f"QwenStudio/{'i2i' if mode == 'i2i' else 't2i'}_",
        )

        prompt_id = CLIENT.queue_prompt(graph)
        CURRENT["prompt_id"] = prompt_id
        started = time.time()
        yield None, render_progress("running", 18, "已提交，开始生成",
                                    f"{mode} / {width}×{height} / {int(steps)} 步")

        # 后台 websocket 抓真实步进（仅用于更细腻的百分比，失败不影响主流程）
        client_id = str(uuid.uuid4())
        watcher = threading.Thread(target=_watch_comfy_progress,
                                   args=(prompt_id, client_id, int(steps), stop_event),
                                   daemon=True)
        watcher.start()

        # 经验速度：本机 8 步约 53 秒，线性外推；下限 60 秒避免小步数过快跑满。
        est_total = max(60.0, int(steps) * 7.0)
        while True:
            entry = CLIENT.history_entry(prompt_id)
            if entry:
                outputs = CLIENT.outputs_or_raise(entry)
                if outputs:
                    break
            elapsed = time.time() - started
            if elapsed > 1800:
                raise ComfyError("生成超时（超过 30 分钟）。")
            # 时间估算进度：从 18% 平滑爬到 96%（真实步进若可用则取较大值）
            est = 18.0 + min(78.0, elapsed / est_total * 78.0)
            ws_pct = float(_PROGRESS.get("pct") or 0.0)
            pct = max(est, min(96.0, ws_pct))
            detail = f"{mode} · {width}×{height} · {int(steps)} 步 · 已用 {elapsed:.0f} 秒"
            yield None, render_progress("running", pct, "生成中", detail)
            time.sleep(0.5)

        stop_event.set()

        images = outputs.get("9", {}).get("images") or []
        if not images:
            raise ComfyError(f"执行完成但没有取到图片，outputs={outputs}")

        first = images[0]
        raw = CLIENT.fetch_image(first["filename"], first.get("subfolder", ""), first.get("type", "output"))
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        elapsed = time.time() - started
        saved = os.path.join(COMFY_DIR, "output", first.get("subfolder", ""), first["filename"])
        detail = f"用时 {elapsed:.1f} 秒 ｜ {mode} {width}×{height} ｜ 种子 {actual_seed}"
        yield image, render_progress("done", 100, "生成完成", detail)
    except ComfyError as error:
        yield None, render_progress("error", _PROGRESS["pct"], "生成失败", str(error))
    except Exception as error:  # noqa: BLE001
        yield None, render_progress("error", _PROGRESS["pct"], "出错了",
                                    f"{type(error).__name__}: {error}")
    finally:
        stop_event.set()
        CURRENT["prompt_id"] = None


def on_interrupt():
    CLIENT.interrupt()
    return render_progress("error", _PROGRESS["pct"], "已中断", "已请求停止当前任务。")


# --------------------------------------------------------------------------
# 界面
# --------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="Qwen Image 2.1 Studio",
                   css=CUSTOM_CSS, js=THEME_JS, fill_width=True) as demo:
        gr.Markdown("# Qwen Image 2.1 Studio", elem_id="app-title")
        status_md = gr.Markdown("正在检查 ComfyUI 状态…", elem_id="app-sub")

        with gr.Row(elem_id="q-main-row"):
            with gr.Column(scale=5, elem_id="q-left-col"):
                files = gr.Gallery(
                    label="参考图片（可多张，最多 16 张；留空 = 文生图）",
                    type="filepath",
                    file_types=["image"],
                    columns=4,
                    rows=1,
                    height=135,
                    object_fit="cover",
                    allow_preview=True,
                    show_download_button=False,
                    interactive=True,
                )
                summary_md = gr.Markdown("")
                prompt_box = gr.Textbox(label="提示词", lines=2,
                                        placeholder="描述你想要的画面…")
                negative_box = gr.Textbox(label="负面提示词", lines=2,
                                          value="bad quality, blurry, low resolution")

                with gr.Row():
                    aspect_box = gr.Dropdown(
                        choices=list(ASPECT_RATIO_LABELS.values()),
                        value=ASPECT_RATIO_LABELS["9:16 (Portrait Widescreen)"],
                        label="宽高比", scale=4,
                    )
                    megapixel_box = gr.Slider(0.1, 16.0, value=1.0, step=0.1, label="百万像素",
                                              scale=4, show_reset_button=False)
                    randomize_box = gr.Checkbox(value=True, label="每次随机种子", scale=4)

                with gr.Row(equal_height=True, elem_id="q-param-row"):
                    steps_box = gr.Slider(1, 50, value=8, step=1, label="步数",
                                          scale=1, min_width=0, show_reset_button=False)
                    cfg_box = gr.Slider(0.0, 20.0, value=1.0, step=0.1, label="CFG",
                                        scale=1, min_width=0, show_reset_button=False)
                    denoise_box = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="降噪",
                                            scale=1, min_width=0, show_reset_button=False)
                    seed_box = gr.Number(value=0, label="种子", precision=0,
                                         scale=1, min_width=0)

                with gr.Row():
                    style_box = gr.Checkbox(value=False, label=f"CG 画风（{STYLE_STRENGTH}）", scale=3)

                with gr.Row():
                    run_button = gr.Button("生成", variant="primary", size="lg")
                    stop_button = gr.Button("中断", size="lg")

            with gr.Column(scale=4, elem_id="q-right-col"):
                output_image = gr.Image(label="结果", type="pil",
                                        elem_id="q-output-image")
                output_info = gr.HTML(value=render_progress("idle", 0, "待命", "点击「生成」开始"),
                                      elem_id="q-output-info")

        # 事件绑定
        summary_inputs = [files, aspect_box, megapixel_box]
        demo.load(fn=on_load, outputs=[status_md])
        demo.load(fn=update_summary, inputs=summary_inputs, outputs=[summary_md])
        files.change(fn=update_summary, inputs=summary_inputs, outputs=[summary_md])
        aspect_box.change(fn=update_summary, inputs=summary_inputs, outputs=[summary_md])
        megapixel_box.change(fn=update_summary, inputs=summary_inputs, outputs=[summary_md])

        run_button.click(
            fn=on_generate,
            inputs=[files, prompt_box, negative_box, aspect_box, megapixel_box,
                    steps_box, cfg_box, denoise_box, seed_box, randomize_box, style_box],
            outputs=[output_image, output_info],
        )
        stop_button.click(fn=on_interrupt, outputs=[output_info])

    return demo


# --------------------------------------------------------------------------
# 独立窗口
# --------------------------------------------------------------------------

def find_edge() -> str | None:
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def open_window(url: str) -> None:
    edge = find_edge()
    if not edge:
        webbrowser.open(url)
        return
    # 窗口尺寸可在 config.json 用 window_width / window_height 覆盖。
    # 默认高度取 1000：实测左列（含「生成」按钮）需要约 893px 视口高，
    # 加 Edge --app 标题栏开销（约 93px）后 ≈ 986，取 1000 留出余量。
    # 关键：Edge --app 模式会记住上次窗口尺寸，导致 --window-size 不生效，
    # 因此给一个独立 user-data-dir，并配合 --new-window 强制使用本次指定尺寸。
    win_w = int(CONFIG.get("window_width") or 1200)
    win_h = int(CONFIG.get("window_height") or 1000)
    app_profile = os.path.join(APP_DIR, "_edge_profile")
    try:
        os.makedirs(app_profile, exist_ok=True)
    except Exception:
        app_profile = None

    # ★ Edge --app 会把窗口位置/尺寸记进 Preferences 的 app_window_placement，
    #   下次启动时优先用它而忽略 --window-size（这是窗口"拉不长"的真凶）。
    #   这里在启动前清掉该记录，保证每次都用本次指定的宽高。
    if app_profile:
        pref_path = os.path.join(app_profile, "Default", "Preferences")
        try:
            if os.path.isfile(pref_path):
                with open(pref_path, encoding="utf-8") as fp:
                    pref = json.load(fp)
                browser = pref.get("browser")
                if isinstance(browser, dict) and "app_window_placement" in browser:
                    browser.pop("app_window_placement", None)
                    tmp = pref_path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as fp:
                        json.dump(pref, fp)
                    os.replace(tmp, pref_path)
        except Exception:
            pass

    args = [edge, f"--app={url}", f"--window-size={win_w},{win_h}",
            "--no-first-run", "--no-default-browser-check"]
    if app_profile:
        args.append(f"--user-data-dir={app_profile}")
    subprocess.Popen(args)


def main() -> None:
    # ---- 卡密授权闸门：已激活 -> 主界面；未激活 -> 激活界面（同端口）----
    activated = False
    try:
        import qlic
        activated = qlic.is_activated(APP_DIR)
    except Exception as _lic_err:  # noqa: BLE001
        print("[授权模块异常]", _lic_err)

    if activated:
        demo = build_ui()
    else:
        try:
            import qactivate
            demo = qactivate.build_activation_ui(APP_DIR)
        except Exception as _act_err:  # noqa: BLE001
            print("[激活界面异常]", _act_err)
            return

    demo.queue()
    try:
        with warnings.catch_warnings():
            # show_api 在 Gradio 6 会改名为 footer_links，这里先静音该弃用提示
            warnings.filterwarnings("ignore", message=r".*show_api.*")
            demo.launch(server_name="127.0.0.1", server_port=UI_PORT,
                        inbrowser=False, prevent_thread_lock=True, show_api=False, quiet=True)
    except Exception as error:  # noqa: BLE001
        print(f"\n[启动失败] 界面无法在 127.0.0.1:{UI_PORT} 上启动。")
        print(f"原因：{error}")
        print("排查：1) 端口是否被占用；2) 是否开着会劫持本地回环的代理软件。")
        return
    open_window(f"http://127.0.0.1:{UI_PORT}")
    print(f"\n界面已启动：http://127.0.0.1:{UI_PORT}\n关闭本窗口即退出（ComfyUI 会继续在后台运行）。\n")
    try:
        demo.block_thread()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
