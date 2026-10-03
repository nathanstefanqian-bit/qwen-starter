"""Qwen Image 2.1 Studio — 一体化桌面启动器 + 操作界面。

双击 启动.bat 后：
  1. 自动拉起 ComfyUI（用实例自带 python，日志写入 comfyui.log）
  2. 打开本地 Gradio 界面（用 Edge 的 --app 模式呈现为独立窗口）
  3. 上传图片即走图生图，不传图自动走文生图
"""
from __future__ import annotations

import html
import io
import itertools
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
    MAX_REF_IMAGES,
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
    # 关键词是文件名的子串（大小写不敏感），改名后只要同步这里即可
    "mode_loras": {
        "t2i": {"keyword": "Qwen Lora", "strength": 0.5},
        "i2i": {"keyword": "Qwen-Image-2.1 Image Edit", "strength": 0.8},
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

/* ---------- 满高布局：逐层撑满 ----------
   实测真实层级是 main > .wrap > .contain > #component-0 > #q-main-row。
   中间每一层都必须有 min-height:0，否则出图后内容会把 .contain 撑到上千像素，
   被 .wrap 的 overflow:hidden 裁掉——状态条和任务队列就再也看不见了。 */
.gradio-container main { flex: 1 1 auto !important; display: flex !important;
                         flex-direction: column !important; min-height: 0 !important; overflow: hidden !important; }
.gradio-container main > .wrap {
    flex: 1 1 auto !important; min-height: 0 !important; height: auto !important;
    display: flex !important; flex-direction: column !important; overflow: hidden !important;
}
.gradio-container main > .wrap > .contain,
.gradio-container main > .wrap > .contain > * {
    flex: 1 1 auto !important; min-height: 0 !important; height: auto !important;
    display: flex !important; flex-direction: column !important; overflow: hidden !important;
}
/* flex-wrap:nowrap 是关键：Gradio 的 Row 默认 wrap，折行后 flex 行高由最高的子项决定，
   出图后结果图会把整行撑到上千像素，子列跟着变高、溢出被父层裁掉——状态条和队列就没了。 */
#q-main-row { flex: 1 1 auto !important; min-height: 0 !important;
              flex-wrap: nowrap !important; align-items: stretch !important;
              overflow: hidden !important; }
/* 左列：正常填满；窗口实在太矮时允许滚动，绝不把输入框压成一条缝。
   nowrap 必须显式声明：Gradio 的 Row/Column 默认 flex-wrap:wrap，
   列方向一旦折行，子项会横向排到「新的一列」里，直接跑到宽度之外被裁掉
   （结果图会把状态条和队列面板挤没，就是这个原因）。 */
#q-left-col { display: flex !important; flex-direction: column !important;
              flex-wrap: nowrap !important;
              min-height: 0 !important; overflow-y: auto !important; overflow-x: hidden !important; }
#q-right-col { display: flex !important; flex-direction: column !important;
               flex-wrap: nowrap !important;
               gap: var(--q-gap) !important;
               min-height: 0 !important; overflow: hidden !important; }

/* 结果图：吃掉右列剩余高度，但下限压低，保证状态条/队列面板永远留在可视区。
   ⚠ 必须带 :not(.fullscreen)：点放大镜时 Gradio 会给这个 Block 加 .fullscreen 类
   （position:fixed + 100vw×100vh + overflow:auto）。若在这里用 height:auto / overflow:hidden
   去覆盖它，容器就会变成「按内容撑开」——图片按原始 1376px 展开，比屏幕还高又滚不动，
   于是放大后只能看到半张图。放大态下这里一条都不该插手。 */
#q-output-image:not(.fullscreen) {
    flex: 1 1 auto !important; min-height: 140px !important; height: auto !important;
    display: flex !important; flex-direction: column !important;
    flex-wrap: nowrap !important; overflow: hidden !important;
}
#q-output-image:not(.fullscreen) > .image-container,
#q-output-image:not(.fullscreen) .image-container,
#q-output-image:not(.fullscreen) [data-testid="image"],
#q-output-image:not(.fullscreen) > div {
    flex: 1 1 auto !important; min-height: 0 !important; max-height: 100% !important; overflow: hidden !important;
}
#q-output-image:not(.fullscreen) img {
    object-fit: contain !important; width: 100% !important; height: 100% !important;
}
/* 放大态：交给 Gradio 自己的 90vw/90vh 约束，只把图片居中，确保整张可见 */
#q-output-image.fullscreen img {
    max-width: 92vw !important; max-height: 92vh !important; object-fit: contain !important;
}
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
    --q-chip-bg:   rgba(26,26,33,.82);
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
    --q-chip-bg:   rgba(255,255,255,.88);
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

/* =====================================================================
   ① 生成按钮：只要「点击」这一下的反馈（注意：#q-run-btn 就是 button 本身）
   —— 不要转圈、不要常驻忙碌态：用户随时可以再点一次把新任务排进队列。
   ===================================================================== */
#q-run-btn, #q-stop-btn { flex: 1 1 0 !important; min-width: 0 !important; }
#q-run-btn {
    display: inline-flex !important; align-items: center !important;
    justify-content: center !important; gap: 9px !important;
    transition: transform .09s ease-out, filter .18s ease-out,
                box-shadow .22s ease-out !important;
}
#q-run-btn:active { transform: translateY(1px) scale(.985) !important; filter: brightness(.9) !important; }
/* 点击瞬间的闪光：JS 加上 .q-press，约 320ms 后自行移除 */
#q-run-btn.q-press {
    transform: translateY(1px) scale(.97) !important;
    filter: brightness(.86) !important;
    box-shadow: 0 0 0 4px var(--q-accent-soft) !important;
}
@keyframes q-press-flash {
    0%   { box-shadow: 0 0 0 4px var(--q-accent-soft); }
    100% { box-shadow: 0 0 0 10px rgba(0,0,0,0); }
}
#q-run-btn.q-press { animation: q-press-flash .32s ease-out 1; }

/* =====================================================================
   ② 任务队列面板（右侧工作区）
   ===================================================================== */
#q-queue { flex: 0 0 auto !important; }
.q-queue {
    background: var(--q-panel) !important;
    border: 1px solid var(--q-border) !important;
    border-radius: var(--q-radius) !important;
    box-shadow: var(--q-shadow-soft);
    padding: 8px 10px !important;
}
.q-queue-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.q-queue-title { flex: 1 1 auto; font-size: .78rem; font-weight: 700; color: var(--q-fg); }
.q-queue-count {
    flex: 0 0 auto; min-width: 20px; text-align: center;
    font-size: .7rem; font-weight: 700; color: var(--q-accent);
    background: var(--q-accent-soft); border-radius: 999px; padding: 1px 7px;
    font-variant-numeric: tabular-nums;
}
.q-queue-none { font-size: .74rem; color: var(--q-muted) !important; padding: 1px 0; }
.q-queue-list { display: flex; flex-direction: column; gap: 5px; max-height: 138px; overflow-y: auto; }
.q-task {
    display: flex; align-items: center; gap: 7px;
    padding: 5px 8px; border-radius: 7px;
    background: var(--q-panel-2); border: 1px solid transparent;
    font-size: .74rem; color: var(--q-fg) !important;
}
.q-task-run { border-color: var(--q-accent) !important; background: var(--q-accent-soft) !important; }
.q-task-done { opacity: .68; }
.q-task-err { border-color: rgba(248,113,113,.55) !important; }
.q-task-cancel-hint { border-color: rgba(148,163,184,.35) !important; }
.q-task-icon { flex: 0 0 auto; font-size: .8rem; line-height: 1; color: var(--q-muted); }
.q-task-run .q-task-icon { color: var(--q-accent); animation: q-spin 1.1s linear infinite; }
.q-task-done .q-task-icon { color: #4ade80; }
.q-task-err .q-task-icon { color: #f87171; }
.q-task-cancel-hint .q-task-icon { color: #94a3b8; }
.q-task-id { flex: 0 0 auto; font-weight: 700; color: var(--q-muted) !important; font-variant-numeric: tabular-nums; }
.q-task-body { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; gap: 1px; }
.q-task-title { color: var(--q-fg) !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.q-task-meta { font-size: .68rem; color: var(--q-muted) !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.q-task-pct { flex: 0 0 auto; font-weight: 700; color: var(--q-accent) !important; font-variant-numeric: tabular-nums; }

/* 每条任务自己的「终止」按钮 */
.q-cancel-btn {
    flex: 0 0 auto;
    display: inline-flex; align-items: center; justify-content: center;
    width: 21px; height: 21px; padding: 0; margin-left: 1px;
    font-size: .72rem; line-height: 1;
    color: var(--q-muted) !important;
    background: transparent;
    border: 1px solid var(--q-border-2);
    border-radius: 6px;
    cursor: pointer;
    transition: background .15s, color .15s, border-color .15s;
}
.q-cancel-btn:hover {
    color: #f87171 !important;
    background: rgba(248,113,113,.16);
    border-color: rgba(248,113,113,.6);
}
.q-cancel-btn:disabled,
.q-cancel-btn.q-cancel-pending {
    opacity: .5; cursor: progress;
    color: var(--q-muted) !important;
    background: transparent;
    border-color: var(--q-border-2);
}

/* =====================================================================
   ②b 结果图下方：「打开输出文件夹」链接标签
   ===================================================================== */
#q-folder-row { flex: 0 0 auto !important; justify-content: flex-end !important;
                align-items: center !important; }
#q-open-folder {
    flex: 0 0 auto !important; width: auto !important; min-width: 0 !important;
    display: inline-flex !important; align-items: center !important; justify-content: center !important;
    padding: 1px 10px !important;
    font-size: .74rem !important; font-weight: 600 !important; line-height: 1.7 !important;
    color: var(--q-accent) !important;
    background: var(--q-accent-soft) !important;
    border: 1px solid transparent !important;
    border-radius: 999px !important;
    box-shadow: none !important;
    cursor: pointer !important;
    transition: background .15s, border-color .15s !important;
}
#q-open-folder:hover { border-color: var(--q-accent) !important; background: transparent !important; }
#q-open-folder:active { transform: translateY(1px) !important; }

/* =====================================================================
   ③ 左右滑动开关：CG 画风 + 每次随机种子共用一套样式（改一处两边都生效）
   ===================================================================== */
#q-style-row { flex: 0 0 auto !important; }
#q-style-toggle { flex: 1 1 auto !important; min-width: 0 !important; }
#q-seed-toggle { min-width: 0 !important; }
/* 随机种子卡片和「宽高比/百万像素」同一行，高度被最高的那张撑开，开关要垂直居中 */
#q-seed-toggle .wrap { justify-content: center !important; }

#q-output-row { flex-wrap: nowrap !important; gap: var(--q-gap) !important; }

#q-style-toggle label,
#q-seed-toggle label {
    display: flex !important; flex-direction: row-reverse !important;
    align-items: center !important; justify-content: space-between !important;
    width: 100% !important; min-height: 22px !important; gap: 10px !important;
    margin: 0 !important; padding: 0 !important;
    background: transparent !important; border: none !important; box-shadow: none !important;
    cursor: pointer !important;
}
#q-style-toggle label > span,
#q-seed-toggle label > span {
    flex: 0 0 auto !important; padding: 0 !important; background: transparent !important;
    font-size: .82rem !important; font-weight: 600 !important; color: var(--q-fg) !important;
    white-space: nowrap !important;
}
#q-style-toggle input[type="checkbox"],
#q-seed-toggle input[type="checkbox"] {
    -webkit-appearance: none !important; appearance: none !important;
    flex: 0 0 auto !important; position: relative !important;
    width: 40px !important; min-width: 40px !important; height: 22px !important;
    margin: 0 !important; padding: 0 !important;
    border-radius: 999px !important;
    background: var(--q-panel-3) !important;
    border: 1px solid var(--q-border-2) !important;
    cursor: pointer !important;
    transition: background .18s, border-color .18s !important;
}
#q-style-toggle input[type="checkbox"]::after,
#q-seed-toggle input[type="checkbox"]::after {
    content: "" !important;
    position: absolute !important; top: 1px !important; left: 1px !important;
    width: 18px !important; height: 18px !important; border-radius: 50% !important;
    background: #fff !important; box-shadow: 0 1px 3px rgba(0,0,0,.35) !important;
    transition: left .18s cubic-bezier(.22,.61,.36,1) !important;
}
#q-style-toggle input[type="checkbox"]:checked,
#q-seed-toggle input[type="checkbox"]:checked {
    background: var(--q-accent) !important; border-color: var(--q-accent) !important;
}
#q-style-toggle input[type="checkbox"]:checked::after,
#q-seed-toggle input[type="checkbox"]:checked::after { left: 19px !important; }

/* ---------------- 百万像素卡片内的实时分辨率徽标 ---------------- */
#q-megapixel .head { gap: 6px !important; }
.q-res-badge {
    flex: 0 0 auto !important;
    padding: 1px 7px !important;
    font-size: .72rem !important; font-weight: 700 !important;
    line-height: 1.55 !important; letter-spacing: .01em !important;
    font-variant-numeric: tabular-nums !important;
    color: var(--q-accent) !important;
    background: var(--q-accent-soft) !important;
    border-radius: 999px !important;
    white-space: nowrap !important;
}

/* 拖拽参考图时的落点高亮（自己接管拖拽后给的反馈） */
#q-ref-gallery.q-drop,
#q-add-ref-row.q-drop {
    box-shadow: 0 0 0 2px var(--q-accent) !important;
    border-radius: var(--q-radius) !important;
    background: var(--q-accent-soft) !important;
}

/* =====================================================================
   ④ 左列撑满：两个文本框吸收多余高度，底部不留白
   —— 真正的 flex-grow:0 在 Gradio 自动生成的 .form 包装层上，用 :has() 命中它
   ===================================================================== */
#q-main-row { align-items: stretch !important; }
#q-left-col { align-self: stretch !important; }
#q-left-col > .form:has(#q-prompt-box),
#q-left-col > .form:has(#q-negative-box) {
    /* 两个文本框共用一个 .form 容器（实测确认），所以最小高度要按「两条」给，
       否则会被压到几十像素，输入区只剩一条缝。 */
    flex: 1 1 0 !important; min-height: 190px !important;
}
#q-prompt-box, #q-negative-box {
    flex: 1 1 auto !important; min-height: 0 !important;
    display: flex !important; flex-direction: column !important;
}
#q-prompt-box > label, #q-negative-box > label {
    flex: 1 1 auto !important; min-height: 0 !important;
    display: flex !important; flex-direction: column !important;
}
#q-prompt-box > label > span, #q-negative-box > label > span { flex: 0 0 auto !important; }
#q-prompt-box .input-container, #q-negative-box .input-container {
    flex: 1 1 auto !important; min-height: 0 !important;
    display: flex !important; flex-direction: column !important;
}
#q-prompt-box textarea, #q-negative-box textarea {
    flex: 1 1 auto !important; height: auto !important;
    min-height: 46px !important;   /* 兜底：任何情况下都不低于约两行 */
    resize: none !important;
}

/* 参考图区、追加按钮、摘要行都不参与压缩：窗口太矮时让左列滚动，而不是把上传区压扁 */
#q-ref-gallery, #q-summary, #q-add-ref-row { flex: 0 0 auto !important; }

/* 缩略图钉成固定小方块：张数少于列数时，Gradio 会把每列按 1fr 拉大
   （实测 3 张时每块 211px、1 张时 648px），直接溢出 150px 的图库区盖住下面的控件。
   把行高与方块尺寸都钉死，多出来的图交给 grid-wrap 纵向滚动。 */
#q-ref-gallery .grid-wrap { grid-auto-rows: 96px !important; }
#q-ref-gallery .thumbnail-item { width: 100% !important; height: 96px !important; }
#q-ref-gallery .thumbnail-item img {
    width: 100% !important; height: 96px !important; object-fit: cover !important;
}

/* 藏掉图库右上角自带的 upload / 清除 图标：
   1) 那个「清除」是一键删光全部参考图的危险按钮，位置又和旧版预览的关闭按钮完全撞车，
      用户很容易误点后以为程序坏了；
   2) 它只在「未选中任何图」时出现，点一下缩略图就消失，行为不一致；
   3) 追加图片已经有常驻且带文字的「+ 添加参考图」，空库时也仍有拖放区。
   单张删除不受影响，仍在各自的缩略图上。 */
#q-ref-gallery .icon-button { display: none !important; }

/* ---------------- 参考图放大镜：每张缩略图右上角的独立入口 ---------------- */
#q-ref-gallery .gallery-item { position: relative !important; }
.q-zoom {
    position: absolute; top: 3px; right: 3px; z-index: 3;
    width: 20px; height: 20px;
    display: inline-flex; align-items: center; justify-content: center;
    color: #fff; background: rgba(12,15,22,.66);
    border: 1px solid rgba(255,255,255,.30); border-radius: 6px;
    cursor: pointer; opacity: .9;
    transition: background .15s, opacity .15s, border-color .15s;
}
.q-zoom:hover { background: var(--q-accent, #6d8cff); border-color: transparent; opacity: 1; }

/* 独立的放大遮罩。不用 Gradio 的预览态——它的工具条会在「关闭预览」和「清除全部」之间
   偷换语义，用户点两下就可能删光参考图并卡在全屏空白里。 */
#q-lightbox {
    position: fixed; inset: 0; z-index: 300;
    display: none; align-items: center; justify-content: center;
    background: rgba(6,8,14,.93);
    backdrop-filter: blur(3px);
}
#q-lightbox.q-open { display: flex; }
#q-lightbox img {
    max-width: 92vw; max-height: 86vh; object-fit: contain;
    border-radius: 8px; box-shadow: 0 18px 60px rgba(0,0,0,.6);
}
#q-lightbox-close {
    position: absolute; top: 14px; right: 18px;
    width: 30px; height: 30px;
    display: inline-flex; align-items: center; justify-content: center;
    color: #fff; background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.28); border-radius: 8px;
    cursor: pointer; font-size: .85rem; line-height: 1;
}
#q-lightbox-close:hover { background: rgba(248,113,113,.85); border-color: transparent; }
#q-lightbox-hint {
    position: absolute; bottom: 16px; left: 0; right: 0; text-align: center;
    color: rgba(255,255,255,.55); font-size: .74rem; pointer-events: none;
}

/* 「+ 添加参考图」：常驻入口。虚线边框表示「在这里添加」，避免用户找不到加第二张的地方 */
#q-add-ref-row { justify-content: flex-start !important; align-items: center !important; }
#q-add-ref {
    flex: 0 0 auto !important; width: auto !important; min-width: 0 !important;
    display: inline-flex !important; align-items: center !important; justify-content: center !important;
    padding: 1px 12px !important;
    font-size: .74rem !important; font-weight: 600 !important; line-height: 1.9 !important;
    color: var(--q-accent) !important;
    background: transparent !important;
    border: 1px dashed var(--q-accent) !important;
    border-radius: 999px !important;
    box-shadow: none !important;
    cursor: pointer !important;
    transition: background .15s !important;
}
#q-add-ref:hover { background: var(--q-accent-soft) !important; }
#q-add-ref:active { transform: translateY(1px) !important; }
/* 注：真正的 <input type=file> 是按钮的兄弟节点（同在这个 row 里），Gradio 已用 .hide 藏好 */

/* =====================================================================
   ⑤ 组件浮动标签：去掉丑贴图，换成与暗黑主题协调的半透明小胶囊
   ===================================================================== */
.gradio-container [data-testid="block-label"] {
    display: inline-flex !important; align-items: center !important;
    max-width: calc(100% - 18px) !important;
    margin: 0 !important; padding: 2px 8px !important;
    background: var(--q-chip-bg) !important;
    border: 1px solid var(--q-border) !important;
    border-radius: 6px !important;
    box-shadow: var(--q-shadow-soft) !important;
    color: var(--q-fg-2) !important;
    font-size: .72rem !important; font-weight: 600 !important;
    line-height: 1.35 !important; letter-spacing: .01em !important;
    white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important;
    backdrop-filter: blur(4px);
    pointer-events: none !important;
}
/* 干掉标签前面那个跟主题不搭的小图标 */
.gradio-container [data-testid="block-label"] > span { display: none !important; }
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

    // 生成按钮：只做「点击这一下」的反馈，不进入常驻忙碌态。
    // 这样用户随时能再点一次，把新任务排进队列（配合后端 trigger_mode="multiple"）。
    const bindRun = () => {
        const btn = document.getElementById('q-run-btn');
        if (!btn || btn.dataset.qPress === '1') return;
        btn.dataset.qPress = '1';
        btn.addEventListener('click', () => {
            btn.classList.remove('q-press');
            void btn.offsetWidth;   // 强制重排，连点也能重放动画
            btn.classList.add('q-press');
            clearTimeout(btn._qPressTimer);
            btn._qPressTimer = setTimeout(() => btn.classList.remove('q-press'), 340);
        }, true);
    };

    // 队列里每条任务的「终止」按钮：打后端 queue=False 的直连端点。
    // （api_name 必须和 app.py 里 cancel_button.click(..., api_name=...) 一致）
    const CANCEL_API = 'qwen_cancel_task';
    const bindQueueCancel = () => {
        if (document.body.dataset.qCancelBound === '1') return;
        document.body.dataset.qCancelBound = '1';
        document.addEventListener('click', (ev) => {
            const el = ev.target instanceof Element ? ev.target.closest('.q-cancel-btn') : null;
            if (!el) return;
            ev.preventDefault();
            ev.stopPropagation();
            if (el.hasAttribute('disabled')) return;
            const id = Number(el.getAttribute('data-task-id'));
            if (!id) return;
            el.setAttribute('disabled', 'disabled');
            el.classList.add('q-cancel-pending');
            el.textContent = '…';
            const cfg = window.gradio_config || {};
            fetch('/gradio_api/run/' + CANCEL_API, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({data: [id], session_hash: cfg.session_hash || ''})
            }).then((r) => {
                if (!r.ok) throw new Error('HTTP ' + r.status);
            }).catch(() => {
                // 失败就把按钮还原，让用户能重试
                el.textContent = '✕';
                el.removeAttribute('disabled');
                el.classList.remove('q-cancel-pending');
            });
        }, true);
    };

    // 参考图放大镜：给每张缩略图右上角挂一个放大镜，点开在独立遮罩里看大图。
    // 用 MutationObserver 补挂，因为图库增删图片时 Gradio 会重绘这些节点。
    const bindZoom = () => {
        if (document.body.dataset.qZoomBound === '1') return;
        const gal = document.getElementById('q-ref-gallery');
        if (!gal) return;
        document.body.dataset.qZoomBound = '1';

        const ICON = '<svg viewBox="0 0 24 24" width="12" height="12" aria-hidden="true">'
            + '<circle cx="10.5" cy="10.5" r="6.5" fill="none" stroke="currentColor" stroke-width="2.2"/>'
            + '<line x1="15.6" y1="15.6" x2="21" y2="21" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>'
            + '</svg>';

        const box = document.createElement('div');
        box.id = 'q-lightbox';
        box.innerHTML =
            '<span id="q-lightbox-close" role="button" tabindex="0" title="关闭（Esc）" aria-label="关闭">✕</span>'
            + '<img alt="参考图放大预览">'
            + '<div id="q-lightbox-hint">← → 切换参考图　｜　Esc 或点击空白处关闭</div>';
        document.body.appendChild(box);

        let srcs = [];
        let idx = 0;

        const paint = () => {
            if (!srcs.length) { box.classList.remove('q-open'); return; }
            idx = ((idx % srcs.length) + srcs.length) % srcs.length;
            box.querySelector('img').src = srcs[idx];
        };
        const collect = () => {
            srcs = [...gal.querySelectorAll('.thumbnail-item img')].map((im) => im.src);
        };
        const openAt = (src) => {
            collect();
            if (!srcs.length) return;
            const found = srcs.indexOf(src);
            idx = found >= 0 ? found : 0;
            paint();
            box.classList.add('q-open');
        };
        const close = () => box.classList.remove('q-open');

        box.querySelector('#q-lightbox-close').addEventListener('click', close);
        box.addEventListener('click', (ev) => { if (ev.target === box) close(); });
        document.addEventListener('keydown', (ev) => {
            if (!box.classList.contains('q-open')) return;
            if (ev.key === 'Escape') { close(); }
            else if (ev.key === 'ArrowLeft') { idx -= 1; paint(); }
            else if (ev.key === 'ArrowRight') { idx += 1; paint(); }
        });

        const inject = () => {
            collect();
            gal.querySelectorAll('.gallery-item').forEach((item) => {
                if (item.dataset.qZoom === '1') return;
                const im = item.querySelector('.thumbnail-item img');
                if (!im) return;
                item.dataset.qZoom = '1';   // 幂等标记，避免反复补挂
                const btn = document.createElement('span');
                btn.className = 'q-zoom';
                btn.setAttribute('role', 'button');
                btn.setAttribute('tabindex', '0');
                btn.setAttribute('aria-label', '放大查看这张参考图');
                btn.title = '放大查看';
                btn.innerHTML = ICON;
                btn.addEventListener('click', (ev) => {
                    ev.preventDefault();
                    ev.stopPropagation();     // 别触发缩略图自身的点击
                    openAt(im.src);
                });
                item.appendChild(btn);
            });
        };

        new MutationObserver(inject).observe(gal, { childList: true, subtree: true });
        inject();
    };

    // 百万像素卡片里的实时分辨率徽标：把隐藏的 #q-res-text 内容搬进卡片显示。
    // 数值由后端 compute_resolution 计算，保证徽标和真正出图用的尺寸是同一个数。
    const bindResBadge = () => {
        if (document.body.dataset.qResBadge === '1') return;
        const card = document.getElementById('q-megapixel');
        const src = document.getElementById('q-res-text');
        if (!card || !src) return;
        const head = card.querySelector('.head');
        if (!head) return;
        document.body.dataset.qResBadge = '1';

        const badge = document.createElement('span');
        badge.className = 'q-res-badge';
        badge.title = '实际输出分辨率';
        const numBox = head.querySelector('.tab-like-container');
        head.insertBefore(badge, numBox || null);   // 放在数值框左边，贴近卡片右侧

        const sync = () => { badge.textContent = (src.textContent || '').trim(); };
        new MutationObserver(sync).observe(src, {
            childList: true, subtree: true, characterData: true
        });
        sync();
    };

    // 拖拽追加参考图：Gradio 自带的拖放区只在图库为空时存在。传完第一张后再拖，
    // 页面上没有落点，浏览器就按默认行为「打开这个文件」——于是弹出新标签页显示图片。
    // 这里自己接管：图库和「+ 添加参考图」都能接住拖入的图，塞进隐藏的上传控件走正常流程。
    const bindDropRefs = () => {
        if (document.body.dataset.qDropRefs === '1') return;
        const gal = document.getElementById('q-ref-gallery');
        const row = document.getElementById('q-add-ref-row');
        const input = row ? row.querySelector('input[type=file]') : null;
        if (!gal || !row || !input) return;
        document.body.dataset.qDropRefs = '1';

        // 兜底：页面上任何位置都不让浏览器自行打开拖入的文件
        ['dragover', 'drop'].forEach((t) => {
            document.addEventListener(t, (ev) => ev.preventDefault());
        });

        const galHasImages = () => !!gal.querySelector('.thumbnail-item');

        const accept = (files) => {
            if (!files || !files.length) return;
            const dt = new DataTransfer();
            Array.prototype.forEach.call(files, (f) => dt.items.add(f));
            input.files = dt.files;                     // 交给 Gradio 自己的上传控件
            input.dispatchEvent(new Event('change', { bubbles: true }));
        };

        [[gal, false], [row, true]].forEach(([el, always]) => {
            // 图库为空时它自带拖放区，让 Gradio 自己处理，避免重复上传
            const mine = () => always || galHasImages();
            ['dragenter', 'dragover'].forEach((t) => {
                el.addEventListener(t, (ev) => {
                    if (!mine()) return;
                    ev.preventDefault();
                    if (ev.dataTransfer) ev.dataTransfer.dropEffect = 'copy';
                    el.classList.add('q-drop');
                });
            });
            el.addEventListener('dragleave', (ev) => {
                if (ev.target === el) el.classList.remove('q-drop');
            });
            el.addEventListener('drop', (ev) => {
                el.classList.remove('q-drop');
                if (!mine()) return;
                ev.preventDefault();
                accept(ev.dataTransfer ? ev.dataTransfer.files : null);
            });
        });
    };

    const boot = () => {
        init(); bindRun(); bindQueueCancel(); bindZoom(); bindResBadge(); bindDropRefs();
    };
    if (document.readyState === 'complete') boot();
    else window.addEventListener('load', boot);
    // Gradio 是异步挂载，兜底再跑几次
    [300, 800, 1500, 2500, 4000].forEach((ms) => setTimeout(boot, ms));
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
# 生成结果直接落在实例根目录下的 output\，而不是 ComfyUI\output\QwenStudio\ ——
# 用户点开根目录的 output 就能看到图，不用往里翻两层。
# 实现方式：启动 ComfyUI 时传 --output-directory（见 Launcher.start），
# 同时工作流的 filename_prefix 不再带子目录（见 on_generate）。
OUTPUT_DIR = os.path.join(CONFIG["instance_dir"], "output")
CLIENT = ComfyClient(COMFY_BASE)


# --------------------------------------------------------------------------
# 授权闸门（fail-closed）
# --------------------------------------------------------------------------
def _licence_gate(where: str) -> str:
    """已授权返回空串；否则返回可直接展示给用户的原因。

    这里刻意**不抛异常**：调用方自己决定是抛（启动器 / 界面构建），
    还是转成界面提示（生成流程是生成器函数，抛出去会炸掉整条输出流）。
    """
    try:
        import qlic
    except Exception as error:  # noqa: BLE001
        return f"授权模块加载失败：{error}"
    try:
        ok, why = qlic.state(APP_DIR)
    except Exception as error:  # noqa: BLE001
        why = f"授权校验异常：{error}"
        ok = False
    if not ok:
        # 只在控制台留一行痕迹方便作者排查；返回给用户的话术不带内部位置信息。
        print(f"[授权闸门] {where} 拒绝：{why}")
        return why or "本机尚未激活"
    return ""


def _licence_require(where: str) -> None:
    """闸门严格版：未授权直接抛 LicenseError。"""
    why = _licence_gate(where)
    if not why:
        return
    try:
        import qlic
        exc = qlic.LicenseError
    except Exception:  # noqa: BLE001
        exc = RuntimeError
    raise exc(why)


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
        # 闸门 1/4：后端进程层。
        # 这一处是最"硬"的——就算有人把界面层的判断改成无条件通过，
        # 这里不放行就起不了 ComfyUI，拿到的只是个没有引擎的空壳。
        _licence_require("Launcher.start")
        if self.started:
            return
        self.started = True
        if not os.path.isfile(self.python):
            raise ComfyError(f"找不到实例 python：{self.python}\n请检查 config.json 里的 instance_dir。")

        log = open(self.log_path, "ab", buffering=0)
        log.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 ComfyUI =====\n".encode("utf-8"))
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            # --output-directory：把出图目录指到实例根目录的 output\，
            # 这样用户不用进 ComfyUI\output\QwenStudio\ 里翻。路径用绝对路径，
            # 整个文件夹搬到任何盘/目录都跟着走。
            [self.python, "main.py", "--port", str(self.config["port"]), "--listen", "127.0.0.1",
             "--output-directory", OUTPUT_DIR],
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
    if not count:
        mode = "**文生图**（未上传参考图）"
    elif count > MAX_REF_IMAGES:
        # 工作流只吃 MAX_REF_IMAGES 张，这里必须写明，否则用户以为全都用上了
        mode = f"**图生图**（{count} 张参考图）　注意：超出上限，本次只用前 {MAX_REF_IMAGES} 张"
    else:
        mode = f"**图生图**（{count} 张参考图）"
    ratio = LABEL_TO_RATIO.get(aspect_label, aspect_label)
    width, height = compute_resolution(ratio, megapixels, 32)
    return f"当前模式：{mode} ｜ 输出尺寸：**{width} × {height}**"


def update_resolution_badge(aspect_label, megapixels):
    """百万像素卡片里那个分辨率徽标的文本。

    必须复用 compute_resolution，徽标显示的数字才会和真正出图用的尺寸完全一致
    （注意它是按 1024×1024 当「百万」算的，不能在这里另写一套换算）。
    """
    ratio = LABEL_TO_RATIO.get(aspect_label, aspect_label)
    width, height = compute_resolution(ratio, megapixels, 32)
    return f"{width}×{height}"


def toggle_seed_box(randomize):
    """「每次随机种子」开着时，种子输入框不参与生成，置灰只读；关掉才可编辑。

    这样用户一眼就能看出「这个框现在算不算数」，而不是敲了数字却没生效。
    """
    return gr.update(interactive=not randomize)


def _norm_ref_items(items) -> list[tuple[str, str | None]]:
    """把参考图列表统一成 Gallery 认的 (路径, 说明) 形式。

    Gallery 的 value 内部是 [(path, caption), ...]；UploadButton 给的是纯路径字符串，
    两者混用会导致追加后格式不一致，所以统一规整一遍。
    """
    out: list[tuple[str, str | None]] = []
    for item in items or []:
        if isinstance(item, (tuple, list)):
            path = item[0] if item else None
            caption = item[1] if len(item) > 1 else None
        else:
            path, caption = item, None
        if path:
            out.append((str(path), caption))
    return out


def append_reference_images(current, new_files, aspect_label, megapixels):
    """把新选的参考图追加到已有列表后面，并重置上传按钮，方便连续追加。

    返回 (Gallery 值, 上传按钮值, 摘要)。
    """
    merged = _norm_ref_items(current) + _norm_ref_items(new_files)
    if len(merged) > MAX_REF_IMAGES:
        merged = merged[:MAX_REF_IMAGES]
    return merged, None, update_summary(merged, aspect_label, megapixels)


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
    _sync_task(_ACTIVE_TASK["task"])
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


# --------------------------------------------------------------------------
# 任务队列（点「生成」即登记，右侧工作区实时展示）
# --------------------------------------------------------------------------

_TASK_LOCK = threading.Lock()
_TASKS: list[dict] = []
_TASK_SEQ = itertools.count(1)
_TASK_TTL = 6.0            # 终态任务在队列里保留几秒，然后自动消失
_TASK_ORPHAN_GRACE = 20.0  # 无任务在跑时，「排队中」超过这么久仍未被认领 → 视为幽灵丢弃
_TASK_STALE_TTL = 1800.0   # 「生成中」超过这么久没有任何进度更新 → 判失败
_ACTIVE_TASK: dict = {"task": None}

_STATUS_STYLE = {
    "排队中": ("wait", "◷"),
    "生成中": ("run", "◐"),
    "完成": ("done", "✓"),
    "失败": ("err", "✕"),
    "已中断": ("err", "■"),
    "已取消": ("cancel-hint", "⊘"),
}

# 被用户点过「终止」的任务 id。排队中的直接判终态；正在跑的需要中断 ComfyUI，
# 并让 on_generate 在收尾时把它记成「已中断」而不是「失败」。
_CANCELLED: set[int] = set()


def _task_brief(files, prompt_text) -> tuple[str, str]:
    count = len(files) if files else 0
    mode = f"图生图 · {count} 图" if count else "文生图"
    text = " ".join((prompt_text or "").split())
    if len(text) > 20:
        text = text[:20] + "…"
    return mode, (text or "（无提示词）")


def register_task(files, prompt_text) -> str:
    """点「生成」时立即登记（queue=False，不占生成队列），返回队列面板 HTML。"""
    mode, title = _task_brief(files, prompt_text)
    now = time.time()
    with _TASK_LOCK:
        _TASKS.append({
            "id": next(_TASK_SEQ),
            "mode": mode,
            "title": title,
            "status": "排队中",
            "pct": 0.0,
            "detail": "",
            "queued_at": now,
            "updated_at": now,
            "finished_at": None,
        })
    return render_queue_html()


def _claim_task(timeout: float = 1.0) -> dict | None:
    """把队首「排队中」的任务标成「生成中」。

    登记事件是 queue=False、和生成事件同时发出的，正常几十毫秒内就到；
    这里留一点等待时间，避免首击时抢跑。已被用户终止的任务会从「排队中」里消失，
    所以它们不会再被认领——这也正是「终止排队中任务」能生效的原因。
    """
    deadline = time.time() + timeout
    while True:
        with _TASK_LOCK:
            for task in _TASKS:
                if task["status"] == "排队中":
                    task["status"] = "生成中"
                    task["updated_at"] = time.time()
                    return task
        if time.time() >= deadline:
            return None
        time.sleep(0.05)


def _update_task(task: dict | None, **fields) -> None:
    if not task:
        return
    with _TASK_LOCK:
        task.update(fields)
        task["updated_at"] = time.time()


def _sync_task(task: dict | None) -> None:
    """把当前进度同步到队列里那条正在跑的任务上。"""
    if not task:
        return
    # 用户已终止的任务不再被进度回写覆盖（否则会从「已中断」变回「失败/完成」）
    if task.get("id") in _CANCELLED:
        return
    state = _PROGRESS["state"]
    if state == "running":
        _update_task(task, pct=float(_PROGRESS["pct"]), detail=str(_PROGRESS["detail"]))
    elif state == "done":
        _update_task(task, status="完成", pct=100.0, detail="", finished_at=time.time())
    elif state == "error":
        _update_task(task, status="失败", detail=str(_PROGRESS["detail"]), finished_at=time.time())


def request_cancel(task_id: int) -> str:
    """终止指定任务：排队中直接取消；生成中则中断 ComfyUI。返回最新队列 HTML。"""
    need_interrupt = False
    with _TASK_LOCK:
        task = next((t for t in _TASKS if t["id"] == task_id), None)
        if task is not None and task.get("finished_at") is None:
            _CANCELLED.add(task_id)
            now = time.time()
            if task["status"] == "排队中":
                task["status"] = "已取消"
                task["detail"] = "已在排队阶段终止"
                task["updated_at"] = now
                task["finished_at"] = now
            else:
                task["detail"] = "正在中断…"
                task["updated_at"] = now
                need_interrupt = True
    if need_interrupt:
        try:
            CLIENT.interrupt()
        except Exception:  # noqa: BLE001
            pass
    return render_queue_html()


def _finish_task(task: dict | None, status: str) -> None:
    """收尾：把任务标记成终态，交给队列面板自动淡出。"""
    if not task:
        return
    now = time.time()
    with _TASK_LOCK:
        task["status"] = status
        if status == "完成":
            task["pct"] = 100.0
            task["detail"] = ""
        task["updated_at"] = now
        task["finished_at"] = now


def render_queue_html() -> str:
    now = time.time()
    with _TASK_LOCK:
        # 是否有任务正在跑：排队中的任务只有在「没人在跑」时才可能是幽灵，
        # 否则它只是在前一单后面正常等待（很可能要等好几分钟）。
        running = any(t.get("finished_at") is None and t["status"] == "生成中" for t in _TASKS)
        kept: list[dict] = []
        for t in _TASKS:
            ended = t.get("finished_at")
            if ended is not None:
                # 终态：过了保留时间就移除
                if now - ended < _TASK_TTL:
                    kept.append(t)
                continue
            if t["status"] == "生成中":
                # 长时间没有任何进度更新 → 判失败
                if now - t.get("updated_at", now) > _TASK_STALE_TTL:
                    t["status"] = "失败"
                    t["detail"] = "任务长时间无响应"
                    t["finished_at"] = now
                kept.append(t)
                continue
            # 排队中：没人跑又迟迟不被认领，说明对应的生成事件已丢失 → 丢弃
            if not running and now - t.get("queued_at", now) > _TASK_ORPHAN_GRACE:
                continue
            kept.append(t)
        _TASKS[:] = kept
        tasks = [dict(t) for t in _TASKS]

    if not tasks:
        body = '<div class="q-queue-none">暂无任务，点击「生成」加入队列</div>'
    else:
        rows = []
        for t in tasks:
            style, icon = _STATUS_STYLE.get(t["status"], ("wait", "◷"))
            meta = t["mode"]
            # 生成中显示进度细节；失败/中断/取消显示原因，方便排查
            if t.get("detail") and t["status"] in ("生成中", "失败", "已中断", "已取消"):
                meta += " · " + t["detail"]
            pct = f'<span class="q-task-pct">{t["pct"]:.0f}%</span>' if t["status"] == "生成中" else ""
            # 只有还没结束的任务才给「终止」按钮
            if t["status"] in ("排队中", "生成中") and t.get("finished_at") is None:
                cancel = ('<button type="button" class="q-cancel-btn" data-task-id="%d"'
                          ' title="终止这个任务" aria-label="终止任务 %d">✕</button>'
                          % (t["id"], t["id"]))
            else:
                cancel = ""
            rows.append(
                '<div class="q-task q-task-%s">'
                '<span class="q-task-icon">%s</span>'
                '<span class="q-task-id">#%s</span>'
                '<span class="q-task-body">'
                '<span class="q-task-title">%s</span>'
                '<span class="q-task-meta">%s</span>'
                '</span>%s%s</div>'
                % (style, icon, t["id"], html.escape(t["title"]), html.escape(meta), pct, cancel)
            )
        body = "".join(rows)

    return (
        '<div class="q-queue">'
        '<div class="q-queue-head">'
        '<span class="q-queue-title">任务队列</span>'
        '<span class="q-queue-count">%d</span>'
        '</div>'
        '<div class="q-queue-list">%s</div>'
        '</div>' % (len(tasks), body)
    )


def on_queue_click(files, prompt_text):
    # 闸门 2/4：入队层。未授权连"排队"这一步都不受理。
    if _licence_gate("on_queue_click"):
        return render_queue_html()
    return register_task(files, prompt_text)


def refresh_queue():
    return render_queue_html()


def on_generate(files, prompt_text, negative_text, aspect_label, megapixels,
                steps, cfg, denoise, seed, randomize, style_on):
    """生成主流程：必要时启动 ComfyUI → 上传图片 → 提交 → 轮询 → 取图。

    三个输出依次是：结果图、状态条、种子框。
    - 结果图在进度阶段用 gr.skip()，不要用 None：None 会立刻清空上一次的预览
      （第二次任务一开始图就没了），gr.skip() 表示「本次不更新」，旧图会留到新图出来。
    - 种子框把本单**实际使用**的种子写回去。开随机种子时界面上的种子框是禁用只读的，
      只有靠这里回填，用户才能看到「这一单到底用了哪个种子」。
    """
    # 闸门 3/4：生成层。这是唯一真正会出图的入口。
    # 注意用 yield 转成界面提示而不是抛异常：这是个生成器函数，
    # 抛出去会把 Gradio 的输出流整个打断，用户只看到界面卡住。
    _why = _licence_gate("on_generate")
    if _why:
        yield (gr.skip(),
               render_progress("error", 0, "未授权", _why),
               gr.skip())
        return

    task = _claim_task()
    if task is None:
        # 没有可认领的排队任务：说明这一单在排队阶段就被用户终止了（或登记事件没到）。
        # 绝不能凭空造一条新任务去跑——那会让「已终止」的任务照样出图。
        _ACTIVE_TASK["task"] = None
        yield (gr.skip(),
               render_progress("idle", 0, "已终止", "该任务已在排队阶段被终止。"),
               gr.skip())
        return

    _ACTIVE_TASK["task"] = task

    if not CLIENT.is_up():
        yield (gr.skip(),
               render_progress("running", 2, "正在启动 ComfyUI", "首次生成需要加载模型，请稍候…"),
               gr.skip())
        try:
            LAUNCHER.start()
        except ComfyError as error:
            yield (gr.skip(), render_progress("error", 0, "启动失败", str(error)), gr.skip())
            return
        started = time.time()
        while not CLIENT.is_up() and time.time() - started < 300:
            el = time.time() - started
            yield (gr.skip(),
                   render_progress("running", min(20, el / 300 * 20),
                                   "正在启动 ComfyUI",
                                   f"已等待 {el:.0f} 秒…"),
                   gr.skip())
            time.sleep(1.5)
        if not CLIENT.is_up():
            yield (gr.skip(),
                   render_progress("error", 0, "ComfyUI 启动超时",
                                   "请查看 comfyui.log 末尾的报错。"),
                   gr.skip())
            return

    stop_event = threading.Event()
    watcher = None
    try:
        # 参考图上传：工作流最多接 MAX_REF_IMAGES 张，超出的先丢掉，
        # 并明确写进摘要，避免「选了 20 张其实只用了 16 张」却无人察觉。
        image_names: list[str] = []
        if files:
            all_items = list(files)
            selected = all_items[:MAX_REF_IMAGES]
            dropped = len(all_items) - len(selected)
            total = len(selected)
            for idx, item in enumerate(selected, 1):
                # Gradio 5.x 的 Gallery value 是 (filepath, caption) 元组，这里统一取出路径
                path = item[0] if isinstance(item, (tuple, list)) else item
                if not path:
                    continue
                yield (gr.skip(),
                       render_progress("running", 5 + idx / max(total, 1) * 10,
                                       "正在上传参考图",
                                       f"{idx}/{total}：{os.path.basename(path)}"),
                       gr.skip())
                image_names.append(CLIENT.upload_image(path))
            if dropped:
                print(f"[参考图] 共选 {len(all_items)} 张，超出上限 {MAX_REF_IMAGES}，本次只用前 {MAX_REF_IMAGES} 张")

        mode = "i2i" if image_names else "t2i"
        ratio = LABEL_TO_RATIO.get(aspect_label, aspect_label)
        width, height = compute_resolution(ratio, megapixels, 32)

        # 随机种子上限取 2**53-1（JS 安全整数上限）：种子要回填到 Number 输入框里显示，
        # 超过这个范围在浏览器里会被四舍五入，导致「看到的数字」和「实际用的数字」对不上。
        actual_seed = random.randint(0, 2 ** 53 - 1) if randomize else int(seed)
        # 加速常开；模式 LoRA 按模式二选一（t2i 用旧的、i2i 用新的）；CG 由界面开关决定
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
            # 不带子目录：配合 --output-directory，图片直接落在根目录 output\ 下
            filename_prefix=f"{mode}_",
        )

        prompt_id = CLIENT.queue_prompt(graph)
        CURRENT["prompt_id"] = prompt_id
        started = time.time()
        # 把本单实际用的种子回填到种子框：开了随机种子后，用户只能从这里看到随到了哪个数
        yield (gr.skip(),
               render_progress("running", 18, "已提交，开始生成",
                               f"{mode} / {width}×{height} / {int(steps)} 步"),
               gr.update(value=int(actual_seed)))

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
            yield gr.skip(), render_progress("running", pct, "生成中", detail), gr.skip()
            time.sleep(0.5)

        stop_event.set()

        images = outputs.get("9", {}).get("images") or []
        if not images:
            raise ComfyError(f"执行完成但没有取到图片，outputs={outputs}")

        first = images[0]
        raw = CLIENT.fetch_image(first["filename"], first.get("subfolder", ""), first.get("type", "output"))
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        elapsed = time.time() - started
        saved = os.path.join(OUTPUT_DIR, first.get("subfolder", ""), first["filename"])
        detail = f"用时 {elapsed:.1f} 秒 ｜ {mode} {width}×{height} ｜ 种子 {actual_seed}"
        yield image, render_progress("done", 100, "生成完成", detail), gr.skip()
    except ComfyError as error:
        # 失败时保留上一张成功的预览图，不要把它清掉
        yield (gr.skip(), render_progress("error", _PROGRESS["pct"], "生成失败", str(error)),
               gr.skip())
    except Exception as error:  # noqa: BLE001
        yield (gr.skip(), render_progress("error", _PROGRESS["pct"], "出错了",
                                          f"{type(error).__name__}: {error}"), gr.skip())
    finally:
        stop_event.set()
        CURRENT["prompt_id"] = None
        # 用户点过终止的，一律记「已中断」，不要被中断引发的报错带成「失败」
        if task is not None and task.get("id") in _CANCELLED:
            _finish_task(task, "已中断")
        else:
            _finish_task(task, {"done": "完成", "error": "失败"}.get(_PROGRESS["state"], "已中断"))
        _ACTIVE_TASK["task"] = None


def on_interrupt():
    CLIENT.interrupt()
    return render_progress("error", _PROGRESS["pct"], "已中断", "已请求停止当前任务。")


def open_output_folder() -> str:
    """在资源管理器里打开输出文件夹，方便直接查看所有生成结果。

    注意：这里不接受任何来自前端的路径参数，永远只打开 OUTPUT_DIR，
    避免把「打开任意本地目录」的能力暴露给页面。
    """
    path = OUTPUT_DIR
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        return f"创建输出文件夹失败：{error}"
    try:
        if hasattr(os, "startfile"):          # Windows 首选
            os.startfile(path)
        else:
            subprocess.Popen(["explorer", path])
    except Exception as error:  # noqa: BLE001
        return f"打开失败：{error}"
    return f"已打开：{path}"


# --------------------------------------------------------------------------
# 界面
# --------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    # 闸门 4/4：界面构建层。
    # 即使有人把 main() 里那句 `if activated:` 直接改成无条件调用 build_ui()，
    # 这里仍然会拦下 —— 未授权就是构造不出主界面。
    _licence_require("build_ui")

    with gr.Blocks(theme=gr.themes.Soft(), title="Qwen Image 2.1 Studio",
                   css=CUSTOM_CSS, js=THEME_JS, fill_width=True) as demo:
        gr.Markdown("# Qwen Image 2.1 Studio", elem_id="app-title")
        status_md = gr.Markdown("正在检查 ComfyUI 状态…", elem_id="app-sub")

        with gr.Row(elem_id="q-main-row"):
            with gr.Column(scale=5, elem_id="q-left-col"):
                files = gr.Gallery(
                    label="参考图片（可多张，最多 16 张）",
                    type="filepath",
                    file_types=["image"],
                    columns=6,
                    rows=1,
                    # 缩略图已被 CSS 钉成 96px 方块，这里按「标签 + 一行方块」给高度即可，
                    # 多出来的图由 grid-wrap 纵向滚动；也把省下的高度还给下面的输入框。
                    height=122,
                    object_fit="cover",
                    # 关掉预览与全屏：Gradio 的预览态会把工具条换成 Fullscreen/Close，
                    # 退出后同一位置又变回「清除（一键删光）」——用户点两下就可能误删全部参考图，
                    # 再叠加全屏遮罩就卡在一片空白里，只能按 ESC 退出。参考图区不需要这套。
                    allow_preview=False,
                    preview=False,
                    show_fullscreen_button=False,
                    show_download_button=False,
                    interactive=True,
                    elem_id="q-ref-gallery",
                )
                # 常驻的追加入口：Gradio 自带的那个上传小图标只在「未选中任何图」时出现，
                # 传完第一张后进入预览态就消失了，用户会找不到加第二张的地方。
                with gr.Row(elem_id="q-add-ref-row"):
                    add_ref_button = gr.UploadButton(
                        "+ 添加参考图",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                        size="sm",
                        elem_id="q-add-ref",
                    )
                summary_md = gr.Markdown("", elem_id="q-summary")
                prompt_box = gr.Textbox(label="提示词", lines=2, elem_id="q-prompt-box",
                                        placeholder="描述你想要的画面…")
                negative_box = gr.Textbox(label="负面提示词", lines=2, elem_id="q-negative-box",
                                          value="bad quality, blurry, low resolution")

                with gr.Row(elem_id="q-output-row"):
                    aspect_box = gr.Dropdown(
                        choices=list(ASPECT_RATIO_LABELS.values()),
                        value=ASPECT_RATIO_LABELS["9:16 (Portrait Widescreen)"],
                        label="宽高比", scale=4, elem_id="q-aspect",
                    )
                    megapixel_box = gr.Slider(0.1, 16.0, value=1.0, step=0.1, label="百万像素",
                                              scale=5, show_reset_button=False,
                                              elem_id="q-megapixel")
                    randomize_box = gr.Checkbox(value=True, label="每次随机种子",
                                                scale=3, elem_id="q-seed-toggle")

                # 分辨率徽标的文本源（视觉隐藏但保留在 DOM 里，前端 JS 才能读到并搬进卡片）。
                # 注意必须用 visible="hidden"：visible=False 会把元素直接从 DOM 移除。
                res_text = gr.HTML(
                    value=update_resolution_badge(
                        ASPECT_RATIO_LABELS["9:16 (Portrait Widescreen)"], 1.0),
                    visible="hidden", elem_id="q-res-text",
                )

                with gr.Row(equal_height=True, elem_id="q-param-row"):
                    steps_box = gr.Slider(1, 50, value=8, step=1, label="步数",
                                          scale=1, min_width=0, show_reset_button=False)
                    cfg_box = gr.Slider(0.0, 20.0, value=1.0, step=0.1, label="CFG",
                                        scale=1, min_width=0, show_reset_button=False)
                    denoise_box = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="降噪",
                                            scale=1, min_width=0, show_reset_button=False)
                    # 默认「每次随机种子」是开着的，此时种子的输入框不参与生成，
                    # 所以初始就是禁用只读状态（开着的开关会把它置灰，关掉才可编辑）。
                    seed_box = gr.Number(value=0, label="种子", precision=0,
                                         scale=1, min_width=0, interactive=False,
                                         elem_id="q-seed")

                with gr.Row(elem_id="q-style-row"):
                    style_box = gr.Checkbox(value=False, label="CG 画风",
                                            elem_id="q-style-toggle")

                with gr.Row():
                    run_button = gr.Button("生成", variant="primary", size="lg",
                                           elem_id="q-run-btn")
                    stop_button = gr.Button("中断", size="lg", elem_id="q-stop-btn")

            with gr.Column(scale=4, elem_id="q-right-col"):
                output_image = gr.Image(label="结果", type="pil",
                                        elem_id="q-output-image")
                with gr.Row(elem_id="q-folder-row"):
                    open_folder_button = gr.Button("打开输出文件夹 ↗", size="sm",
                                                   elem_id="q-open-folder")
                output_info = gr.HTML(value=render_progress("idle", 0, "待命", "点击「生成」开始"),
                                      elem_id="q-output-info")
                queue_html = gr.HTML(value=render_queue_html(), elem_id="q-queue")

        # 事件绑定
        summary_inputs = [files, aspect_box, megapixel_box]
        demo.load(fn=on_load, outputs=[status_md])
        demo.load(fn=update_summary, inputs=summary_inputs, outputs=[summary_md])
        files.change(fn=update_summary, inputs=summary_inputs, outputs=[summary_md])
        aspect_box.change(fn=update_summary, inputs=summary_inputs, outputs=[summary_md])
        megapixel_box.change(fn=update_summary, inputs=summary_inputs, outputs=[summary_md])

        # 分辨率徽标：拖动百万像素时也要跟着变，所以 input（拖动中）和 change（松手）都绑上
        res_inputs = [aspect_box, megapixel_box]
        aspect_box.change(fn=update_resolution_badge, inputs=res_inputs,
                          outputs=[res_text], queue=False)
        megapixel_box.input(fn=update_resolution_badge, inputs=res_inputs,
                            outputs=[res_text], queue=False)
        megapixel_box.change(fn=update_resolution_badge, inputs=res_inputs,
                             outputs=[res_text], queue=False)

        # 随机种子开关联动种子输入框：开着时置灰只读（种子由程序随机），关掉才可编辑
        randomize_box.change(fn=toggle_seed_box, inputs=[randomize_box],
                             outputs=[seed_box], queue=False)

        # 追加参考图：把新选的图并到已有列表后面（Gradio 自带的入口在预览态会消失）
        add_ref_button.upload(
            fn=append_reference_images,
            inputs=[files, add_ref_button, aspect_box, megapixel_box],
            outputs=[files, add_ref_button, summary_md],
        )

        # 点「生成」时先立即登记队列（queue=False，不等生成槽位）
        # trigger_mode="multiple"：允许重复点击排队，否则 Gradio 默认 "once"
        # 会在上一单没跑完时直接把新点击丢掉（队列里只剩一条永远排队的行）。
        run_button.click(
            fn=on_queue_click,
            inputs=[files, prompt_box],
            outputs=[queue_html],
            queue=False,
            trigger_mode="multiple",
        )
        run_button.click(
            fn=on_generate,
            inputs=[files, prompt_box, negative_box, aspect_box, megapixel_box,
                    steps_box, cfg_box, denoise_box, seed_box, randomize_box, style_box],
            outputs=[output_image, output_info, seed_box],
            trigger_mode="multiple",
        )
        stop_button.click(fn=on_interrupt, outputs=[output_info],
                          trigger_mode="multiple")

        # 「打开输出文件夹」：纯本地动作，queue=False 保证立刻响应（不等生成槽位）
        open_folder_button.click(fn=open_output_folder, inputs=None, outputs=None,
                                 queue=False)

        # 队列里每条任务的「终止」按钮：面板是纯 HTML，点击由前端 JS 直接 POST 这个端点。
        # 必须 queue=False，否则生成任务占着队列时这个请求会排在后面，根本终止不了。
        cancel_box = gr.Number(value=-1, precision=0, visible=False, elem_id="q-cancel-id")
        cancel_button = gr.Button("cancel", visible=False, elem_id="q-cancel-btn")
        cancel_button.click(
            fn=request_cancel,
            inputs=[cancel_box],
            outputs=[queue_html],
            queue=False,
            api_name="qwen_cancel_task",
        )

        # 队列面板轮询刷新（queue=False，生成中也能刷新）
        queue_timer = gr.Timer(1.0)
        queue_timer.tick(fn=refresh_queue, outputs=[queue_html], queue=False)

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
        try:
            demo = build_ui()
        except Exception as _ui_err:  # noqa: BLE001
            # build_ui 顶部还有一道闸门。极端情况下（授权文件在启动那一瞬间
            # 被删掉或写坏）它会抛，这里兜底回激活界面 —— 别让正版用户看到崩溃堆栈。
            print("[主界面构建被拦截]", _ui_err)
            try:
                import qactivate
                demo = qactivate.build_activation_ui(APP_DIR)
            except Exception as _act_err:  # noqa: BLE001
                print("[激活界面异常]", _act_err)
                return
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
