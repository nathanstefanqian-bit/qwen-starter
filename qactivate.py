"""激活界面（与主界面同端口，作为 Gradio Blocks 的一个分支）。

app.py 启动时：
  - 已激活 -> build_ui() 返回主界面
  - 未激活 -> build_ui() 返回本模块的激活界面；激活成功后提示用户重启
             （或直接在该界面内切换到主界面，见 app.py 逻辑）

设计：为了简单可靠，激活成功后要求**关闭并重新打开**软件即可进入主界面。
（避免同一进程内动态切换 Blocks 的复杂性。）
"""
from __future__ import annotations

import gradio as gr

import qlic

ACT_CSS = """
body { background:#15151c; }
.gradio-container { max-width: 620px !important; margin: 30px auto !important;
                    font-family: "Microsoft YaHei UI", system-ui, sans-serif; }
#act-card { background:#1f1f28; border:1px solid #33333e; border-radius:14px; padding:22px 24px; }
#act-title { font-size:1.4rem; font-weight:800; color:#ececf1; margin-bottom:2px; }
#act-sub { color:#8b8b98; font-size:.86rem; margin-bottom:16px; }
#act-mc textarea { font-family: Consolas, monospace !important; font-size:1.2rem !important;
                   text-align:center !important; letter-spacing:2px !important;
                   background:#2a2a36 !important; color:#7ee787 !important;
                   border:1px solid #33333e !important; }
#act-in textarea { font-family: Consolas, monospace !important; font-size:.9rem !important;
                   background:#2a2a36 !important; color:#ececf1 !important;
                   border:1px solid #33333e !important; }
#act-btn { background:#3b82f6 !important; color:#fff !important; border:none !important;
           font-weight:700 !important; font-size:1rem !important; }
#act-msg * { color:#ececf1 !important; }
"""


def build_activation_ui(app_dir: str) -> gr.Blocks:
    with gr.Blocks(title="软件激活", css=ACT_CSS) as act:
        with gr.Column(elem_id="act-card"):
            gr.Markdown("软件激活", elem_id="act-title")
            gr.Markdown("本软件需要**卡密**才能使用。请把下面的机器码发给作者获取卡密；"
                        "激活后将自动进入软件。", elem_id="act-sub")
            gr.Textbox(value=qlic.machine_code(), label="本机机器码（发给作者）",
                       interactive=False, elem_id="act-mc", lines=1, show_copy_button=True)
            card_in = gr.Textbox(label="卡密", placeholder="粘贴作者发给你的卡密…",
                                 elem_id="act-in", lines=3)
            btn = gr.Button("激活", elem_id="act-btn")
            msg = gr.Markdown("", elem_id="act-msg")

        def do_activate(card):
            if not card or not card.strip():
                return "⚠️ 请输入卡密。"
            if qlic.save_activation(app_dir, card):
                return "✅ 激活成功！请**关闭本窗口并重新打开软件**即可使用。"
            return "❌ 授权信息无效，请检查后重试，或联系作者。"

        btn.click(fn=do_activate, inputs=[card_in], outputs=[msg])
    return act
