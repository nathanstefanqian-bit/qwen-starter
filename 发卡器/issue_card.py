"""发卡器（仅作者使用）。

用途：输入用户的「机器码」 -> 生成一条卡密 -> 复制发给用户。
私钥：keys/private_key.pem（本目录下，切勿外传）
"""
from __future__ import annotations

import base64
import hashlib
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PRIV_PATH = os.path.join(HERE, "keys", "private_key.pem")
_B32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"


def _machine_hash_from_code(code: str) -> bytes:
    """把用户发来的机器码（XXXX-XXXX-XXXX-XXXX）还原成指纹前 10 字节。

    机器码来自 base32(指纹[:10])[:16]，此处反解出前 10 字节，
    再取前 6 字节作为卡内绑定值。
    """
    s = "".join(ch for ch in str(code).upper() if ch in _B32)
    # base32 每 8 字符 -> 5 字节
    pad = "=" * ((8 - len(s) % 8) % 8)
    return base64.b32decode(s + pad)


def make_card(machine_code: str) -> str:
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    priv = load_pem_private_key(open(PRIV_PATH, "rb").read(), password=None)

    fp10 = _machine_hash_from_code(machine_code)
    if len(fp10) < 6:
        raise ValueError("机器码无效（太短）")
    fp6 = fp10[:6]
    serial = struct.pack(">H", int(time.time()) & 0xFFFF)  # 2 字节序号
    payload = fp6 + serial                                  # 8 字节
    sig = priv.sign(payload)                                # 64 字节
    raw = payload + sig                                     # 72 字节
    card = base64.b32encode(raw).decode().rstrip("=")
    return "-".join(card[i:i + 4] for i in range(0, len(card), 4))


def _gui():
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.title("Qwen Studio 发卡器")
    root.geometry("560x360")
    root.configure(bg="#1f1f28")

    tk.Label(root, text="机器码（用户微信发来的）：", bg="#1f1f28", fg="#ececf1",
             font=("Microsoft YaHei UI", 10)).pack(anchor="w", padx=16, pady=(16, 4))
    code_var = tk.StringVar()
    tk.Entry(root, textvariable=code_var, font=("Consolas", 13), width=40,
             bg="#2a2a36", fg="#ececf1", insertbackground="#ececf1").pack(padx=16, pady=4, fill="x")

    tk.Label(root, text="卡密（复制发给用户）：", bg="#1f1f28", fg="#ececf1",
             font=("Microsoft YaHei UI", 10)).pack(anchor="w", padx=16, pady=(14, 4))
    card_box = tk.Text(root, height=5, font=("Consolas", 11), wrap="word",
                       bg="#2a2a36", fg="#7ee787", insertbackground="#ececf1")
    card_box.pack(padx=16, pady=4, fill="both", expand=True)

    def gen():
        code = code_var.get().strip()
        if not code:
            messagebox.showwarning("提示", "请先粘贴机器码")
            return
        try:
            card = make_card(code)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("生成失败", f"机器码可能不对：\n{e}")
            return
        card_box.delete("1.0", "end")
        card_box.insert("1.0", card)
        root.clipboard_clear()
        root.clipboard_append(card)
        messagebox.showinfo("成功", "卡密已生成，并已复制到剪贴板 ✅")

    def copy():
        c = card_box.get("1.0", "end").strip()
        if c:
            root.clipboard_clear()
            root.clipboard_append(c)
            messagebox.showinfo("已复制", "卡密已复制到剪贴板")

    bar = tk.Frame(root, bg="#1f1f28")
    bar.pack(padx=16, pady=(8, 16), fill="x")
    tk.Button(bar, text="生成卡密", command=gen, font=("Microsoft YaHei UI", 10),
              bg="#3b82f6", fg="white", relief="flat", padx=18, pady=6).pack(side="left")
    tk.Button(bar, text="复制卡密", command=copy, font=("Microsoft YaHei UI", 10),
              bg="#374151", fg="white", relief="flat", padx=18, pady=6).pack(side="left", padx=10)

    root.mainloop()


if __name__ == "__main__":
    if not os.path.exists(PRIV_PATH):
        print("找不到私钥：", PRIV_PATH)
        sys.exit(1)
    _gui()
