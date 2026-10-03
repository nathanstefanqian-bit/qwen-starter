"""授权校验（客户端）。—— 内部工具，请勿随意改动。

职责：
  1. 计算本机"机器码"（稳定硬件指纹）
  2. 用内嵌公钥校验用户卡密（Ed25519 签名）
  3. 读写本地授权文件，实现"激活一次、永久免输"
  4. 对外提供运行时闸门 require() / state()，供 app.py 多处调用

卡密结构：
  payload(8B) = 指纹前 6 字节 + 序号 2 字节
  sig(64B)    = Ed25519(私钥, payload) 完整签名
  card        = Base32(payload + sig)   —— 用户复制粘贴，无需手打
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import time

# ---- 验签公钥（可公开；私钥只在作者发卡器里）----
# 这里刻意不放 PEM 文本：PEM 首行那个公钥标记是个极显眼的路标，
# 任何人 grep 一下就定位到授权校验，进而顺藤摸瓜改掉闸门。
# 改成「Ed25519 原始 32 字节 -> 掩码异或 -> Base64 -> 拆三段」存放：
# 语义与原 PEM 完全等价（重建出的公钥逐字节相同），但源码里看不出这是什么。
_P1 = "Hr06YFD0WlJtFB"
_P2 = "nS+qDV6OVQ5ChIM"
_P3 = "Tb1cnyN+zFSOik="
_MASK_SEED = b"qwen-image-2.1/licence/pub"

_MACHINE_CACHE = None
_FP_CACHE = None
_B32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"


class LicenseError(RuntimeError):
    """未授权 / 授权失效。

    调用方**必须**捕获它并给出友好提示，不要让它冒泡成崩溃堆栈——
    对正版用户来说这是「还没激活」，不该表现成程序坏了。
    """


def _ps(script: str) -> str:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return (out.stdout or "").strip()
    except Exception:
        return ""


def _vol_serial() -> str:
    """取 C 盘卷序列号。

    这里必须自己处理编码：`vol` 在中文系统上吐的是 GBK 字节，而本进程可能
    处于 UTF-8 模式，`text=True` 会按 UTF-8 解码并在**读线程里**抛
    UnicodeDecodeError —— 异常发生在子线程，主流程毫无察觉，
    后果是卷序列号**静默变成空串**，指纹里白白少一项。

    更麻烦的是它不稳定：换个启动方式（是否带 UTF-8 模式）指纹就变了，
    已经发出去的卡会莫名失效。所以这里读原始字节、按 utf-8 -> gbk 依次尝试，
    让结果与运行环境无关。
    """
    try:
        out = subprocess.run(["cmd", "/c", "vol", "C:"], capture_output=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        blob = out.stdout or b""
        text = ""
        for enc in ("utf-8", "gbk", "latin-1"):
            try:
                text = blob.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        for line in text.splitlines():
            if "Serial" in line or "序列号" in line:
                return line.split("is")[-1].strip() if "is" in line else line.split(":")[-1].strip()
    except Exception:
        pass
    return ""


def machine_fingerprint() -> bytes:
    """稳定硬件因子 -> SHA-256。故意不取 MAC / 网卡 / 外设，减少误伤。

    结果缓存：指纹要跑 3 次 PowerShell + 1 次 cmd，单次开销 1~3 秒。
    闸门在多处调用，不缓存的话每次生成都要白等好几秒。
    """
    global _FP_CACHE
    if _FP_CACHE is None:
        parts = [
            _ps("(Get-CimInstance Win32_ComputerSystemProduct).UUID"),
            _ps("(Get-CimInstance Win32_Processor).ProcessorId"),
            _vol_serial(),
        ]
        raw = "|".join(parts)
        _FP_CACHE = hashlib.sha256(raw.encode("utf-8")).digest()
    return _FP_CACHE


def machine_code() -> str:
    """展示给用户 / 微信发送的机器码，形如 XXXX-XXXX-XXXX-XXXX。"""
    global _MACHINE_CACHE
    if _MACHINE_CACHE is None:
        h = machine_fingerprint()
        s = base64.b32encode(h[:10]).decode().rstrip("=")[:16]
        _MACHINE_CACHE = "-".join(s[i:i + 4] for i in range(0, 16, 4))
    return _MACHINE_CACHE


def _b32e(data: bytes) -> str:
    return base64.b32encode(data).decode().rstrip("=")


def _b32d(text: str) -> bytes:
    pad = "=" * ((8 - len(text) % 8) % 8)
    return base64.b32decode(text + pad)


def _pub():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    blob = base64.b64decode(_P1 + _P2 + _P3)
    mask = hashlib.sha256(_MASK_SEED).digest()[: len(blob)]
    raw = bytes(a ^ b for a, b in zip(blob, mask))
    return Ed25519PublicKey.from_public_bytes(raw)


def _check(card: str) -> bool:
    """核心校验：验签 + 指纹匹配。失败一律返回 False（不暴露原因）。"""
    try:
        norm = "".join(ch for ch in str(card).upper() if ch in _B32)
        raw = _b32d(norm)
        if len(raw) != 72:
            return False
        payload, sig = raw[:8], raw[8:]
        fp6 = payload[:6]
        # 1) 验签：确认是作者签发
        _pub().verify(sig, payload)
        # 2) 指纹匹配：确认是本机
        if fp6 != machine_fingerprint()[:6]:
            return False
        return True
    except Exception:
        return False


# ---------------- 本地授权文件 ----------------
def _lic_path(app_dir: str) -> str:
    return os.path.join(app_dir, "license.dat")


def is_activated(app_dir: str) -> bool:
    return state(app_dir)[0]


def state(app_dir: str) -> tuple[bool, str]:
    """返回 (是否已授权, 原因)。**不抛异常**，方便界面直接展示原因。

    分成不同的失败原因不是给攻击者看的（本地代码本来就能读），
    而是让正版用户能自己分清「没激活」和「激活文件被我弄坏了」。
    """
    p = _lic_path(app_dir)
    if not os.path.exists(p):
        return False, "本机尚未激活，请输入卡密激活后使用"
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False, "license.dat 读取失败，请重新激活"
    card = (data or {}).get("card")
    if not card:
        return False, "本机尚未激活，请输入卡密激活后使用"
    if not _check(card):
        return False, "授权信息无效或与本机不匹配，请联系作者重新发卡"
    return True, ""


def save_activation(app_dir: str, card: str) -> bool:
    if not _check(card):
        return False
    try:
        with open(_lic_path(app_dir), "w", encoding="utf-8") as f:
            json.dump({"card": card.strip().upper(), "ts": int(time.time())}, f)
        return True
    except Exception:
        return False


# ---------------- 运行时闸门 ----------------
def require(app_dir: str) -> None:
    """未授权直接抛 LicenseError。

    设计意图（重要）：闸门**故意分散在多处调用**——
    app.py 的 build_ui / on_generate / Launcher.start / on_queue_click 各校验一次。

    原因：只在 main() 里判一个布尔量（if activated: ... else: ...）最省事，
    但攻击者改一行 `activated = True` 就全盘失守。
    分散校验后，绕过者必须同时定位并修改全部调用点，且**任何一处漏改**
    都会让软件 fail-closed（界面出不来 / 出不了图），而不是默默放行。
    这是本地方案里性价比最高的一层；真正不可破的是服务端授权，见文档。
    """
    ok, why = state(app_dir)
    if not ok:
        raise LicenseError(why or "未授权")
