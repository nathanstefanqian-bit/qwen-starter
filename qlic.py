"""授权校验（客户端）。—— 内部工具，请勿随意改动。

职责：
  1. 计算本机"机器码"（稳定硬件指纹）
  2. 用内嵌公钥校验用户卡密（Ed25519 签名）
  3. 读写本地授权文件，实现"激活一次、永久免输"

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

# ---- 公钥（可公开；私钥只在作者发卡器里）----
_PUB_PEM = """-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAj4jOD695u0Q+LCEWbUujJ1rW2SIukC3IY0RAlE/50S0=
-----END PUBLIC KEY-----"""

_MACHINE_CACHE = None
_B32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"


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
    try:
        out = subprocess.run(["cmd", "/c", "vol", "C:"], capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for line in (out.stdout or "").splitlines():
            if "Serial" in line or "序列号" in line:
                return line.split("is")[-1].strip() if "is" in line else line.split(":")[-1].strip()
    except Exception:
        pass
    return ""


def machine_fingerprint() -> bytes:
    """稳定硬件因子 -> SHA-256。故意不取 MAC / 网卡 / 外设，减少误伤。"""
    parts = [
        _ps("(Get-CimInstance Win32_ComputerSystemProduct).UUID"),
        _ps("(Get-CimInstance Win32_Processor).ProcessorId"),
        _vol_serial(),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).digest()


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
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    return load_pem_public_key(_PUB_PEM.encode())


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
    p = _lic_path(app_dir)
    if not os.path.exists(p):
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("card")) and _check(data["card"])
    except Exception:
        return False


def save_activation(app_dir: str, card: str) -> bool:
    if not _check(card):
        return False
    try:
        with open(_lic_path(app_dir), "w", encoding="utf-8") as f:
            json.dump({"card": card.strip().upper(), "ts": int(time.time())}, f)
        return True
    except Exception:
        return False
