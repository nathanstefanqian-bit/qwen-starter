"""Qwen Image 2.1 Studio — ComfyUI HTTP API 客户端与工作流图构建。

只依赖标准库（urllib），避免给 ComfyUI 环境增加依赖。
"""
from __future__ import annotations

import json
import math
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

# --------------------------------------------------------------------------
# 分辨率计算（与 ResolutionSelector 节点算法一致）
# --------------------------------------------------------------------------

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}

ASPECT_RATIO_LABELS = {
    "1:1 (Square)": "1:1 正方形",
    "2:3 (Portrait Photo)": "2:3 竖版照片",
    "3:2 (Photo)": "3:2 横版照片",
    "3:4 (Portrait Standard)": "3:4 竖版标准",
    "4:3 (Standard)": "4:3 横版标准",
    "9:16 (Portrait Widescreen)": "9:16 竖版宽屏",
    "16:9 (Widescreen)": "16:9 横版宽屏",
    "21:9 (Ultrawide)": "21:9 超宽",
}

LABEL_TO_RATIO = {v: k for k, v in ASPECT_RATIO_LABELS.items()}


def compute_resolution(aspect_ratio: str, megapixels: float, multiple: int = 32) -> tuple[int, int]:
    """按宽高比与百万像素目标算出宽高，结果对齐到 multiple 的整数倍。"""
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
    width = round(w_ratio * scale / multiple) * multiple
    height = round(h_ratio * scale / multiple) * multiple
    return max(width, multiple), max(height, multiple)


# --------------------------------------------------------------------------
# 工作流图构建
# --------------------------------------------------------------------------

def build_graph(
    *,
    mode: str,
    unet_name: str,
    clip_name: str,
    vae_name: str,
    loras: list[dict],
    prompt: str,
    negative_prompt: str,
    ref_resolution: int,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    seed: int,
    images: list[str],
    filename_prefix: str,
) -> dict:
    """构建 API 格式的 prompt 图。mode 为 "i2i" 或 "t2i"。"""
    graph: dict[str, dict] = {}

    graph["1"] = {"class_type": "UNETLoader",
                  "inputs": {"unet_name": unet_name, "weight_dtype": "default"}}
    graph["2"] = {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": clip_name, "type": "qwen_image", "device": "default"}}
    graph["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}}

    lora_inputs: dict = {"model": ["1", 0], "clip": ["2", 0]}
    for index, lora in enumerate(loras, start=1):
        lora_inputs[f"lora_{index}"] = {
            "on": True,
            "lora": lora["name"],
            "strength": float(lora["strength"]),
        }
    graph["15"] = {"class_type": "Power Lora Loader (rgthree)", "inputs": lora_inputs}

    text_inputs: dict = {
        "clip": ["15", 1],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "resolution": int(ref_resolution),
    }
    if mode == "i2i":
        text_inputs["vae"] = ["3", 0]
        for index, filename in enumerate(images[:16], start=1):
            node_id = str(100 + index)
            graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": filename}}
            text_inputs[f"images.image_{index}"] = [node_id, 0]
    graph["5"] = {"class_type": "TextEncodeQwenImage21", "inputs": text_inputs}

    graph["13"] = {"class_type": "EmptyLatentImage",
                   "inputs": {"width": int(width), "height": int(height), "batch_size": 1}}
    graph["7"] = {"class_type": "KSampler", "inputs": {
        "model": ["15", 0],
        "positive": ["5", 0],
        "negative": ["5", 1],
        "latent_image": ["13", 0],
        "seed": int(seed),
        "steps": int(steps),
        "cfg": float(cfg),
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "denoise": float(denoise),
    }}
    graph["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    graph["9"] = {"class_type": "SaveImage",
                  "inputs": {"images": ["8", 0], "filename_prefix": filename_prefix}}
    return graph


# --------------------------------------------------------------------------
# HTTP 客户端
# --------------------------------------------------------------------------

class ComfyError(RuntimeError):
    pass


def _safe_filename(name: str) -> str:
    """上传文件名只保留 ASCII，避免 multipart 头出现非 ASCII 导致编码问题。"""
    stem, ext = os.path.splitext(os.path.basename(name))
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "image"
    return stem + (ext.lower() if ext else ".png")


class ComfyClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    # -- 基础请求 ---------------------------------------------------------

    def _request(self, method: str, path: str, *, data: bytes | None = None,
                 headers: dict | None = None, timeout: float | None = None):
        url = self.base_url + path
        request = urllib.request.Request(url, data=data, method=method,
                                         headers=headers or {})
        return urllib.request.urlopen(request, timeout=timeout or self.timeout)

    def _get_json(self, path: str, timeout: float | None = None):
        with self._request("GET", path, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post_json(self, path: str, payload: dict, timeout: float | None = None):
        body = json.dumps(payload).encode("utf-8")
        with self._request("POST", path, data=body,
                           headers={"Content-Type": "application/json"},
                           timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    # -- 状态与元信息 -----------------------------------------------------

    def is_up(self, timeout: float = 2.0) -> bool:
        try:
            self._get_json("/system_stats", timeout=timeout)
            return True
        except Exception:
            return False

    def object_info(self, node_class: str | None = None) -> dict:
        path = "/object_info" + (f"/{urllib.parse.quote(node_class)}" if node_class else "")
        return self._get_json(path)

    def list_loras(self) -> list[str]:
        """从 LoraLoader 的可选项里取权威的 LoRA 列表。"""
        try:
            info = self.object_info("LoraLoader")
            options = info["LoraLoader"]["input"]["required"]["lora_name"][0]
            return [item for item in options if item and item != "None"]
        except Exception:
            return []

    def list_choices(self, node_class: str, input_name: str) -> list[str]:
        try:
            info = self.object_info(node_class)
            spec = info[node_class]["input"]
            for section in ("required", "optional"):
                if input_name in spec.get(section, {}):
                    value = spec[section][input_name]
                    if isinstance(value, list) and value and isinstance(value[0], list):
                        return list(value[0])
        except Exception:
            pass
        return []

    # -- 上传与执行 -------------------------------------------------------

    def upload_image(self, filepath: str) -> str:
        with open(filepath, "rb") as handle:
            content = handle.read()
        filename = _safe_filename(filepath)
        boundary = "----QwenStudio" + uuid.uuid4().hex
        parts: list[bytes] = []

        def add_field(name: str, value: str) -> None:
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
            )

        add_field("type", "input")
        add_field("overwrite", "true")
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n".encode("utf-8")
        )
        parts.append(content)
        parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(parts)

        with self._request("POST", "/upload/image", data=body,
                           headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}) as response:
            result = json.loads(response.read().decode("utf-8"))

        name = result.get("name", filename)
        subfolder = result.get("subfolder") or ""
        return f"{subfolder}/{name}" if subfolder else name

    def queue_prompt(self, graph: dict) -> str:
        payload = {"prompt": graph, "client_id": self.client_id}
        try:
            result = self._post_json("/prompt", payload)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise ComfyError(f"ComfyUI 拒绝了工作流（HTTP {error.code}）：{_format_validation_error(detail)}") from error
        if "prompt_id" not in result:
            raise ComfyError(f"提交失败：{result}")
        return result["prompt_id"]

    def history_entry(self, prompt_id: str) -> dict | None:
        """查询单次执行的历史记录；未完成时返回 None。"""
        try:
            return self._get_json(f"/history/{prompt_id}").get(prompt_id)
        except Exception:
            return None

    @staticmethod
    def outputs_or_raise(entry: dict) -> dict | None:
        """已完成返回 outputs，出错抛 ComfyError，仍在跑返回 None。"""
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            raise ComfyError(_format_execution_error(status))
        return entry.get("outputs") or None

    def fetch_image(self, filename: str, subfolder: str = "", image_type: str = "output") -> bytes:
        query = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": image_type})
        with self._request("GET", f"/view?{query}") as response:
            return response.read()

    def interrupt(self) -> None:
        try:
            self._post_json("/interrupt", {})
        except Exception:
            pass


# --------------------------------------------------------------------------
# 错误信息格式化
# --------------------------------------------------------------------------

def _format_validation_error(detail: str) -> str:
    try:
        payload = json.loads(detail)
    except Exception:
        return detail[:600]
    node_errors = payload.get("node_errors") or {}
    lines = []
    for node_id, info in node_errors.items():
        for err in info.get("errors", []):
            message = err.get("message") or err.get("details") or str(err)
            lines.append(f"节点 {node_id}: {message}")
    if lines:
        return "\n".join(lines)
    return payload.get("error", {}).get("message", detail[:600])


def _format_execution_error(status: dict) -> str:
    lines = []
    for message in status.get("messages", []):
        if message and message[0] == "execution_error":
            payload = message[1]
            lines.append(
                f"{payload.get('node_type')} (节点 {payload.get('node_id')}): "
                f"{payload.get('exception_type')}: {payload.get('exception_message')}"
            )
    return "\n".join(lines) or "执行出错，请查看 ComfyUI 控制台日志。"
