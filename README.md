# Qwen Image 2.1 Studio

一体化桌面软件：双击即用，后台自动拉起 ComfyUI 服务，前台打开本地操作界面。
上传参考图即走**图生图**，不传图自动走**文生图**；加速 LoRA、NSFW LoRA 全部内置，界面上只保留分辨率、采样步数、CG 画风开关等常用参数。

> **本仓库只包含「工具层」源码与配置**，不含模型权重，也不含 ComfyUI 本体与 Python 运行环境。
> 模型请按下文「[模型下载](#模型下载魔搭-modelscope)」从魔搭（ModelScope）获取。
> 开箱即用的完整整合包（含 ComfyUI + Python 运行环境 + 模型）另行分发。

---

## 目录结构

```
qwen-starter/
├─ app.py                    # 主程序：界面 + 启动器 + 生成流程
├─ comfy_client.py           # ComfyUI HTTP API 客户端与工作流图构建（仅用标准库）
├─ qlic.py                   # 授权校验：机器码 / Ed25519 验签 / 本地授权文件读写
├─ qactivate.py              # 激活界面
├─ config.json               # 模型名、端口、LoRA 规则等配置
├─ 启动.bat                   # 入口批处理（调用 python\python.exe app.py）
├─ 启动.exe                   # 带图标的便携入口，跨盘移动仍可用
├─ app.ico                   # 应用图标
├─ 启动器源码/启动器.cs          # 启动.exe 的源码（C#，可用 csc 重新编译）
├─ keys/public_key.pem       # 验签公钥（可公开；对应私钥只在作者手里）
├─ docs/LICENSE_DESIGN.md    # 卡密授权系统设计文档
├─ requirements.txt          # 自建 Python 环境时的依赖清单
└─ 发卡器/                    # 作者专用发卡工具
   ├─ issue_card.py          # 发卡逻辑 + 界面
   ├─ 发卡器.bat              # 双击启动
   ├─ 使用说明.md             # 发卡 / 换机流程说明
   └─ keys/public_key.pem    # 公钥副本
```

---

## 模型下载（魔搭 ModelScope）

把下列文件放进 **ComfyUI\models\** 对应的子目录即可（文件名需与 `config.json` 中的配置一致）。

| 放置位置（相对 `ComfyUI\models\`） | 魔搭下载页 |
|---|---|
| `diffusion_models\qwen_image_2_1_bf16.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://modelscope.cn/models/Comfy-Org/Qwen-Image-2.1)　或　官方 [Qwen/Qwen-Image-2.1](https://modelscope.cn/models/Qwen/Qwen-Image-2.1) |
| `text_encoders\qwen3vl_8b_int8_convrot.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://modelscope.cn/models/Comfy-Org/Qwen-Image-2.1) → `text_encoders/` |
| `vae\qwen_image_2.1_vae_bf16.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://modelscope.cn/models/Comfy-Org/Qwen-Image-2.1) → `vae/` |
| `loras\Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors` | [Viggle/Qwen-Image-2.1-viggle-turbo](https://modelscope.cn/models/Viggle/Qwen-Image-2.1-viggle-turbo)（4 步加速 LoRA） |

命令行下载示例（魔搭官方工具，支持断点续传）：

```bash
pip install modelscope
modelscope download --model Comfy-Org/Qwen-Image-2.1 --local_dir ./_dl
```

> **文件名对不上怎么办？**
> Comfy-Org 仓库里扩散模型叫 `qwen_image_2.1_bf16.safetensors`（带点），本软件配置里写的是 `qwen_image_2_1_bf16.safetensors`（带下划线）。
> 二者是同一个权重，**要么把下载到的文件改名**，**要么改 `config.json` 里的 `unet_name`**，保证一致即可。

### 本仓库不提供的模型

以下 LoRA 属于第三方训练内容，需自行获取（多为社区模型站）：

| 文件 | 说明 |
|---|---|
| `loras\QWEN-IMAGE-2-1_CG画风_c1-st6000.safetensors` | CG 画风风格 LoRA，界面上那个「CG 画风」开关用的就是它 |
| `loras\NSFW Qwen Lora.safetensors` | 文生图模式下自动加载的 NSFW LoRA |
| `loras\Qwen-Image-2.1 NSFW Image Edit.safetensors` | 图生图模式下自动加载的 NSFW Image Edit LoRA |

---

## 运行环境

本工具层需要与下面两样东西配合才能跑起来，缺一不可：

1. **ComfyUI 本体**（含其自身依赖）—— 放在程序目录的 `ComfyUI\` 下；
2. **Python 运行环境** —— 放在程序目录的 `python\` 下，`启动.bat` 会调用 `python\python.exe app.py`。

放好之后，双击 **`启动.exe`**（或 `启动.bat`）即可。
`config.json` 里的 `instance_dir` 留空，就表示「使用程序自身所在目录」，整个文件夹拷到任何盘 / 目录都能直接运行。

自建 Python 环境时，装好依赖即可：

```bash
python -m pip install -r requirements.txt
```

---

## 配置说明（config.json）

| 字段 | 作用 |
|---|---|
| `instance_dir` | 留空 = 用程序自身所在目录；只有要指向别的 ComfyUI 实例时才填绝对路径 |
| `port` | ComfyUI 服务端口，默认 `8188` |
| `unet_name` / `clip_name` / `vae_name` | 三个主模型文件名，需与 `ComfyUI\models\` 下的实际文件名一致 |
| `sampler_name` / `scheduler` | 采样器与调度器，固定在配置里，界面上不暴露 |
| `always_loras` | 常开的内置 LoRA（按文件名关键词匹配），默认是 `viggle-turbo` 加速 LoRA，强度 1.0 |
| `mode_loras` | 按模式二选一的内置 LoRA：`t2i` 用 `NSFW Qwen Lora`（0.5），`i2i` 用 `NSFW Image Edit`（0.8） |
| `style_lora_keyword` / `style_lora_strength` | 界面上唯一保留的 LoRA 开关，默认匹配 `CG画风`，强度 0.8 |

---

## 授权与发卡

软件带**卡密授权**：一码一机，未激活时只显示激活页。

- 客户端：`qlic.py`（机器码 + Ed25519 验签）+ `qactivate.py`（激活界面）+ `keys/public_key.pem`（公钥）。
- 发卡：`发卡器\` 目录，作者专用。
- 设计文档：`docs/LICENSE_DESIGN.md`。

> ⚠️ **本仓库必须保持私有。**
> `发卡器\keys\private_key.pem`（发卡签名私钥）**刻意没有放进本仓库**，`.gitignore` 也已把它和 `license.dat` 列为忽略项。
> 但 `issue_card.py` 等发卡代码仍在此仓库内 —— 一旦仓库转为公开，等于把发卡方案连同流程一起交出去。**请勿把本仓库设为 Public。**

---

## 常见问题

**双击没反应 / 闪退？**
多为系统代理劫持了本地回环请求。`app.py` 已在 `import gradio` 之前把 `127.0.0.1`、`localhost` 写进 `NO_PROXY`，一般无需处理；若仍异常，先关掉系统代理再试。

**换台电脑要重新激活？**
是。机器指纹取「主板 UUID + CPU ID + 系统盘卷序列号」，换机 / 重装系统 / 换主板都会变，需要重新发卡。

**出图很慢？**
确认加速 LoRA（`viggle-turbo`）已加载、步数按 4~8 步设置；显存吃紧时降低分辨率，并使用分块解码（VAEDecodeTiled）。
