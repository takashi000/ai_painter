# ai_painter

Diffusionモデルを使用したお絵かきツールです。

## アップデート

2025/10/15：初回リリース(Stable Diffusion 1.5のみ対応)\
2025/10/19：0.0.2リリース(メニューバー整理、undo, redo機能追加)\
2025/10/19：0.0.3リリース(ステップごとにVAEデコード画像の出力機能追加(ControlNet未対応))
2025/10/20：0.0.4リリース(ステップごとにVAEデコード画像の出力機能追加(ControlNet対応))

## インストール

```sh
uv venv -p 3.12 .venv
```

```sh
git clone https://github.com/takashi000/ai_painter.git
```

```sh
uv pip install ai_painter
```

## 使い方

起動方法

```sh
python -m ai_painter
```

その他\
モデルはHaggingFaceHubなどからダウンロードして使用できます。
ControlNetやUperNetモデルはHaggingFaceHubのモデルIDを指定することで、
自動的にダウンロードされます。
Checkpoint、Lora、VAEはsafetensorsファイルのみ対応しています。\
ControlNetはネットワーク上のモデルIDまたはローカルフォルダからの読み込みに対応しています。\
UperNetモデルはControlNetの画像領域解析のために使用されます。HaggingfaceHubなどのモデルIDもしくはローカルにダウンロードしたフォルダパスを指定してください。

## 動作確認環境

OS: Windows 11\
CPU: Intel Core Ultra 7 265K\
GPU: NVIDIA Geforce RTX 5050 VRAM 8G\
メモリ:32GB\
NVIDIA CUDA Tool Kit 12.9
