# 安裝支援 CUDA 的 PyTorch（.venv）

目前若為 CPU 版 PyTorch（`2.x.x+cpu`），訓練會很慢。請在 **已連接 GPU（如 GX10）的環境** 下執行：

## 1. 確認系統 CUDA 版本（可選）

```bash
nvidia-smi
```

看右上角 **CUDA Version**（例如 12.1 / 12.4 / 13.0）。PyTorch 的 cu121/cu124/cu126 可與較新驅動相容，不必完全一致。

## 2. 在 .venv 中安裝 CUDA 版 PyTorch

**若為 ARM64 (aarch64，例如 GX10)**：請使用 cu124（cu121 無 aarch64 wheel）：

```bash
cd /home/iih/task6
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
```

下載約 2.3GB，需稍候。若原本為 CPU 版，請先解除安裝再裝：

```bash
.venv/bin/pip uninstall torch -y
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**若為 x86_64**：可擇一使用 cu121 / cu124 / cu126：

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
# 或 cu124 / cu126
```

## 3. 驗證

```bash
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

應顯示 `CUDA: True` 與 GPU 型號。

## 4. 執行訓練

```bash
.venv/bin/python train.py
```

程式會自動使用 `device='cuda:0'`，模型與所有張量都會在 GPU 上執行。
