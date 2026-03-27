# TNM 模型調整指南

訓練結束後，可依驗證集表現與需求調整以下項目。

---

## 一、查看訓練結果

- **日誌**：`outputs/train.log`（每個 Epoch 的 train/val loss、Exact-Match）
- **設定與最佳 EM**：`outputs/train_config.json`（含 `best_val_em`）
- **最佳權重**：`outputs/best_tnm.pt`

```bash
tail -30 outputs/train.log
cat outputs/train_config.json
```

---

## 二、超參數調整（`train.py` 頂部常數）

| 參數 | 目前 | 建議調整方向 |
|------|------|----------------|
| **EPOCHS** | 5 | 若 val EM 仍上升 → 改 8～10；若早過擬合 → 改 3～4 |
| **LR** | 2e-5 | 收斂太慢 → 試 3e-5 或 5e-5；不穩/震盪 → 試 1e-5 |
| **BATCH_SIZE** | 2 | GPU 記憶體夠 → 4 或 8（訓練更快、梯度較穩） |
| **STRIDE** | 128 | 報告很長、想更多 overlap → 64；想省算力 → 256 |
| **MAX_LENGTH** | 512 | 一般維持 512；顯存不足可改 384（需一併改 CONTENT_LEN 邏輯） |

---

## 三、損失與類別不平衡

- **M01 權重**：已用訓練集計算並寫入 `train_config.json`。若 M1 仍明顯欠擬合，可手動放大 M1 權重（在 `compute_m01_class_weights` 或對 `criterion_m` 的 `weight` 乘係數）。
- **總損失**：目前為 `Loss_T + Loss_N + Loss_M`。若某一頭（如 M）特別難學，可改為加權，例如：
  `loss = loss_t + loss_n + 2.0 * loss_m`

---

## 四、模型結構（`TNMModel`）

- **Dropout**：預設 0.1。過擬合可調高到 0.2～0.3；欠擬合可調低到 0.05。
- **分類頭**：目前為單一線性層。若驗證 EM 卡住，可改為兩層 MLP，例如：
  `Linear(hidden, 256) → ReLU → Dropout → Linear(256, num_class)`
- **Pooling**：目前為對所有 chunks 的 [CLS] 做 mean pooling。可嘗試只取前 K 個 chunk 的 mean，或加上 attention 權重（依 chunk 重要性加權平均）。

---

## 五、資料與切分

- **Stratify**：目前以 T14 做分層。若希望 N 或 M 分布更一致，可改為以多欄位 stratify（例如 `stratify=df["T14"].astype(str)+df["N03"].astype(str)` 或依資料量簡化）。
- **驗證比例**：目前 85% 中 20% 為驗證。可改 `VAL_RATIO_OF_TRAIN_VAL`（例如 0.15 或 0.25）。

---

## 六、再訓練與評估

- 修改常數或模型後，重新執行：
  ```bash
  .venv/bin/python train.py
  ```
- 若要用 CPU 避免 GPU OOM：
  ```bash
  PYTORCH_DEVICE=cpu .venv/bin/python train.py
  ```
- 訓練完成後，可用測試集評估：載入 `best_tnm.pt`，在 `test_df` 上算 Exact-Match 與各任務的 accuracy/F1。
