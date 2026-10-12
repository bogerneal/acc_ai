# W4A4 卷積到全連接邊界轉換

本資料夾實作 `RemoveCNVtoFCFlatten()`，將 FC 前的 NHWC→NCHW 轉置吸收到權重排列中，再移除該 Transpose 與 Flatten。來源模型與前一步產物保持不變。此步驟不包含硬體分區、RTL 生成或 FPGA 效能量測。

## 輸入與結果

- 來源：[03_hw_layers.onnx](../mnist_hw_prepare/local_check_2/03_hw_layers.onnx)，使用者確認為 W4A4。
- 輸入：[input.npy](../mnist_hw_prepare/local_check_2/input.npy)，18 筆合成資料，每筆 1×28×28、float32。
- 成功的正式輸出：[04_fc_boundary.onnx](verified_local/04_fc_boundary.onnx)。
- 成功報告：[report.json](verified_local/report.json)。

```text
轉換前：
StreamingMaxPool → Transpose → Flatten → MVAU（512→128）
[1,4,4,32]        [1,32,4,4]   [1,512]

轉換後（供後續硬體準備）：
StreamingMaxPool ────────────────────→ MVAU（512→128）
[1,4,4,32]                            FC 權重已重排
```

最上方的輸入 Transpose 與前處理 MultiThreshold 保留，末端 Mul／Add 也保留。W4A4 量化設定沒有改變。

## 驗證結果與界線

本次成功執行使用 `/home/za_warudo/projects/mnist_cnn/.venv-finn/bin/python`。

| 檢查 | 結果 |
|---|---|
| FC 的 Transpose／Flatten 移除 | 通過 |
| FC 輸入直接接到 pooling 輸出 | 通過 |
| FC 權重符合 NCHW→NHWC 重排公式 | 逐元素完全一致 |
| 其他節點、其他 initializer | 完全不變 |
| 18 筆實際 pooling 輸出的 FC 矩陣乘法 | 轉換前後完全一致 |
| 驗證副本的最終 logits 與來源圖 | 最大絕對誤差 0 |
| 預測類別一致率 | 18/18，100% |
| 來源 ONNX SHA-256 | 執行前後相同 |

本機 FINN 的抽象 MVAU 以 NumPy matmul 執行，不會自動將 [1,4,4,32] 攤平後乘上 [512,128] 權重，因此直接對已移除 Flatten 的硬體準備圖執行原有 Python interpreter 不適用。

為此，程式另建 `verification_only.onnx`，只在 FC 前加入 **NHWC 順序的 Flatten**，不加回原本的 Transpose，也不還原權重。它模擬 NHWC 串流以 512 個元素進入 FC 的排列，供 Python 功能比對。此副本不能作為下一步硬體分區的輸入；正式輸出是 `04_fc_boundary.onnx`。

驗證包括權重重排的精確檢查，以及 18 筆樣本的 FC 乘積與完整輸出比對。這不是直接執行正式硬體圖的驗證、RTL 模擬或完整 MNIST 準確率測試。

## 檔案用途

| 檔案 | 用途 |
|---|---|
| `prepare_fc_boundary.py` | 可重跑的轉換與驗證程式，檢查預期邊界、共享張量限制及結果。 |
| `verified_local/04_fc_boundary_candidate.onnx` | 轉換完成、驗證前保存的候選圖。 |
| `verified_local/04_fc_boundary.onnx` | 所有檢查通過後保存的正式硬體準備圖。 |
| `verified_local/verification_only.onnx` | 含驗證專用 Flatten 的 Python 執行副本。 |
| `verified_local/input.npy` | 此次驗證輸入副本。 |
| `verified_local/source_output.npy` | 原始 03 模型的輸出。 |
| `verified_local/verification_output.npy` | 驗證副本的輸出。 |
| `verified_local/report.json` | 來源／輸入 SHA-256、節點統計、重排檢查、逐筆誤差與總耗時。 |
| `verified/report.json` | 首次失敗紀錄：使用家目錄 `.venv-finn` 時缺少 pyverilator，未完成轉換。 |

目錄名稱不代表通過狀態，請以 `report.json` 的 `passed` 為準。報告的 `total_seconds` 是整個腳本工作流程的秒數，不是 FPGA 推論耗時。

## 重跑指令

在 WSL 執行：

```bash
cd /home/za_warudo/projects/mnist_hw_prepare/fc_boundary_w4a4
source /home/za_warudo/projects/mnist_cnn/.venv-finn/bin/activate

# 預設使用原 local_check_2 的 03 模型與 input.npy
python prepare_fc_boundary.py --output-dir rerun_01
```

`rerun_01` 必須是尚未存在的新目錄。已完成或失敗的目錄均不會被覆寫。

也可以指定來源與輸入：

```bash
python prepare_fc_boundary.py \
  --model ../mnist_hw_prepare/local_check_2/03_hw_layers.onnx \
  --input ../mnist_hw_prepare/local_check_2/input.npy \
  --output-dir rerun_02
```

程式要求恰好一組預期的 Transpose→Flatten→MVAU 邊界；若將已轉換的 04 模型再次當作輸入，會拒絕執行。新資料或其他模型須另外驗證，不能直接套用本次 W4A4 的通過結論。

## 下一步

已在獨立的 [dataflow_partition_w4a4](../dataflow_partition_w4a4/README.md) 資料夾完成分區與 Python 驗證。後續依目標 FPGA、時脈與 HLS／RTL 後端進行特化、平行度配置與硬體驗證；硬體建置尚未執行。

