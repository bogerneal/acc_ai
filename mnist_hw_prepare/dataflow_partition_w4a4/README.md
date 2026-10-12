# W4A4 硬體資料流分區

建立日期：2026-09-28。本資料夾從已完成 FC 邊界處理的 04 模型建立 FINN 資料流分區，保存父圖與硬體核心子圖。已實際執行結構檢查與 Python 功能驗證，尚未進行 FPGA 後端特化、硬體生成或 RTL 模擬。

## 分區結果

```text
05_dataflow_parent.onnx
global_in：FLOAT32 [1,1,28,28]
    ↓ MultiThreshold（輸入量化）
    ↓ Transpose（NCHW→NHWC）
    ↓ UINT8 [1,28,28,1]
StreamingDataflowPartition ── model → 05_dataflow_core.onnx
    ↓ INT32 [1,10]
    ↓ Mul
    ↓ Add
global_out：FLOAT32 [1,10]
```

核心子圖包含 8 個 FINN 硬體抽象節點：

| 節點 | 數量 |
|---|---:|
| ConvolutionInputGenerator | 2 |
| MVAU | 4 |
| StreamingMaxPool | 2 |

核心不含一般 Transpose／Flatten／Mul／Add。四個 MVAU 的權重為 INT4，隱藏 activation 沿用 UINT4；第一層輸入為 UINT8、最後輸出為 INT32。這是 W4A4 模型的硬體核心，不代表輸入、累加器與輸出皆為 4-bit。

## 檔案與用途

| 檔案 | 用途 |
|---|---|
| [partition_w4a4.py](partition_w4a4.py) | 執行分區、檢查節點與權重不變，並驗證父圖＋子圖的功能。 |
| [verified/05_dataflow_parent.onnx](verified/05_dataflow_parent.onnx) | 正式父圖，保留主機前後處理及一個 StreamingDataflowPartition。 |
| [verified/05_dataflow_core.onnx](verified/05_dataflow_core.onnx) | 正式硬體子圖，後續後端特化與硬體建置的輸入。 |
| `verified/05_dataflow_parent_candidate.onnx` | 驗證前保存的父圖，指向候選子圖。 |
| `verified/05_dataflow_core_candidate.onnx` | 驗證前保存的子圖。 |
| `verified/partitions/` | FINN CreateDataflowPartition 直接生成的中間分區模型。正式產物以 05 檔名為準。 |
| `verified/verification_parent.onnx` | 指向驗證專用子圖的父圖，可用於本機 Python 執行。 |
| `verified/verification_core.onnx` | 在 FC 前額外加上 NHWC Flatten 的驗證副本，不用於硬體建置。 |
| `verified/input.npy` | 18 筆完整模型輸入，形狀 [18,1,28,28]。 |
| `verified/reference_output.npy` | 原始 03 模型的 logits，形狀 [18,10]。 |
| `verified/partition_output.npy` | 分區驗證父圖的 logits，形狀 [18,10]。 |
| `verified/core_input.npy` | 父圖前處理後的核心輸入，形狀 [18,28,28,1]。 |
| `verified/core_expected_output.npy` | 驗證子圖的整數語意输出，形狀 [18,10]，供後續核心驗證參考。 |
| [verified/report.json](verified/report.json) | 來源與 SHA-256、分區數、節點統計、介面型別、逐樣本數值比對與腳本總耗時。 |

NPY 的核心輸入／輸出是 Python 模擬張量；UINT8／INT32 指 ONNX 的數值型別語意，不代表檔案已打包成 FPGA 串流。後續還須依 folding、資料型別與串流寬度打包。

父圖的 StreamingDataflowPartition 使用子圖的**絕對路徑**。搬移目錄到另一部電腦或更改路徑後，須重新執行腳本或更新節點的 model 屬性；只有父圖檔案不足以執行子圖。

## 驗證方式與結果

來源：
- [04_fc_boundary.onnx](../fc_boundary_w4a4/verified_local/04_fc_boundary.onnx)
- 對照模型：[03_hw_layers.onnx](../mnist_hw_prepare/local_check_2/03_hw_layers.onnx)
- 驗證資料：[前一階段 input.npy](../fc_boundary_w4a4/verified_local/input.npy)

程式確認：

1. 恰好建立一個 StreamingDataflowPartition。
2. 子圖僅包含預期的 8 個硬體節點，節點屬性／連線及權重與 04 模型完全一致。
3. 父圖前後處理節點與參數不變。
4. 核心介面為 UINT8 [1,28,28,1] → INT32 [1,10]。
5. 18 筆合成輸入經父圖＋子圖的 Python 驗證副本執行後，與原始 03 模型輸出一致。
6. 來源模型、對照模型與輸入檔案的 SHA-256 執行前後不變。

本次結果：**passed=true；18 筆；最大絕對誤差 0；argmax 一致率 100%。**

本機抽象 MVAU 的 Python 實作不會將四維 FC 輸入自動攤平，因此驗證子圖加入單一 NHWC Flatten 適配；正式 05 子圖完全沒有此節點。此驗證覆蓋分區的父子圖呼叫及前後處理，但不是直接執行正式 05 核心的 RTL 驗證。argmax 一致率也不是完整 MNIST 分類準確率。

`total_seconds` 是腳本分區、檢查、Python 驗證與存檔的總耗時，不是 FPGA latency，也不是純分區時間。

## 重跑方式

在 WSL 使用已具備 FINN 與 PyVerilator 相依套件的環境：

```bash
cd /home/za_warudo/projects/mnist_hw_prepare/dataflow_partition_w4a4
source /home/za_warudo/projects/mnist_cnn/.venv-finn/bin/activate
python partition_w4a4.py --output-dir rerun_01
```

`--output-dir` 必須是新目錄，不會覆寫既有結果。預設來源為上節列出的模型與樣本；若需替換：

```bash
python partition_w4a4.py \
  --model ../fc_boundary_w4a4/verified_local/04_fc_boundary.onnx \
  --reference ../mnist_hw_prepare/local_check_2/03_hw_layers.onnx \
  --input ../fc_boundary_w4a4/verified_local/input.npy \
  --output-dir rerun_02
```

此脚本針對目前 W4A4 的結構與介面設有檢查，不能不經調整便套用其他模型。轉換或比對失敗時，報告保存錯誤，成功檔案只在全部檢查通過後寫入。

## 下一步

使用 `verified/05_dataflow_core.onnx`，指定 FPGA part 與時脈，執行 SpecializeLayers、PE／SIMD／folding 設定及硬體生成。後續仍須驗證硬體核心與父圖介面的等價性，才能進入 RTL 模擬與上板量測。

