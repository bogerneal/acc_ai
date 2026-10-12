# MNIST FINN 硬體準備

2026-09-30 新增：[AUP-ZU3 W4A4 HLS／RTL 基準流程](aup_zu3_w4a4/README.md)。已完成 HLS 特化、4 個資料寬度轉換器、18 筆 Python 參考比對及 12 個節點原始碼生成；缺少 vendor 工具，尚未完成 HLS 綜合與 RTL 驗證。

2026-09-28 新增：[W4A4 子圖逐層規格與 PE／SIMD 候選](core_specs_w4a4/README.md)。已產生 8 層規格、217 組 MVAU 整除候選與 3 套探索配置；未指定晶片，未修改原父子圖。

2026-09-28 新增：[W4A4 硬體資料流分區](dataflow_partition_w4a4/README.md)。已建立正式父圖與 8 節點硬體子圖；18 筆輸入經 Python 驗證副本比對原始 03 模型，最大誤差為 0，尚未生成 RTL。

2026-09-28 新增：[W4A4 FC 邊界轉換與驗證](fc_boundary_w4a4/README.md)。已產生 `fc_boundary_w4a4/verified_local/04_fc_boundary.onnx`；原始 03 模型保持不變，驗證範圍與執行方式見該文件。

更新日期：2026-09-27。本文依目前的 `prepare_mnist_hw.py`、`README.txt` 與三份 `report.json` 整理；此次僅建立文件，未重新執行轉換。

本目錄將 FINN-ONNX 整理、streamline 並映射成 FINN 硬體抽象節點，同時逐階段比對模型輸出。原始準備程式截至抽象節點；新增的 `aup_zu3_w4a4` 已生成 HLS 原始碼，尚未產生綜合後 RTL 或 FPGA bitstream。

## 目錄位置

目前有兩層同名目錄，本 README 放在外層：

```text
projects/
├── mn/                              # FP32／QAT 訓練、測試及 QONNX→FINN 轉換
└── mnist_hw_prepare/
    ├── README.md                    # 本文件
    └── mnist_hw_prepare/
        ├── prepare_mnist_hw.py
        ├── mnist_finn.onnx
        ├── README.txt
        ├── verified/
        ├── local_check/
        └── local_check_2/
```

下列程式執行範例均先進入內層：
`/home/za_warudo/projects/mnist_hw_prepare/mnist_hw_prepare`。

## 與 .pth、W×A×、RTL 的關係

```text
mnist_qat.py：W×A× 的 best_qat.pth
    ↓ export
QONNX → cleaned QONNX
    ↓ convert_mnist_finn.py
FINN-ONNX
    ↓ prepare_mnist_hw.py（本目錄）
01_tidy → 02_streamlined → 03_hw_layers
    ↓ 已實作於各獨立子資料夾
FC 邊界轉換／資料流分區／HLS 後端特化／保留基準平行度
    ↓
HLS 原始碼（已生成）
    ↓ 待工具鏈就緒
RTL／IP、FIFO 與串接、RTL 驗證、布局繞線、bitstream
```

本程式的輸入是 `.onnx`，不直接讀取 `.pth`，也沒有 `--wbits`／`--abits` 參數。量化設定應由上游 QAT 匯出並保留在模型中；此處依圖中的資料型別和量化資訊做轉換。

`03_hw_layers.onnx` 是硬體抽象節點與一般算子並存的模型，並非 RTL。現有報告沒有逐層權重／activation 位元表，也沒有來源 checkpoint 識別資訊，不能僅由 `mnist_finn.onnx` 這個檔名認定它屬於哪一組 W×A×，或認定它對應 `mn/` 最新訓練結果。

各組量化實驗應各自匯出 FINN-ONNX，使用獨立輸出目錄，並保留來源 checkpoint、量化設定與後續硬體設定。W×A× 描述權重與 activation 精度，不代表累加器、輸入與輸出皆為相同位元。

## 程式與原始檔案

| 檔案 | 用途 |
|---|---|
| [prepare_mnist_hw.py](mnist_hw_prepare/prepare_mnist_hw.py) | 命令列入口；整理圖、streamline、映射硬體抽象節點，逐階段執行 Python 功能比對並保存報告。 |
| [mnist_finn.onnx](mnist_hw_prepare/mnist_finn.onnx) | 本目錄現有的來源 FINN-ONNX；程式不修改此來源檔案。 |
| [README.txt](mnist_hw_prepare/README.txt) | 原有套件說明，包含歷史驗證資訊與環境備註。 |
| `README.md` | 本文件，補充目前目錄狀態、指令與硬體流程界線。 |

### 三個轉換階段

| 階段 | 主要工作 | 通過驗證後的檔案 |
|---|---|---|
| `01_tidy` | 推導形狀與型別、常數折疊、整理節點與張量名稱、移除靜態圖輸入。 | `01_tidy.onnx` |
| `02_streamlined` | 合併可吸收的縮放／偏移、展開卷積成 Im2Col／MatMul、調整資料布局並移除未使用張量。 | `02_streamlined.onnx` |
| `03_hw_layers` | 將可映射的節點轉為 FINN 硬體抽象算子，再整理名稱、布局、形狀與型別。 | `03_hw_layers.onnx` |

程式刻意保留卷積到全連接之間的 Transpose／Flatten，供目前硬體抽象節點的 Python 模擬使用；後續後端準備階段仍須處理此邊界。

## 驗證資料與 batch

預設不讀取 MNIST 資料集，而是以 NumPy seed 42 產生 18 筆合成輸入：

- 8 筆 [0,1) 隨機浮點影像。
- 8 筆整數像素 0～255 除以 255 的影像。
- 1 筆全零、1 筆全一影像。

輸入必須是非空、有限值的 `float32 [N,1,28,28]`。雖然檔案可包含 N 筆樣本，`infer()` 實際以每次 1 筆、batch size 1 執行，最後合併輸出。

也可透過 `--input` 提供自己的 NPY 樣本；若同時提供 `--expected`，程式先比較原始 FINN 模型與該參考輸出。各階段一律與原始 FINN 模型的輸出比較，容許誤差為 `rtol=1e-4`、`atol=1e-5`。

這些樣本沒有用於計算真實標籤準確率。`argmax_agreement=1.0` 只表示轉換前後預測類別一致，不表示 MNIST 辨識率為 100%。完整 10,000 張 MNIST 測試準確率應使用 `mn/` 的 `test` 模式。

## 輸出檔案用途

以下檔名相對於指定的 `--output-dir`：

| 檔案 | 用途 |
|---|---|
| `input.npy` | 本次使用的全部驗證輸入。 |
| `source_output.npy` | 原始 FINN 模型對上述輸入的參考 logits。 |
| `01_tidy_candidate.onnx` | 整理完成、數值驗證前保存的候選圖。 |
| `02_streamlined_candidate.onnx` | Streamlining 完成、數值驗證前保存的候選圖。 |
| `03_hw_layers_candidate.onnx` | 硬體抽象節點映射完成、數值驗證前保存的候選圖。 |
| `01_tidy.onnx` | 整理階段數值驗證通過的模型。 |
| `02_streamlined.onnx` | Streamlining 階段數值驗證通過的模型。 |
| `03_hw_layers.onnx` | 硬體節點映射階段數值驗證通過的模型。 |
| `01_tidy_output.npy` | 整理後模型的輸出。 |
| `02_streamlined_output.npy` | Streamlining 後模型的輸出。 |
| `03_hw_layers_output.npy` | 硬體抽象圖的輸出。 |
| `report.json` | 來源、輸入說明、各階段誤差、argmax 一致率、節點數量、整體通過狀態；失敗時記錄錯誤。全部成功後另列硬體與非硬體節點。 |

候選圖的存在不代表驗證通過。程式於階段比對失敗時停止；已進入執行區段的錯誤會寫入 `report.json`。輸出目錄必須不存在，否則建立目錄時即停止，不覆寫舊結果。

## 目前保存的驗證結果

| 目錄／報告 | 完成狀態 | 最大絕對誤差 | 說明 |
|---|---|---:|---|
| [verified/report.json](mnist_hw_prepare/verified/report.json) | 三階段通過 | 9.5367431640625e-7 | 既有驗證產物，報告來源路徑為先前工作環境；未保存 candidate 檔案。 |
| [local_check/report.json](mnist_hw_prepare/local_check/report.json) | 前兩階段通過，整體失敗 | 已完成階段最高 9.5367431640625e-7 | 載入 `finn.custom_op.fpgadataflow` 失敗，沒有第三階段產物。 |
| [local_check_2/report.json](mnist_hw_prepare/local_check_2/report.json) | 三階段通過 | 9.5367431640625e-7 | 保存三階段 candidate、已通過模型與對應輸出。 |

三份報告均使用上述 18 筆合成輸入；已完成階段的 argmax 一致率皆為 1.0。整理階段最大絕對誤差為 0。

兩份成功報告的最終圖包含：

| 節點類型 | 數量 | 狀態 |
|---|---:|---|
| ConvolutionInputGenerator | 2 | FINN 硬體抽象節點 |
| MVAU | 4 | FINN 硬體抽象節點 |
| StreamingMaxPool | 2 | FINN 硬體抽象節點 |
| MultiThreshold | 1 | 尚留在一般運算圖 |
| Transpose | 2 | 尚留在一般運算圖 |
| Flatten | 1 | 尚留在一般運算圖 |
| Mul | 1 | 尚留在一般運算圖 |
| Add | 1 | 尚留在一般運算圖 |

目前沒有記錄轉換耗時、RTL cycle 數、硬體延遲、吞吐量、LUT／DSP／BRAM 使用量、功耗或能耗。

## 執行方式

使用可正常匯入 FINN、QONNX、NumPy 與 FINN 硬體自訂算子的 Python 環境。此階段本身不需要指定 FPGA part，也不呼叫 HLS 或 FPGA 綜合工具。

```bash
cd /home/za_warudo/projects/mnist_hw_prepare/mnist_hw_prepare

# 啟用你已安裝 FINN 的環境後執行；輸出名稱須尚未存在
python prepare_mnist_hw.py \
  --model mnist_finn.onnx \
  --output-dir check_new
```

使用同一個模型對應的上游驗證資料：

```bash
python prepare_mnist_hw.py \
  --model /absolute/path/to/mnist_finn.onnx \
  --input /absolute/path/to/export_qonnx/input.npy \
  --expected /absolute/path/to/export_qonnx/expected_output.npy \
  --output-dir check_with_reference
```

範例中的絕對路徑須替換為實際位置。`--expected` 必須搭配 `--input`；不得將另一個量化組別或 checkpoint 的參考輸出混用。

| 參數 | 必要性 | 說明 |
|---|---|---|
| `--model` | 必填 | 來源 FINN-ONNX 路徑。 |
| `--output-dir` | 必填 | 新輸出目錄，可包含尚未建立的父目錄。 |
| `--input` | 選填 | 驗證輸入 NPY；未指定則產生 18 筆合成資料。 |
| `--expected` | 選填 | 來源模型應對應的參考輸出 NPY，形狀須與來源模型輸出一致。 |

若出現 `Could not load custom opset finn.custom_op.fpgadataflow`，先查看完整 traceback，確認目前 Python 環境能載入 FINN 自訂算子及其相依套件。`local_check` 只記錄外層錯誤，不能從該 JSON 單獨判定缺少哪個套件。原 `README.txt` 提及 PyVerilator 與 tkinter 相依問題，可作為排查線索。

原 `README.txt` 記錄的歷史來源版本為：

- FINN：`39f0c9a6b7675f62d47390fbf9a591707bcbac9b`
- QONNX：`fd61cfeebbdaba351abf7e9d54cd785d7776fa4f`

這些是歷史環境備註；目前程式未自動將實際執行環境版本寫入報告。

## 下一階段：從抽象圖到 FPGA

本目錄目前未實作以下工作：

1. 決定 FPGA part、工具版本、時脈及 HLS／RTL 後端。
2. 處理一般運算節點與輸入／輸出邊界，建立硬體資料流分區。
3. 配置後端特化、平行度／folding 及記憶體等硬體參數。
4. 產生硬體原始碼，進行適用後端的模擬與 RTL 功能驗證。
5. 綜合、布局繞線與 bitstream 生成，收集資源、時序及上板量測結果。

比較不同 W×A× 時，需明確記錄各組硬體配置及共同約束。此處的 Python 功能驗證不能取代後續 RTL／上板驗證，也不能用來推算 FPGA 的實際耗時或功耗。

