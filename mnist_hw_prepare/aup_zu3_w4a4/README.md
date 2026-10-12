# W4A4 核心 → AUP-ZU3 HLS／RTL 基準流程

更新：2026-09-30。依使用者提供的 [流程文件](provenance/AUP_ZU3_HLS_RTL_FLOW.md)，從既有 `05_dataflow_core.onnx` 開始。實際輸入與 FINN／Python 版本已在本機找到，文件中的「尚未包含模型」是規劃當時狀態，不適用於此目錄。

**目前已生成 HLS C++ 與綜合 Tcl，尚未取得 HLS 綜合後 RTL／IP，也未完成完整核心 RTL 模擬。** 全部原始父子圖保持不變。

## 已執行與待完成

| 文件階段 | 本次結果 |
|---|---|
| A：輸入與環境 | 備份核心、父圖、測試資料、上游腳本；保存 SHA-256、Python 套件版本、CPU／RAM／WSL 配額。工具版本及器件支援仍待檢查。 |
| B：後端特化 | 8 個原始硬體節點全部明確選 HLS，完成 `SpecializeLayers(part)`。 |
| C：並行度與介面 | 保留四個 MVAU 的 PE=1／SIMD=1；插入 4 個 DWC，檢查相鄰串流寬度與每張圖元素數匹配。FIFO 尚未完成。 |
| D：先行驗證 | 18 筆既有合成資料，轉換前後 Python 參考語意與核心 golden output 逐元素相同。C++／RTL 模擬未執行。 |
| E：原始碼生成 | 12 個 HLS 節點均生成 C++、Tcl 及所需權重／閾值檔；相依 HLS 函式庫已鎖定。 |
| F：HLS → RTL／IP | 已提供有版本、part、timeout 與產物檢查的入口；當前缺少 WSL Vivado／Vitis HLS，尚未綜合。 |
| G：FIFO 與 IP 串接 | 待工具鏈可用後實作及驗證，不把現有預設 FIFO 深度當成已驗證設定。 |
| H：完整核心 RTL 模擬 | 尚未執行。必須檢查多筆輸入、背壓、順序、數量、數值與死鎖。 |
| I／J：bitstream／上板 | 不在目前完成範圍。 |

## 基準設定

| 項目 | 設定 |
|---|---|
| 參考板／核心 part | AUP-ZU3／`xczu3eg-sfvc784-2-e`；工具是否安裝該 part 尚待確認 |
| HLS 約束 | 10 ns，僅是目標，不是實測 100 MHz |
| FINN commit | `39f0c9a6b7675f62d47390fbf9a591707bcbac9b` |
| finn-hlslib commit | `16e5847a5e3ef76cffe84c8fad2f010d593457d3`，取自該 FINN 的 `fetch-repos.sh` |
| 預期 vendor 工具 | 本機鎖定 FINN 文件指定的 Vivado／Vitis HLS 2022.2；尚未確認可執行版本、授權及 part |
| 執行環境 | 既有 WSL Python venv，非 FINN Docker；尚未證明等同官方完整工具環境 |
| 後端政策 | 核心 8 節點及 4 個 DWC 均指定 HLS；不讓 FINN 自動挑選不同後端 |
| 四個 MVAU | PE=1／SIMD=1、`internal_decoupled` 權重、乘法資源 `auto`、累加器 INT32 |
| SWU SIMD | 第一層 1、第二層 16，沿用來源 |
| 自動 folding／位寬最小化 | 關閉；目前保留來源位寬與平行度 |
| FIFO | 屬性預設深度 2 已列入報告；**尚未插入、定深或驗證無死鎖** |

目前 FINN 的 `pynq_part_map` 沒有 AUP-ZU3 board key。本階段以 part 生成層級核心原始碼，不臆測 board 字串；後續板級整合必須另行核實。

全部選 HLS 是這次可重現的基準政策，不表示 FINN 只能使用 HLS。後續切換 SWU 或 MVAU 到 RTL 後端應另建實驗，避免把後端改變混入位寬比較。

## 新增的 4 個資料寬度轉換器

| 位置 | 每拍串流位元數 |
|---|---|
| MVAU_0 → MaxPool_0 | 4 → 64 |
| SWU_1 → MVAU_1 | 64 → 4 |
| MVAU_1 → MaxPool_1 | 4 → 128 |
| MaxPool_1 → MVAU_2 | 128 → 4 |

DWC 只重新打包同一串數值，不改變量化精度。2D MaxPool 的介面一次攜帶全部通道，不能由表中的 PE=1 推論其串流只有 4-bit。

`layers_with_dwc.json` 的週期欄是 FINN 分析模型輸出，不是模擬或實測值。此版本 DWC 的 `get_exp_cycles()` 回傳 0 是未提供估算的預設值，**不代表資料轉換不需要週期**；MaxPool 估算也有可能偏低。不得把這些欄位直接相加當作端到端延遲。

## 檔案用途

```text
aup_zu3_w4a4/
├── README.md
├── configs/
│   ├── aup_zu3_baseline.json       # part、時脈、版本與後端政策
│   └── folding_baseline.json      # 實際繼承的逐層設定快照，不是自動調參命令
├── models/
│   ├── 05_dataflow_core.onnx      # 原始核心的完整副本
│   ├── 05_dataflow_parent_original.onnx # 原父圖逐位元組副本
│   └── 05_dataflow_parent.onnx    # partition 路徑指向本目錄核心
├── verification/                 # 核心 UINT8 輸入、INT32 golden、父圖輸入與結果
├── provenance/                   # 使用者流程文件、上游轉換腳本
├── scripts/
│   ├── build_hls_rtl.py           # 前置、介面檢查、Python 參考驗證、原始碼生成
│   ├── check_part.tcl            # Vivado 檢查器件是否存在
│   └── synthesize_layers.py      # 工具就緒後執行層級綜合；尚未驗證成功路徑
├── deps/
│   ├── finn-hlslib/              # 下載並固定 commit 的相依函式庫
│   └── finn_runtime/             # custom_hls、finn-rtllib 快照及 hlslib 連結
├── build/aup_zu3/<run_id>/       # 每次獨立輸出，不覆蓋既有建置
└── reports/                      # 版本、雜湊、驗證、分段計時與重複實驗摘要
```

每次建置的主要 ONNX：

| 檔案 | 意義 |
|---|---|
| `06_specialized.onnx` | 8 個 HLS 特化節點 |
| `07_widths.onnx` | 加入 4 個 DWC，尚未完成 FIFO／串接 |
| `08_codegen.onnx` | 附帶 12 個節點原始碼目錄的模型 |
| `verification_only_*.onnx` | 僅供 Python 參考驗證，包含 FC reshape adapter，不能當硬體輸入 |
| `generated/code_gen_ipgen_*/top_*.cpp` | 個別節點的 HLS C++ |
| `generated/code_gen_ipgen_*/hls_syn_*.tcl` | 個別節點的 HLS 綜合與 IP 匯出腳本 |
| `report.json` | 實際執行狀態與各計時事件 |

ONNX 中產物路徑是絕對路徑。搬移本資料夾或換主機時應重新建置，不可直接假設舊路徑有效。`internal_decoupled` 模式的 MVAU 還需要後續記憶體串流包裝；僅有 MVAU C++ 不等於完整核心。

## 重現目前已完成的步驟

以下指令在 **Ubuntu WSL 終端**執行。既有 venv 路徑如下：

```bash
cd /home/za_warudo/projects/mnist_hw_prepare/aup_zu3_w4a4
/home/za_warudo/projects/mnist_cnn/.venv-finn/bin/python scripts/build_hls_rtl.py
```

預設使用新時間戳目錄；指定已存在的 run ID 會拒絕覆蓋。

只重建前置／原始碼階段三次並統計中位數與範圍：

```bash
/home/za_warudo/projects/mnist_cnn/.venv-finn/bin/python scripts/build_hls_rtl.py --run-id experiment_02 --repeat 3
```

下載依賴在計時前已完成，不包含於建置耗時。本次 `baseline_r1`～`baseline_r3` 為三個乾淨原始碼建置；`baseline_initial` 是開發時第一輪，不能混入正式三次統計。

計時獨立記錄 `T_prepare`、`T_stream_interfaces`、`T_verify_python`、`T_codegen`。這裡的 `T_stream_interfaces` **只有 DWC 插入與介面檢查**，不是文件中包含 FIFO 自動定深與重建的完整 `T_stream`。

本次三次實測摘要見 [baseline_summary.json](reports/baseline_summary.json)：

| 指標 | 中位數（秒） | 最小～最大（秒） |
|---|---:|---:|
| 前置與特化 | 0.149 | 0.133～0.389 |
| DWC 插入／介面檢查 | 0.011 | 0.010～0.012 |
| Python 參考驗證 | 29.003 | 27.634～29.081 |
| HLS 原始碼生成 | 1.773 | 1.592～1.792 |

這些時間不包含 HLS 綜合。下一階段實際環境檢查結果為 [blocked 報告](build/aup_zu3/baseline_r1/ipgen_20260930T114551_023864Z/report.json)，原因為 WSL PATH 缺少 Vivado／Vitis HLS。

## 工具就緒後的下一步

目前 WSL PATH 未找到 `vitis_hls`、`vivado`、`g++`、`verilator`；常見安裝目錄也未發現 vendor 工具。Python 中存在 pyverilator 套件，不代表已安裝 Verilator 執行檔。

先提供／安裝匹配的工具鏈，載入其 `settings64.sh`，確認器件、授權與 C++ 模擬依賴。**不能用更新版 FINN 的工具需求取代這個鎖定版本的要求。** 不需要實體板子。

F 階段入口（先檢查版本與 part，再以一個小型 DWC 做工具鏈綜合檢查，成功後逐層綜合）：

```bash
/home/za_warudo/projects/mnist_cnn/.venv-finn/bin/python scripts/synthesize_layers.py \
  --run-dir build/aup_zu3/baseline_r1
```

缺少工具時回傳 exit code 2 並保存 `blocked` 報告；沒有綜合耗時時使用 `null`，不填 0。每層有 timeout、原始日誌、工具命令及產物存在檢查。成功後才產生 `09_ipgen.onnx`，並讀取 HLS XML 報告中的 latency／II／資源估計。

此入口沒有替代 C++ 數值驗證，成功路徑目前也未在 vendor 工具上執行過。工具就緒後先完成 C++ 檢查，再執行 F；F 成功後才繼續 FIFO 策略、IP 串接及多筆 RTL 模擬。需要依鎖定 FINN builder 實作這些後續階段，不能直接把 `08_codegen.onnx` 當完整 FPGA 工程。

## 驗證界線

- 18 筆是上游既有合成樣本，不是完整 MNIST 測試集，不報告分類準確率。
- 核心輸入為父圖前處理後的 `float32` 容器，數值必須是 UINT8 整數，形狀 `[N,28,28,1]`；逐筆 batch=1 驗證。
- 核心輸出 `[N,10]` 使用 INT32 語意；仍以浮點 ONNX 容器表示，這不表示將來硬體用浮點乘法。
- 特化後驗證使用「去掉後端標記」的抽象 Python 參考副本；FC 前加 Flatten 只解決 Python 抽象執行器的形狀要求。
- 這證明參數、打包前後的功能語意保持一致，**不證明生成的 C++／RTL 已執行正確**。
- 核心成功驗證後仍需接回父圖前後處理；本次未重新量測完整 MNIST 準確率、板上耗時或功耗。
- 三次原始碼生成不等於三次完整硬體建置。完整生成時間、FIFO 深度、RTL 延遲及實際資源仍待後續完成。
