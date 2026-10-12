# W4A4 子圖規格與 PE／SIMD 候選

2026-09-28。維持原父子圖配置，不指定 FPGA part、後端或時脈；本資料夾只讀取模型並產生分析文件，沒有輸出修改後的 ONNX，也沒有執行 SpecializeLayers、SetFolding、綜合或 RTL 模擬。

來源：[05_dataflow_core.onnx](../dataflow_partition_w4a4/verified/05_dataflow_core.onnx)。

## 查看結果

| 檔案 | 用途 |
|---|---|
| [SPECIFICATION.md](reports_verified/SPECIFICATION.md) | 可直接閱讀的 8 層規格、MVAU 候選因數及三套配置比較。 |
| [layer_specs.json](reports_verified/layer_specs.json) | 完整形狀、型別、節點明示屬性、權重／閾值尺寸、目前配置、輸入 SHA-256 與檢查結果。 |
| [layer_specs.csv](reports_verified/layer_specs.csv) | 適合試算表查看的逐層摘要。 |
| [mvau_candidates.csv](reports_verified/mvau_candidates.csv) | 四個 MVAU 共 217 組 PE／SIMD 整除組合；不是整網組合數。 |
| [profile_comparison.csv](reports_verified/profile_comparison.csv) | 三套配置、四個 MVAU 的理想 MAC 迭代數及串流寬度比較。 |
| [serial_proposal.json](reports_verified/serial_proposal.json) | MVAU PE／SIMD 與輸入生成器 SIMD 均選 1 的探索方案；2D pooling 不改動。 |
| [small_proposal.json](reports_verified/small_proposal.json) | 分別選取不超過 4 的最大合法因數。 |
| [wider_proposal.json](reports_verified/wider_proposal.json) | 分別選取不超過 8 的最大合法因數。 |
| [analyze_core.py](analyze_core.py) | 可重跑的唯讀分析程式。 |

配置是探索候選，不表示符合某個晶片的資源或時序限制，也未套用到原模型。serial 並非現有圖的完整原樣配置：現有第二個輸入生成器 SIMD=16；serial 建議 SIMD=1。原始配置另存於規格表。

## 候選規則與限制

MVAU：

- PE 必須整除 MH（輸出通道／神經元數）。
- SIMD 必須整除 MW（每個輸出累加的輸入數）。
- 理想 MAC 迭代數／影像 = numInputVectors 各維乘積 × (MW/SIMD) × (MH/PE)。
- PE×SIMD 是平行 MAC lane 數的描述，不是 DSP 使用數。
- 串流寬度以未補齊的 payload bits 表示：輸入 SIMD×輸入位元；輸出 PE×輸出位元。後續實作可能另外補齊介面位寬。

輸入生成器：

- SIMD 必須整除 IFMChannels。
- 第一層 IFMChannels=1，通道 SIMD 只能取 1；不妨礙後續 MVAU 從展開後的 25 元素視窗選擇 SIMD=5 或 25，但須處理串流寬度與供資料速率。
- 本分析假設視窗依序輸出，不啟用後端專屬 parallel_window。抽象類別的輸出寬度方法依賴後端實作，所以此處明確使用 SIMD×位元計算該假設下的寬度。

Pooling：

- 目前兩層均為 2D StreamingMaxPool。本機 FINN 的 PE folding 僅適用 1D pooling。
- 因此不替 2D pooling 任意調 PE，介面按全通道輸出計算；即使原節點含 PE=1，也不表示實際僅一個通道寬。

各 proposal 的 stream_edges 列出相鄰節點的 payload 位寬、是否需要轉寬，以及寬度比是否為整數。整數比只是初步檢查，不是 DWC 實作已驗證；沒有插入 DWC 或 FIFO，也沒有檢查整網動態吞吐。

## 如何理解結果

例如 MVAU_0 的 MW=25，SIMD 因數為 1、5、25。因此 small 的上限 4 仍只能選 SIMD=1；wider 的上限 8 則選 SIMD=5。MVAU_3 的 MH=10，其 PE 候選只有 1、2、5、10，不能直接指定 4 或 8。

MAC 迭代數只是理想化運算工作量，未包含視窗生成、pooling、pipeline、記憶體頻寬、FIFO、反壓及控制成本。不能將各層數值直接加總當作網路延遲，也不能除以任意時脈就宣稱實際 FPS。

accumulator_type 是模型屬性或 FINN 預設值，不是最後綜合出的累加器位寬。未選晶片的階段不提供 LUT、DSP、BRAM、功耗或實際 latency 預測。

proposal JSON 包含說明與分析欄位，**不是可直接傳给 FINN folding_config_file 的設定檔**。日後特化會改變節點名稱／型別，須將 node_config 對應到特化後節點，確認後端限制，再產生可套用的 folding 設定。另需處理 DWC、FIFO 與完整功能驗證。

## 重跑

在 WSL：

```bash
cd /home/za_warudo/projects/mnist_hw_prepare/core_specs_w4a4
source /home/za_warudo/projects/mnist_cnn/.venv-finn/bin/activate
python analyze_core.py --output-dir reports_02
```

輸出目錄須尚未存在。可用 --model 指定相同結構的來源子圖；目前腳本會確認八節點結構與 MVAU INT4 權重，不應將其視為任意 CNN 的通用分析器。

## 本次檢查

已成功執行分析，並核對：

- 8 層，4 個 MVAU。
- 217 組 MVAU 候選：15＋90＋80＋32。
- 3 套探索配置，所有 PE／SIMD 均滿足各自整除條件。
- 來源子圖 SHA-256 執行前後相同。
- 未指定 FPGA part、backend 或 clock，未修改父圖／子圖。

此階段只產生規格與候選，不重新執行數值推論；前一階段的功能驗證結果請見資料流分區文件。

