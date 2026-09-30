# FINN 硬體子圖 → HLS／RTL：AUP-ZU3 基準流程

更新日期：2026-09-30

## 1. 目標與目前狀態

以 AUP-ZU3（Zynq UltraScale+ ZU3EG）作為暫定目標，從既有 FINN 硬體子圖生成各層 HLS C++／RTL，接著進行 HLS 綜合、IP 整合與模擬。先完成離線流程，不需要實體開發板。

- 使用者表示已取得硬體子圖；預期輸入為 `05_dataflow_core.onnx`。
- 目前儲存庫尚未包含該 ONNX、外部權重、轉換程式或工具版本資訊。
- 本文件是後續執行計畫，不代表 HLS／RTL 已生成或驗證通過。
- ONNX 圖片僅供閱讀，不能代替真正的 `.onnx` 模型。
- 本次不重新訓練 CNN／QAT，也不重做已驗證的 QONNX 匯出。

## 2. 基準設定

| 項目 | 初始設定 |
|---|---|
| 參考開發板 | AUP-ZU3 |
| 晶片 | XCZU3EG-2SFVC784E |
| 工具使用的 part 名稱 | `xczu3eg-sfvc784-2-e`，執行前確認工具已安裝此器件 |
| HLS 目標週期 | 10 ns（100 MHz） |
| 系統目標週期 | 初始同為 10 ns |
| PE／SIMD | 若確認既有四個 MVAU 皆為 1／1，先沿用；以實際子圖為準 |
| 後端 | 每個節點明確選擇支援的 HLS 或 RTL 實作，保存實際選擇 |
| 自動吞吐量調整 | 基準實驗不以 target FPS 自動改變並行度 |
| FIFO | 明確記錄深度與插入策略，驗證無死鎖後固定 |
| 實驗輸出 | 原始碼、IP／RTL、模擬結果、資源估計、分段耗時 |

10 ns 是綜合約束，不是已達成的硬體頻率。HLS 估計與實作後時序也不是同一項結果。

先以晶片 part 進行核心生成。完整板級建置時，再確認該 FINN 版本的 AUP-ZU3 board key、板卡檔案及平台設定，不臆測 board 字串。

## 3. 整體流程

| 階段 | 工作 | 完成標準 |
|---|---|---|
| A | 準備輸入與鎖定環境 | 模型可載入、參數齊全、版本可重現 |
| B | 檢查／特化硬體節點 | 每個節點有可用後端 |
| C | 設定 folding 與資料流介面 | PE、SIMD、串流寬度及 FIFO 策略明確 |
| D | Python／適用的 C++ 驗證 | 與硬體子圖參考輸出一致 |
| E | 生成 HLS C++／RTL 原始碼 | 所有節點原始碼與工程設定完整 |
| F | HLS 綜合，取得各層 RTL／IP | 綜合成功、報告無未處理錯誤 |
| G | FIFO／寬度轉換器完善與 IP 串接 | 完整核心可整合 |
| H | 核心 RTL 模擬 | 數值、輸出數量及串流握手正確 |
| I（延伸） | 板級整合與 bitstream | 實作完成並滿足時序 |
| J（未來） | 上板測試 | MNIST 準確率、吞吐量及延遲實測 |

這是工作里程碑，不是適用所有 FINN 版本的固定 API 呼叫順序。FIFO 自動定深可能先建立暫時 IP／模擬，再重新生成；須依鎖定版本的 builder 相依關係安排。

## 4. A：輸入與環境準備

需要收集：

- 真正的硬體子圖 ONNX，以及它引用的 external data／權重檔案。
- 父圖：若有 `StreamingDataflowPartition`，保留其子圖路徑與外部前後處理。
- 產生硬體子圖的 Python 程式與設定。
- FINN commit、QONNX／Brevitas／Python 版本、容器映像識別。
- 與此 FINN commit 相容的 Vitis HLS、Vivado，以及對應授權／器件支援。
- 相同測試輸入與參考輸出；記錄 tensor 名稱、shape、layout、datatype、數值範圍。
- 主機 CPU、RAM、作業系統、工具執行緒數、WSL／容器配額。

基準模型與建置設定都保存 SHA-256。執行前先做單一小型節點的工具鏈檢查，再跑整個模型。

## 5. B：檢查硬體子圖與選擇後端

1. 確認輸入是硬體核心子圖，不是只有 partition 節點的父圖。
2. 確認 shape、量化 datatype、initializer 與外部資料可解析。
3. 逐節點列出 op type、後端、輸入／輸出維度及 PE／SIMD。
4. 若節點仍為抽象硬體節點，使用該版本支援的 `SpecializeLayers`／`step_specialize_layers` 選擇 HLS／RTL。
5. 已特化的節點不重複套用前端轉換，也不重複建立資料流分區。
6. 檢查尚未映射的普通 ONNX 節點；不能假設每個 Add、Mul、Reshape 都可直接生成硬體。

重要區分：

- 選 HLS 後端的節點：生成 C++，再由 Vitis HLS 綜合成 RTL／IP。
- 選 RTL 後端的節點：使用 RTL 模板／實作，不經 HLS 綜合。
- MVAU 是否內含門檻運算，須檢查 `noActivation` 與門檻參數，不能只憑節點名稱判斷。
- 最終可能是 HLS 與 RTL 混合核心；需記錄實際後端，以免量化實驗混入後端差異。

## 6. C／D：設定並行度與先行驗證

- 先固定目前各層 PE／SIMD；確認其符合矩陣維度及後端限制。
- 記錄權重記憶體模式、乘法資源偏好、threshold 配置及串流寬度。
- 確認相鄰節點介面需要的資料寬度轉換與 FIFO。
- 自動 FIFO 定深和最終插入順序依 FINN 版本執行；不要為了縮短流程任意略過。
- 特化／folding 前後，用同一批核心輸入比對數值。
- 對支援的 HLS 節點執行 C++ 模擬；直接 RTL 節點則使用適用的 RTL 模擬途徑。

核心輸入不一定是原始 MNIST 的 NCHW 浮點圖片。若父圖先完成量化、transpose 或 reshape，驗證資料必須先經同樣前處理。核心輸出若還要經過外部 Mul／Add，評估完整分類結果時也必須接回。

整數核心輸出以逐元素完全相同為優先標準；若比較跨越浮點前後處理，另訂並記錄 rtol／atol、最大絕對誤差、分類一致率與樣本數。

## 7. E：生成 HLS C++／RTL 原始碼

對應概念：`step_hw_codegen`／`PrepareIP`。

- 傳入目標 part 與時脈。
- HLS 節點生成頂層 C++、權重／門檻參數及綜合腳本；相依 FINN HLS 函式庫也需可取得。
- RTL 節點填入模板並收集其 RTL 與包裝檔。
- 保存節點生成目錄及更新後 ONNX；檢查引用的路徑有效。
- 備份完整原始碼與相依設定，不只保留最上層 C++。

此階段完成表示「已生成硬體原始碼」，不代表 HLS 節點已完成綜合，也不代表整個 CNN 已形成單一可直接上板的 Verilog。

## 8. F：HLS 綜合至 RTL／IP

對應概念：`step_hw_ipgen`／`HLSSynthIP`。

- 對 HLS 節點執行 Vitis HLS，產出 RTL／IP 與綜合報告。
- 直接 RTL 節點不需經 HLS 綜合，但後續仍需整合、模擬與 FPGA 綜合。
- 收集每層 latency、II、時脈估計及 LUT／FF／DSP／BRAM 估計。
- 失敗時保存失敗節點、日誌、命令與版本；失敗／超時不能填成耗時 0。

HLS 資源估計不等於 Vivado 實作後數字；不同階段的報告分欄記錄。

## 9. G／H：串接與 RTL 驗證

- 按實際 builder 流程插入／調整 FIFO 與資料寬度轉換器，必要時重新生成受影響 IP。
- 使用 `CreateStitchedIP`／對應 builder 步驟整合串流核心。
- 使用相容的模擬器執行核心 RTL 模擬。
- 檢查連續多筆輸入、輸出數量、順序、ready／valid 握手與背壓，設定超時以偵測死鎖。
- 比對核心參考結果；接回父圖前後處理後，再量測端到端分類一致率。
- 保存測試樣本、結果摘要，以及失敗案例的波形。

這一階段通過後，才稱為「完整資料流核心已完成 RTL 模擬驗證」。

## 10. 生成時間與量化實驗

以同一主機、工具版本、part、時脈、folding 與後端政策比較各量化組。每次使用乾淨建置目錄，固定工具並行工作數，排除安裝及下載時間。

| 指標 | 計時邊界 |
|---|---|
| T_prepare | 載入硬體子圖至特化／folding 等前置工作完成 |
| T_codegen | 各節點 HLS／RTL 原始碼生成 |
| T_ipgen | HLS 綜合及層級 IP 生成 |
| T_stream | FIFO 定深、寬度轉換與因此觸發的重建 |
| T_stitch | 核心 IP 串接 |
| T_verify | Python／C++／RTL 驗證，獨立報告 |
| T_impl | FPGA 綜合與布局繞線，延伸實驗獨立報告 |

各區間不得重複計時。自動 FIFO 定深中的暫時綜合／模擬歸入 T_stream，並記錄子項；若某項在該版本穿插執行，使用事件計時累加，不用檔案編號推估耗時。

主要區分兩種終點：

- 各層 RTL／IP：前置工作、原始碼與層級 IP 生成所需耗時。
- 可模擬的完整核心：再包含串流調整、必要重建與串接耗時。

每個已指定 checkpoint 重建 3 次，報告中位數與範圍。分類準確率與轉換一致率分開；現有少量樣本比對不能代替完整 MNIST 測試集準確率。

## 11. 預計產物與完成清單

以下為建議專案整理名稱，不是 FINN 保證的原生輸出檔名：

| 路徑 | 用途 |
|---|---|
| models/05_dataflow_core.onnx | 原始硬體核心 |
| configs/aup_zu3_baseline.json | part、時脈及建置設定 |
| configs/folding_baseline.json | 各層 PE／SIMD 與記憶體設定 |
| scripts/build_hls_rtl.py | 核對版本後實作的建置入口 |
| verification/ | 核心測試輸入與 golden outputs |
| build/aup_zu3/<run_id>/ | 每次獨立建置輸出 |
| reports/ | 計時、驗證與資源摘要 |

- [ ] 實際 ONNX、參數與父圖齊全
- [ ] FINN／工具版本及器件支援已確認
- [ ] 每個硬體節點已明確指定後端
- [ ] 並行度及串流設定已保存
- [ ] HLS／RTL 原始碼生成完成
- [ ] 各層 RTL／IP 生成完成
- [ ] 完整核心串接與 RTL 驗證通過
- [ ] 各階段耗時、資源及數值結果已保存

大型建置中間檔不預設提交 Git；程式、設定、文件與小型結果摘要優先版本控制。後續落實腳本前先取得模型及實際環境資訊，避免把未核實的 API 組合當成可直接執行的程式。

## 12. 後續上板範圍

核心完成後，才加入 AUP-ZU3 板級平台、PS／PL 介面、DMA、時脈／reset 與驅動。執行 Vivado 綜合、布局繞線及 bitstream 生成，檢查完整設計資源和時序。最後取得實體板子後，再量測實際延遲、吞吐量與端到端準確率。

## 13. 官方參考

以下為查閱時的官方文件；實際步驟以專案鎖定的 FINN commit 為準。

- [AUP-ZU3 板卡與晶片](https://xilinx.github.io/AUP-ZU3/overview.html)
- [FINN 環境與平台支援](https://finn.readthedocs.io/en/latest/getting_started.html)
- [FINN Network Preparation](https://finn.readthedocs.io/en/latest/nw_prep.html)
- [FINN Hardware Build and Deployment](https://finn.readthedocs.io/en/latest/hw_build.html)
- [FINN builder 流程](https://finn.readthedocs.io/en/latest/command_line.html)
- [FINN builder 步驟原始碼](https://github.com/Xilinx/finn/blob/main/src/finn/builder/build_dataflow_steps.py)
