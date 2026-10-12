# W4A4 子圖逐層規格（自動產生）

| 節點 | 類型 | 輸入形狀／型別 | 輸出形狀／型別 | 權重 | 現有 PE／SIMD |
|---|---|---|---|---|---|
| ConvolutionInputGenerator_0 | ConvolutionInputGenerator | [1, 28, 28, 1] / UINT8 | [1, 24, 24, 25] / UINT8 |  |  / 1 |
| MVAU_0 | MVAU | [1, 24, 24, 25] / UINT8 | [1, 24, 24, 16] / UINT4 | INT4 | 1 / 1 |
| StreamingMaxPool_0 | StreamingMaxPool | [1, 24, 24, 16] / UINT4 | [1, 12, 12, 16] / UINT4 |  | 1 /  |
| ConvolutionInputGenerator_1 | ConvolutionInputGenerator | [1, 12, 12, 16] / UINT4 | [1, 8, 8, 400] / UINT4 |  |  / 16 |
| MVAU_1 | MVAU | [1, 8, 8, 400] / UINT4 | [1, 8, 8, 32] / UINT4 | INT4 | 1 / 1 |
| StreamingMaxPool_1 | StreamingMaxPool | [1, 8, 8, 32] / UINT4 | [1, 4, 4, 32] / UINT4 |  | 1 /  |
| MVAU_2 | MVAU | [1, 4, 4, 32] / UINT4 | [1, 128] / UINT4 | INT4 | 1 / 1 |
| MVAU_3 | MVAU | [1, 128] / UINT4 | [1, 10] / INT32 | INT4 | 1 / 1 |

## MVAU PE／SIMD 整除候選

候選值須取笛卡兒積；滿足整除並不保證任何後端或晶片皆可實現。

| 節點 | MW×MH | PE 候選（MH 因數） | SIMD 候選（MW 因數） |
|---|---|---|---|
| MVAU_0 | 25×16 | [1, 2, 4, 8, 16] | [1, 5, 25] |
| MVAU_1 | 400×32 | [1, 2, 4, 8, 16, 32] | [1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 200, 400] |
| MVAU_2 | 512×128 | [1, 2, 4, 8, 16, 32, 64, 128] | [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] |
| MVAU_3 | 128×10 | [1, 2, 5, 10] | [1, 2, 4, 8, 16, 32, 64, 128] |

## 範例配置比較

MAC 迭代數 = 向量數 × (MW/SIMD) × (MH/PE)，僅為理想化運算工作量，不是硬體量測週期。

| 配置 | MVAU | PE | SIMD | 理想 MAC 迭代數／影像 |
|---|---|---:|---:|---:|
| serial | MVAU_0 | 1 | 1 | 230400 |
| serial | MVAU_1 | 1 | 1 | 819200 |
| serial | MVAU_2 | 1 | 1 | 65536 |
| serial | MVAU_3 | 1 | 1 | 1280 |
| small | MVAU_0 | 4 | 1 | 57600 |
| small | MVAU_1 | 4 | 4 | 51200 |
| small | MVAU_2 | 4 | 4 | 4096 |
| small | MVAU_3 | 2 | 4 | 160 |
| wider | MVAU_0 | 8 | 5 | 5760 |
| wider | MVAU_1 | 8 | 8 | 12800 |
| wider | MVAU_2 | 8 | 8 | 1024 |
| wider | MVAU_3 | 5 | 8 | 32 |

2D StreamingMaxPool 的 PE 不作為候選變數；本機 FINN 僅於 1D pooling 支援 PE folding。
輸入生成器 SIMD 必須整除輸入通道數；不使用尚未選定後端的 parallel_window。
各方案的串流寬度相容性與可能需要的 DWC 已列在 proposal JSON；沒有插入 DWC／FIFO。
