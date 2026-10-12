MNIST FINN 網路整理與硬體節點映射

本套件是對你上傳的 mnist_finn.onnx 實際執行轉換後的結果。

內容：
- mnist_finn.onnx：原始輸入模型。
- prepare_mnist_hw.py：可重跑的整理與驗證程式。
- verified/01_tidy.onnx：完成名稱、常數、尺寸及型別整理。
- verified/02_streamlined.onnx：縮放與偏移合併進閾值，卷積展開為 Im2Col/MatMul。
- verified/03_hw_layers.onnx：映射成 FINN 硬體抽象節點的混合圖。
- verified/report.json：各階段驗證與節點清單。
- verified/input.npy：18 筆合成輸入；不是原先的 22 筆驗證資料。
- verified/*output.npy：各階段輸出。

結果：
整理、streamlining、硬體節點映射均通過 Python 功能比對。
最大絕對誤差 9.5367431640625e-07，18 筆 argmax 全部一致。
這不是 MNIST 準確率評估、C++ 模擬或 RTL 模擬。

硬體抽象節點：
ConvolutionInputGenerator x2、MVAU x4、StreamingMaxPool x2。
仍有 MultiThreshold x1、Transpose x2、Flatten x1、Mul x1、Add x1。
輸入浮點量化與最終分數縮放/偏移仍在一般運算節點中。
為維持硬體抽象節點的 Python 模擬，保留卷積到全連接的 Transpose/Flatten。
資料流分區、後端選擇及 Flatten 消除要在下一階段處理。
此檔案尚非可直接合成的完整硬體資料流分區，沒有生成 HLS、RTL 或 bitstream。

在 Ubuntu 重跑：
1. 解壓縮後進入本資料夾。
2. 啟用先前 FINN 環境：
   source ~/projects/mnist_cnn/.venv-finn/bin/activate
3. 執行：
   python prepare_mnist_hw.py --model mnist_finn.onnx --output-dir local_check
   local_check 必須是尚未存在的新資料夾。

以你先前的 22 筆資料重跑（路徑請依實際位置調整）：
   python prepare_mnist_hw.py --model mnist_finn.onnx --input /absolute/path/export_qonnx/input.npy --expected /absolute/path/export_qonnx/expected_output.npy --output-dir local_check_22

--expected 會先檢查原 FINN 模型與 QAT 參考輸出。
各轉換階段則與原 FINN 模型輸出比較，rtol=1e-4、atol=1e-5。
任何一階段失敗都會停止，report.json 保留原因。

依賴：沿用 FINN 及其配套 QONNX 環境。
此階段會載入 FINN 硬體節點，需安裝 FINN 使用的 PyVerilator fork。
若錯誤底層顯示 ModuleNotFoundError: pyverilator，可在 FINN 環境執行：
   python -m pip install "git+https://github.com/maltanar/pyverilator.git@ce0a08c20cb8c1d1e84181d6f392390f846adbd1"
Ubuntu 如缺 tkinter，需安裝與 Python 版本對應的 tkinter 套件。
使用官方 FINN Docker 通常已具備這些依賴。

本次來源版本：
FINN: 39f0c9a6b7675f62d47390fbf9a591707bcbac9b
QONNX: fd61cfeebbdaba351abf7e9d54cd785d7776fa4f

下一步需要選定 FPGA part、HLS/RTL 後端與工具版本，再進行資料流分區、
並行度設定、C++ 模擬及硬體生成。
