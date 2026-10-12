import torch
import torch.nn as nn
import brevitas.nn as qnn

class QuantizedMNIST(nn.Module):
    def __init__(self, weight_bit_width=4, act_bit_width=4):
        super(QuantizedMNIST, self).__init__()
        
        # 1. 輸入量化層 (FINN 轉換時通常需要明確的輸入量化)
        self.quant_inp = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)
        
        # 2. 第一層卷積層 (Conv -> ReLU -> MaxPool)
        self.conv1 = qnn.QuantConv2d(
            in_channels=1, 
            out_channels=16, 
            kernel_size=3, 
            padding=1, 
            weight_bit_width=weight_bit_width,
            bias=False # FINN 硬體實現中通常省略 bias 以節省資源
        )
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3. 第二層卷積層 (Conv -> ReLU -> MaxPool)
        self.conv2 = qnn.QuantConv2d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=3, 
            padding=1, 
            weight_bit_width=weight_bit_width,
            bias=False
        )
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 4. 全連接層 (Fully Connected)
        # MNIST 圖片大小為 28x28，經過兩次 2x2 MaxPool 後縮小為 7x7
        self.fc1 = qnn.QuantLinear(
            in_features=32 * 7 * 7, 
            out_features=10, # 10 個數字類別 (0~9)
            weight_bit_width=weight_bit_width, 
            bias=False
        )

    def forward(self, x):
        # 確保輸入被量化
        x = self.quant_inp(x)
        
        # 特徵提取
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 攤平張量以進入全連接層
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        
        return x

# 測試模型結構與輸出形狀
if __name__ == '__main__':
    # 初始化 4-bit 模型
    model = QuantizedMNIST(weight_bit_width=4, act_bit_width=4)
    print("模型結構建立成功！\n", model)
    
    # 建立一個虛擬的 MNIST 輸入 (Batch Size=1, Channels=1, Height=28, Width=28)
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    
    print(f"\n輸入形狀: {dummy_input.shape}")
    print(f"輸出形狀: {output.shape} (對應 10 個類別的預測分數)")