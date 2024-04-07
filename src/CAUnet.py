from typing import Dict
import torch
import torch.nn as nn
from .unet import Up, OutConv, DoubleConv, Down
from .CloAttention import EfficientAttention


class CAUnet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(CAUnet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.ea_in_conv = EfficientAttention(base_c)
        # self.ea_down1 = EfficientAttention(base_c)
        self.down1 = Down(base_c, base_c * 2)
        # self.ea_down2 = EfficientAttention(base_c*2)
        self.down2 = Down(base_c * 2, base_c * 4)       
        # self.ea_down3 = EfficientAttention(base_c*4)
        self.down3 = Down(base_c * 4, base_c * 8)

        factor = 2 if bilinear else 1
        # self.ea_down4 = EfficientAttention(base_c*8)
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        # self.ea_up1 = EfficientAttention(base_c * 8 // factor)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        # self.ea_up2 = EfficientAttention(base_c * 4 // factor)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        # self.ea_up3 = EfficientAttention(base_c * 2 // factor)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        # self.ea_up4 = EfficientAttention(base_c)
        self.out_conv = OutConv(base_c, num_classes)
        # self.ea_out_conv = EfficientAttention(num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x1 = self.ea_in_conv(x)
        x1 = self.in_conv(x)
        x1 = self.ea_in_conv(x1)

        # x2 = self.ea_down1(x1)
        x2 = self.down1(x1)
        # x3 = self.ea_down2(x2)
        x3 = self.down2(x2)
        # x4 = self.ea_down3(x3)
        x4 = self.down3(x3)

        # x5 = self.ea_down4(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        # x = self.ea_up1(x)
        x = self.up2(x, x3)
        # x = self.ea_up2(x)
        x = self.up3(x, x2)
        # x = self.ea_up3(x)
        x = self.up4(x, x1)
        # x = self.ea_up4(x)
        
        logits = self.out_conv(x)
        # logits = self.ea_out_conv(logits)

        return {"out": logits}


# 输入 N C HW,  输出 N C H W
if __name__ == '__main__':
    block = CAUnet(in_channels=3,num_classes=2,bilinear=False,base_c=32)
    total_trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    input = torch.rand(1, 3, 128, 128)
    output = block(input)["out"]
    print(input.size(), output.size())