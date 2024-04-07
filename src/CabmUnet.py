from .unet import UNet
from typing import Dict
import torch
from .CABM import CBAM

class CabmUnet(UNet):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(CabmUnet, self).__init__(in_channels,num_classes,bilinear,base_c)
        
        self.CabmBlock = CBAM(base_c)

        

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x1 = self.CabmBlock(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}
    
if __name__ == '__main__':
    block = CabmUnet(in_channels=3,num_classes=2,bilinear=False,base_c=32)
    total_trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    input = torch.rand(1, 3, 128, 128)
    output = block(input)["out"]
    print(input.size(), output.size())