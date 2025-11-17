import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.mobilenetv3 import InvertedResidual
from torchvision.ops.misc import SqueezeExcitation  # <-- REAL SE class used inside MobileNetV3

# ============================================================
# Swish Activation
# ============================================================
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# ============================================================
# Squeeze-and-Excitation Block
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4, debug=False):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.act = Swish()
        self.sigmoid = nn.Sigmoid()
        self.debug = debug

        # storage variable for debug output
        self.last_scale = None

    def forward(self, x):
        scale = torch.mean(x, dim=(2,3), keepdim=True)
        scale = self.act(self.fc1(scale))
        scale = self.sigmoid(self.fc2(scale))

        # store for later use (convert to 2D later)
        if self.debug:
            self.last_scale = scale.detach().cpu().squeeze().numpy()  # shape (C,)
            
        return x * scale



# ============================================================
# Custom MobileNetV3 Block
# ============================================================
class MBConvCustom(nn.Module):
    def __init__(self, in_c, out_c, expansion=4, stride=1):
        super().__init__()
        hidden = in_c * expansion
        self.use_res = (stride == 1 and in_c == out_c)

        self.block = nn.Sequential(
            nn.Conv2d(in_c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            Swish(),

            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            Swish(),

            SEBlock(hidden),

            nn.Conv2d(hidden, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        out = self.block(x)
        return out + x if self.use_res else out

# ============================================================
# ✅ Pretrained MobileNetV3 for Federated Learning
# ============================================================




class MobileNetFL(nn.Module):
    def __init__(self, num_classes=38, se_debug=False, debug_replace=True):
        super().__init__()

        # ✅ Load pretrained MobileNetV3
        self.base = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )

        # ✅ Replace ALL SE layers in MobileNetV3 with custom SEBlock
        replaced = 0

        def replace_se_layers(module):
            nonlocal replaced
            for name, child in module.named_children():
                # If this layer is torchvision's SE block, replace it
                if isinstance(child, SqueezeExcitation):
                    in_c = child.fc2.out_channels  # get channel count
                    setattr(module, name, SEBlock(in_c, debug=se_debug))
                    replaced += 1
                else:
                    replace_se_layers(child)

        replace_se_layers(self.base)

        if debug_replace:
            print(f"[MobileNetFL] ✅ Replaced {replaced} SE blocks with custom SEBlock")

        # ✅ Replace final classifier layer
        in_feat = self.base.classifier[-1].in_features
        self.base.classifier[-1] = nn.Linear(in_feat, num_classes)

    def forward(self, x):
        return self.base(x)
