import torch
import torch.nn as nn
from torchvision import models

class CSRNet(nn.Module):
    """
    CSRNet: VGG16 front-end + dilated conv back-end
    Output: 1xHxW density (stride ~8)
    """
    def __init__(self, load_pretrained_vgg=True):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if load_pretrained_vgg else None)
        features = list(vgg.features.children())

        # VGG front-end: up to conv4_3 (includes 3 maxpools => stride 8)
        self.frontend = nn.Sequential(*features[:23])

        # CSRNet backend (dilated convs)
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128,  64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        # Kendi conv katmanlarını init et (VGG zaten pretrained)
        for m in list(self.backend.modules()) + list(self.output_layer.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
