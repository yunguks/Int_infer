from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from .layers import *


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 100, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes, bias=True),
        )
    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

class INTVGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 100, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = features
        # self.avgpool = IntPool(7,stride=7,mode=1)
        self.classifier = nn.Sequential(
            IntLinear(25088, 4096),
            QuantReLU(),
            IntLinear(4096, 4096),
            QuantReLU(),
            IntLinear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def cuda(self):
        for layer in self.features.modules():
            if 'Int' in str(type(layer)):
                layer.cuda()

        for layer in self.classifier.modules():
            if 'Int' in str(type(layer)):
                layer.cuda()


class FLOATVGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 100, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = features
        # self.avgpool = FLOATPool(7,stride=7,mode=1)
        self.classifier = nn.Sequential(
            FLOATLinear(25088, 4096),
            nn.ReLU(),
            FLOATLinear(4096, 4096),
            nn.ReLU(),
            FLOATLinear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def cuda(self):
        for layer in self.features.modules():
            if 'FLOAT' in str(type(layer)):
                layer.cuda()

        for layer in self.classifier.modules():
            if 'FLOAT' in str(type(layer)):
                layer.cuda()


def make_layers(cfg: List[Union[str, int]]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 4
    for vs in cfg:
        for v in vs:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU()]
            in_channels = v
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


def int_make_layers(cfg: List[Union[str, int]]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 4
    for vs in cfg:
        for v in vs:
            v = cast(int, v)
            conv2d = IntConv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, QuantReLU()]
            in_channels = v
        layers += [IntPool(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


def float_make_layers(cfg: List[Union[str, int]]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 4
    for vs in cfg:
        for v in vs:
            v = cast(int, v)
            conv2d = FLOATConv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU()]
            in_channels = v
        layers += [FLOATPool(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "D": [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512],[512, 512, 512]],
    # "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}


def vgg16(cfg: str = "D", batch_norm: bool=False, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg]), **kwargs)
    return model

def int_vgg16(cfg: str = "D", batch_norm: bool=False, **kwargs: Any) -> INTVGG:
    model = INTVGG(int_make_layers(cfgs[cfg]), **kwargs)
    return model

def float_vgg16(cfg: str = "D", batch_norm: bool=False, **kwargs: Any) -> FLOATVGG:
    model = FLOATVGG(float_make_layers(cfgs[cfg]), **kwargs)
    return model