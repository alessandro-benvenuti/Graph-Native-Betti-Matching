from collections import OrderedDict
from typing import Any, Mapping, Sequence, Type, Union

import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.squeeze_and_excitation import (
    SEBottleneck,
    SEResNetBottleneck,
    SEResNeXtBottleneck,
)
from monai.networks.layers.factories import Act, Conv, Norm, Pool


class SENet(nn.Module):
    """Squeeze-and-excitation ResNet feature encoder adapted from MONAI."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        block: Type[Union[SEBottleneck, SEResNetBottleneck, SEResNeXtBottleneck]],
        layers: Sequence[int],
        groups: int,
        reduction: int,
        inplanes: int = 128,
        downsample_kernel_size: int = 3,
        input_3x3: bool = True,
        strides: Sequence[int] = (1, 2, 2, 2),
    ) -> None:

        super().__init__()

        relu_type: Type[nn.ReLU] = Act[Act.RELU]
        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        if len(strides) != 4:
            raise ValueError("strides must contain one value per residual stage")

        self.inplanes = inplanes
        self.spatial_dims = spatial_dims

        layer0_modules: list[tuple[str, Any]]

        if input_3x3:
            layer0_modules = [
                (
                    "conv1",
                    conv_type(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
                ),
                ("bn1", norm_type(num_features=64)),
                ("relu1", relu_type(inplace=True)),
                ("conv2", conv_type(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)),
                ("bn2", norm_type(num_features=64)),
                ("relu2", relu_type(inplace=True)),
                (
                    "conv3",
                    conv_type(in_channels=64, out_channels=inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                ),
                ("bn3", norm_type(num_features=inplanes)),
                ("relu3", relu_type(inplace=True)),
            ]
        else:
            layer0_modules = [
                (
                    "conv1",
                    conv_type(
                        in_channels=in_channels, out_channels=inplanes, kernel_size=7, stride=2, padding=3, bias=False
                    ),
                ),
                ("bn1", norm_type(num_features=inplanes)),
                ("relu1", relu_type(inplace=True)),
            ]

        layer0_modules.append(("pool", pool_type(kernel_size=3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            stride=strides[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=strides[1],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=strides[2],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=strides[3],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
        )
        self.num_features = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _make_layer(
        self,
        block: Type[Union[SEBottleneck, SEResNetBottleneck, SEResNeXtBottleneck]],
        planes: int,
        blocks: int,
        groups: int,
        reduction: int,
        stride: int = 1,
        downsample_kernel_size: int = 1,
    ) -> nn.Sequential:

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.inplanes,
                out_channels=planes * block.expansion,
                strides=stride,
                kernel_size=downsample_kernel_size,
                act=None,
                norm=Norm.BATCH,
                bias=False,
            )

        layers = []
        layers.append(
            block(
                spatial_dims=self.spatial_dims,
                inplanes=self.inplanes,
                planes=planes,
                groups=groups,
                reduction=reduction,
                stride=stride,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for _num in range(1, blocks):
            layers.append(
                block(
                    spatial_dims=self.spatial_dims,
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=groups,
                    reduction=reduction,
                )
            )

        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
    
    
def build_seresnet(config: Mapping) -> SENet:
    encoder = config["model"]["encoder"]
    return SENet(
        block=SEResNetBottleneck,
        spatial_dims=int(config["data"]["spatial_dims"]),
        in_channels=int(encoder["input_channels"]),
        layers=tuple(int(value) for value in encoder["depths"]),
        groups=1,
        reduction=16,
        inplanes=64,
        downsample_kernel_size=1,
        input_3x3=False,
        strides=tuple(int(value) for value in encoder["strides"]),
    )


__all__ = ["SENet", "build_seresnet"]
