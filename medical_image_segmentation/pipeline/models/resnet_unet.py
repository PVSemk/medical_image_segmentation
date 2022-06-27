import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from models.enc1_dec3 import Conv3dNormAct


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv3d:
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3,3,3), stride=(stride,stride,stride),
                     padding=(1,dilation,dilation), groups=groups, bias=False, dilation=(dilation,dilation,dilation))


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=(stride, stride, stride), bias=False)


class BasicBlock_3D(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super(BasicBlock_3D, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm3d(planes, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm3d(planes, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck_3D(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super(Bottleneck_3D, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.InstanceNorm3d(width, affine=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.InstanceNorm3d(width, affine=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.InstanceNorm3d(planes * self.expansion, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_3D(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock_3D, Bottleneck_3D]],
        layers: List[int],
        in_channels = 3,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
    ) -> None:
        super(ResNet_3D, self).__init__()
        norm_layer = nn.InstanceNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.InstanceNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_3D):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock_3D):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock_3D, Bottleneck_3D]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                ))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResNetEncoder_3D(ResNet_3D):
    def __init__(self, config, depth=5, **kwargs):
        self._depth = depth
        if config is not None:
            config_data = config['data']
            self._in_channels = len(config_data.get('channels'))
            self.out_channels = config["network"].get("out_channels")
        else:
            self._in_channels = 4
        super().__init__(**kwargs, in_channels=self._in_channels)
        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


class DecoderBlock_3D(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
    ):
        super().__init__()
        self.conv1 = Conv3dNormAct(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv3dNormAct(out_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CenterBlock_3D(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        conv1 = Conv3dNormAct(in_channels, out_channels, kernel_size=3, padding=1)
        conv2 = Conv3dNormAct(out_channels, out_channels, kernel_size=3, padding=1)
        super().__init__(conv1, conv2)


class UnetDecoder_3D(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            center=False,
            remove_first=True
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        self.remove_first = remove_first
        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock_3D(
                head_channels, head_channels
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        blocks = [
            DecoderBlock_3D(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        if self.remove_first:
            features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x

import torch.nn.functional as F
class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv3d(32, 16, kernel_size=1)
        self.bn_conv_1x1_1 = nn.InstanceNorm3d(16, affine=True)

        self.conv_3x3_1 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.InstanceNorm3d(16, affine=True)

        self.conv_3x3_2 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.InstanceNorm3d(16, affine=True)

        self.conv_3x3_3 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.InstanceNorm3d(16, affine=True)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.conv_1x1_2 = nn.Conv3d(32, 16, kernel_size=1)

        self.conv_1x1_3 = nn.Conv3d(80, 16, kernel_size=1)
        self.bn_conv_1x1_3 = nn.InstanceNorm3d(16, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)


    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]
        feature_map_c = feature_map.size()[4]

        out_1x1 = self.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = self.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = self.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = self.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = self.relu(self.conv_1x1_2(out_img))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w, feature_map_c), mode='trilinear', align_corners=True)

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = self.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

        return out

class ResNet50UNet(nn.Module):
    def __init__(
        self,
        config=None,
        encoder_depth = 5,
        decoder_channels: List[int] = (512, 256, 128, 64, 32),
    ):
        super().__init__()

        self.encoder = ResNetEncoder_3D(config, depth=encoder_depth, block=Bottleneck_3D, layers=[3, 4, 6, 3])

        self.decoder_pathA = UnetDecoder_3D(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            center=False
        )
        self.aspp = ASPP()
        # self.decoder_pathB = UnetDecoder_3D(
        #     encoder_channels=self.encoder.out_channels,
        #     decoder_channels=decoder_channels,
        #     n_blocks=encoder_depth,
        #     center=False
        # )
        # self.decoder_pathC = UnetDecoder_3D(
        #     encoder_channels=self.encoder.out_channels,
        #     decoder_channels=decoder_channels,
        #     n_blocks=encoder_depth,
        #     center=False
        # )
        self.conv_pathA = nn.Conv3d(16, 2, kernel_size=1)

        ############### Path B

        self.conv_pathB = nn.Conv3d(16, 2, kernel_size=1)

        ############### Path C

        self.conv_pathC = nn.Conv3d(16, 2, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output_pathA = self.decoder_pathA(*features)
        decoder_output_pathA = self.aspp(decoder_output_pathA)
        # decoder_output_pathB = self.decoder_pathB(*features)
        # decoder_output_pathC = self.decoder_pathC(*features)
        x_pathA = self.conv_pathA(decoder_output_pathA)
        x_pathB = self.conv_pathB(decoder_output_pathA)
        x_pathC = self.conv_pathC(decoder_output_pathA)

        return x_pathA, x_pathB, x_pathC


if __name__ == '__main__':
    from utils.parse_yaml import parse_yaml_config
    config = parse_yaml_config("config.yaml")
    net = ResNet50UNet(config).cuda().train()
    # net.print_model_parameters()

    x = torch.randn(3, 4, 96, 96, 96).cuda()
    ya, yb, yc = net(x)
    print(ya.shape, yb.shape, yc.shape)
    loss = ya.sum() + yb.sum() + yc.sum()
    loss.backward()
