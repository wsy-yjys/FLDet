# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Convolution modules
"""

import math
from typing import Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# from .block import conv_bn
__all__ = ('Conv', 'Conv2', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'GhostConv',
           'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'RepConv', 'RepDWConv', 'RepConv2','Conv2x2Sp',
           'SpaceToDepthModule','Down_SpaceToDepthModule','ConvRepbn','RepOrthoConv')

def get_activation(name='silu', inplace=True):
    if name is None or name==False or name == 'identity':
        return nn.Identity()
    elif name==True:
        name = 'silu'

    if isinstance(name, str):
        name = name.lower()
        if name == 'silu':
            module = nn.SiLU(inplace=inplace)
        elif name == 'gelu':
            module = nn.GELU()
        elif name == 'relu':
            module = nn.ReLU(inplace=inplace)
        elif name == 'relu6':
            module = nn.ReLU6(inplace=inplace)
        elif name == 'lrelu':
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'hardswish':
            module = nn.Hardswish(inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        else:
            raise AttributeError('Unsupported act type: {}'.format(name))
        return module
    elif isinstance(name, nn.Module):
        return name
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, act=True, p=None, g=1, d=1, bias=False):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        # self.act = self.default_act if act is True or act.lower()=="silu" else act if isinstance(act, nn.Module) else nn.Identity()
        self.act = get_activation(act)


    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

    def forward_fuse_noact(self, x):
        """Perform transposed convolution of 2D data."""
        return self.conv(x)


class ConvRepbn(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, act=True, p=None, g=1, d=1, bias=False):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.repbn = RepBN(c2)
        self.act = get_activation(act)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.repbn(self.conv(x)))

    def switch_to_deploy(self, ):
        miu = self.repbn.bn.running_mean
        sigma2 = self.repbn.bn.running_var
        gamma = self.repbn.bn.weight
        beta = self.repbn.bn.bias
        eps = self.repbn.bn.eps
        alpha = self.repbn.alpha

        weight = self.conv.weight.data
        bias = self.conv.bias.data if self.conv.bias else 0

        w_n = (gamma / torch.sqrt(sigma2 + eps) + alpha).reshape(-1, 1, 1, 1) * weight
        b_n = gamma * (bias - miu) / torch.sqrt(sigma2 + eps) + beta

        self.conv.weight.data = w_n
        # self.conv.bias = nn.Parameter(torch.zeros(self.conv.out_channels))
        # self.conv.bias.data = b_n
        setattr(self.conv, 'bias', nn.Parameter(b_n, requires_grad=False))
        self.__delattr__('repbn')
        if isinstance(self.act, nn.Identity):
            self.forward = self.forward_fuse_noact
            self.__delattr__('act')
        else:
            self.forward = self.forward_fuse

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

    def forward_fuse_noact(self, x):
        """Perform transposed convolution of 2D data."""
        return self.conv(x)


class RepBN(nn.Module):
    def __init__(self, channels):
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(x) + self.alpha * x

class SymmetricPadding:
    """对称填充"""
    def __init__(self):
        self.zeros =  [ (0, 1, 0, 1),
                        (1, 0, 1, 0),
                        (1, 0, 0, 1),
                        (0, 1, 1, 0) ]

    def __call__(self, x):
        num_split = 4 * [x.size(1) // 4]
        x_slide = torch.split(x, num_split, dim=1)
        x_pad = [F.pad(x_slide[i], self.zeros[i]) for i in range(4)]

        return torch.cat(x_pad, dim=1)

class Conv2x2Sp(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    def __init__(self, c1, c2, k=2, s=1, act=True, p=0, g=1, d=1):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = get_activation(act)
        self.symmetric_padding = SymmetricPadding()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(self.symmetric_padding(x))))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(self.symmetric_padding(x)))

    def forward_fuse_noact(self, x):
        """Perform transposed convolution of 2D data."""
        return self.conv(self.symmetric_padding(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution with args(ch_in, ch_out, kernel).
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)

class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status. This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act="silu", bn=False, input_method="1input", fuse_method="add"):
        super().__init__()
        # assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = get_activation(act) if act else nn.Identity()
        self.input_method = input_method
        self.fuse_method = fuse_method
        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=0, g=g, act=False)

    def forward_fuse_finalact(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward_fuse(self, x):
        """Forward process"""
        return self.conv(x)

    def forward(self, x):
        """Forward process"""
        if self.fuse_method=="add":
            if self.input_method=="2input":
                id_out = 0 if self.bn is None else self.bn(x[1])
                return self.act(self.conv1(x[0]) + self.conv2(x[1])+ id_out)
            else:
                id_out = 0 if self.bn is None else self.bn(x)
                return self.act(self.conv1(x) + self.conv2(x)+ id_out)
        elif self.fuse_method == "concat":
            if self.input_method=="2input":
                id_out = 0 if self.bn is None else self.bn(x[1])
                cat = torch.cat([self.conv1(x[0]),self.conv2(x[1]),id_out],dim=1) if isinstance(id_out, torch.Tensor) else torch.cat([self.conv1(x[0]),self.conv2(x[1])],dim=1)
                return self.act(cat)
            else:
                id_out = 0 if self.bn is None else self.bn(x)
                cat = torch.cat([self.conv1(x),self.conv2(x),id_out],dim=1) if isinstance(id_out, torch.Tensor) else torch.cat([self.conv1(x),self.conv2(x)],dim=1)
                return self.act(cat)


    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        if self.fuse_method == "add":
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
        elif self.fuse_method == "concat":
            return torch.cat([kernel3x3, self._pad_1x1_to_3x3_tensor(kernel1x1), kernelid], dim=0), torch.cat([bias3x3, bias1x1, biasid], dim=0)

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        if self.fuse_method=="add":
            self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                                  out_channels=self.conv1.conv.out_channels,
                                  kernel_size=self.conv1.conv.kernel_size,
                                  stride=self.conv1.conv.stride,
                                  padding=self.conv1.conv.padding,
                                  dilation=self.conv1.conv.dilation,
                                  groups=self.conv1.conv.groups,
                                  bias=True).requires_grad_(False)
        elif self.fuse_method=="concat":
            self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                                  out_channels=self.conv1.conv.out_channels * 3,
                                  kernel_size=self.conv1.conv.kernel_size,
                                  stride=self.conv1.conv.stride,
                                  padding=self.conv1.conv.padding,
                                  dilation=self.conv1.conv.dilation,
                                  groups=self.conv1.conv.groups,
                                  bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        if isinstance(self.act, nn.Identity):
            self.__delattr__('act')
            self.forward = self.forward_fuse
        else:
            self.forward = self.forward_fuse_finalact


class RepConv2(nn.Module):
    """    4x4+2x2+bn    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=4, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        # assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=1, g=g, act=False)
        self.conv2 = Conv(c1, c2, 2, s, p=p, g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel4x4, bias4x4 = self._fuse_bn_tensor(self.conv1)
        kernel2x2, bias2x2 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel4x4 + self._pad_2x2_to_4x4_tensor(kernel2x2) + kernelid, bias4x4 + bias2x2 + biasid

    def _pad_2x2_to_4x4_tensor(self, kernel2x2):
        """Pads a 2x2 tensor to a 4x4 tensor."""
        if kernel2x2 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel2x2, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():      # 模型的所有参数从计算图中分离出来
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.forward = self.forward_fuse

class RepDWConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        use_act: bool = True,
        use_scale_branch: bool = True,
        use_skip_branch: bool = True,
        num_conv_branches: int = 1,
        activation: nn.Module = nn.SiLU(),
    ) -> None:
        super(RepDWConv, self).__init__()
        self.groups = groups
        self.stride = stride
        self.padding = kernel_size//2
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.activation = activation if use_act else nn.Identity()

        # Re-parameterizable skip connection
        self.rbr_skip = (
            nn.BatchNorm2d(num_features=in_channels)
            if out_channels == in_channels and stride == 1 and use_skip_branch
            else None
        )

        # Re-parameterizable 3x3 branches
        if num_conv_branches > 0 and kernel_size >= 3:
            conv3x3 = list()
            for _ in range(self.num_conv_branches):
                conv3x3.append(
                    self._conv_bn(kernel_size=3, padding=1)
                )
            self.conv3x3 = nn.ModuleList(conv3x3)
        else:
            self.conv3x3 = None

        # Re-parameterizable 1x1 branch
        self.conv1x1 = None
        if (kernel_size >= 1) and use_scale_branch:
            self.conv1x1 = self._conv_bn(kernel_size=1, padding=0)

        # Re-parameterizable 5x5 branches
        self.conv5x5 = None
        if kernel_size >= 5:
            self.conv5x5 = self._conv_bn(kernel_size=5, padding=2)

        # Re-parameterizable 7x7 branches
        self.conv7x7 = None
        if kernel_size >= 7:
            self.conv7x7 = self._conv_bn(kernel_size=7, padding=3)

        # Re-parameterizable 9x9 branches
        self.conv9x9 = None
        if kernel_size >= 9:
            self.conv9x9 = self._conv_bn(kernel_size=9, padding=4)

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        # Inference mode forward pass.
        return self.activation(self.reparam_conv(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        # Multi-branched train-time forward pass.

        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # 1x1 branch output
        scale_out = 0
        if self.conv1x1 is not None:
            scale_out = self.conv1x1(x)

        # 3x3 branches
        out = scale_out + identity_out
        if self.conv3x3 is not None:
            for ix in range(self.num_conv_branches):
                out += self.conv3x3[ix](x)
        # 5x5 branches
        if self.conv5x5 is not None:
            out += self.conv5x5(x)

        if self.conv7x7 is not None:
            out += self.conv7x7(x)

        if self.conv9x9 is not None:
            out += self.conv9x9(x)

        return self.activation(out)

    def switch_to_deploy(self):
        if hasattr(self, "reparam_conv"):
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():  # 将参数的梯度追踪功能关闭，使得参数不再具有梯度信息
            para.detach_()
        if hasattr(self, "conv9x9"):
            self.__delattr__("conv9x9")
        if hasattr(self, "conv7x7"):
            self.__delattr__("conv7x7")
        if hasattr(self, "conv5x5"):
            self.__delattr__("conv5x5")
        self.__delattr__("conv3x3")
        self.__delattr__("conv1x1")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.forward = self.forward_fuse

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # get weights and bias of conv1x1 branch
        kernel_1x1 = 0
        bias_1x1 = 0
        if self.conv1x1 is not None:
            kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.conv1x1)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_1x1 = torch.nn.functional.pad(kernel_1x1, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv3x3 branches
        kernel_3x3 = 0
        bias_3x3 = 0
        if self.conv3x3 is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(self.conv3x3[ix])
                kernel_3x3 += _kernel
                bias_3x3 += _bias
            pad = (self.kernel_size-3) // 2
            kernel_3x3 = torch.nn.functional.pad(kernel_3x3, [pad, pad, pad, pad])

        # get weights and bias of conv5x5 branches
        kernel_5x5 = 0
        bias_5x5 = 0
        if self.conv5x5 is not None:
            kernel_5x5, bias_5x5 = self._fuse_bn_tensor(self.conv5x5)
            pad = (self.kernel_size-5) // 2
            kernel_5x5 = torch.nn.functional.pad(kernel_5x5, [pad, pad, pad, pad])

        # get weights and bias of conv7x7 branches
        kernel_7x7 = 0
        bias_7x7 = 0
        if self.conv7x7 is not None:
            kernel_7x7, bias_7x7 = self._fuse_bn_tensor(self.conv7x7)
            pad = (self.kernel_size-7) // 2
            kernel_7x7 = torch.nn.functional.pad(kernel_7x7, [pad, pad, pad, pad])

        # get weights and bias of conv9x9 branches
        conv9x9 = 0
        bias_9x9 = 0
        if self.conv9x9 is not None:
            conv9x9, bias_9x9 = self._fuse_bn_tensor(self.conv9x9)


        kernel_final = conv9x9 + kernel_7x7 + kernel_5x5 + kernel_3x3 + kernel_1x1 + kernel_identity
        bias_final = bias_9x9 + bias_7x7 + bias_5x5 + bias_3x3 + bias_1x1 + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch: Union[nn.Sequential, nn.BatchNorm2d]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        mod_list = nn.Sequential()
        mod_list.add_module("conv",
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=kernel_size,
                stride=self.stride, padding=padding, groups=self.groups, bias=False,),)
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list

class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))

def projection(u, v):
    return (v * u).sum() / (u * u).sum() * u


def gram_schmidt(input):
    output = []
    for x in input:
        x.view(-1)
        for y in output:
            x -= torch.sum(y * x) / torch.sum(y * y) * y
        x /= torch.norm(x, p=2)
        output.append(x)
    return torch.stack(output)


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x

@torch.jit.script
class SpaceToDepthJitx2(object):
    def __call__(self, x: torch.Tensor):
        # assuming hard-coded that block_size==2 for acceleration
        N, C, H, W = x.size()
        # x = x.view(N, C, H // 2, 2, W // 2, 2)  # (N, C, H//bs, bs, W//bs, bs)
        # x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        # x = x.view(N, C * 4, H // 2, W // 2)  # (N, C*bs^2, H//bs, W//bs)
        return x.view(N, C, H // 2, 2, W // 2, 2).permute(0, 3, 5, 1, 2, 4).contiguous().view(N, C * 4, H // 2, W // 2)

@torch.jit.script
class SpaceToDepthJitx4(object):
    def __call__(self, x: torch.Tensor):
        # assuming hard-coded that block_size==4 for acceleration
        N, C, H, W = x.size()
        # x = x.view(N, C, H // 4, 4, W // 4, 4)  # (N, C, H//bs, bs, W//bs, bs)
        # x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        # x = x.view(N, C * 16, H // 4, W // 4)  # (N, C*bs^2, H//bs, W//bs)
        return x.view(N, C, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 2, 4).contiguous().view(N, C * 16, H // 4, W // 4)

class SpaceToDepthModule(nn.Module):
    def __init__(self, block_size=4, remove_model_jit=False):
        super().__init__()
        if not remove_model_jit:
            self.op = SpaceToDepthJitx4() if block_size==4 else SpaceToDepthJitx2()
        else:
            self.op = SpaceToDepth(block_size)

    def forward(self, x):
        return self.op(x)

class Down_SpaceToDepthModule(nn.Module):
    def __init__(self, c1, c2, k=3, block_size=2, g=1, act=True):
        super().__init__()
        self.down = SpaceToDepthModule(block_size=block_size)
        self.conv = Conv(c1*block_size*block_size, c2, k, 1, g=g, act=act)

    def forward(self, x):
        return self.conv(self.down(x))


class RepOrthoConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status. This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.stride = s
        self.act = get_activation(act)

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)        # 3x3
        # self.conv2 = Conv(c1, c2, k, s, p=p, g=g, act=False)        # Ortho3x3
        # self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    # def initialize_orthogonal_filters(self, weight):
    #     # copy_weight = weight.clone()
    #     c2,c1,h,w = weight.shape
    #     # Gram-Schmidt正交化方法是一组对线性无关的向量操作，对于一个c1hw维向量组，最多可以有c1hw个线性无关的向量。因此当c2>c1hw时，需要对c2进行分段，即c2//c1hw段，段内线性无关
    #     if c1*h*w < c2:
    #         n = c2//(c1*h*w)
    #         gram = []
    #         for i in range(n):
    #             # gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))   随机初始化
    #             # gram.append(gram_schmidt(weight[i*c1*h*w:(i+1)*c1*h*w, :, :, :]))
    #             gram.append(gram_schmidt_nonorm(weight[i*c1*h*w:(i+1)*c1*h*w, :, :, :]))  # 不进行归一化
    #         # weight[:n*(c1*h*w), :, :, :] *= torch.cat(gram, dim=0)
    #         weight[:n*(c1*h*w), :, :, :] = torch.cat(gram, dim=0)
    #     else:
    #         # return gram_schmidt(torch.rand([c, 1, h, w]))
    #         # weight*=gram_schmidt(weight)
    #         # weight*=gram_schmidt_nonorm(weight)           # 不进行归一化
    #         weight = gram_schmidt_nonorm(weight)           # 不进行归一化
    #     # return F.conv2d(input=x, weight=weight, stride=1, padding=1, groups=self.g, bias=False)
    #     return weight

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        ortho_weight = nn.Parameter(gram_schmidt(self.conv1.conv.weight.data))
        conv2_out = F.conv2d(input=x, weight=ortho_weight, stride=self.stride, padding=1, groups=self.g)
        # self.conv2.conv.weight.data = self.initialize_orthogonal_filters(self.conv1.conv.weight.data)
        # return self.act(self.conv1(x) + self.conv2(x) + id_out)
        return self.act(self.conv1(x) + conv2_out + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        # kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelOrtho3x3, biasOrtho3x3 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        # return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
        return kernel3x3 + kernelOrtho3x3 + kernelid, bias3x3 + biasOrtho3x3 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        pass
        # if hasattr(self, 'conv'):
        #     return
        # kernel, bias = self.get_equivalent_kernel_bias()
        # self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
        #                       out_channels=self.conv1.conv.out_channels,
        #                       kernel_size=self.conv1.conv.kernel_size,
        #                       stride=self.conv1.conv.stride,
        #                       padding=self.conv1.conv.padding,
        #                       dilation=self.conv1.conv.dilation,
        #                       groups=self.conv1.conv.groups,
        #                       bias=True).requires_grad_(False)
        # self.conv.weight.data = kernel
        # self.conv.bias.data = bias
        # for para in self.parameters():
        #     para.detach_()
        # self.__delattr__('conv1')
        # # self.__delattr__('conv2')
        # if hasattr(self, 'nm'):
        #     self.__delattr__('nm')
        # if hasattr(self, 'bn'):
        #     self.__delattr__('bn')
        # if hasattr(self, 'id_tensor'):
        #     self.__delattr__('id_tensor')
        # self.forward = self.forward_fuse


if __name__ == '__main__':
    tensor = torch.randn(1, 32, 40, 40)
    ConvRepbn = ConvRepbn(32, 32, 3)
    ConvRepbn.eval()
    out = ConvRepbn(tensor)
    ConvRepbn.switch_to_deploy()
    out_deploy = ConvRepbn(tensor)
    print('========================== The diff is')
    print(((out - out_deploy) ** 2).sum())





