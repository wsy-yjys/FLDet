"""
FLDet Block modules
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, RepBN
# from conv import Conv, RepBN

__all__ = ('IEPR', 'conv_bn', 'RepBlock3',  "CSNeck3in", 'ECM')


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
        elif name == 'sigmoid':
            module = nn.Sigmoid()
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        else:
            raise AttributeError('Unsupported act type: {}'.format(name))
        return module
    elif isinstance(name, nn.Module):
        return name
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))

def channel_shuffle(x, groups: int):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class conv_bn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=None,
                 groups=1,
                 dilation=1,
                 bias=False):

        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              dilation=dilation,
                              bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self,x):
        return self.bn(self.conv(x))

    def forward_fuse(self,x):
        return self.conv(x)

    def switch_to_deploy(self):
        kernel = self.conv.weight
        bias = self.conv.bias if self.conv.bias is not None else 0
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        kernel_fused, bias_fused = kernel * t, (bias - running_mean) * gamma / std + beta
        # Reparameterization
        self.conv.weight.data = kernel_fused
        setattr(self.conv, 'bias', torch.nn.Parameter(bias_fused,requires_grad=False))
        self.__delattr__('bn')
        self.forward = self.forward_fuse

class FFN(nn.Module):
    def __init__(self, c1=3,
                 c2=3,
                 act="SiLU",
                 e=2,
                 shortcut=True):

        super().__init__()
        self.m = nn.Sequential(
                    conv_bn(c1, int(e * c2), 1, 1, 0),
                    get_activation(act),
                    conv_bn(int(e * c2), c2, 1, 1, 0),
                )
        self.shortcut = shortcut
        # self.res_scale = Scale(dim=c1, init_value=csp_scale_init_value) if scale_init_value else nn.Identity()

    def forward(self, x):
        return x + self.m(x) if self.shortcut else self.m(x)

    def switch_to_deploy(self):
        for m in self.m:
            if isinstance(m, (conv_bn,)):
                m.switch_to_deploy()


class Partial_conv3(nn.Module):
    def __init__(self,
                 dim,
                 n_div=2,
                 forward='split_cat',
                 act=False,
                 bn=True,
                 type="normal",
                 k=3,
                 d=1,
                 groups=False,
                 deepextend=4,
                 shortcut=False,
                 grouprep=False):       # Pconvshortcut: RepBlock3 use shortcut

        super().__init__()
        self.dim_conv3 = int(dim // n_div)
        self.dim_untouched = dim - self.dim_conv3
        self.type = type

        if type=="normal":
            self.partial_conv3 = Conv(self.dim_conv3, self.dim_conv3, k, 1, act=act, d=d, g=groups) if bn else nn.Conv2d(self.dim_conv3, self.dim_conv3, k, 1, k//2, bias=False,dilation=d)
        elif type=="deepextend":
            self.partial_conv3 = RepBlock3(self.dim_conv3, self.dim_conv3, e=deepextend, groups=groups if groups else self.dim_conv3*deepextend, shortcut=shortcut, grouprep=grouprep)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        return torch.cat((self.partial_conv3(x1), x2), 1)

    def forward_slicing(self, x):
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def switch_to_deploy(self):
        self.forward = self.forward_slicing
        # conv_bn deploy
        if isinstance(self.partial_conv3, (conv_bn,RepBlock3)):
            self.partial_conv3.switch_to_deploy()


class RepBlock3(nn.Module):
    """LocalMixer"""
    def __init__(self,
                 c1=3,
                 c2=512,
                 k=(1, 3, 1),
                 stride=(1, 1, 1),
                 groups=1,
                 act="silu",
                 finalact=True,
                 e=1,
                 shortcut=False,
                 grouprep=False,
                 bn=True,
                 ):

        super().__init__()
        self.groups = groups
        self.c1 = c1
        self.c2 = c2
        self.grouprep = grouprep
        self.kernel = k
        self.stride = stride
        self.pad_pixels = k[1]//2
        self.add = shortcut and c1 == c2 and stride[1]==1
        self.finalact = finalact     # Whether to use the activation function at the end
        self.bn = bn     # Whether to use conv_bn
        if bn:
            self.cv1 = conv_bn(in_channels=c1, out_channels=c2 * e, kernel_size=k[0], stride=stride[0], padding=k[0] // 2, groups=1, bias=False)
            if self.grouprep:
                self.cv2 = GroupRepConv(c1=c2 * e, c2=c2 * e, k=k[1], s=stride[1], p=0)
            else:
                self.cv2 = conv_bn(in_channels=c2 * e, out_channels=c2 * e, kernel_size=k[1], stride=stride[1], padding=0, groups=groups, bias=False)
            self.cv3 = conv_bn(in_channels=c2 * e, out_channels=c2, kernel_size=k[2], stride=stride[2], padding=k[2] // 2, groups=1, bias=False)
        else:
            self.cv1 = nn.Conv2d(in_channels=c1, out_channels=c2 * e, kernel_size=k[0], stride=stride[0], padding=k[0] // 2)
            self.cv2 = nn.Conv2d(in_channels=c2 * e, out_channels=c2 * e, kernel_size=k[1], stride=stride[1], padding=0, groups=groups)
            self.cv3 = nn.Conv2d(in_channels=c2 * e, out_channels=c2, kernel_size=k[2], stride=stride[2], padding=k[2] // 2)

        # rbr_identity
        if self.add:
            self.rbr_identity = nn.BatchNorm2d(num_features=c1)
        if finalact:
            self.activation = get_activation(act)

    # deploy method（as a single block）
    def forward_fuse_finalact(self, x):
        return self.activation(self.cv(x))

    # deploy method（as a block in PC2f_Decay）
    def forward_fuse(self, x):
        return self.cv(x)

    def forward(self,x):
        if self.add:
            x_identity = x.clone()
        x = self.cv1(x)
        if self.kernel[1]>=3:
            x_pad = self.padlayer(x, self._fuse_bn_tensor(self.cv1.bn if self.bn else self.cv1)[1])    # wo bias
            # x_pad = self.padlayer(x, self._fuse_bn_tensor(self.cv1)[1])     # with bias
        x = self.cv2(x_pad)
        x = self.cv3(x)
        x = self.rbr_identity(x_identity) + x if self.add else x
        if self.finalact:       # normal act
            x = self.activation(x)
        return x

    def padlayer(self, x ,pad_values):
        x = F.pad(x, [self.pad_pixels] * 4)
        # pad_pixels=K//2
        pad_values = pad_values.view(1, -1, 1, 1)
        x[:, :, 0:self.pad_pixels, :] = pad_values
        x[:, :, -self.pad_pixels:, :] = pad_values
        x[:, :, :, 0:self.pad_pixels] = pad_values
        x[:, :, :, -self.pad_pixels:] = pad_values
        return x

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.cv1)
        self.group2vanilla()
        if self.grouprep:
            kernel2, bias2 = self.cv2.conv.weight.data,self.cv2.conv.bias.data
        else:
            kernel2, bias2 = self._fuse_bn_tensor(self.cv2)
        kernel3, bias3 = self._fuse_bn_tensor(self.cv3)
        # Vertically fuse cv1 and cv2
        k = F.conv2d(kernel2, kernel1.permute(1, 0, 2, 3))  # [input, weight]
        b = (kernel2 * bias1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + bias2
        # Vertically fuse cv2 and cv3
        weight_fused = torch.einsum('oi,icjk->ocjk', kernel3.squeeze(3).squeeze(2), k)
        bias_fused = bias3 + (b.view(1, -1, 1, 1) * kernel3).sum(3).sum(2).sum(1)
        # Horizontal fusion cv and BN
        if self.add:
            kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
            weight_fused += kernelid
            bias_fused += biasid
        # Reparameterization
        self.cv = nn.Conv2d(in_channels=self.c1, out_channels=self.c2,
                            kernel_size=self.kernel[1], stride=self.stride[1],
                            padding=self.pad_pixels, dilation=self.cv2.conv.dilation if self.bn else self.cv2.dilation,
                            groups=1, bias=True)
        self.cv.weight.data = weight_fused
        self.cv.bias.data = bias_fused
        # Remove excess branches
        self.__delattr__('cv1')
        self.__delattr__('cv2')
        self.__delattr__('cv3')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        # Set inference mode
        if self.finalact:
            self.forward = self.forward_fuse_finalact
        else:
            self.forward = self.forward_fuse

    def group2vanilla(self, ):
        if self.grouprep:
            self.cv2.switch_to_deploy()
        elif not self.bn:           # conv wo bn
            kernel = self.cv2.weight.data
            group_out_channels = self.cv2.out_channels // self.cv2.groups     # Output channel for each group
            group_kernel_list = []
            for i in range(self.groups):
                zeros_kernel = torch.zeros([group_out_channels, self.cv2.in_channels, 3, 3]).to(self.cv2.weight.device)
                zeros_kernel[:, group_out_channels * i:group_out_channels * (i + 1), :, :] = kernel[group_out_channels * i:group_out_channels * (i + 1),:, :, :]
                group_kernel_list.append(zeros_kernel)
            # obtained the equivalent weights of conv3x3 after BasicBlock reargument
            weight = torch.cat(group_kernel_list, dim=0)
            self.cv2.weight.data = weight
        else:                               # conv_bn
            # group conv——>vanilla conv
            kernel = self.cv2.conv.weight.data
            group_out_channels = self.cv2.conv.out_channels // self.cv2.conv.groups     # Output channel for each group
            group_kernel_list = []
            for i in range(self.groups):
                zeros_kernel = torch.zeros([group_out_channels, self.cv2.conv.in_channels, 3, 3]).to(self.cv2.conv.weight.device)
                zeros_kernel[:, group_out_channels * i:group_out_channels * (i + 1), :, :] = kernel[group_out_channels * i:group_out_channels * (i + 1),:, :, :]
                group_kernel_list.append(zeros_kernel)
            # obtained the equivalent weights of conv3x3 after BasicBlock reargument
            weight = torch.cat(group_kernel_list, dim=0)
            self.cv2.conv.weight.data = weight

    def _fuse_bn_tensor(self, branch):
        if hasattr(branch, "conv"):
            kernel = branch.conv.weight
            bias = branch.conv.bias if branch.conv.bias is not None else 0
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, (nn.Conv2d, )):     # conv wo bn
            kernel = branch.weight
            bias = branch.bias if branch.bias is not None else 0
            return kernel, bias
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            input_dim = branch.num_features
            kernel_value = np.zeros((branch.num_features, input_dim, self.kernel[1], self.kernel[1]), dtype=np.float32)
            for i in range(branch.num_features):
                kernel_value[i, i % input_dim, (self.kernel[1]-1)//2, (self.kernel[1]-1)//2] = 1     # 中间元素变1
            kernel = torch.from_numpy(kernel_value).to(branch.weight.device)
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            bias = 0
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, (bias - running_mean) * gamma / std + beta

class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones([1,dim,1,1]), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale

class GroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, p=None, bias=False, bn=True):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=p if p!= None else (kernel_size-1)//2, groups=groups, bias=bias)
        self.group = groups
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_in_channels = in_channels // groups
        self.group_out_channels = out_channels // groups
        self.bn = nn.BatchNorm2d(num_features=out_channels) if bn else nn.Identity()

    def forward(self, x):
        return self.bn(self.conv(x))

    def forward_fuse(self, x):
        return self.conv(x)

    def _fuse_bn_tensor(self, conv_weight,bias, bn):
        kernel = conv_weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

    def switch_to_deploy(self):
        # group conv——>normal conv
        kernel = self.conv.weight
        bias = self.conv.bias.data if self.conv.bias is not None else 0
        group_kernel_list = []
        for i in range(self.group):
            zeros_kernel = torch.zeros([self.group_out_channels, self.in_channels, self.kernel_size, self.kernel_size]).to(self.conv.weight.device)
            zeros_kernel[:, self.group_in_channels * i:self.group_in_channels * (i + 1), :, :] = kernel[self.group_out_channels * i:self.group_out_channels * (i + 1),:, :, :]
            group_kernel_list.append(zeros_kernel)
        # 得到BasicBlock重参后convkxk的等价权重
        weight = torch.cat(group_kernel_list, dim=0)
        # fuse conv and bn
        if isinstance(self.bn, (nn.BatchNorm2d,)):
            weight, bias = self._fuse_bn_tensor(weight,bias, self.bn)
        # self.cv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
        #                     kernel_size=self.conv.kernel_size, stride=self.conv.stride,
        #                     padding=self.conv.padding, groups=1)
        # 重参数化
        self.conv.weight.data = weight
        self.conv.bias.data = bias if self.conv.bias.data is not None else setattr(self, 'bias', bias)
        self.conv.groups=1
        # 删除无用属性
        self.__delattr__('bn')
        # 切换推理模式
        self.forward = self.forward_fuse


class GroupRepConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, act=False, bn=True):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.act = get_activation(act)
        # self.num_of_convs = math.floor(math.log2(min(c1, c2))) + 1
        self.num_of_convs = self.max_power_of_two_common_factor(c1, c2) + 1
        self.groupconvs = nn.Sequential(
            GroupConv(c1, c2, k, s, groups=1, p=p, bias=True, bn=bn),
            GroupConv(c1, c2, k, s, groups=min(c1,c2), p=p, bias=True, bn=bn))


    def max_power_of_two_common_factor(self, c1, c2):
        # Find the greatest common factor
        while c2:
            c1, c2 = c2, c1 % c2
        # Find the power of 2 in the greatest common factor
        power_of_two = 0
        while c1 % 2 == 0:
            c1 = c1 // 2
            power_of_two += 1
        # print(2 ** power_of_two)
        return power_of_two

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        # id_out = 0 if self.bn is None else self.bn(x)
        y=[conv(x) for conv in self.groupconvs]
        return self.act(sum(y))

    def switch_to_deploy(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = [],[]
        for gconv in self.groupconvs:
            gconv.switch_to_deploy()
            kernel.append(gconv.conv.weight)
            bias.append(gconv.conv.bias)
        kernel, bias = sum(kernel),sum(bias)
        self.conv = nn.Conv2d(
            in_channels=self.groupconvs[0].conv.in_channels,
            out_channels=self.groupconvs[0].conv.out_channels,
            kernel_size=self.groupconvs[0].conv.kernel_size,
            stride=self.groupconvs[0].conv.stride,
            padding=self.groupconvs[0].conv.padding,
            dilation=self.groupconvs[0].conv.dilation,
            groups=self.groupconvs[0].conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("groupconvs")
        self.forward = self.forward_fuse


class IEPR(nn.Module):
    """Local-Aware Modules"""
    def __init__(self,
                 c1,
                 c2,
                 n_div=2,
                 e=2,               # expansion rate
                 shortcut=True,
                 Pconvtype="normal",
                 groups=False,
                 block_type="Partial_conv3",
                 grouprep=False,
                 scale_init_value=None,
                 deepextend=4,
                 act="SiLU",
                 ):

        super().__init__()
        self.shortcut = shortcut
        if block_type == "Partial_conv3":
            self.cv = nn.Sequential(
                Conv(c1, int(e * c2), 1, 1, act),
                Partial_conv3(int(e * c2), n_div=n_div, act=act, type=Pconvtype, groups=groups, deepextend=deepextend, grouprep=grouprep),
                Conv(int(e * c2), c2, 1, 1, False))
        elif block_type == "lk":
            self.cv = nn.Sequential(
                Conv(c1, int(e * c2), 1, 1, act),
                Conv(int(e * c2), int(e * c2), 7, 1, g=int(e * c2)),
                Conv(int(e * c2), c2, 1, 1, False))

        if shortcut:
            self.res_scale = Scale(dim=c1, init_value=scale_init_value) if scale_init_value else nn.Identity()

    def forward(self, x):
        return self.res_scale(x) + self.cv(x) if self.shortcut else self.cv(x)

    def switch_to_deploy(self):
        self.cv[1].switch_to_deploy()


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class CSNeck3in(nn.Module):
    """Partial Interaction Neck"""
    def __init__(self, c_l, c_m, c_h, shortcut=False, n_div=2, partial_ratio=0.5, fuse_block_num=1, blocktype="IEPR", groups=False, grouprep=False, scale_init_value=0) -> None:
        super().__init__()
        # self.act = h_sigmoid()
        # self.downsample = nn.functional.adaptive_avg_pool2d
        # self.unsample = nn.Upsample(None, 4, 'nearest')
        self.c_l = int(c_l*partial_ratio)
        self.c_l_id = c_l - self.c_l
        self.c_h = int(c_h*partial_ratio)
        self.c_h_id = c_h - self.c_h
        self.splitchannels = [self.c_l, self.c_h]
        self.splitchannels_l = [self.c_l_id, self.c_l]
        self.splitchannels_h = [self.c_h_id, self.c_h]
        self.c = self.c_l + self.c_h

        if blocktype=="IEPR":
            fuse_block = [IEPR(self.c+c_m, self.c, n_div, 2, shortcut, "deepextend", groups=groups, scale_init_value=scale_init_value, grouprep=grouprep)]
            fuse_block.extend(IEPR(self.c, self.c, n_div, 2, shortcut, "deepextend", groups=groups, scale_init_value=scale_init_value, grouprep=grouprep) for i in range(fuse_block_num-1))

            self.fuse = nn.Sequential(*fuse_block)
            self.branch_l = IEPR(c_l, c_l, n_div, 2, shortcut, "deepextend", groups=groups, scale_init_value=scale_init_value, grouprep=grouprep)
            self.branch_h = IEPR(c_h, c_h, n_div, 2, shortcut, "deepextend", groups=groups, scale_init_value=scale_init_value, grouprep=grouprep)

        if blocktype=="MIX":
            fuse_block = [IEPR(self.c+c_m, self.c, n_div, 2, shortcut, "deepextend", groups=groups, scale_init_value=scale_init_value, grouprep=grouprep)]
            if fuse_block_num%2==0:     # even number
                mix_block = [ECM(self.c, shortcut=shortcut)]
            else:                       # odd number
                mix_block = [
                    ECM(self.c, shortcut=shortcut),
                    IEPR(self.c, self.c, n_div, 2, shortcut, "deepextend", groups=groups, scale_init_value=scale_init_value, grouprep=grouprep),
                    ]
            for i in range(0, fuse_block_num - 1, 2):
                fuse_block.extend(mix_block)

            self.fuse = nn.Sequential(*fuse_block)
            self.branch_l = IEPR(c_l, c_l, n_div, 2, shortcut, "deepextend", groups=groups, scale_init_value=scale_init_value, grouprep=grouprep)
            self.branch_h = IEPR(c_h, c_h, n_div, 2, shortcut, "deepextend", groups=groups, scale_init_value=scale_init_value, grouprep=grouprep)

    def forward(self, x):
        '''
        x[0]: low features(P2)
        x[1]: mid features(P3)
        x[2]: high features(P4)
        '''
        low1, low2 = x[0].split(self.splitchannels_l, dim=1)
        high1, high2 = x[2].split(self.splitchannels_h, dim=1)
        fuse_l,fuse_h = self.fuse(torch.cat(
            [F.interpolate(low2, scale_factor=0.5, mode='bilinear'),
            x[1],
            F.interpolate(high2, scale_factor=2, mode='bilinear')],dim=1)
        ).split(self.splitchannels, dim=1)
        return [self.branch_l(torch.cat([low1, F.interpolate(fuse_l, scale_factor=2, mode='bilinear')],dim=1)), \
            self.branch_h(torch.cat([high1, F.interpolate(fuse_h, scale_factor=0.5, mode='bilinear')],dim=1))]

    def forward_slicing(self, x):
        fuse_l,fuse_h = self.fuse(torch.cat(
            [F.interpolate(x[0][:, self.c_l_id:, :, :], scale_factor=0.5, mode='bilinear'),
             x[1],
            F.interpolate(x[2][:, self.c_h_id:, :, :], scale_factor=2, mode='bilinear')],dim=1)
        ).split(self.splitchannels, dim=1)
        x[0][:, self.c_l_id:, :, :] = F.interpolate(fuse_l, scale_factor=2, mode='bilinear')
        x[2][:, self.c_h_id:, :, :] = F.interpolate(fuse_h, scale_factor=0.5, mode='bilinear')
        return [self.branch_l(x[0]), self.branch_h(x[2])]

    def switch_to_deploy(self):
        self.forward = self.forward_slicing


class PConv(nn.Module):
    def __init__(self, dim, partial, k=3, d=1, groups=1, forward='split_cat', act=False, bn=True, type="normal"):
        super().__init__()
        self.padding = k // 2
        self.dim_conv3 = int(dim*partial)
        self.dim_untouched = dim - self.dim_conv3
        self.type = type

        if type=="normal":
            self.partial_conv = Conv(self.dim_conv3, self.dim_conv3, k, 1, act=act, d=d, g=groups) if bn else nn.Conv2d(self.dim_conv3, self.dim_conv3, k, 1, k//2, bias=False,dilation=d)
        elif type=="deepextend":
            self.partial_conv3 = RepBlock3(self.dim_conv3, self.dim_conv3, e=4, groups=groups, shortcut=False,)

        if forward == 'slicing':
            self.forward = self.forward_slicing_bacth1

    def forward(self, x, att):
        # n, c, h, w = x.shape
        # topk_indices = att.view(n, c).topk(self.dim_conv3, dim=1)[-1]
        # expanded_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        # return x.scatter(1, expanded_indices, self.partial_conv(torch.gather(x, 1, expanded_indices)))
        topk_indices = att.topk(self.dim_conv3, dim=1)[-1].expand(-1, -1, *x.shape[2:])
        return x.scatter(1, topk_indices, self.partial_conv(torch.gather(x, 1, topk_indices).contiguous()))

    def switch_to_deploy(self):
        if isinstance(self.partial_conv, (RepBlock3)):
            self.partial_conv.switch_to_deploy()
        self.forward = self.forward_slicing

    def forward_slicing(self, x, att):
        # step by step for debug
        # topk_indices = att.topk(self.dim_conv3, dim=1)[-1]
        # expanded_indices = topk_indices.expand(-1, -1, *x.shape[2:])
        # return x.scatter_(1, expanded_indices, self.partial_conv(torch.gather(x, 1, expanded_indices)))
        # infer for bacth=n
        topk_indices = att.topk(self.dim_conv3, dim=1)[-1].expand(-1, -1, *x.shape[2:])
        return x.scatter_(1, topk_indices, self.partial_conv(torch.gather(x, 1, topk_indices).contiguous()))


class RepCTXBlock(nn.Module):
    """ Rep context blcok
    """
    def __init__(self, in_channels, band_kernel_size=11, branch_ratio=0.25, start_k=1,  dilated=False, fc=True):
        super().__init__()

        self.gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.split_indexes = (self.gc, self.gc, in_channels - 2 * self.gc)

        self.hcb = HorizontalContextBlockDilated(self.gc, band_kernel_size, start_k=start_k) if dilated else HorizontalContextBlock(self.gc, band_kernel_size, start_k=start_k)
        self.vcb = VerticalContextBlockDilated(self.gc, band_kernel_size, start_k=start_k) if dilated else VerticalContextBlock(self.gc, band_kernel_size, start_k=start_k)
        # self.fc = Conv(in_channels, in_channels, 1) if fc else nn.Identity()
        if fc==True:
            self.fc = Conv(in_channels, in_channels, 1, act=False)
        elif fc=="mlp":
            self.fc = RepBlock3(in_channels, in_channels, groups=in_channels * 4, e=4, act=False)
        else:
            self.fc = nn.Identity()
        self.act = get_activation('silu')

    def forward(self, x):
        x_h, x_v, x_id = torch.split(x, self.split_indexes, dim=1)
        return self.fc(torch.cat((self.act(self.hcb(x_h)), self.act(self.vcb(x_v)), x_id), dim=1))

    def switch_to_deploy(self):
        self.hcb.switch_to_deploy()
        self.vcb.switch_to_deploy()
        if isinstance(self.fc, (RepBlock3,)):
            self.fc.switch_to_deploy()
        # self.forward = self.forward_slicing

    def forward_slicing(self, x):
        x[:, :self.gc, :, :] = self.hcb(x[:, :self.gc, :, :])
        x[:, self.gc:self.gc*2, :, :] = self.vcb(x[:, self.gc:self.gc*2, :, :])
        return self.fc(x)


class RepMLP(nn.Module):
    """RepBlock3-act with shortcut"""
    def __init__(self, in_channels, groups=False):
        super().__init__()
        self.mlp = RepBlock3(in_channels, in_channels, groups=groups if groups else in_channels*4, e=4)
        
    def forward(self, x):
        return self.mlp(x)+x

    def switch_to_deploy(self):
        self.mlp.switch_to_deploy()


class ECM(nn.Module):
    """ Efficient Context-Aware Modulation (ECAM)"""
    def __init__(self, in_channels, band_kernel_size=13, branch_ratio=0.25, start_k=3, shortcut=True, mode="normal", dilated=False, groups=False, shuffle=False):
        super().__init__()
        self.mode = mode
        self.shortcut = shortcut
        self.shuffle = shuffle
        if self.mode == "convnext":
            fc = True
        elif self.mode == "preposition":        # mlp preposition
            fc = "mlp"
        else:
            fc = True

        if self.mode != "delcab":   # del contextaware_branch
            self.ctx = RepCTXBlock(in_channels, band_kernel_size, branch_ratio, dilated=dilated, fc=fc, start_k=start_k)
        # self.project = Conv(in_channels, in_channels, 1, act=False)

        if self.mode != "convnext":
            self.project = Conv(in_channels, in_channels, 1, act=True)
        if fc == "mlp":
            self.ffn = Conv(in_channels, in_channels, 1, act=True)
        if self.mode == "metanext":
            self.ffn = RepMLP(in_channels)
        if self.mode == "idlm":        # identity local mixer
            self.ffn = nn.Identity()
        else:
            self.ffn = RepBlock3(in_channels, in_channels, groups=groups if groups else in_channels*4, e=4)


    def forward(self, x):
        if self.shuffle:
            x = channel_shuffle(x, 2)
            return self.ffn(self.project(x) * self.ctx(x)) + x if self.shortcut else self.ffn(
                self.project(x) * self.ctx(x))
        if self.mode == "muladd":
            ctx = self.ctx(x)
            return self.ffn(self.project(x) * ctx + ctx) + x if self.shortcut else self.ffn(self.project(x) * ctx + ctx)
        if self.mode == "convnext":
            return self.ffn(self.ctx(x)) + x if self.shortcut else self.ffn(self.ctx(x))
        if self.mode == "metanext":
            return self.ffn(self.project(x) * self.ctx(x) + x)
        if self.mode == "delcab":
            return self.ffn(self.project(x)) + x if self.shortcut else self.ffn(self.project(x))

        return self.ffn(self.project(x)*self.ctx(x))+x if self.shortcut else self.ffn(self.project(x)*self.ctx(x))


    def switch_to_deploy(self):
        if self.mode != "delcab":
            self.ctx.switch_to_deploy()
        if isinstance(self.ffn, (RepBlock3,)):
            self.ffn.switch_to_deploy()

class HorizontalContextBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, start_k=1):
        super(HorizontalContextBlock, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        # Define the strip depthwise convolutions
        self.strips = nn.ModuleList([
                conv_bn(in_channels, in_channels, (1, k), stride=1, padding=(0, k // 2), groups=in_channels, bias=False)
            for k in range(start_k, kernel_size + 1, 2)
        ])

    def forward(self, x):
        # Apply each strip depthwise convolution and sum the results
        conv_results = [strip(x) for strip in self.strips]
        out = sum(conv_results)
        # Pass through the 1x1 convolution and activation function
        # out = self.fc(out)
        return out

    def switch_to_deploy(self):
        # Initialize an empty weight tensor for the fused convolution
        device = self.strips[0].conv.weight.device
        fused_kernel_size = self.kernel_size
        fused_weight = torch.zeros(self.strips[0].conv.weight.size(0), 1, 1, fused_kernel_size).to(device)
        fused_bias = torch.zeros(self.strips[0].conv.weight.size(0)).to(device)

        # Reparameterize each strip and accumulate their contributions
        for strip in self.strips:
            strip.switch_to_deploy()
            kernel_size = strip.conv.kernel_size[1]
            padding_left = (fused_kernel_size - kernel_size) // 2
            padding_right = fused_kernel_size - kernel_size - padding_left

            weight_padded = torch.nn.functional.pad(strip.conv.weight, (padding_left, padding_right))
            fused_weight += weight_padded
            fused_bias += strip.conv.bias

        # Create the fused convolution layer
        self.fused_conv = nn.Conv2d(
            in_channels=self.strips[0].conv.in_channels,
            out_channels=self.strips[0].conv.out_channels,
            kernel_size=(1, fused_kernel_size),
            padding=(0, fused_kernel_size // 2),
            groups=self.strips[0].conv.groups,
            bias=True
        )
        self.fused_conv.weight.data = fused_weight
        self.fused_conv.bias.data = fused_bias

        # Remove the original strips and replace forward method to use the fused convolution
        del self.strips
        self.forward = self.forward_fuse

    def forward_fuse(self, x):
        out = self.fused_conv(x)
        return out

class VerticalContextBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, start_k=1):
        super(VerticalContextBlock, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size must be odd"

        # Define the strip depthwise convolutions
        self.strips = nn.ModuleList([
                conv_bn(in_channels, in_channels, (k, 1), stride=1, padding=(k // 2, 0), groups=in_channels, bias=False)
            for k in range(start_k, kernel_size + 1, 2)
        ])

    def forward(self, x):
        # Apply each strip depthwise convolution and sum the results
        conv_results = [strip(x) for strip in self.strips]
        out = sum(conv_results)
        # Pass through the 1x1 convolution and activation function
        # out = self.fc(out)
        return out

    def switch_to_deploy(self):
        # Initialize an empty weight tensor for the fused convolution
        device = self.strips[0].conv.weight.device
        fused_kernel_size = max([strip.conv.kernel_size[0] for strip in self.strips])
        fused_weight = torch.zeros(self.strips[0].conv.weight.size(0), 1, fused_kernel_size, 1).to(device)
        fused_bias = torch.zeros(self.strips[0].conv.weight.size(0)).to(device)

        # Reparameterize each strip and accumulate their contributions
        for strip in self.strips:
            strip.switch_to_deploy()
            kernel_size = strip.conv.kernel_size[0]
            padding_top = (fused_kernel_size - kernel_size) // 2
            padding_bottom = fused_kernel_size - kernel_size - padding_top

            weight_padded = F.pad(strip.conv.weight, (0, 0, padding_top, padding_bottom))
            fused_weight += weight_padded
            fused_bias += strip.conv.bias

        # Create the fused convolution layer
        self.fused_conv = nn.Conv2d(
            in_channels=self.strips[0].conv.in_channels,
            out_channels=self.strips[0].conv.out_channels,
            kernel_size=(fused_kernel_size, 1),
            padding=(fused_kernel_size // 2, 0),
            groups=self.strips[0].conv.groups,
            bias=True
        )
        self.fused_conv.weight.data = fused_weight
        self.fused_conv.bias.data = fused_bias

        # Remove the original strips and replace forward method to use the fused convolution
        del self.strips
        self.forward = self.forward_fuse

    def forward_fuse(self, x):
        out = self.fused_conv(x)
        # out = self.fc(out)
        return out

class HorizontalContextBlockDilated(nn.Module):
    def __init__(self, in_channels, kernel_size, start_k=1):
        super(HorizontalContextBlockDilated, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        # Define the strip depthwise convolutions
        self.strips = nn.ModuleList([
                conv_bn(in_channels, in_channels, (1, k), stride=1, padding=(0, k // 2), groups=in_channels, bias=False)
            for k in range(start_k, kernel_size + 1, 2)
        ])
        # Define horizontal dilated convolutions
        dilations = [3, 4, 5, 2]
        dilated_kernels = [3, 3, 3, 5]
        self.dilated_convs = nn.ModuleList([
            conv_bn(in_channels, in_channels, (1, k), stride=1, padding=(0, ((k-1)*dil+1) // 2), dilation=(1, dil), groups=in_channels, bias=False)
            for k, dil in zip(dilated_kernels, dilations)
        ])

    def forward(self, x):
        # Apply each strip depthwise convolution and sum the results
        conv_results = [strip(x) for strip in self.strips]
        dilated_results = [dilated_conv(x) for dilated_conv in self.dilated_convs]

        out = sum(conv_results)+sum(dilated_results)
        # Pass through the 1x1 convolution and activation function
        # out = self.fc(out)
        return out

    def switch_to_deploy(self):
        # Initialize an empty weight tensor for the fused convolution
        device = self.strips[0].conv.weight.device
        fused_kernel_size = self.kernel_size
        fused_weight = torch.zeros(self.strips[0].conv.weight.size(0), 1, 1, fused_kernel_size).to(device)
        fused_bias = torch.zeros(self.strips[0].conv.weight.size(0)).to(device)

        # Reparameterize each strip and accumulate their contributions
        for strip in self.strips:
            strip.switch_to_deploy()
            kernel_size = strip.conv.kernel_size[1]
            padding_left = (fused_kernel_size - kernel_size) // 2
            padding_right = fused_kernel_size - kernel_size - padding_left

            weight_padded = F.pad(strip.conv.weight, (padding_left, padding_right))
            fused_weight += weight_padded
            fused_bias += strip.conv.bias

        # Reparameterize each dilated convolution
        identity_kernel = torch.ones((1, 1, 1, 1)).to(device)
        for dilated_conv in self.dilated_convs:
            dilated_conv.switch_to_deploy()
            kernel_size = dilated_conv.conv.kernel_size[1]
            dilation = dilated_conv.conv.dilation[1]
            equivalent_kernel_size = dilation * (kernel_size - 1) + 1
            equivalent_weight = F.conv_transpose2d(dilated_conv.conv.weight, identity_kernel, stride=dilation)
            padding_left = (fused_kernel_size - equivalent_kernel_size) // 2
            padding_right = fused_kernel_size - equivalent_kernel_size - padding_left

            weight_padded = F.pad(equivalent_weight, (padding_left, padding_right))
            fused_weight += weight_padded
            fused_bias += dilated_conv.conv.bias

        # Create the fused convolution layer
        self.fused_conv = nn.Conv2d(
            in_channels=self.strips[0].conv.in_channels,
            out_channels=self.strips[0].conv.out_channels,
            kernel_size=(1, fused_kernel_size),
            padding=(0, fused_kernel_size // 2),
            groups=self.strips[0].conv.groups,
            bias=True
        )
        self.fused_conv.weight.data = fused_weight
        self.fused_conv.bias.data = fused_bias

        # Remove the original strips and replace forward method to use the fused convolution
        del self.strips
        del self.dilated_convs

        self.forward = self.forward_fuse

    def forward_fuse(self, x):
        return self.fused_conv(x)

class VerticalContextBlockDilated(nn.Module):
    def __init__(self, in_channels, kernel_size, start_k=1):
        super(VerticalContextBlockDilated, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size must be odd"

        # Define the strip depthwise convolutions
        self.strips = nn.ModuleList([
                conv_bn(in_channels, in_channels, (k, 1), stride=1, padding=(k // 2, 0), groups=in_channels, bias=False)
            for k in range(start_k, kernel_size + 1, 2)
        ])
        # Define horizontal dilated convolutions
        dilations = [3, 4, 5, 2]
        dilated_kernels = [3, 3, 3, 5]
        self.dilated_convs = nn.ModuleList([
            conv_bn(in_channels, in_channels, (k, 1), stride=1, padding=(((k-1)*dil+1) // 2, 0), dilation=(dil,1), groups=in_channels, bias=False)
            for k, dil in zip(dilated_kernels, dilations)
        ])


    def forward(self, x):
        # Apply each strip depthwise convolution and sum the results
        conv_results = [strip(x) for strip in self.strips]
        dilated_results = [dilated_conv(x) for dilated_conv in self.dilated_convs]

        out = sum(conv_results)+sum(dilated_results)
        # Pass through the 1x1 convolution and activation function
        # out = self.fc(out)
        return out

    def switch_to_deploy(self):
        # Initialize an empty weight tensor for the fused convolution
        device = self.strips[0].conv.weight.device
        fused_kernel_size = max([strip.conv.kernel_size[0] for strip in self.strips])
        fused_weight = torch.zeros(self.strips[0].conv.weight.size(0), 1, fused_kernel_size, 1).to(device)
        fused_bias = torch.zeros(self.strips[0].conv.weight.size(0)).to(device)

        # Reparameterize each strip and accumulate their contributions
        for strip in self.strips:
            strip.switch_to_deploy()
            kernel_size = strip.conv.kernel_size[0]
            padding_top = (fused_kernel_size - kernel_size) // 2
            padding_bottom = fused_kernel_size - kernel_size - padding_top

            weight_padded = F.pad(strip.conv.weight, (0, 0, padding_top, padding_bottom))
            fused_weight += weight_padded
            fused_bias += strip.conv.bias

        # Reparameterize each dilated convolution
        identity_kernel = torch.ones((1, 1, 1, 1)).to(device)
        for dilated_conv in self.dilated_convs:
            dilated_conv.switch_to_deploy()
            kernel_size = dilated_conv.conv.kernel_size[0]
            dilation = dilated_conv.conv.dilation[0]
            equivalent_kernel_size = dilation * (kernel_size - 1) + 1
            equivalent_weight = F.conv_transpose2d(dilated_conv.conv.weight, identity_kernel, stride=dilation)
            padding_left = (fused_kernel_size - equivalent_kernel_size) // 2
            padding_right = fused_kernel_size - equivalent_kernel_size - padding_left

            weight_padded = F.pad(equivalent_weight, (0, 0, padding_left, padding_right))
            fused_weight += weight_padded
            fused_bias += dilated_conv.conv.bias

        # Create the fused convolution layer
        self.fused_conv = nn.Conv2d(
            in_channels=self.strips[0].conv.in_channels,
            out_channels=self.strips[0].conv.out_channels,
            kernel_size=(fused_kernel_size, 1),
            padding=(fused_kernel_size // 2, 0),
            groups=self.strips[0].conv.groups,
            bias=True
        )
        self.fused_conv.weight.data = fused_weight
        self.fused_conv.bias.data = fused_bias

        # Remove the original strips and replace forward method to use the fused convolution
        del self.strips
        del self.dilated_convs
        self.forward = self.forward_fuse

    def forward_fuse(self, x):
        return self.fused_conv(x)






