import torch
import torch.nn as nn
import torch.nn.functional as F
from DEAnet.modules.deconv import DEConv
from TransUNet.unet.unet_parts import *
from torch.nn import LayerNorm
import math

class color_net(nn.Module):

    def __init__(self, d_hist=64):
        super(color_net, self).__init__()
        self.g_net = DEU_Net()
        self.c_net = c_net(d_hist)
        self.r_net = SNE_Net()
        
    def forward(self, inputs):
        haze_img = inputs[:, 0:4, :, :]
        polar_prior = inputs[:, 4:9, :, :] # IrawD & IrawS & AoP & DoP
        viewing_encoding = inputs[:, 9:12, :, :]

        dehaze_img = self.g_net(haze_img)
        inputs_cnet = torch.cat([polar_prior, dehaze_img], dim=1)
        normal_hist, color_feature = self.c_net(inputs_cnet)
        outputs = self.r_net(haze_img, dehaze_img, viewing_encoding, color_feature)
        # outputs = self.r_net(haze_img, dehaze_img, color_feature)
        
        return dehaze_img, normal_hist, outputs
    
class pce(nn.Module):
    # parmid color embedding

    def __init__(self):
        super(pce, self).__init__()

        self.cma_3 = cma(128, 64)
        self.cma_2 = cma(64, 32)
        self.cma_1 = cma(32, 16)
        
    def forward(self, c, shortcuts):
        
        # change channels
        x_3_color, c_2 = self.cma_3(c, shortcuts[2])
        x_2_color, c_1 = self.cma_2(c_2, shortcuts[1])
        x_1_color, _ = self.cma_1(c_1, shortcuts[0])
        
        return [x_1_color, x_2_color, x_3_color]
        
class cma(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(cma, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.InstanceNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode='nearest'))
        
    def forward(self, c, x):
        # x: gray image features 
        # c: color features

        # l1 distance
        channels = c.shape[1]
        sim_mat_l1 = -torch.abs(x-c) # <0  (b,c,h,w)
        sim_mat_l1 = torch.sum(sim_mat_l1, dim=1, keepdim=True) # (b,1,h,w)
        sim_mat_l1 = torch.sigmoid(sim_mat_l1) # (0, 0.5) (b,1,h,w)
        sim_mat_l1 = sim_mat_l1.repeat(1,channels,1, 1)
        sim_mat_l1 = 2*sim_mat_l1 # (0, 1)

        # cos distance
        sim_mat_cos = x*c # >0 (b,c,h,w)
        sim_mat_cos = torch.sum(sim_mat_cos, dim=1, keepdim=True) # (b,1,h,w)       
        sim_mat_cos = torch.tanh(sim_mat_cos) # (0, 1) (b,1,h,w)
        sim_mat_cos = sim_mat_cos.repeat(1,channels,1, 1) # (0, 1)
        
        # similarity matrix
        sim_mat = sim_mat_l1 * sim_mat_cos # (0, 1)
        
        # color embeding
        x_color = x + c*sim_mat
        
        # color features upsample
        c_up = self.conv(c)
        
        return x_color, c_up
           
class r_net(nn.Module):

    def __init__(self, depth=[2, 2, 2, 2]):
        super(r_net, self).__init__()
        
        base_channel = 32
        
        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel*2, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Down_scale(base_channel*2),
            BasicConv(base_channel*4, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            Down_scale(base_channel*4),
        ])
        
        # Middle
        self.middle = nn.Sequential(*[RB(base_channel*8) for _ in range(depth[3])])
        
        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale(base_channel*8),
            BasicConv(base_channel*8, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            Up_scale(base_channel*4),
            BasicConv(base_channel*4, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Up_scale(base_channel*2),
            BasicConv(base_channel*2, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
        ])

        # conv
        self.conv_first = BasicConv(11, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 3, 3, 1, 1)
        self.pce = pce()

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts
    
    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i//3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x
       
    def forward(self, img_low, gray, viewing_encoding, color_feature):
        x = torch.cat([img_low, gray, viewing_encoding], 1)
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        x =  self.middle(x)
        shortcuts = self.pce(color_feature, shortcuts)
        x = self.decoder(x, shortcuts)
        img_color = self.conv_last(x)
        # img_color = (torch.tanh(x) + 1) / 2
        return img_color

class c_net(nn.Module):

    def __init__(self, d_hist, depth=[2, 2, 2]):
        super(c_net, self).__init__()
        
        base_channel = 32
        
        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel*2, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Down_scale(base_channel*2),
            BasicConv(base_channel*4, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
        ])

        self.conv_first = BasicConv(9, base_channel, 3, 1)
        
        # color hist
        self.conv_color = BasicConv(base_channel*4, 256*3, 3, 1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, d_hist)
        self.softmax = nn.Softmax(dim=2)

        self.d_hist = d_hist
        
    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts
    
    def color_forward(self, x):
        x = self.conv_color(x)
        x = self.pooling(x)
        x = torch.reshape(x, (-1, 3, 256))
        color_hist = self.softmax(self.fc(x))
        return color_hist
        
    def forward(self, x):
        # # 将取值范围为[-1,1]的法向量归一化回0-1，以保证后续公平比较
        # x = (x + 1) / 2.0
        x = self.conv_first(x)
        x, _ = self.encoder(x)
        color_hist = self.color_forward(x)
        
        return color_hist, x
 
class g_net(nn.Module):

    def __init__(self, depth=[2, 2, 2, 2]):
        super(g_net, self).__init__()

        base_channel = 32
        
        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv2DEConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB_for_gnet(base_channel) for _ in range(depth[0])]),
            Down_scale_for_gnet(base_channel),
            BasicConv2DEConv(base_channel*2, base_channel*2, 3, 1),
            nn.Sequential(*[RB_for_gnet(base_channel*2) for _ in range(depth[1])]),
            Down_scale_for_gnet(base_channel*2),
            BasicConv2DEConv(base_channel*4, base_channel*4, 3, 1),
            nn.Sequential(*[RB_for_gnet(base_channel*4) for _ in range(depth[2])]),
            Down_scale_for_gnet(base_channel*4),
        ])
        
        # Middle
        self.middle = nn.Sequential(*[RB_for_gnet(base_channel*8) for _ in range(depth[3])])
        
        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale_for_gnet(base_channel*8),
            BasicConv2DEConv(base_channel*8, base_channel*4, 3, 1),
            nn.Sequential(*[RB_for_gnet(base_channel*4) for _ in range(depth[2])]),
            Up_scale_for_gnet(base_channel*4),
            BasicConv2DEConv(base_channel*4, base_channel*2, 3, 1),
            nn.Sequential(*[RB_for_gnet(base_channel*2) for _ in range(depth[1])]),
            Up_scale_for_gnet(base_channel*2),
            BasicConv2DEConv(base_channel*2, base_channel, 3, 1),
            nn.Sequential(*[RB_for_gnet(base_channel) for _ in range(depth[0])]),
        ])

        # conv
        self.conv_first = BasicConv2DEConv(4, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 4, 3, 1, 1)

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts
    
    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i//3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x
        
    def forward(self, x):
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        x =  self.middle(x)
        x = self.decoder(x, shortcuts)
        x = self.conv_last(x)
        gray = (torch.tanh(x) + 1) / 2
        return gray

class BasicConv2DEConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, activation=True, transpose=False):
        super(BasicConv2DEConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, activation=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.LeakyReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class RB_for_gnet(nn.Module):
    def __init__(self, channels):
        super(RB_for_gnet, self).__init__()
        self.layer_1 = BasicConv2DEConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv2DEConv(channels, channels, 3, 1)

    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x

class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.layer_1 = BasicConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv(channels, channels, 3, 1)
        
    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x

class Down_scale_for_gnet(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale_for_gnet, self).__init__()
        self.main = BasicConv2DEConv(in_channel, in_channel*2, 3, 2)

    def forward(self, x):
        return self.main(x)
class Down_scale(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel*2, 3, 2)

    def forward(self, x):
        return self.main(x)

class Up_scale_for_gnet(nn.Module):
    def __init__(self, in_channel):
        super(Up_scale_for_gnet, self).__init__()
        self.main = BasicConv2DEConv(in_channel, in_channel//2, kernel_size=4, activation=True, stride=2, transpose=True)

    def forward(self, x):
        return self.main(x)

class Up_scale(nn.Module):
    def __init__(self, in_channel):
        super(Up_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel//2, kernel_size=4, activation=True, stride=2, transpose=True)

    def forward(self, x):
        return self.main(x)


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class DEconv_block(nn.Module):
    """
    Detail Enhance Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(DEconv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DEConv(out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class DEU_Net(nn.Module):

    def __init__(self, in_ch=4, out_ch=4):
        super(DEU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DEconv_block(in_ch, filters[0])
        self.Conv2 = DEconv_block(filters[0], filters[1])
        self.Conv3 = DEconv_block(filters[1], filters[2])
        self.Conv4 = DEconv_block(filters[2], filters[3])
        self.Conv5 = DEconv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = DEconv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = DEconv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = DEconv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = DEconv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out


class Mlp(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(dim, dim * mult)
        self.fc2 = nn.Linear(dim * mult, dim)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = nn.Dropout(dropout)  # 防止过拟合

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):  # query_dim=512
        super(Attention, self).__init__()
        self.heads = heads  # # 8个Attention
        context_dim = context_dim or query_dim
        hidden_dim = max(query_dim, context_dim)
        # self.dim_head = int(hidden_dim / self.heads)
        self.dim_head = dim_head  # 64个维度
        self.all_head_dim = self.heads * self.dim_head  # 512

        ## All linear layers (including query, key, and value layers and dense block layers)
        ## preserve the dimensionality of their inputs and are tiled over input index dimensions #
        # (i.e. applied as a 1 × 1 convolution).

        self.query = nn.Linear(query_dim, self.all_head_dim)  # (b n d_q) -> (b n hd) q和k的维度一定是一样的
        self.key = nn.Linear(context_dim, self.all_head_dim)  # (b m d_c) -> (b m hd)
        self.value = nn.Linear(context_dim, self.all_head_dim)  # (b m d_c) -> (b m hd)
        self.out = nn.Linear(self.all_head_dim, query_dim)  # (b n hd) -> (b n d_q)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.heads, self.dim_head)  # (6, 1024, 8, 64)即512拆成8*64
        x = x.view(*new_x_shape)  # (b n hd) -> (b n h d)
        return x.permute(0, 2, 1, 3)  # (b n h d) -> (b h n d)

    # b为batch_size 6
    # n=1024
    # h=8
    # d_q为query_dim
    # hd=512 8*64
    # m=1024
    # d=64
    # d_c为context_dim=512
    def forward(self, query, context=None):
        if context is None:
            context = query
        mixed_query_layer = self.query(query)  # (b n d_q) -> (b n hd)
        mixed_key_layer = self.key(context)  # (b m d_c) -> (b m hd)
        mixed_value_layer = self.value(context)  # (b m d_c) -> (b m hd)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (b h n d)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (b h m d)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (b h m d)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (b h n m)
        attention_scores = attention_scores / math.sqrt(self.dim_head)  # (b h n m)
        attention_probs = self.softmax(attention_scores)  # (b h n m)
        attention_probs = self.attn_dropout(attention_probs)  # (b h n m)

        context_layer = torch.matmul(attention_probs, value_layer)  # (b h n m) , (b h m d) -> (b h n d)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (b n h d)    contiguous()时，会强制拷贝一份tensor
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Block(nn.Module):
    def __init__(self, hidden_size, droppath=0., dropout=0.0):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size)
        self.attn = Attention(hidden_size, dropout=dropout)
        # self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.drop_path = nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.attn(self.attention_norm(x))) + x
        x = self.drop_path(self.ffn(self.ffn_norm(x))) + x
        return x

class TransUnet(nn.Module):
    def __init__(self, n_channels=15, n_classes=3, dim=64, residual_num=16, bilinear=True, norm='bn', dropout=0.0,
                 skip_res=True):
        super(TransUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.skip_res = skip_res

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(dim, dim * 2, norm=norm)
        self.down2 = Down(dim * 2, dim * 4, norm=norm)
        self.down3 = Down(dim * 4, dim * 8, norm=norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(dim * 8, dim * 16 // factor, norm=norm)
        self.up1 = Up(dim * 16, dim * 8 // factor, bilinear)
        self.up2 = Up(dim * 8, dim * 4 // factor, bilinear)
        self.up3 = Up(dim * 4, dim * 2 // factor, bilinear)
        self.up4 = Up(dim * 2, dim, bilinear)
        self.outc = OutConv(dim, n_classes)
        self.resblock_layers = nn.ModuleList([])

        # Missing positinal encoding
        for i in range(residual_num):
            self.resblock_layers.append(Block(dim * 8, dropout=dropout))  # 512

            # self.resblock_layers.append(BasicBlock(512, 512, norm_layer=nn.LayerNorm))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        b, c, h, w = x5.size()
        x5 = torch.reshape(x5, [b, c, h * w]).permute(0, 2, 1)
        # print(x5.size())
        for resblock in self.resblock_layers:
            residual = resblock(x5)
            if self.skip_res:
                # print("residual", residual[0,0,0])
                # import ipdb; ipdb.set_trace()
                x5 = residual
                # print("x5", x5[0,0,0])
            else:
                x5 = x5 + residual
        x5 = torch.reshape(x5.permute(0, 2, 1), [b, c, h, w])
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # print(x.device)
        # print(b)
        return logits

class SNE_Net(nn.Module):
    def __init__(self, n_channels=11, n_classes=3, dim=32, residual_num=16, bilinear=True, norm='bn', dropout=0.0, skip_res=True):
        super(SNE_Net, self).__init__()

        self.skip_res = skip_res
        factor = 2 if bilinear else 1

        # Encoder
        self.Encoder = nn.ModuleList([
            Down(dim, dim * 2, norm=norm),
            Down(dim * 2, dim * 4, norm=norm),
            Down(dim * 4, dim * 8, norm=norm),
        ])

        # Middle
        self.middle = nn.Sequential(*[Block(dim * 8, dropout=dropout) for _ in range(residual_num)])

        # Decoder
        self.Decoder = nn.ModuleList([
            Up(dim * 8, dim * 4 // factor, bilinear=False),
            Up(dim * 4, dim * 2 // factor, bilinear),
            Up(dim * 2, dim, bilinear),
        ])

        # Conv
        self.conv_first = DoubleConv(n_channels, dim)
        self.conv_last = OutConv(dim, n_classes)
        self.pce = pce()

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            shortcuts.append(x)
            x = self.Encoder[i](x)
        return x, shortcuts

    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            index = len(shortcuts) - (i // 1 + 1)
            # x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x, shortcuts[index])
        return x

    def forward(self, haze_img, dehaze_img, viewing_encoding, color_feature):
        x = torch.cat([haze_img, dehaze_img, viewing_encoding], 1)
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        b, c, h, w = x.size()
        x = torch.reshape(x, [b, c, h * w]).permute(0, 2, 1)
        x = self.middle(x)
        x = torch.reshape(x.permute(0, 2, 1), [b, c, h, w])
        shortcuts = self.pce(color_feature, shortcuts)
        x = self.decoder(x, shortcuts)
        img_color = self.conv_last(x)
        # img_color = (torch.tanh(x) + 1) / 2
        return img_color