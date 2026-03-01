import torch
import torch.nn as nn
from torch.nn.functional import normalize
from DEAnet.modules.deconv import DEConv
from torch.nn import LayerNorm
import math

class NetWork(nn.Module):

    def __init__(self, d_hist=64):
        super(NetWork, self).__init__()
        self.Descattering_Net = descattering_net()
        self.Polarization_Prior_Net = polarization_prior_net(d_hist)
        self.Normal_Prediction_Net = normal_prediction_net()

    def forward(self, inputs):
        # UD_SfP Input_1
        scattering_img = inputs[:, 0:4, :, :]
        polar_prior = inputs[:, 4:9, :, :] # IrawD & IrawS & AoP & DoP
        viewing_encoding = inputs[:, 9:12, :, :]

        # UD_SfP Input_2
        # scattering_img = inputs[:, 0:4, :, :]
        # polar_prior = inputs[:, 4:13, :, :] # Diffuse & Specular1 & Specular2
        # viewing_encoding = inputs[:, 13:16, :, :]

        descattering_img = self.Descattering_Net(scattering_img)
        inputs_polar_prior_net = torch.cat([polar_prior, descattering_img], dim=1)
        normal_hist, normal_feature = self.Polarization_Prior_Net(inputs_polar_prior_net)
        outputs = self.Normal_Prediction_Net(scattering_img, descattering_img, viewing_encoding, normal_feature)

        return descattering_img, normal_hist, outputs

class descattering_net(nn.Module):
    def __init__(self, in_ch=4, out_ch=4):
        super(descattering_net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Encoder = nn.ModuleList([
            DEconv_block(in_ch, filters[0]),
            nn.MaxPool2d(2),
            DEconv_block(filters[0], filters[1]),
            nn.MaxPool2d(2),
            DEconv_block(filters[1], filters[2]),
            nn.MaxPool2d(2),
            DEconv_block(filters[2], filters[3]),
            nn.MaxPool2d(2),
        ])

        self.middle = DEconv_block(filters[3], filters[4])

        self.Decoder = nn.ModuleList([
            up_conv(filters[4], filters[3]),
            DEconv_block(filters[4], filters[3]),
            up_conv(filters[3], filters[2]),
            DEconv_block(filters[3], filters[2]),
            up_conv(filters[2], filters[1]),
            DEconv_block(filters[2], filters[1]),
            up_conv(filters[1], filters[0]),
            DEconv_block(filters[1], filters[0]),
        ])

        self.conv_last = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 0) % 2 == 0:
                shortcuts.append(x)
        return x, shortcuts

    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 1) % 2 == 0:
                index = len(shortcuts) - (i // 2 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x

    def forward(self, x):
        x, shortcuts = self.encoder(x)
        x =  self.middle(x)
        x = self.decoder(x, shortcuts)
        DeScatterImg = self.conv_last(x)
        return DeScatterImg

class polarization_prior_net(nn.Module):

    def __init__(self, d_hist, depth=[2, 2, 2]):
        super(polarization_prior_net, self).__init__()

        base_channel = 32

        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel * 2, base_channel * 2, 3, 1),
            nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
            Down_scale(base_channel * 2),
            BasicConv(base_channel * 4, base_channel * 4, 3, 1),
            nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
        ])

        self.conv_first = BasicConv(9, base_channel, 3, 1)

        # prior hist
        self.conv_prior = BasicConv(base_channel * 4, 256 * 3, 3, 1)
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

    def prior_forward(self, x):
        x = self.conv_prior(x)
        x = self.pooling(x)
        x = torch.reshape(x, (-1, 3, 256))
        prior_hist = self.softmax(self.fc(x))
        return prior_hist

    def forward(self, x):
        # # Normalize the normal vector with a value range of [-1,1] back to 0-1 to ensure fair comparison in the future
        # x = (x + 1) / 2.0
        x = self.conv_first(x)
        x, _ = self.encoder(x)
        prior_hist = self.prior_forward(x)

        return prior_hist, x

class normal_prediction_net(nn.Module):
    def __init__(self, n_channels=4, n_classes=3, dim=32, residual_num=8, bilinear=True, norm='bn', dropout=0.0, skip_res=True):
        super(normal_prediction_net, self).__init__()

        self.skip_res = skip_res
        factor = 2 if bilinear else 1

        # Encoder
        self.Encoder = nn.ModuleList([
            nn.MaxPool2d(2),
            DEconv_block(dim, dim * 2),
            nn.MaxPool2d(2),
            DEconv_block(dim * 2, dim * 4),
            nn.MaxPool2d(2),
            DEconv_block(dim * 4, dim * 8),
        ])

        # Middle
        self.middle = nn.Sequential(*[Transformer_Block(dim * 8, dropout=dropout) for _ in range(residual_num)])

        # Decoder
        self.Decoder = nn.ModuleList([
            nn.ConvTranspose2d(dim * 8, dim * 8 // 2, kernel_size=2, stride=2),
            DEconv_block(dim * 8, dim * 4 // factor),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DEconv_block(dim * 4, dim * 2 // factor),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DEconv_block(dim * 2, dim),
        ])

        # Conv
        self.conv_first = DEconv_block(n_channels, dim)
        self.conv_last = nn.Conv2d(dim, n_classes, kernel_size=1)
        self.pce = pce()

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            if (i + 0) % 2 == 0:
                shortcuts.append(x)
            x = self.Encoder[i](x)
        return x, shortcuts

    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 1) % 2 == 0:
                index = len(shortcuts) - (i // 2 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x

    def forward(self, scattering_img, descattering_img, viewing_encoding, normal_feature):
        # x = torch.cat([scattering_img, descattering_img], 1)
        x = descattering_img
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        b, c, h, w = x.size()
        x = torch.reshape(x, [b, c, h * w]).permute(0, 2, 1)
        x = self.middle(x)
        x = torch.reshape(x.permute(0, 2, 1), [b, c, h, w])
        shortcuts = self.pce(normal_feature, shortcuts)
        x = self.decoder(x, shortcuts)
        img_prior = self.conv_last(x)
        img_prior = normalize(img_prior, p=2, dim=1)
        # img_prior = (torch.tanh(x) + 1) / 2
        return img_prior

# -----------------------------------------------------------------------------
# Other Modules
# -----------------------------------------------------------------------------
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


class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.layer_1 = BasicConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv(channels, channels, 3, 1)

    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x

class Down_scale(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel*2, 3, 2)

    def forward(self, x):
        return self.main(x)

class pce(nn.Module):
    # parmid prior embedding

    def __init__(self):
        super(pce, self).__init__()

        self.cma_3 = cma(128, 64)
        self.cma_2 = cma(64, 32)
        self.cma_1 = cma(32, 16)

    def forward(self, c, shortcuts):
        # change channels
        x_3_prior, c_2 = self.cma_3(c, shortcuts[2])
        x_2_prior, c_1 = self.cma_2(c_2, shortcuts[1])
        x_1_prior, _ = self.cma_1(c_1, shortcuts[0])

        return [x_1_prior, x_2_prior, x_3_prior]


class cma(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(cma, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.InstanceNorm2d(out_channels),
                                  nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, c, x):
        # x: gray image features
        # c: prior features

        # l1 distance
        channels = c.shape[1]
        sim_mat_l1 = -torch.abs(x - c)  # <0  (b,c,h,w)
        sim_mat_l1 = torch.sum(sim_mat_l1, dim=1, keepdim=True)  # (b,1,h,w)
        sim_mat_l1 = torch.sigmoid(sim_mat_l1)  # (0, 0.5) (b,1,h,w)
        sim_mat_l1 = sim_mat_l1.repeat(1, channels, 1, 1)
        sim_mat_l1 = 2 * sim_mat_l1  # (0, 1)

        # cos distance
        sim_mat_cos = x * c  # >0 (b,c,h,w)
        sim_mat_cos = torch.sum(sim_mat_cos, dim=1, keepdim=True)  # (b,1,h,w)
        sim_mat_cos = torch.tanh(sim_mat_cos)  # (0, 1) (b,1,h,w)
        sim_mat_cos = sim_mat_cos.repeat(1, channels, 1, 1)  # (0, 1)

        # similarity matrix
        sim_mat = sim_mat_l1 * sim_mat_cos  # (0, 1)

        # prior embeding
        x_prior = x + c * sim_mat

        # prior features upsample
        c_up = self.conv(c)

        return x_prior, c_up

class Transformer_Block(nn.Module):
    def __init__(self, hidden_size, droppath=0., dropout=0.0):
        super(Transformer_Block, self).__init__()
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

class Mlp(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(dim, dim * mult)
        self.fc2 = nn.Linear(dim * mult, dim)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = nn.Dropout(dropout)  # avoid overfitting

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):  # query_dim=512
        super(Attention, self).__init__()
        self.heads = heads  # # 8 Attention
        context_dim = context_dim or query_dim
        hidden_dim = max(query_dim, context_dim)
        # self.dim_head = int(hidden_dim / self.heads)
        self.dim_head = dim_head  # 64 dim
        self.all_head_dim = self.heads * self.dim_head  # 512

        ## All linear layers (including query, key, and value layers and dense block layers)
        ## preserve the dimensionality of their inputs and are tiled over input index dimensions #
        # (i.e. applied as a 1 × 1 convolution).

        self.query = nn.Linear(query_dim, self.all_head_dim)  # (b n d_q) -> (b n hd) The dimensions of q and k must be the same
        self.key = nn.Linear(context_dim, self.all_head_dim)  # (b m d_c) -> (b m hd)
        self.value = nn.Linear(context_dim, self.all_head_dim)  # (b m d_c) -> (b m hd)
        self.out = nn.Linear(self.all_head_dim, query_dim)  # (b n hd) -> (b n d_q)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.heads, self.dim_head)  # (6, 1024, 8, 64) Divide 512 into 8 * 64
        x = x.view(*new_x_shape)  # (b n hd) -> (b n h d)
        return x.permute(0, 2, 1, 3)  # (b n h d) -> (b h n d)

    # b=batch_size 6
    # n=1024
    # h=8
    # d_q=query_dim
    # hd=512 8*64
    # m=1024
    # d=64
    # d_c=context_dim=512
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
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (b n h d)    When contiguous (), a tensor will be forcibly copied
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output
