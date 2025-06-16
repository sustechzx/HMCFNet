import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
import math

from model.hyp_crossvit import DropPath, Mlp


class SequentialEncoder(nn.Sequential):
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Apply self-attention
        attn_output,_ = self.attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)

        # Apply feed forward network
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoderBlock, self).__init__()
        self.layers = SequentialEncoder(
            *[TransformerEncoderLayer(input_dim, num_heads, hidden_dim)
              for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x



class Attention_img(nn.Module):
    def __init__(self, dim=512, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,):
        x_img = x[:, :1, :]
        x_pps = x[:, 1:, :]

        B, N, C = x_img.shape
        kv_img = self.kv(x_img).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv_pps = self.kv(x_pps).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k_img, v_img = kv_img.unbind(0) # make torchscript happy (cannot use tensor as tuple)
        k_pps, v_pps = kv_pps.unbind(0)

        q_img = x_pps.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q_pps = x_pps.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q_img @ k_img.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_img = (attn @ v_pps).transpose(1, 2).reshape(B, N, C)

        # attn = (q_pps @ k_img.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        #
        # x_img = (attn @ v_img).transpose(1, 2).reshape(B, N, C)


        x_img = self.proj(x_img)
        x_img = self.proj_drop(x_img)

        return x_img

class Attention_pps(nn.Module):
    def __init__(self, dim=512, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,):
        x_img = x[:, :1, :]
        x_pps = x[:, 1:, :]

        B, N, C = x_pps.shape
        kv_img = self.kv(x_img).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv_pps = self.kv(x_pps).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k_img, v_img = kv_img.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        k_pps, v_pps = kv_pps.unbind(0)

        q_img = x_pps.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q_pps = x_pps.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q_pps @ k_pps.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_pps = (attn @ v_img).transpose(1, 2).reshape(B, N, C)

        # attn = (q_img @ k_pps.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        #
        # x_pps = (attn @ v_pps).transpose(1, 2).reshape(B, N, C)


        x_pps = self.proj(x_pps)
        x_pps = self.proj_drop(x_pps)

        return x_pps


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_img = Attention_img(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.attn_lm = Attention_pps(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x_img = x[:,:1, :]
        x_lm = x[:,1:, :]
        x_img = x_img + self.drop_path(self.attn_img(self.norm1(x)))
        x_img = x_img + self.drop_path(self.mlp1(self.norm2(x_img)))

        x_lm = x_lm + self.drop_path(self.attn_lm(self.norm3(x)))
        x_lm = x_lm + self.drop_path(self.mlp2(self.norm4(x_lm)))
        x = torch.cat((x_img, x_lm), dim=1)
        x = self.conv(x)

        return x
        # return x, x_img, x_lm


class FusionModel2(nn.Module):
    def __init__(self):
        super(FusionModel2, self).__init__()

        self.blocks = Block(dim=512, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)

        self.fusion_encoder = TransformerEncoderBlock(512, num_heads=2, hidden_dim=512, num_layers=1)

        self.dropout = nn.Dropout(0.)

        self.layer_norm = nn.LayerNorm(512)

        self.linear = nn.Linear(1024, 512)

        # self.classifier = nn.Sequential(nn.Linear(1024, 512),
        #                                 # nn.ReLU(),
        #                                 # nn.Linear(512, 512),
        #                                 # nn.ReLU(),
        #                                 # nn.Linear(512, 512),
        #                                 nn.Sigmoid(),
        #                                 nn.Linear(512, 2)
        #                                 )

        self.classifier = nn.Sequential(nn.Linear(512, 256),
                                        # nn.ReLU(),
                                        # nn.Linear(512, 512),
                                        # nn.ReLU(),
                                        # nn.Linear(512, 512),
                                        nn.Sigmoid(),
                                        nn.Linear(256, 4)
                                        )

    def forward(self, x1, x2):

        # JCAM+DCAFM
        joint_x = self.linear(torch.cat((x1, x2), dim=1))

        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        joint_x = joint_x.unsqueeze(1)

        # x1 = self.layer_norm(x1)
        # x2 = self.layer_norm(x2)
        # joint_x = self.layer_norm(joint_x)

        ############################ fusion ##################################
        d = x1.shape[-1]

        scores = torch.bmm(joint_x, x2.transpose(1, 2) / math.sqrt(d))

        sm = F.softmax(scores, dim=-1)

        output_A = torch.bmm(self.dropout(F.softmax(scores, dim=-1)), x2)
        scores = torch.bmm(joint_x, x1.transpose(1, 2)) / math.sqrt(d)
        output_B = torch.bmm(self.dropout(F.softmax(scores, dim=-1)), x1)

        joint_attention = torch.cat((output_A, output_B), dim=1)

        out = self.blocks(joint_attention)
        # out, x_img, x_lm = self.blocks(joint_attention)

        out = out.squeeze(1)

        attention = self.fusion_encoder(out)

        output = self.classifier(attention)


        ######################################### B ############################
        # x = torch.cat((x1, x2), dim=1)
        #
        # attention = self.blocks(x).squeeze(1)
        #
        # attention = self.fusion_encoder(attention)
        #
        # output = self.classifier(attention)

        # return output, output_A, output_B, x_img, x_lm
        return output, attention


# x1 = torch.randn(128, 512)
# x2 = torch.randn(128, 512)

# model = FusionModel2()
#
# output = model(x1,x2)
#
# print(output)