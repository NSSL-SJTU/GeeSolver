import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import named_apply, adapt_input_conv

_logger = logging.getLogger(__name__)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num=-1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, num=num)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., num=-1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.num = num

    def forward(self, x):
        num = self.num
        if num == -1:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = q.reshape(B, self.num_heads, 7, 20, C // self.num_heads).permute(0, 1, 3, 2, 4)
            k = k.reshape(B, self.num_heads, 7, 20, C // self.num_heads).permute(0, 1, 3, 2, 4)
            v = v.reshape(B, self.num_heads, 7, 20, C // self.num_heads).permute(0, 1, 3, 2, 4)

            k_ls = []
            for i in range(num, 0, -1):
                k_ls.append(F.pad(k[:, :, :-num, :, :], pad=(0, 0, 0, 0, num, 0), mode='constant', value=0))
            k_ls.append(k)
            for i in range(1, num+1, 1):
                k_ls.append(F.pad(k[:, :, num:, :, :], pad=(0, 0, 0, 0, 0, num), mode='constant', value=0))

            k_new = torch.stack(k_ls).permute(1, 2, 3, 0, 4, 5)
            k_new = k_new.transpose(3, 4).reshape(B, self.num_heads, 20, 7 * (2 * num + 1), C // self.num_heads)

            v_ls = []
            for i in range(num, 0, -1):
                v_ls.append(F.pad(v[:, :, :-num, :, :], pad=(0, 0, 0, 0, num, 0), mode='constant', value=0))
            v_ls.append(v)
            for i in range(1, num + 1, 1):
                v_ls.append(F.pad(v[:, :, num:, :, :], pad=(0, 0, 0, 0, 0, num), mode='constant', value=0))

            v_new = torch.stack(v_ls).permute(1, 2, 3, 0, 4, 5)
            v_new = v_new.transpose(3, 4).reshape(B, self.num_heads, 20, 7 * (2 * num + 1), C // self.num_heads)

            attn = (q @ k_new.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v_new).permute(0, 3, 2, 1, 4).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class OriginVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, num=-1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, num=num)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
