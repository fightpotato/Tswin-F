# # ------------------------------------------
# # CSWin Transformer
# # Copyright (c) Microsoft Corporation.
# # Licensed under the MIT License.
# # written By Xiaoyi Dong
# # ------------------------------------------
#
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import partial
# from .helpers import build_model_with_cfg ,named_apply, checkpoint_seq
# from .vision_transformer import checkpoint_filter_fn,get_init_weights_vit
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.helpers import load_pretrained
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
# from einops.layers.torch import Rearrange
# import torch.utils.checkpoint as checkpoint
# import numpy as np
# import time
# from einops import rearrange, repeat
# import math
# from drloc import DenseRelativeLoc
# from munch import Munch
# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
#         'crop_pct': .9, 'interpolation': 'bicubic',
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
#         'first_conv': 'patch_embed.proj', 'classifier': 'head',
#         **kwargs
#     }
#
#
# default_cfgs = {
#     'CSWin_BOAT_64_24322_small_224': _cfg(),
#     'cswin_384': _cfg(
#         crop_pct=1.0
#     ),
#
# }
# class MyConvDila(nn.Module):
#     def __init__(self):
#         super().__init__()
#         #定义第一次空洞卷积
#         # self.Para1 = nn.Parameter(torch.ones(32,3136,64))
#         self.conv1 = nn.Sequential(
#             # nn.Conv2d(in_channels=64,out_channels=512,kernel_size=5,stride=4,dilation=7),#这是不用avg的时候
#             nn.Conv2d(in_channels=64,out_channels=512,kernel_size=5,stride=3,dilation=3),
#             nn.GELU(),
#             nn.AvgPool2d(kernel_size=2,stride=2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=128,out_channels=512,kernel_size=3,stride=2,dilation=1,padding=1),
#             nn.GELU(),
#             nn.AvgPool2d(kernel_size=2,stride=2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,dilation=1,padding=1),
#             nn.GELU()
#         )
#     def forward(self,z):
#         y=list()
#         for i in range(len(z)):
#             if i<2:
#                 O = rearrange(z[i].transpose(1,2),'b c (h w) -> b c h w',h=56)
#                 O = self.conv1(O)
#                 y.append(O)
#             elif i>1 and i<7:
#                 P = rearrange(z[i].transpose(1,2),'b c (h w) -> b c h w',h=28)
#                 P = self.conv2(P)
#                 y.append(P)
#             elif i>6 and i<40 :
#                 J = rearrange(z[i].transpose(1,2),'b c (h w) -> b c h w',h=14)
#                 J = self.conv3(J)
#                 y.append(J)
#             elif i>39 and i<43:
#                 K = rearrange(z[i].transpose(1,2),'b c (h w) -> b c h w',h=7)
#                 y.append(K)
#         U = torch.stack(y).mean(0)
#         U = rearrange(U,'b c h w -> b c (h w)',h=7)
#         U = U.transpose(1,2)
#         return [U]
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# class LePEAttention(nn.Module):
#     def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
#                  qk_scale=None):
#         super().__init__()
#         self.dim = dim
#         self.dim_out = dim_out or dim
#         self.resolution = resolution
#         self.split_size = split_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#         if idx == -1:
#             H_sp, W_sp = self.resolution, self.resolution
#         elif idx == 0:
#             H_sp, W_sp = self.resolution, self.split_size
#         elif idx == 1:
#             W_sp, H_sp = self.resolution, self.split_size
#         else:
#             print("ERROR MODE", idx)
#             exit(0)
#         self.H_sp = H_sp
#         self.W_sp = W_sp
#         stride = 1
#         self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#
#     def im2cswin(self, x):
#         B, N, C = x.shape
#         H = W = int(np.sqrt(N))
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#         x = img2windows(x, self.H_sp, self.W_sp)
#         x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
#         return x
#
#     def get_lepe(self, x, func):
#         B, N, C = x.shape
#         H = W = int(np.sqrt(N))
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#
#         H_sp, W_sp = self.H_sp, self.W_sp
#         x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
#         x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'
#
#         lepe = func(x)  ### B', C, H', W'
#         lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
#
#         x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
#         return x, lepe
#
#     def forward(self, qkv):
#         """
#         x: B L C
#         """
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         ### Img2Window
#         H = W = self.resolution
#         B, L, C = q.shape
#         assert L == H * W, "flatten img_tokens has wrong size"
#
#         q = self.im2cswin(q)
#         k = self.im2cswin(k)
#         v, lepe = self.get_lepe(v, self.get_v)
#
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
#         attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v) + lepe
#         x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C
#
#         ### Window2Img
#         x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C
#
#         return x
#
#
# class ContentAttention(nn.Module):
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.ws = window_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softmax = nn.Softmax(dim=-1)
#         self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
#
#     def forward(self, x, mask=None):
#         # B_, W, H, C = x.shape
#         # x = x.view(B_,W*H,C)
#         B_, N, C = x.shape
#
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
#                                                                                          4)  # 3, B_, self.num_heads,N,D
#         if True:
#             q_pre = qkv[0].reshape(B_ * self.num_heads, N, C // self.num_heads).permute(0, 2,
#                                                                                         1)  # qkv_pre[:,0].reshape(b*self.num_heads,qkvhd//3//self.num_heads,hh*ww)
#             ntimes = int(math.log(N // 49, 2))
#             q_idx_last = torch.arange(N).cuda().unsqueeze(0).expand(B_ * self.num_heads, N)
#             for i in range(ntimes):
#                 bh, d, n = q_pre.shape
#                 q_pre_new = q_pre.reshape(bh, d, 2, n // 2)
#                 q_avg = q_pre_new.mean(dim=-1)  # .reshape(b*self.num_heads,qkvhd//3//self.num_heads,)
#                 q_avg = torch.nn.functional.normalize(q_avg, dim=-2)
#                 iters = 2
#                 for i in range(iters):
#                     q_scores = torch.nn.functional.normalize(q_pre.permute(0, 2, 1), dim=-1).bmm(q_avg)
#                     soft_assign = torch.nn.functional.softmax(q_scores * 100, dim=-1).detach()
#                     q_avg = q_pre.bmm(soft_assign)
#                     q_avg = torch.nn.functional.normalize(q_avg, dim=-2)
#                 q_scores = torch.nn.functional.normalize(q_pre.permute(0, 2, 1), dim=-1).bmm(q_avg).reshape(bh, n,
#                                                                                                             2)  # .unsqueeze(2)
#                 q_idx = (q_scores[:, :, 0] + 1) / (q_scores[:, :, 1] + 1)
#                 _, q_idx = torch.sort(q_idx, dim=-1)
#                 q_idx_last = q_idx_last.gather(dim=-1, index=q_idx).reshape(bh * 2, n // 2)
#                 q_idx = q_idx.unsqueeze(1).expand(q_pre.size())
#                 q_pre = q_pre.gather(dim=-1, index=q_idx).reshape(bh, d, 2, n // 2).permute(0, 2, 1, 3).reshape(bh * 2,
#                                                                                                                 d,
#                                                                                                                 n // 2)
#
#             q_idx = q_idx_last.view(B_, self.num_heads, N)
#             _, q_idx_rev = torch.sort(q_idx, dim=-1)
#             q_idx = q_idx.unsqueeze(0).unsqueeze(4).expand(qkv.size())
#             qkv_pre = qkv.gather(dim=-2, index=q_idx)
#             q, k, v = rearrange(qkv_pre, 'qkv b h (nw ws) c -> qkv (b nw) h ws c', ws=49)
#
#             k = k.view(B_ * ((N // 49)) // 2, 2, self.num_heads, 49, -1)
#             k_over1 = k[:, 1, :, :20].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
#             k_over2 = k[:, 0, :, 29:].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
#             k_over = torch.cat([k_over1, k_over2], 1)
#             k = torch.cat([k, k_over], 3).contiguous().view(B_ * ((N // 49)), self.num_heads, 49 + 20, -1)
#
#             v = v.view(B_ * ((N // 49)) // 2, 2, self.num_heads, 49, -1)
#             v_over1 = v[:, 1, :, :20].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
#             v_over2 = v[:, 0, :, 29:].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
#             v_over = torch.cat([v_over1, v_over2], 1)
#             v = torch.cat([v, v_over], 3).contiguous().view(B_ * ((N // 49)), self.num_heads, 49 + 20, -1)
#
#             # v = rearrange(v[:,:,:49,:], '(b nw) h ws d -> b h d (nw ws)', h=self.num_heads, b=B_)
#             # W = int(math.sqrt(N))
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = self.softmax(attn)
#
#         attn = self.attn_drop(attn)
#         out = attn @ v
#
#         out = rearrange(out, '(b nw) h ws d -> b (h d) nw ws', h=self.num_heads, b=B_)
#         out = out.reshape(B_, self.num_heads, C // self.num_heads, -1)
#         q_idx_rev = q_idx_rev.unsqueeze(2).expand(out.size())
#         x = out.gather(dim=-1, index=q_idx_rev).reshape(B_, C, N).permute(0, 2, 1)
#
#         v = rearrange(v[:, :, :49, :], '(b nw) h ws d -> b h d (nw ws)', h=self.num_heads, b=B_)
#         W = int(math.sqrt(N))
#         v = v.gather(dim=-1, index=q_idx_rev).reshape(B_, C, W, W)
#         v = self.get_v(v)
#         v = v.reshape(B_, C, N).permute(0, 2, 1)
#         x = x + v
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#
# class CSWinBlock(nn.Module):
#
#     def __init__(self, dim, reso, num_heads,
#                  split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm,
#                  last_stage=False, content=False):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.patches_resolution = reso
#         self.split_size = split_size
#         self.mlp_ratio = mlp_ratio
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.norm1 = norm_layer(dim)
#
#         if self.patches_resolution == split_size:
#             last_stage = True
#         if last_stage:
#             self.branch_num = 1
#         else:
#             self.branch_num = 2
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(drop)
#         self.content = content
#         if last_stage:
#             self.attns = nn.ModuleList([
#                 LePEAttention(
#                     dim, resolution=self.patches_resolution, idx=-1,
#                     split_size=split_size, num_heads=num_heads, dim_out=dim,
#                     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#                 for i in range(self.branch_num)])
#         else:
#             self.attns = nn.ModuleList([
#                 LePEAttention(
#                     dim // 2, resolution=self.patches_resolution, idx=i,
#                     split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
#                     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#                 for i in range(self.branch_num)])
#         if self.content:
#             self.content_attn = ContentAttention(dim=dim, window_size=split_size, num_heads=num_heads,
#                                                  qkv_bias=qkv_bias, qk_scale=qkv_bias, attn_drop=attn_drop,
#                                                  proj_drop=attn_drop)
#             self.norm3 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
#                        drop=drop)
#         self.norm2 = norm_layer(dim)
#
#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#
#         H = W = self.patches_resolution
#         B, L, C = x.shape
#         assert L == H * W, "flatten img_tokens has wrong size"
#         img = self.norm1(x)
#         qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
#
#         if self.branch_num == 2:
#             x1 = self.attns[0](qkv[:, :, :, :C // 2])
#             x2 = self.attns[1](qkv[:, :, :, C // 2:])
#             attened_x = torch.cat([x1, x2], dim=2)
#         else:
#             attened_x = self.attns[0](qkv)
#         attened_x = self.proj(attened_x)
#         x = x + self.drop_path(attened_x)
#         if self.content:
#             x = x + self.drop_path(self.content_attn(self.norm3(x)))
#
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#
#         return x
#
#
# def img2windows(img, H_sp, W_sp):
#     """
#     img: B C H W
#     """
#     B, C, H, W = img.shape
#     img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
#     img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
#     return img_perm
#
#
# def windows2img(img_splits_hw, H_sp, W_sp, H, W):
#     """
#     img_splits_hw: B' H W C
#     """
#     B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
#
#     img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
#     img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return img
#
#
# class Merge_Block(nn.Module):
#     def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
#         self.norm = norm_layer(dim_out)
#
#     def forward(self, x):
#         B, new_HW, C = x.shape
#         H = W = int(np.sqrt(new_HW))
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#         x = self.conv(x)
#         B, C = x.shape[:2]
#         x = x.view(B, C, -1).transpose(-2, -1).contiguous()
#         x = self.norm(x)
#
#         return x
#
#
# class CSWinTransformer(nn.Module):
#     """ Vision Transformer with support for patch or hybrid CNN input stage
#     """
#
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=96, depth=[2, 2, 6, 2],
#                  split_size=[3, 5, 7],
#                  num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0.,
#                  hybrid_backbone=None,
#                  norm_layer=nn.LayerNorm,
#                  use_chk=False,
#                  use_drloc = False,
#                  sample_size=32,
#                  use_multiscale = False,
#                  drloc_mode='l1',
#                  use_abs=False,
#                  with_lca=True
#
#                  ):
#         super().__init__()
#         self.with_lca = with_lca
#         self.use_drloc = use_drloc
#         self.use_multiscale = use_multiscale
#         self.num_layers = len(depth)
#         self.use_chk = use_chk
#         self.num_classes = num_classes
#         # self.pos_embed = nn.Parameter(torch.zeros(1,nu))
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.num_feature = int(embed_dim*2**(self.num_layers-1))
#         heads = num_heads
#         if with_lca:
#             self.LCA = MyConvDila()
#
#         self.stage1_conv_embed = nn.Sequential(
#             nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
#             Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
#             nn.LayerNorm(embed_dim)
#         )
#
#         curr_dim = embed_dim
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
#         self.stage1 = nn.ModuleList([
#             CSWinBlock(
#                 dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
#                 drop=drop_rate, attn_drop=attn_drop_rate,
#                 drop_path=dpr[i], norm_layer=norm_layer, content=(i % 2 == 0))
#             for i in range(depth[0])])
#
#         self.merge1 = Merge_Block(curr_dim, curr_dim * 2)
#         curr_dim = curr_dim * 2
#         self.stage2 = nn.ModuleList(
#             [CSWinBlock(
#                 dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
#                 drop=drop_rate, attn_drop=attn_drop_rate,
#                 drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer, content=(i % 2 == 0))
#                 for i in range(depth[1])])
#
#         self.merge2 = Merge_Block(curr_dim, curr_dim * 2)
#         curr_dim = curr_dim * 2
#         temp_stage3 = []
#         temp_stage3.extend(
#             [CSWinBlock(
#                 dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
#                 drop=drop_rate, attn_drop=attn_drop_rate,
#                 drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer, content=(i % 2 == 0))
#                 for i in range(depth[2])])
#
#         self.stage3 = nn.ModuleList(temp_stage3)
#
#         self.merge3 = Merge_Block(curr_dim, curr_dim * 2)
#         curr_dim = curr_dim * 2
#         self.stage4 = nn.ModuleList(
#             [CSWinBlock(
#                 dim=curr_dim, num_heads=heads[3], reso=img_size // 32, mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
#                 drop=drop_rate, attn_drop=attn_drop_rate,
#                 drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True, content=(i % 2 == 0))
#                 for i in range(depth[-1])])
#         if self.use_drloc:
#             self.drloc = nn.ModuleList()
#             if self.use_multiscale:
#                 for i_layer in range(self.num_layers):
#                     self.drloc.append(DenseRelativeLoc(
#                         in_dim=min(int(embed_dim * 2 ** (i_layer + 1)), self.num_feature),
#                         out_dim=2 if drloc_mode == "l1" else max(img_size // (4 * 2 ** i_layer),
#                                                                  img_size // (4 * 2 ** (self.num_layers - 1))),
#                         sample_size=sample_size,
#                         drloc_mode=drloc_mode,
#                         use_abs=use_abs
#                     ))
#             else:
#                 self.drloc.append(DenseRelativeLoc(
#                     in_dim=self.num_feature,
#                     out_dim=2 if drloc_mode == "l1" else img_size // (4 * 2 ** (self.num_layers - 1)),
#                     sample_size=sample_size,
#                     drloc_mode=drloc_mode,
#                     use_abs=use_abs
#                 ))
#         self.norm = norm_layer(curr_dim)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         # Classifier head
#         self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()
#         self.head1 = nn.Linear(num_classes,13)
#         trunc_normal_(self.head1.weight, std=0.02)
#         self.apply(self._init_weights)
#
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}
#
#     def get_classifier(self):
#         return self.head
#
#     def reset_classifier(self, num_classes, global_pool=''):
#         if self.num_classes != num_classes:
#             print('reset head to', num_classes)
#             self.num_classes = num_classes
#             self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
#             self.head1= nn.Linear(num_classes,13)
#             # self.head = self.head.cuda()
#             self.head1 = self.head1.cuda()
#             trunc_normal_(self.head1.weight, std=.02)
#             if self.head1.bias is not None:
#                 nn.init.constant_(self.head1.bias, 0)
#
#     def forward_features(self, x):
#         B = x.shape[0]
#         x = self.stage1_conv_embed(x)
#         c = list()
#         for blk in self.stage1:
#             if self.use_chk:
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)
#                 c.append(x)
#         for pre, blocks in zip([self.merge1, self.merge2, self.merge3],
#                                [self.stage2, self.stage3, self.stage4]):
#             x = pre(x)
#             c.append(x)
#             for blk in blocks:
#                 if self.use_chk:
#                     x = checkpoint.checkpoint(blk, x)
#                 else:
#                     x = blk(x)
#                     c.append(x)
#         # x = self.norm(x)
#         # return torch.mean(x, dim=1)
#         return [x],c
#
#     def forward(self, x):
#         x ,L= self.forward_features(x)
#         if self.with_lca:
#             x = self.LCA(L)
#         x_last = self.norm(x[-1])
#         pool = self.avgpool(x_last.transpose(1,2))
#         sup = self.head(torch.flatten(pool,1))
#         sup = self.head1(sup)
#         outs = Munch(sup=sup)
#         if self.use_drloc:
#             outs.drloc = []
#             outs.deltaxy = []
#             outs.plz = []
#
#             for idx, x_cur in enumerate(x):
#                 x_cur = x_cur.transpose(1, 2)  # [B, C, L]
#                 B, C, HW = x_cur.size()
#                 H = W = int(math.sqrt(HW))
#                 feats = x_cur.view(B, C, H, W)  # [B, C, H, W]
#
#                 drloc_feats, deltaxy = self.drloc[idx](feats)
#                 outs.drloc.append(drloc_feats)
#                 outs.deltaxy.append(deltaxy)
#                 outs.plz.append(H)  # plane size
#         # x = self.head(x)
#         # x = self.head1(x)
#         return outs
#
#
# def _conv_filter(state_dict, patch_size=16):
#     """ convert patch embedding weight from manual patchify + linear proj to conv"""
#     out_dict = {}
#     for k, v in state_dict.items():
#         if 'patch_embed.proj.weight' in k:
#             v = v.reshape((v.shape[0], 3, patch_size, patch_size))
#         out_dict[k] = v
#     return out_dict
# # def _create_cswin_transformer(variant, pretrained=False, **kwargs):
# #     model = build_model_with_cfg(
# #         CSWinTransformer, variant, pretrained,
# #         pretrained_filter_fn=checkpoint_filter_fn,
# #         **kwargs)
# #     return model
# # ### 224 models
# # @register_model
# # def CSWin_BOAT_64_24322_small_224(pretrained=False, **kwargs):
# #     model_kwargs = dict(
# #         patch_size=4,embed_dim=64, depth = [2,4,32,2],drop_path_rate = 0.4,
# #         split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs
# #     )
# #     return _create_cswin_transformer('CSWin_BOAT_64_24322_small_224',pretrained=pretrained, **model_kwargs)
# @register_model
# def CSWin_64_12211_tiny_224(pretrained=False, **kwargs):
#     model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[1, 2, 21, 1],
#                              split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs)
#     model.default_cfg = default_cfgs['cswin_224']
#     return model
#
#
# @register_model
# def CSWin_BOAT_64_24322_small_224(pretrained=False, **kwargs):
#     model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[2, 4, 32, 2],
#                              split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs)
#     model.default_cfg = default_cfgs['CSWin_BOAT_64_24322_small_224']
#     return model
#
#
# @register_model
# def CSWin_96_24322_base_224(pretrained=False, **kwargs):
#     model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
#                              split_size=[1, 2, 7, 7], num_heads=[4, 8, 16, 32], mlp_ratio=4., **kwargs)
#     model.default_cfg = default_cfgs['cswin_224']
#     return model
#
#
# @register_model
# def CSWin_144_24322_large_224(pretrained=False, **kwargs):
#     model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2, 4, 32, 2],
#                              split_size=[1, 2, 7, 7], num_heads=[6, 12, 24, 24], mlp_ratio=4., **kwargs)
#     model.default_cfg = default_cfgs['cswin_224']
#     return model
#
#
# ### 384 models
#
# @register_model
# def CSWin_96_24322_base_384(pretrained=False, **kwargs):
#     model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
#                              split_size=[1, 2, 12, 12], num_heads=[4, 8, 16, 32], mlp_ratio=4., **kwargs)
#     model.default_cfg = default_cfgs['cswin_384']
#     return model
#
#
# @register_model
# def CSWin_144_24322_large_384(pretrained=False, **kwargs):
#     model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2, 4, 32, 2],
#                              split_size=[1, 2, 12, 12], num_heads=[6, 12, 24, 24], mlp_ratio=4., **kwargs)
#     model.default_cfg = default_cfgs['cswin_384']
#     return model
#
###########++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++############################################
###########++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++############################################
###########++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++############################################
###########++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++############################################
###########++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++############################################
###########++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++############################################
# 上面代码是cswinboatwithdrlocwithlca的完全代码，能够运行的
# 下面的代码是使用了Parameter的代码
# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------
# =================================================================================================================================================================# ------------------------------------------
# # =================================================================================================================================================================# ------------------------------------------
# # =================================================================================================================================================================# ------------------------------------------
# # =================================================================================================================================================================# ------------------------------------------
# # =================================================================================================================================================================
# 下面是对13类数据-使用了cswin_boat_drloc_fuse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import einsum
# from functools import partial
# from .helpers import build_model_with_cfg ,named_apply, checkpoint_seq
# from .vision_transformer import checkpoint_filter_fn,get_init_weights_vit
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.helpers import load_pretrained
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
# from einops.layers.torch import Rearrange
# import torch.utils.checkpoint as checkpoint
# import numpy as np
# import time
# from einops import rearrange, repeat
# import math
# from drloc import DenseRelativeLoc
# from munch import Munch
# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
#         'crop_pct': .9, 'interpolation': 'bicubic',
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
#         'first_conv': 'patch_embed.proj', 'classifier': 'head',
#         **kwargs
#     }
#
#
# default_cfgs = {
#     'CSWin_BOAT_64_24322_small_224': _cfg(),
#     'cswin_384': _cfg(
#         crop_pct=1.0
#     ),
#
# }
# class MyConvDila(nn.Module):
#     def __init__(self):
#         super().__init__()
#         #定义第一次空洞卷积
#         # self.Para1 = nn.Parameter(torch.ones(32,3136,64))
#         self.conv1 = nn.Sequential(
#             # nn.Conv2d(in_channels=64,out_channels=512,kernel_size=5,stride=4,dilation=7),#这是不用avg的时候
#             nn.Conv2d(in_channels=64,out_channels=512,kernel_size=5,stride=3,dilation=3),
#             nn.GELU(),
#             nn.AvgPool2d(kernel_size=2,stride=2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=128,out_channels=512,kernel_size=3,stride=2,dilation=1,padding=1),
#             nn.GELU(),
#             nn.AvgPool2d(kernel_size=2,stride=2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,dilation=1,padding=1),
#             nn.GELU()
#         )
#         self.SIG=torch.nn.Sigmoid()
#         self.Para1 = nn.Parameter(torch.ones(43, 1,1,1,1))
#         self.Para2 = nn.Parameter(torch.ones(32, 128, 784))
#         self.Para3 = nn.Parameter(torch.ones(32, 256, 196))
#         self.Para4 = nn.Parameter(torch.ones(32, 512, 49))
#     def forward(self,z):
#         y=list()
#         for i in range(len(z)):
#             if i<2:
#                 # z[i] = z[i]@self.Para1
#                 O = rearrange(z[i].transpose(1,2),'b c (h w) -> b c h w',h=56)
#                 O = self.conv1(O)
#                 y.append(O)
#             elif i>1 and i<7:
#                 # z[i] = z[i]@self.Para2
#                 P = rearrange(z[i].transpose(1,2),'b c (h w) -> b c h w',h=28)
#                 P = self.conv2(P)
#                 y.append(P)
#             elif i>6 and i<40 :
#                 # z[i] = z[i]@self.Para3
#                 J = rearrange(z[i].transpose(1,2),'b c (h w) -> b c h w',h=14)
#                 J = self.conv3(J)
#                 y.append(J)
#             elif i>39 and i<43:
#                 # z[i] = z[i]@self.Para4
#                 K = rearrange(z[i].transpose(1,2),'b c (h w) -> b c h w',h=7)
#                 y.append(K)
#         U = torch.stack(y)
#         U = U*self.Para1
#         U = self.SIG(U)
#         U = U.mean(0)
#         U = rearrange(U,'b c h w -> b c (h w)',h=7)
#         U = U.transpose(1,2)
#         return [U]
# class LCAAttention(nn.Module):
#     def __init__(self,dim,heads=8,dim_head=64,dropout= 0):
#         super().__init__()
#         inner_dim = dim_head*heads
#         projec_out = not(heads==1 and dim_head ==dim)
#         self.heads = heads
#         self.scale = dim_head**-0.5
#         self.to_qkv = nn.Linear(dim,inner_dim*3,bias=False)
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim,dim),
#             nn.Dropout(dropout)
#         )if projec_out else nn.Identity()
#
#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
#
#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
#
#         attn = dots.softmax(dim=-1)
#
#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#         return out
#
#
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# class LePEAttention(nn.Module):
#     def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
#                  qk_scale=None):
#         super().__init__()
#         self.dim = dim
#         self.dim_out = dim_out or dim
#         self.resolution = resolution
#         self.split_size = split_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#         if idx == -1:
#             H_sp, W_sp = self.resolution, self.resolution
#         elif idx == 0:
#             H_sp, W_sp = self.resolution, self.split_size
#         elif idx == 1:
#             W_sp, H_sp = self.resolution, self.split_size
#         else:
#             print("ERROR MODE", idx)
#             exit(0)
#         self.H_sp = H_sp
#         self.W_sp = W_sp
#         stride = 1
#         self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#
#     def im2cswin(self, x):
#         B, N, C = x.shape
#         H = W = int(np.sqrt(N))
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#         x = img2windows(x, self.H_sp, self.W_sp)
#         x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
#         return x
#
#     def get_lepe(self, x, func):
#         B, N, C = x.shape
#         H = W = int(np.sqrt(N))
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#
#         H_sp, W_sp = self.H_sp, self.W_sp
#         x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
#         x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'
#
#         lepe = func(x)  ### B', C, H', W'
#         lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
#
#         x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
#         return x, lepe
#
#     def forward(self, qkv):
#         """
#         x: B L C
#         """
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         ### Img2Window
#         H = W = self.resolution
#         B, L, C = q.shape
#         assert L == H * W, "flatten img_tokens has wrong size"
#
#         q = self.im2cswin(q)
#         k = self.im2cswin(k)
#         v, lepe = self.get_lepe(v, self.get_v)
#
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
#         attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v) + lepe
#         x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C
#
#         ### Window2Img
#         x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C
#
#         return x
#
#
# class ContentAttention(nn.Module):
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.ws = window_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softmax = nn.Softmax(dim=-1)
#         self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
#
#     def forward(self, x, mask=None):
#         # B_, W, H, C = x.shape
#         # x = x.view(B_,W*H,C)
#         B_, N, C = x.shape
#
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
#                                                                                          4)  # 3, B_, self.num_heads,N,D
#         if True:
#             q_pre = qkv[0].reshape(B_ * self.num_heads, N, C // self.num_heads).permute(0, 2,
#                                                                                         1)  # qkv_pre[:,0].reshape(b*self.num_heads,qkvhd//3//self.num_heads,hh*ww)
#             ntimes = int(math.log(N // 49, 2))
#             q_idx_last = torch.arange(N).cuda().unsqueeze(0).expand(B_ * self.num_heads, N)
#             for i in range(ntimes):
#                 bh, d, n = q_pre.shape
#                 q_pre_new = q_pre.reshape(bh, d, 2, n // 2)
#                 q_avg = q_pre_new.mean(dim=-1)  # .reshape(b*self.num_heads,qkvhd//3//self.num_heads,)
#                 q_avg = torch.nn.functional.normalize(q_avg, dim=-2)
#                 iters = 2
#                 for i in range(iters):
#                     q_scores = torch.nn.functional.normalize(q_pre.permute(0, 2, 1), dim=-1).bmm(q_avg)
#                     soft_assign = torch.nn.functional.softmax(q_scores * 100, dim=-1).detach()
#                     q_avg = q_pre.bmm(soft_assign)
#                     q_avg = torch.nn.functional.normalize(q_avg, dim=-2)
#                 q_scores = torch.nn.functional.normalize(q_pre.permute(0, 2, 1), dim=-1).bmm(q_avg).reshape(bh, n,
#                                                                                                             2)  # .unsqueeze(2)
#                 q_idx = (q_scores[:, :, 0] + 1) / (q_scores[:, :, 1] + 1)
#                 _, q_idx = torch.sort(q_idx, dim=-1)
#                 q_idx_last = q_idx_last.gather(dim=-1, index=q_idx).reshape(bh * 2, n // 2)
#                 q_idx = q_idx.unsqueeze(1).expand(q_pre.size())
#                 q_pre = q_pre.gather(dim=-1, index=q_idx).reshape(bh, d, 2, n // 2).permute(0, 2, 1, 3).reshape(bh * 2,
#                                                                                                                 d,
#                                                                                                                 n // 2)
#
#             q_idx = q_idx_last.view(B_, self.num_heads, N)
#             _, q_idx_rev = torch.sort(q_idx, dim=-1)
#             q_idx = q_idx.unsqueeze(0).unsqueeze(4).expand(qkv.size())
#             qkv_pre = qkv.gather(dim=-2, index=q_idx)
#             q, k, v = rearrange(qkv_pre, 'qkv b h (nw ws) c -> qkv (b nw) h ws c', ws=49)
#
#             k = k.view(B_ * ((N // 49)) // 2, 2, self.num_heads, 49, -1)
#             k_over1 = k[:, 1, :, :20].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
#             k_over2 = k[:, 0, :, 29:].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
#             k_over = torch.cat([k_over1, k_over2], 1)
#             k = torch.cat([k, k_over], 3).contiguous().view(B_ * ((N // 49)), self.num_heads, 49 + 20, -1)
#
#             v = v.view(B_ * ((N // 49)) // 2, 2, self.num_heads, 49, -1)
#             v_over1 = v[:, 1, :, :20].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
#             v_over2 = v[:, 0, :, 29:].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
#             v_over = torch.cat([v_over1, v_over2], 1)
#             v = torch.cat([v, v_over], 3).contiguous().view(B_ * ((N // 49)), self.num_heads, 49 + 20, -1)
#
#             # v = rearrange(v[:,:,:49,:], '(b nw) h ws d -> b h d (nw ws)', h=self.num_heads, b=B_)
#             # W = int(math.sqrt(N))
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = self.softmax(attn)
#
#         attn = self.attn_drop(attn)
#         out = attn @ v
#
#         out = rearrange(out, '(b nw) h ws d -> b (h d) nw ws', h=self.num_heads, b=B_)
#         out = out.reshape(B_, self.num_heads, C // self.num_heads, -1)
#         q_idx_rev = q_idx_rev.unsqueeze(2).expand(out.size())
#         x = out.gather(dim=-1, index=q_idx_rev).reshape(B_, C, N).permute(0, 2, 1)
#
#         v = rearrange(v[:, :, :49, :], '(b nw) h ws d -> b h d (nw ws)', h=self.num_heads, b=B_)
#         W = int(math.sqrt(N))
#         v = v.gather(dim=-1, index=q_idx_rev).reshape(B_, C, W, W)
#         v = self.get_v(v)
#         v = v.reshape(B_, C, N).permute(0, 2, 1)
#         x = x + v
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#
# class CSWinBlock(nn.Module):
#
#     def __init__(self, dim, reso, num_heads,
#                  split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm,
#                  last_stage=False, content=False):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.patches_resolution = reso
#         self.split_size = split_size
#         self.mlp_ratio = mlp_ratio
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.norm1 = norm_layer(dim)
#
#         if self.patches_resolution == split_size:
#             last_stage = True
#         if last_stage:
#             self.branch_num = 1
#         else:
#             self.branch_num = 2
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(drop)
#         self.content = content
#         if last_stage:
#             self.attns = nn.ModuleList([
#                 LePEAttention(
#                     dim, resolution=self.patches_resolution, idx=-1,
#                     split_size=split_size, num_heads=num_heads, dim_out=dim,
#                     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#                 for i in range(self.branch_num)])
#         else:
#             self.attns = nn.ModuleList([
#                 LePEAttention(
#                     dim // 2, resolution=self.patches_resolution, idx=i,
#                     split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
#                     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#                 for i in range(self.branch_num)])
#         if self.content:
#             self.content_attn = ContentAttention(dim=dim, window_size=split_size, num_heads=num_heads,
#                                                  qkv_bias=qkv_bias, qk_scale=qkv_bias, attn_drop=attn_drop,
#                                                  proj_drop=attn_drop)
#             self.norm3 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
#                        drop=drop)
#         self.norm2 = norm_layer(dim)
#
#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#
#         H = W = self.patches_resolution
#         B, L, C = x.shape
#         assert L == H * W, "flatten img_tokens has wrong size"
#         img = self.norm1(x)
#         qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
#
#         if self.branch_num == 2:
#             x1 = self.attns[0](qkv[:, :, :, :C // 2])
#             x2 = self.attns[1](qkv[:, :, :, C // 2:])
#             attened_x = torch.cat([x1, x2], dim=2)
#         else:
#             attened_x = self.attns[0](qkv)
#         attened_x = self.proj(attened_x)
#         x = x + self.drop_path(attened_x)
#         if self.content:
#             x = x + self.drop_path(self.content_attn(self.norm3(x)))
#
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#
#         return x
#
#
# def img2windows(img, H_sp, W_sp):
#     """
#     img: B C H W
#     """
#     B, C, H, W = img.shape
#     img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
#     img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
#     return img_perm
#
#
# def windows2img(img_splits_hw, H_sp, W_sp, H, W):
#     """
#     img_splits_hw: B' H W C
#     """
#     B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
#
#     img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
#     img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return img
#
#
# class Merge_Block(nn.Module):
#     def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
#         self.norm = norm_layer(dim_out)
#
#     def forward(self, x):
#         B, new_HW, C = x.shape
#         H = W = int(np.sqrt(new_HW))
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#         x = self.conv(x)
#         B, C = x.shape[:2]
#         x = x.view(B, C, -1).transpose(-2, -1).contiguous()
#         x = self.norm(x)
#
#         return x
#
#
# class CSWinTransformer(nn.Module):
#     """ Vision Transformer with support for patch or hybrid CNN input stage
#     """
#
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=96, depth=[2, 2, 6, 2],
#                  split_size=[3, 5, 7],
#                  num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0.,
#                  hybrid_backbone=None,
#                  norm_layer=nn.LayerNorm,
#                  use_chk=False,
#                  use_drloc = False,
#                  sample_size=32,
#                  use_multiscale = False,
#                  drloc_mode='l1',
#                  use_abs=False,
#                  with_lca=True
#
#                  ):
#         super().__init__()
#         self.with_lca = with_lca
#         self.use_drloc = use_drloc
#         self.use_multiscale = use_multiscale
#         self.num_layers = len(depth)
#         self.use_chk = use_chk
#         self.num_classes = num_classes
#         # self.pos_embed = nn.Parameter(torch.zeros(1,nu))
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.num_feature = int(embed_dim*2**(self.num_layers-1))
#         heads = num_heads
#         if with_lca:
#             self.LCA = MyConvDila()
#             self.LCATTen = LCAAttention(dim=512,heads=4,dim_head=128,dropout=0)
#
#         self.stage1_conv_embed = nn.Sequential(
#             nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
#             Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
#             nn.LayerNorm(embed_dim)
#         )
#
#         curr_dim = embed_dim
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
#         self.stage1 = nn.ModuleList([
#             CSWinBlock(
#                 dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
#                 drop=drop_rate, attn_drop=attn_drop_rate,
#                 drop_path=dpr[i], norm_layer=norm_layer, content=(i % 2 == 0))
#             for i in range(depth[0])])
#
#         self.merge1 = Merge_Block(curr_dim, curr_dim * 2)
#         curr_dim = curr_dim * 2
#         self.stage2 = nn.ModuleList(
#             [CSWinBlock(
#                 dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
#                 drop=drop_rate, attn_drop=attn_drop_rate,
#                 drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer, content=(i % 2 == 0))
#                 for i in range(depth[1])])
#
#         self.merge2 = Merge_Block(curr_dim, curr_dim * 2)
#         curr_dim = curr_dim * 2
#         temp_stage3 = []
#         temp_stage3.extend(
#             [CSWinBlock(
#                 dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
#                 drop=drop_rate, attn_drop=attn_drop_rate,
#                 drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer, content=(i % 2 == 0))
#                 for i in range(depth[2])])
#
#         self.stage3 = nn.ModuleList(temp_stage3)
#
#         self.merge3 = Merge_Block(curr_dim, curr_dim * 2)
#         curr_dim = curr_dim * 2
#         self.stage4 = nn.ModuleList(
#             [CSWinBlock(
#                 dim=curr_dim, num_heads=heads[3], reso=img_size // 32, mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
#                 drop=drop_rate, attn_drop=attn_drop_rate,
#                 drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True, content=(i % 2 == 0))
#                 for i in range(depth[-1])])
#         if self.use_drloc:
#             self.drloc = nn.ModuleList()
#             if self.use_multiscale:
#                 for i_layer in range(self.num_layers):
#                     self.drloc.append(DenseRelativeLoc(
#                         in_dim=min(int(embed_dim * 2 ** (i_layer + 1)), self.num_feature),
#                         out_dim=2 if drloc_mode == "l1" else max(img_size // (4 * 2 ** i_layer),
#                                                                  img_size // (4 * 2 ** (self.num_layers - 1))),
#                         sample_size=sample_size,
#                         drloc_mode=drloc_mode,
#                         use_abs=use_abs
#                     ))
#             else:
#                 self.drloc.append(DenseRelativeLoc(
#                     in_dim=self.num_feature,
#                     out_dim=2 if drloc_mode == "l1" else img_size // (4 * 2 ** (self.num_layers - 1)),
#                     sample_size=sample_size,
#                     drloc_mode=drloc_mode,
#                     use_abs=use_abs
#                 ))
#         self.norm = norm_layer(curr_dim)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         # Classifier head
#         self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()
#         self.head1 = nn.Linear(num_classes,13)
#         trunc_normal_(self.head1.weight, std=0.02)
#         self.apply(self._init_weights)
#
#
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}
#
#     def get_classifier(self):
#         return self.head
#
#     def reset_classifier(self, num_classes, global_pool=''):
#         if self.num_classes != num_classes:
#             print('reset head to', num_classes)
#             self.num_classes = num_classes
#             self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
#             self.head1= nn.Linear(num_classes,13)
#             # self.head = self.head.cuda()
#             self.head1 = self.head1.cuda()
#             trunc_normal_(self.head1.weight, std=.02)
#             if self.head1.bias is not None:
#                 nn.init.constant_(self.head1.bias, 0)
#
#     def forward_features(self, x):
#         B = x.shape[0]
#         x = self.stage1_conv_embed(x)
#         c = list()
#         for blk in self.stage1:
#             if self.use_chk:
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)
#                 c.append(x)
#         for pre, blocks in zip([self.merge1, self.merge2, self.merge3],
#                                [self.stage2, self.stage3, self.stage4]):
#             x = pre(x)
#             c.append(x)
#             for blk in blocks:
#                 if self.use_chk:
#                     x = checkpoint.checkpoint(blk, x)
#                 else:
#                     x = blk(x)
#                     c.append(x)
#         # x = self.norm(x)
#         # return torch.mean(x, dim=1)
#         return [x],c
#
#     def forward(self, x):
#         x ,L= self.forward_features(x)
#         if self.with_lca:
#             x = self.LCA(L)
#             # x = self.LCATTen(x)
#         x_last = self.norm(x[-1])
#         pool = self.avgpool(x_last.transpose(1,2))
#         sup = self.head(torch.flatten(pool,1))
#         sup = self.head1(sup)
#         outs = Munch(sup=sup)
#         if self.use_drloc:
#             outs.drloc = []
#             outs.deltaxy = []
#             outs.plz = []
#
#             for idx, x_cur in enumerate(x):
#                 x_cur = x_cur.transpose(1, 2)  # [B, C, L]
#                 B, C, HW = x_cur.size()
#                 H = W = int(math.sqrt(HW))
#                 feats = x_cur.view(B, C, H, W)  # [B, C, H, W]
#
#                 drloc_feats, deltaxy = self.drloc[idx](feats)
#                 outs.drloc.append(drloc_feats)
#                 outs.deltaxy.append(deltaxy)
#                 outs.plz.append(H)  # plane size
#         # x = self.head(x)
#         # x = self.head1(x)
#         return outs
#
#
# def _conv_filter(state_dict, patch_size=16):
#     """ convert patch embedding weight from manual patchify + linear proj to conv"""
#     out_dict = {}
#     for k, v in state_dict.items():
#         if 'patch_embed.proj.weight' in k:
#             v = v.reshape((v.shape[0], 3, patch_size, patch_size))
#         out_dict[k] = v
#     return out_dict
# # def _create_cswin_transformer(variant, pretrained=False, **kwargs):
# #     model = build_model_with_cfg(
# #         CSWinTransformer, variant, pretrained,
# #         pretrained_filter_fn=checkpoint_filter_fn,
# #         **kwargs)
# #     return model
# # ### 224 models
# # @register_model
# # def CSWin_BOAT_64_24322_small_224(pretrained=False, **kwargs):
# #     model_kwargs = dict(
# #         patch_size=4,embed_dim=64, depth = [2,4,32,2],drop_path_rate = 0.4,
# #         split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs
# #     )
# #     return _create_cswin_transformer('CSWin_BOAT_64_24322_small_224',pretrained=pretrained, **model_kwargs)
# @register_model
# def CSWin_64_12211_tiny_224(pretrained=False, **kwargs):
#     model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[1, 2, 21, 1],
#                              split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs)
#     model.default_cfg = default_cfgs['cswin_224']
#     return model
#
#
# @register_model
# def CSWin_BOAT_64_24322_small_224(pretrained=False, **kwargs):
#     model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[2, 4, 32, 2],
#                              split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs)
#     model.default_cfg = default_cfgs['CSWin_BOAT_64_24322_small_224']
#     return model
#
#
# @register_model
# def CSWin_96_24322_base_224(pretrained=False, **kwargs):
#     model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
#                              split_size=[1, 2, 7, 7], num_heads=[4, 8, 16, 32], mlp_ratio=4., **kwargs)
#     model.default_cfg = default_cfgs['cswin_224']
#     return model
#
#
# @register_model
# def CSWin_144_24322_large_224(pretrained=False, **kwargs):
#     model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2, 4, 32, 2],
#                              split_size=[1, 2, 7, 7], num_heads=[6, 12, 24, 24], mlp_ratio=4., **kwargs)
#     model.default_cfg = default_cfgs['cswin_224']
#     return model
#
#
# ### 384 models
#
# @register_model
# def CSWin_96_24322_base_384(pretrained=False, **kwargs):
#     model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
#                              split_size=[1, 2, 12, 12], num_heads=[4, 8, 16, 32], mlp_ratio=4., **kwargs)
#     model.default_cfg = default_cfgs['cswin_384']
#     return model
#
#
# @register_model
# def CSWin_144_24322_large_384(pretrained=False, **kwargs):
#     model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2, 4, 32, 2],
#                              split_size=[1, 2, 12, 12], num_heads=[6, 12, 24, 24], mlp_ratio=4., **kwargs)
#     model.default_cfg = default_cfgs['cswin_384']
#     return model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from functools import partial
# from .helpers import build_model_with_cfg ,named_apply, checkpoint_seq
# from .vision_transformer import checkpoint_filter_fn,get_init_weights_vit
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np
import time
from einops import rearrange, repeat
import math
from drloc import DenseRelativeLoc
from munch import Munch


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'CSWin_BOAT_64_24322_small_224': _cfg(),
    'cswin_384': _cfg(
        crop_pct=1.0
    ),

}




class MyConvDila(nn.Module):
    def __init__(self):
        super().__init__()


        # 定义第一次空洞卷积
        # self.Para1 = nn.Parameter(torch.ones(32,3136,64))
        # self.conv1 = nn.Sequential(
        #     # nn.Conv2d(in_channels=64,out_channels=512,kernel_size=5,stride=4,dilation=7),#这是不用avg的时候
        #     nn.Conv2d(in_channels=64,out_channels=512,kernel_size=5,stride=3,dilation=3),
        #     nn.GELU(),
        #     nn.AvgPool2d(kernel_size=2,stride=2)
        # )#这个是原始conv1
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels=64,out_channels=512,kernel_size=5,stride=4,dilation=7),#这是不用avg的时候
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=0),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, dilation=2, padding=0),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, dilation=5, padding=0),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=0),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, dilation=2, padding=0),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, dilation=5, padding=0),
            nn.GELU(),
            # nn.AvgPool2d(kernel_size=2,stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=2, dilation=1, padding=1),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, dilation=1, padding=1),
            nn.GELU()
        )
        self.SIG = torch.nn.Sigmoid()
        self.Para1 = nn.Parameter(torch.ones(43, 1, 1, 1, 1))
        self.Para2 = nn.Parameter(torch.ones(32, 128, 784))
        self.Para3 = nn.Parameter(torch.ones(32, 256, 196))
        self.Para4 = nn.Parameter(torch.ones(32, 512, 49))

    def forward(self, z):
        y = list()
        for i in range(len(z)):
            if i < 2:
                # z[i] = z[i]@self.Para1
                O = rearrange(z[i].transpose(1, 2), 'b c (h w) -> b c h w', h=56)
                O = self.conv1(O)
                y.append(O)
            elif i > 1 and i < 7:
                # z[i] = z[i]@self.Para2
                P = rearrange(z[i].transpose(1, 2), 'b c (h w) -> b c h w', h=28)
                P = self.conv2(P)
                y.append(P)
            elif i > 6 and i < 40:
                # z[i] = z[i]@self.Para3
                J = rearrange(z[i].transpose(1, 2), 'b c (h w) -> b c h w', h=14)
                J = self.conv3(J)
                y.append(J)
            elif i > 39 and i < 43:
                # z[i] = z[i]@self.Para4
                K = rearrange(z[i].transpose(1, 2), 'b c (h w) -> b c h w', h=7)
                y.append(K)
        U = torch.stack(y)
        U = U + U * self.Para1
        U = self.SIG(U)
        U = U.mean(0)
        U = rearrange(U, 'b c h w -> b c (h w)', h=7)
        U = U.transpose(1, 2)
        # return [U]
        return U


# class LCAAttention(nn.Module):
#     def __init__(self,dim,heads=8,dim_head=64,dropout= 0):
#         super().__init__()
#         inner_dim = dim_head*heads
#         projec_out = not(heads==1 and dim_head ==dim)
#         self.heads = heads
#         self.scale = dim_head**-0.5
#         self.to_qkv = nn.Linear(dim,inner_dim*3,bias=False)
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim,dim),
#             nn.Dropout(dropout)
#         )if projec_out else nn.Identity()
#
#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
#
#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
#
#         attn = dots.softmax(dim=-1)
#
#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#         return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class ContentAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.ws = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, mask=None):
        # B_, W, H, C = x.shape
        # x = x.view(B_,W*H,C)
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                         4)  # 3, B_, self.num_heads,N,D
        if True:
            q_pre = qkv[0].reshape(B_ * self.num_heads, N, C // self.num_heads).permute(0, 2,
                                                                                        1)  # qkv_pre[:,0].reshape(b*self.num_heads,qkvhd//3//self.num_heads,hh*ww)
            ntimes = int(math.log(N // 49, 2))
            q_idx_last = torch.arange(N).cuda().unsqueeze(0).expand(B_ * self.num_heads, N)
            for i in range(ntimes):
                bh, d, n = q_pre.shape
                q_pre_new = q_pre.reshape(bh, d, 2, n // 2)
                q_avg = q_pre_new.mean(dim=-1)  # .reshape(b*self.num_heads,qkvhd//3//self.num_heads,)
                q_avg = torch.nn.functional.normalize(q_avg, dim=-2)
                iters = 2
                for i in range(iters):
                    q_scores = torch.nn.functional.normalize(q_pre.permute(0, 2, 1), dim=-1).bmm(q_avg)
                    soft_assign = torch.nn.functional.softmax(q_scores * 100, dim=-1).detach()
                    q_avg = q_pre.bmm(soft_assign)
                    q_avg = torch.nn.functional.normalize(q_avg, dim=-2)
                q_scores = torch.nn.functional.normalize(q_pre.permute(0, 2, 1), dim=-1).bmm(q_avg).reshape(bh, n,
                                                                                                            2)  # .unsqueeze(2)
                q_idx = (q_scores[:, :, 0] + 1) / (q_scores[:, :, 1] + 1)
                _, q_idx = torch.sort(q_idx, dim=-1)
                q_idx_last = q_idx_last.gather(dim=-1, index=q_idx).reshape(bh * 2, n // 2)
                q_idx = q_idx.unsqueeze(1).expand(q_pre.size())
                q_pre = q_pre.gather(dim=-1, index=q_idx).reshape(bh, d, 2, n // 2).permute(0, 2, 1, 3).reshape(bh * 2,
                                                                                                                d,
                                                                                                                n // 2)

            q_idx = q_idx_last.view(B_, self.num_heads, N)
            _, q_idx_rev = torch.sort(q_idx, dim=-1)
            q_idx = q_idx.unsqueeze(0).unsqueeze(4).expand(qkv.size())
            qkv_pre = qkv.gather(dim=-2, index=q_idx)
            q, k, v = rearrange(qkv_pre, 'qkv b h (nw ws) c -> qkv (b nw) h ws c', ws=49)

            k = k.view(B_ * ((N // 49)) // 2, 2, self.num_heads, 49, -1)
            k_over1 = k[:, 1, :, :20].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
            k_over2 = k[:, 0, :, 29:].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
            k_over = torch.cat([k_over1, k_over2], 1)
            k = torch.cat([k, k_over], 3).contiguous().view(B_ * ((N // 49)), self.num_heads, 49 + 20, -1)

            v = v.view(B_ * ((N // 49)) // 2, 2, self.num_heads, 49, -1)
            v_over1 = v[:, 1, :, :20].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
            v_over2 = v[:, 0, :, 29:].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
            v_over = torch.cat([v_over1, v_over2], 1)
            v = torch.cat([v, v_over], 3).contiguous().view(B_ * ((N // 49)), self.num_heads, 49 + 20, -1)

            # v = rearrange(v[:,:,:49,:], '(b nw) h ws d -> b h d (nw ws)', h=self.num_heads, b=B_)
            # W = int(math.sqrt(N))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        out = attn @ v

        out = rearrange(out, '(b nw) h ws d -> b (h d) nw ws', h=self.num_heads, b=B_)
        out = out.reshape(B_, self.num_heads, C // self.num_heads, -1)
        q_idx_rev = q_idx_rev.unsqueeze(2).expand(out.size())
        x = out.gather(dim=-1, index=q_idx_rev).reshape(B_, C, N).permute(0, 2, 1)

        v = rearrange(v[:, :, :49, :], '(b nw) h ws d -> b h d (nw ws)', h=self.num_heads, b=B_)
        W = int(math.sqrt(N))
        v = v.gather(dim=-1, index=q_idx_rev).reshape(B_, C, W, W)
        v = self.get_v(v)
        v = v.reshape(B_, C, N).permute(0, 2, 1)
        x = x + v

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False, content=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.content = content
        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        if self.content:
            self.content_attn = ContentAttention(dim=dim, window_size=split_size, num_heads=num_heads,
                                                 qkv_bias=qkv_bias, qk_scale=qkv_bias, attn_drop=attn_drop,
                                                 proj_drop=attn_drop)
            self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        if self.content:
            # x = x + self.drop_path(self.content_attn(self.norm3(x)))
            x = x + self.drop_path(self.norm3(self.content_attn(x)))

        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x


class CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=96, depth=[2, 2, 6, 2],
                 split_size=[3, 5, 7],
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.,
                 hybrid_backbone=None,
                 norm_layer=nn.LayerNorm,
                 use_chk=False,
                 use_drloc=False,
                 sample_size=32,
                 use_multiscale=False,
                 drloc_mode='l1',
                 use_abs=False,
                 with_lca=True

                 ):
        super().__init__()
        self.with_lca = with_lca
        self.use_drloc = use_drloc
        self.use_multiscale = use_multiscale
        self.num_layers = len(depth)
        self.use_chk = use_chk
        self.num_classes = num_classes
        # self.pos_embed = nn.Parameter(torch.zeros(1,nu))
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_feature = int(embed_dim * 2 ** (self.num_layers - 1))
        heads = num_heads
        if with_lca:
            self.LCA = MyConvDila()
            # self.LCATTen = LCAAttention(dim=512,heads=4,dim_head=128,dropout=0)

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, content=(i % 2 == 0))
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer, content=(i % 2 == 0))
                for i in range(depth[1])])

        self.merge2 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer, content=(i % 2 == 0))
                for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)

        self.merge3 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size // 32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True, content=(i % 2 == 0))
                for i in range(depth[-1])])
        if self.use_drloc:
            self.drloc = nn.ModuleList()
            if self.use_multiscale:
                for i_layer in range(self.num_layers):
                    self.drloc.append(DenseRelativeLoc(
                        in_dim=min(int(embed_dim * 2 ** (i_layer + 1)), self.num_feature),
                        out_dim=2 if drloc_mode == "l1" else max(img_size // (4 * 2 ** i_layer),
                                                                 img_size // (4 * 2 ** (self.num_layers - 1))),
                        sample_size=sample_size,
                        drloc_mode=drloc_mode,
                        use_abs=use_abs
                    ))
            else:
                self.drloc.append(DenseRelativeLoc(
                    in_dim=self.num_feature,
                    out_dim=2 if drloc_mode == "l1" else img_size // (4 * 2 ** (self.num_layers - 1)),
                    sample_size=sample_size,
                    drloc_mode=drloc_mode,
                    use_abs=use_abs
                ))
        self.norm = norm_layer(curr_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # Classifier head
        self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head1 = nn.Linear(num_classes, 10)
        trunc_normal_(self.head1.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        if self.num_classes != num_classes:
            print('reset head to', num_classes)
            self.num_classes = num_classes
            self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head1 = nn.Linear(num_classes, 10)
            # self.head = self.head.cuda()
            self.head1 = self.head1.cuda()
            trunc_normal_(self.head1.weight, std=.02)
            if self.head1.bias is not None:
                nn.init.constant_(self.head1.bias, 0)

    def forward_features(self, x):

        B = x.shape[0]

        x = self.stage1_conv_embed(x).cuda()
        c = list()
        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
                c.append(x)
        for pre, blocks in zip([self.merge1, self.merge2, self.merge3],
                               [self.stage2, self.stage3, self.stage4]):
            x = pre(x)
            # c.append(x)
            for blk in blocks:
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
                    c.append(x)
        # x = self.norm(x)
        # return torch.mean(x, dim=1)
        return [x], c

    def forward(self, x):
        x = x.cuda()

        x, L = self.forward_features(x)
        if self.with_lca:
            x = self.LCA(L)
            x = [x]
        # x_last = self.LCATTen(x)
        # x = [x]#因为这个x需要被变成列表被下面的drloc所使用，所以被改变
        x_last = self.norm(x[-1])  # 注意，这个是不加lcatten才生效的，而且要将lca（L）返回值改为【U】。
        pool = self.avgpool(x_last.transpose(1, 2))
        sup = self.head(torch.flatten(pool, 1))
        sup = self.head1(sup)
        outs = Munch(sup=sup)
        if self.use_drloc:
            outs.drloc = []
            outs.deltaxy = []
            outs.plz = []

            for idx, x_cur in enumerate(x):
                x_cur = x_cur.transpose(1, 2)  # [B, C, L]
                B, C, HW = x_cur.size()
                H = W = int(math.sqrt(HW))
                feats = x_cur.view(B, C, H, W)  # [B, C, H, W]

                drloc_feats, deltaxy = self.drloc[idx](feats)
                outs.drloc.append(drloc_feats)
                outs.deltaxy.append(deltaxy)
                outs.plz.append(H)  # plane size
        # x = self.head(x)
        # x = self.head1(x)
        return outs


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


# def _create_cswin_transformer(variant, pretrained=False, **kwargs):
#     model = build_model_with_cfg(
#         CSWinTransformer, variant, pretrained,
#         pretrained_filter_fn=checkpoint_filter_fn,
#         **kwargs)
#     return model
# ### 224 models
# @register_model
# def CSWin_BOAT_64_24322_small_224(pretrained=False, **kwargs):
#     model_kwargs = dict(
#         patch_size=4,embed_dim=64, depth = [2,4,32,2],drop_path_rate = 0.4,
#         split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs
#     )
#     return _create_cswin_transformer('CSWin_BOAT_64_24322_small_224',pretrained=pretrained, **model_kwargs)
@register_model
def CSWin_64_12211_tiny_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[1, 2, 21, 1],
                             split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model


@register_model
def CSWin_BOAT_64_24322_small_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['CSWin_BOAT_64_24322_small_224']
    return model


@register_model
def CSWin_96_24322_base_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 7, 7], num_heads=[4, 8, 16, 32], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model


@register_model
def CSWin_144_24322_large_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 7, 7], num_heads=[6, 12, 24, 24], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model


### 384 models

@register_model
def CSWin_96_24322_base_384(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 12, 12], num_heads=[4, 8, 16, 32], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model


@register_model
def CSWin_144_24322_large_384(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 12, 12], num_heads=[6, 12, 24, 24], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model
