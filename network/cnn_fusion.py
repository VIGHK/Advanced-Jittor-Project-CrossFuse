# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: lihui.cv@jiangnan.edu.cn
# @Project : TransFuse
# @File : cnn_fusion.py
# @Time : 2025/12/20
# 
# 消融实验：用简单的CNN Fusion替换Cross-Attention Module (CAM)

import jittor as jt
from jittor import nn


class CNNFusion(nn.Module):
    """
    简单的CNN融合模块，用于消融实验
    输入：两个特征图 f_ir, f_vis (B, C, H, W)
    输出：融合后的特征图 (B, C, H, W)
    """
    def __init__(self, channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv(channels * 2, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv(channels, channels, 3, padding=1)
        )

    def execute(self, f_ir, f_vis):
        x = jt.concat([f_ir, f_vis], dim=1)  # (B, 2*C, H, W)
        return self.fusion(x)  # (B, C, H, W)


class cnn_fusion_encoder(nn.Module):
    """
    消融版本的融合编码器：用CNN Fusion替换Cross-Attention
    保持与cross_encoder相同的接口，以便直接替换
    """
    def __init__(self, img_size, patch_size, embed_dim, num_patches, depth_self, depth_cross, n_heads=16,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        # 注意：这里depth_self和depth_cross参数保留以保持接口一致，但depth_cross不再使用
        # 实际通道数从embed_dim计算：embed_dim = part_out * patch_size * patch_size
        # 例如：part_out=128, patch_size=2, embed_dim=512
        # 但输入特征图的通道数是 part_out = 128
        from network.transformer_cam import self_atten
        self.num_patches = num_patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.shift_size = int(img_size / 2)
        
        # 保留self-attention部分（与原始方法一致）
        self.self_atten_block1 = self_atten(self.patch_size, embed_dim, num_patches, depth_self,
                                              n_heads, mlp_ratio, qkv_bias, p, attn_p)
        self.self_atten_block2 = self_atten(self.patch_size, embed_dim, num_patches, depth_self,
                                                   n_heads, mlp_ratio, qkv_bias, p, attn_p)
        
        # 用CNN Fusion替换Cross-Attention
        # 输入特征的通道数是 part_out (通常是128)
        # 我们需要从embed_dim反推part_out
        part_out = embed_dim // (patch_size * patch_size)  # 128
        self.cnn_fusion = CNNFusion(part_out)

    def execute(self, x1, x2, shift_flag=True):
        """
        x1: IR特征 (B, C, H, W), C=128
        x2: VIS特征 (B, C, H, W), C=128
        返回格式与cross_encoder保持一致
        """
        # x1 -->> ir, x2 -->> vi
        # self-attention (保留，与原始方法一致)
        x1_atten, x2_atten, paddings = self.self_atten_block1(x1, x2)
        x1_a, x2_a = x1_atten, x2_atten
        
        # shift (保留，与原始方法一致)
        if shift_flag:
            shifted_x1 = jt.roll(x1_atten, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            shifted_x2 = jt.roll(x2_atten, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            x1_atten, x2_atten, _ = self.self_atten_block2(shifted_x1, shifted_x2)
            roll_x_self1 = jt.roll(x1_atten, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            roll_x_self2 = jt.roll(x2_atten, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x1_atten, x2_atten, _ = self.self_atten_block2(x1_atten, x2_atten)
            roll_x_self1 = x1_atten
            roll_x_self2 = x2_atten
        
        # -------------------------------------
        # CNN Fusion (替换Cross-Attention)
        out = self.cnn_fusion(roll_x_self1, roll_x_self2)
        
        # 为了保持接口一致，返回与cross_encoder相同格式的中间特征
        x_self1 = roll_x_self1
        x_self2 = roll_x_self2
        x_cross1 = out  # CNN融合结果作为cross特征1
        x_cross2 = out  # CNN融合结果作为cross特征2
        
        # -------------------------------------
        return out, x1_a, x2_a, roll_x_self1, roll_x_self2, x_cross1, x_cross2
