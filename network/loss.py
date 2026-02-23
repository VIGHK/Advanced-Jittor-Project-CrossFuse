# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: lihui.cv@jiangnan.edu.cn
# @Project : TransFuse (Jittor migration)
# @File : loss.py
# @Time : 2021/11/8 18:36

import jittor as jt
from jittor import nn
import numpy as np
import tools.utils as utils

EPSILON = 1e-6


# ---------- basic losses ----------
def mse_loss(x, y):
    return jt.mean((x - y) ** 2)


def l1_loss(x, y):
    return jt.mean(jt.abs(x - y))


# ---------- SSIM / MSSSIM (Jittor port, fixed window) ----------
def _create_window(window_size, channel, sigma=1.5):
    # 对齐 PyTorch 版本：使用 gaussian 函数创建 1D window，然后通过矩阵乘法创建 2D window
    # PyTorch: _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    #          _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    #          window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    coords = np.arange(window_size) - window_size // 2
    gauss = np.exp(-(coords ** 2) / (2 * sigma * sigma))
    gauss = gauss / gauss.sum()
    # 对齐 PyTorch: 使用 outer product 创建 2D window（等价于 mm）
    kernel_2d = np.outer(gauss, gauss).astype(np.float32)
    # 对齐 PyTorch: unsqueeze(0).unsqueeze(0) -> reshape(1, 1, window_size, window_size)
    window = jt.array(kernel_2d).reshape(1, 1, window_size, window_size)
    # 对齐 PyTorch: expand(channel, 1, window_size, window_size) -> 使用 repeat 而不是 broadcast
    # PyTorch 的 expand 是 view-like 操作，不复制数据；Jittor 没有 expand，使用 repeat
    window = window.repeat(channel, 1, 1, 1)
    return window


def _ssim(img1, img2, window, window_size, channel, data_range=None, size_average=True):
    # 对齐 PyTorch 版本：val_range推断逻辑
    # PyTorch: if val_range is None: 在ssim函数内部推断
    if data_range is None:
        # 对齐 PyTorch: torch.max(img1) 对多维张量返回标量，直接比较
        # Jittor: jt.max(img1) 对多维张量返回标量（0维张量），转换为Python float
        img1_max = float(jt.max(img1))
        img1_min = float(jt.min(img1))
        # 对齐 PyTorch: if torch.max(img1) > 128: max_val = 255 else: max_val = 1
        max_val = 255.0 if img1_max > 128 else 1.0
        min_val = -1.0 if img1_min < -0.5 else 0.0
        L = max_val - min_val
    else:
        L = float(data_range)
    
    # 对齐 PyTorch 版本：动态调整window大小
    # PyTorch: if window is None: real_size = min(window_size, height, width)
    # 对齐 PyTorch: 每次调用时检查window大小是否匹配图像尺寸
    _, _, height, width = img1.shape
    real_size = min(window_size, height, width)
    # 如果window大小不匹配，重新创建
    if window is None or window.shape[2] != real_size or window.shape[3] != real_size:
        window = _create_window(real_size, channel)
    
    # 对齐 PyTorch 版本：padding=0（PyTorch中padd=0）
    # PyTorch: F.conv2d(img1, window, padding=padd, groups=channel)
    # Jittor: nn.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
    padd = 0
    mu1 = nn.conv2d(img1, window, None, 1, padd, 1, channel)
    mu2 = nn.conv2d(img2, window, None, 1, padd, 1, channel)

    # 对齐 PyTorch 版本：使用 pow(2) 而不是 * mu1
    # PyTorch: mu1_sq = mu1.pow(2), mu2_sq = mu2.pow(2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nn.conv2d(img1 * img1, window, None, 1, padd, 1, channel) - mu1_sq
    sigma2_sq = nn.conv2d(img2 * img2, window, None, 1, padd, 1, channel) - mu2_sq
    sigma12 = nn.conv2d(img1 * img2, window, None, 1, padd, 1, channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # 对齐 PyTorch 版本：v1 = 2.0 * sigma12 + C2（使用2.0而不是2）
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    # 对齐 PyTorch 版本：cs = torch.mean(v1 / v2)，先计算mean
    cs = jt.mean(v1 / v2)

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        return ssim_map.mean(), cs
    # Jittor: 多维度 mean 需要逐维度计算，或使用 reshape
    # 对齐 PyTorch: mean(1).mean(1).mean(1)
    return ssim_map.mean(dim=1).mean(dim=1).mean(dim=1), cs


def msssim(img1, img2, window_size=11, size_average=True, weights=None, normalize=False, **kwargs):
    # 对齐 PyTorch 版本：weights 的创建方式
    # PyTorch: weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    if weights is None:
        weights = jt.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=jt.float32)
    # Jittor: 使用 len() 或 shape[0] 获取元素数量
    if hasattr(weights, 'numel'):
        levels = weights.numel()
    else:
        levels = weights.shape[0] if len(weights.shape) > 0 else len(weights)
    channel = img1.shape[1]
    # 对齐 PyTorch 版本：默认使用 0-255 的数值范围（输入未经归一化时）
    # 如果显式传入 val_range，则使用该值；否则固定为 255.0，避免自动推断带来的尺度漂移
    val_range = kwargs.get('val_range', 255.0)
    # 对齐 PyTorch 版本：window在_ssim内部动态创建，这里传入None
    window = None
    mssim = []
    mcs = []
    for _ in range(levels):
        # 对齐 PyTorch 版本：每次循环都传递val_range（如果为None，在_ssim内部推断）
        # 这样每次下采样后都会重新推断val_range，与PyTorch版本一致
        # window在_ssim内部根据当前图像尺寸动态创建
        ssim_map, cs_map = _ssim(img1, img2, window, window_size, channel, data_range=val_range, size_average=size_average)
        mssim.append(ssim_map)
        mcs.append(cs_map)
        # downsample - 对齐 PyTorch: F.avg_pool2d(img, (2, 2))
        img1 = nn.pool(img1, 2, op="mean", stride=2)
        img2 = nn.pool(img2, 2, op="mean", stride=2)
    mssim = jt.stack(mssim)
    mcs = jt.stack(mcs)
    
    # 可選的 normalize：對齊 pytorch_msssim 的 (x+1)/2 寫法
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2
    
    # 对齐 PyTorch 版本的计算方式：
    # PyTorch: pow1 = mcs ** weights, pow2 = mssim ** weights
    #          output = torch.prod(pow1[:-1] * pow2[-1])
    # 注意：weights 是 shape (5,)，mcs 和 mssim 也是 shape (5,)
    # pow1[i] = mcs[i] ** weights[i], pow2[i] = mssim[i] ** weights[i]
    pow1 = mcs ** weights  # shape: (5,)
    pow2 = mssim ** weights  # shape: (5,)
    # From Matlab implementation: prod(pow1[:-1] * pow2[-1])
    # 即：prod([pow1[0]*pow2[-1], pow1[1]*pow2[-1], pow1[2]*pow2[-1], pow1[3]*pow2[-1]])
    result = (pow1[:-1] * pow2[-1]).prod()
    return result


# keep same name for compatibility
ssim_loss = msssim


class Gradient_loss(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # average
        patch_szie = 3
        reflection_padding = int(np.floor(patch_szie / 2))
        weight = jt.ones([1, 1, patch_szie, patch_szie])
        self.conv_avg = nn.Conv(channels, channels, (patch_szie, patch_szie),
                                stride=1, padding=reflection_padding, bias=False)
        self.conv_avg.weight.assign((1 / (patch_szie * patch_szie)) * weight.repeat(channels, 1, 1, 1).float())
        self.conv_avg.weight.stop_grad()
        
        weight = jt.array([[[[0.,  1., 0.], 
                             [1., -4., 1.], 
                             [0.,  1., 0.]]]], dtype=jt.float32)
        self.conv_two = nn.Conv(channels, channels, (3, 3), stride=1, padding=1, bias=False)
        self.conv_two.weight.assign(weight.repeat(channels, 1, 1, 1).float())
        self.conv_two.weight.stop_grad()

        # LoG
        weight_log = jt.array([[[[0., 0., -1, 0., 0.],
                                  [0., -1, -2, -1, 0.],
                                  [-1., -2., 16., -2., -1.],
                                  [0., -1., -2., -1., 0.],
                                  [0., 0., -1., 0., 0.]]]], dtype=jt.float32)
        self.conv_log = nn.Conv(channels, channels, (5, 5), stride=1, padding=3, bias=False)
        self.conv_log.weight.assign(weight_log.repeat(channels, 1, 1, 1).float())
        self.conv_log.weight.stop_grad()

        # sobel
        weight_s1 = jt.array([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]], dtype=jt.float32)
        weight_s2 = jt.array([[[[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]]], dtype=jt.float32)
        self.conv_sx = nn.Conv(channels, channels, (3, 3), stride=1, padding=1, bias=False)
        self.conv_sx.weight.assign(weight_s2.repeat(channels, 1, 1, 1).float())
        self.conv_sx.weight.stop_grad()
        self.conv_sy = nn.Conv(channels, channels, (3, 3), stride=1, padding=1, bias=False)
        self.conv_sy.weight.assign(weight_s1.repeat(channels, 1, 1, 1).float())
        self.conv_sy.weight.stop_grad()

        # average
        patch_szie = 3
        reflection_padding = int(np.floor(patch_szie / 2))
        weight_avg = jt.ones([1, 1, patch_szie, patch_szie])
        self.conv_avg = nn.Conv(channels, channels, (patch_szie, patch_szie),
                                stride=1, padding=reflection_padding, bias=False)
        self.conv_avg.weight.assign((1 / (patch_szie * patch_szie)) * weight_avg.repeat(channels, 1, 1, 1).float())
        self.conv_avg.weight.stop_grad()

    def execute(self, out, x_ir, x_vi):
        channels = x_ir.shape[1]
        channels_t = out.shape[1]
        assert channels == channels_t, \
            f"The channels of x ({channels}) doesn't match the channels of target ({channels_t})."
        g_o = jt.clamp(self.conv_two(out), min_v=0)
        g_xir = jt.clamp(self.conv_two(x_ir), min_v=0)
        g_xvi = jt.clamp(self.conv_two(x_vi), min_v=0)

        g_target = jt.maximum(g_xir, g_xvi)
        loss = mse_loss(g_o, g_target)
        return loss, g_o, g_xir, g_xvi, g_target


class Order_loss(nn.Module):
    def __init__(self, channels, patch_szie=11):
        super().__init__()

        reflection_padding = int(np.floor(patch_szie / 2))
        weight = jt.ones([1, 1, patch_szie, patch_szie])
        self.conv_two = nn.Conv(channels, channels, (patch_szie, patch_szie),
                                stride=1, padding=reflection_padding, bias=False)
        self.conv_two.weight.assign((1 / (patch_szie * patch_szie)) * weight.repeat(channels, 1, 1, 1).float())
        self.conv_two.weight.stop_grad()

        # LoG
        weight_log = jt.array([[[[0., 0., -1, 0., 0.],
                                  [0., -1., -2., -1., 0.],
                                  [-1., -2., 16., -2., -1.],
                                  [0., -1., -2., -1., 0.],
                                  [0., 0., -1., 0., 0.]]]], dtype=jt.float32)
        self.conv_log = nn.Conv(channels, channels, (5, 5), stride=1, padding=2, bias=False)
        self.conv_log.weight.assign(weight_log.repeat(channels, 1, 1, 1).float())
        self.conv_log.weight.stop_grad()

    def execute(self, out, x, y):
        channels1 = x.shape[1]
        channels2 = y.shape[1]
        assert channels1 == channels2, \
            f"The channels of x ({channels1}) doesn't match the channels of target ({channels2})."
        s_x = self.conv_two(x)
        s_y = self.conv_two(y)
        w_x = s_x / (s_x + s_y + EPSILON)
        w_y = s_y / (s_x + s_y + EPSILON)
        
        t_one = jt.ones_like(w_x)
        mask = jt.clamp(w_x - w_y, min_v=0.0)
        mask = jt.where(mask > 0, t_one, mask)
        target = mask * x + (1 - mask) * y
        loss_p = mse_loss(out, target)
        
        return loss_p, target


def gram_matrix(y):
    (b, ch, h, w) = y.shape
    features = y.reshape(b, ch, w * h)
    features_t = features.transpose(0, 2, 1)
    gram = jt.matmul(features, features_t) / (ch * h * w)
    return gram


def feature_loss(vgg, ir, vi, f):
    f_fea = vgg(f)
    ir_fea = vgg(ir)
    vi_fea = vgg(vi)

    loss_rgb = 0.
    loss_fea = 0.
    loss_gram = 0.
    # feature loss
    t_idx = 0
    w_fea = [0.01, 0.01, 200.0]
    w_ir = [0.0, 2.0, 4.0]
    w_vi = [1.0, 1.0, 1.0]
    for _vi, _ir, _f, w1, w2, w3 in zip(vi_fea, ir_fea, f_fea, w_fea, w_ir, w_vi):
        if t_idx == 0:
            loss_rgb += w1 * mse_loss(_f, w2 * _ir + w3 * _vi)
        if t_idx == 1:
            loss_fea += w1 * mse_loss(_f, w2 * _ir + w3 * _vi)
        if t_idx == 2:
            gram_s = gram_matrix(_f)
            gram_t = w2 * gram_matrix(_ir) + w3 * gram_matrix(_vi)
            loss_gram += w1 * mse_loss(gram_s, gram_t)
        t_idx += 1
    return loss_rgb, loss_fea, loss_gram


# ------------------------------ Patch loss ------------------------------ #
def padding_tensor(x, patch_size):
    b, c, h, w = x.shape
    h_patches = int(np.ceil(h / patch_size))
    w_patches = int(np.ceil(w / patch_size))

    h_padding = h_patches * patch_size - h
    w_padding = w_patches * patch_size - w
    reflection_padding = [0, w_padding, 0, h_padding]
    x = nn.pad(x, reflection_padding, mode="reflect")
    return x, [h_patches, w_patches, h_padding, w_padding]


class get_patch_tensor(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def execute(self, x):
        b, c, h, w = x.shape
        x, patches_paddings = padding_tensor(x, self.patch_size)
        h_patches = patches_paddings[0]
        w_patches = patches_paddings[1]
        patch_matrix = None
        for i in range(h_patches):
            for j in range(w_patches):
                patch_one = x[:, :, i * self.patch_size: (i + 1) * self.patch_size,
                            j * self.patch_size: (j + 1) * self.patch_size]
                patch_one = patch_one.reshape(-1, c, 1, self.patch_size * self.patch_size)
                if i == 0 and j == 0:
                    patch_matrix = patch_one
                else:
                    patch_matrix = jt.concat((patch_matrix, patch_one), dim=2)
        return patch_matrix, patches_paddings


class Patch_loss(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.get_patch = get_patch_tensor(patch_size)

    def execute(self, out, x1, x2):
        B, C, H, W = out.shape
        patch0, hw_p0 = self.get_patch(out)
        patch1, hw_p1 = self.get_patch(x1)
        patch2, hw_p2 = self.get_patch(x2)

        b0, c0, n0, p0 = patch0.shape
        b1, c1, n1, p1 = patch1.shape
        b2, c2, n2, p2 = patch2.shape
        assert n0 == n1 == n2 and p0 == p1 == p2, \
                f"The number of patches ({n0}, {n1} and {n2}) or the patch size ({p0}, {p1} and {p2}) doesn't match."

        mu1 = jt.mean(patch1, dim=3)
        mu2 = jt.mean(patch2, dim=3)

        mu1_re = mu1.reshape(b1, c1, n1, 1).repeat(1, 1, 1, p1)
        mu2_re = mu2.reshape(b2, c2, n2, 1).repeat(1, 1, 1, p2)

        sd1 = jt.sqrt(jt.sum(((patch1 - mu1_re) ** 2), dim=3) / p1)
        sd2 = jt.sqrt(jt.sum(((patch2 - mu2_re) ** 2), dim=3) / p2)

        w1 = sd1 / (sd1 + sd2 + EPSILON)
        w2 = sd2 / (sd1 + sd2 + EPSILON)
        w1 = w1.reshape(b1, c1, n1, 1).repeat(1, 1, 1, p1)
        w2 = w2.reshape(b2, c2, n2, 1).repeat(1, 1, 1, p2)

        weights = [w1.reshape(b1, c1, hw_p1[0] * self.patch_size, hw_p1[1] * self.patch_size),
                   w2.reshape(b1, c1, hw_p1[0] * self.patch_size, hw_p1[1] * self.patch_size)]
        out_loss = mse_loss(patch0, w1 * patch1 + w2 * patch2)
        return out_loss, weights
