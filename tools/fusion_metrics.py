#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像融合评估指标计算模块

包含六个常用评估指标：
1. EN (Entropy) - 信息熵
2. SD (Standard Deviation) - 标准差
3. MI (Mutual Information) - 互信息
4. Qabf (Gradient-based Fusion Performance) - 基于梯度的融合质量
5. SSIM (Structural Similarity Index) - 结构相似性
6. AG (Average Gradient) - 平均梯度
"""
import numpy as np
from scipy import ndimage
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim_skimage
import cv2


def normalize_image(img):
    """归一化图像到[0, 255]范围"""
    if img.max() > 1.0:
        img = img.astype(np.float64)
    else:
        img = (img * 255.0).astype(np.float64)
    return np.clip(img, 0, 255).astype(np.uint8)


def calculate_EN(img):
    """
    计算信息熵 (Entropy)
    
    信息熵衡量图像的平均信息量，反映图像的细节和纹理丰富度。
    熵值越高，说明图像包含的信息越多。
    
    Args:
        img: 输入图像 (numpy array, 0-255)
    
    Returns:
        float: 信息熵值
    """
    img = normalize_image(img)
    # 使用skimage的shannon_entropy计算
    entropy_value = shannon_entropy(img)
    return entropy_value


def calculate_SD(img):
    """
    计算标准差 (Standard Deviation)
    
    标准差衡量图像灰度值的离散程度，反映图像的对比度。
    标准差越大，说明图像的对比度越高，图像越清晰。
    
    Args:
        img: 输入图像 (numpy array)
    
    Returns:
        float: 标准差值
    """
    img = normalize_image(img).astype(np.float64)
    # 使用numpy的标准差函数，更准确
    std_value = np.std(img)
    return std_value


def calculate_MI(img1, img2, img_f):
    """
    计算互信息 (Mutual Information)
    
    互信息衡量融合图像与源图像之间的共享信息量。
    MI = MI(img_f, img1) + MI(img_f, img2)
    互信息越大，说明融合图像保留了更多源图像的信息。
    
    Args:
        img1: 源图像1 (IR图像)
        img2: 源图像2 (VI图像)
        img_f: 融合图像
    
    Returns:
        float: 互信息值
    """
    img1 = normalize_image(img1)
    img2 = normalize_image(img2)
    img_f = normalize_image(img_f)
    
    def mutual_information(img_a, img_b):
        """计算两幅图像的互信息"""
        # 确保图像是整数类型（0-255）
        img_a = img_a.astype(np.uint8)
        img_b = img_b.astype(np.uint8)
        
        # 计算联合直方图
        # histogram2d(x, y) 返回: H, xedges, yedges
        # H[i, j] 表示 x在bin j，y在bin i 的计数
        # 即: H的行对应y（第二个参数），列对应x（第一个参数）
        hist_2d, _, _ = np.histogram2d(img_a.ravel(), img_b.ravel(), bins=256, range=[[0, 256], [0, 256]])
        
        # 归一化为概率
        hist_2d = hist_2d / (img_a.size + 1e-10)
        
        # 计算边际概率
        # p_a[j] = sum_i hist_2d[i, j] (对行求和，得到x的边际分布)
        # p_b[i] = sum_j hist_2d[i, j] (对列求和，得到y的边际分布)
        p_a = np.sum(hist_2d, axis=0)  # shape: (256,)，对应img_a的分布
        p_b = np.sum(hist_2d, axis=1)  # shape: (256,)，对应img_b的分布
        
        # 计算互信息: MI = sum_i sum_j p_ab(i,j) * log2(p_ab(i,j) / (p_a(i) * p_b(j)))
        mi = 0
        for i in range(256):  # y的bin索引（行）
            for j in range(256):  # x的bin索引（列）
                if hist_2d[i, j] > 1e-10 and p_a[j] > 1e-10 and p_b[i] > 1e-10:
                    mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (p_a[j] * p_b[i] + 1e-10))
        
        return mi
    
    MI_1f = mutual_information(img1, img_f)
    MI_2f = mutual_information(img2, img_f)
    MI = MI_1f + MI_2f
    
    return MI


def calculate_AG(img):
    """
    计算平均梯度 (Average Gradient)
    
    平均梯度反映图像的清晰度和细节表现能力。
    平均梯度越大，说明图像的边缘和细节越清晰。
    
    Args:
        img: 输入图像 (numpy array)
    
    Returns:
        float: 平均梯度值
    """
    img = normalize_image(img).astype(np.float64)
    
    # 使用Sobel算子计算梯度
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    
    # 计算平均梯度
    ag_value = np.mean(gradient_magnitude)
    return ag_value


def calculate_SSIM(img1, img2):
    """
    计算结构相似性 (Structural Similarity Index)
    
    用于衡量两幅图像之间的结构相似性。
    SSIM值范围在[-1, 1]之间，值越大表示两幅图像越相似。
    
    Args:
        img1: 图像1 (numpy array)
        img2: 图像2 (numpy array)
    
    Returns:
        float: SSIM值
    """
    img1 = normalize_image(img1)
    img2 = normalize_image(img2)
    
    # 确保图像大小相同
    if img1.shape != img2.shape:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
    
    # 使用skimage计算SSIM
    ssim_value = ssim_skimage(img1, img2, data_range=255)
    return ssim_value


def calculate_Qabf(img1, img2, img_f):
    """
    计算基于梯度的融合质量指标 (Qabf)
    
    Qabf基于梯度信息评估融合质量，考虑源图像的边缘和细节信息。
    Qabf值范围在[0, 1]之间，值越大表示融合质量越好。
    
    参考: Xydeas and Petrovic, "Objective image fusion performance measure"
    
    Args:
        img1: 源图像1 (IR图像)
        img2: 源图像2 (VI图像)
        img_f: 融合图像
    
    Returns:
        float: Qabf值
    """
    img1 = normalize_image(img1).astype(np.float64)
    img2 = normalize_image(img2).astype(np.float64)
    img_f = normalize_image(img_f).astype(np.float64)
    
    # 使用Sobel算子计算梯度
    def sobel_gradient(img):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        angle = np.arctan2(sobely, sobelx + 1e-10)
        return magnitude, angle
    
    g1, a1 = sobel_gradient(img1)
    g2, a2 = sobel_gradient(img2)
    gf, af = sobel_gradient(img_f)
    
    # 计算梯度相似性（向量化操作提高效率）
    g_max = np.maximum(g1, g2)
    g_min = np.minimum(g1, g2)
    Qg = np.where(g_max > 1e-10, g_min / (g_max + 1e-10), np.zeros_like(g1))
    
    # 计算角度相似性
    alpha1 = np.abs(a1 - af)
    alpha1 = np.where(alpha1 > np.pi / 2, np.pi - alpha1, alpha1)
    Qa1 = 1 - 2 * alpha1 / np.pi
    
    alpha2 = np.abs(a2 - af)
    alpha2 = np.where(alpha2 > np.pi / 2, np.pi - alpha2, alpha2)
    Qa2 = 1 - 2 * alpha2 / np.pi
    
    # 计算融合图像的梯度保留度
    Qg_f1 = np.where(g1 > 1e-10, np.minimum(gf / (g1 + 1e-10), 1.0), np.zeros_like(g1))
    Qg_f2 = np.where(g2 > 1e-10, np.minimum(gf / (g2 + 1e-10), 1.0), np.zeros_like(g2))
    
    # 计算权重
    g_sum = g1 + g2 + 1e-10
    w1 = g1 / g_sum
    w2 = g2 / g_sum
    
    # 计算Qabf (分别对两个源图像计算，然后加权平均)
    Q1 = Qg_f1 * Qa1 * w1
    Q2 = Qg_f2 * Qa2 * w2
    Q = Q1 + Q2
    
    Qabf_value = np.mean(Q)
    
    return Qabf_value


def calculate_all_metrics(img_ir, img_vi, img_fused):
    """
    计算所有六个评估指标
    
    Args:
        img_ir: 红外图像 (IR)
        img_vi: 可见光图像 (VI)
        img_fused: 融合图像
    
    Returns:
        dict: 包含所有指标值的字典
    """
    metrics = {}
    
    # 1. EN (信息熵)
    metrics['EN'] = calculate_EN(img_fused)
    
    # 2. SD (标准差)
    metrics['SD'] = calculate_SD(img_fused)
    
    # 3. MI (互信息)
    metrics['MI'] = calculate_MI(img_ir, img_vi, img_fused)
    
    # 4. Qabf (基于梯度的融合质量)
    metrics['Qabf'] = calculate_Qabf(img_ir, img_vi, img_fused)
    
    # 5. SSIM (结构相似性) - 计算融合图像与两个源图像的平均SSIM
    ssim_ir = calculate_SSIM(img_ir, img_fused)
    ssim_vi = calculate_SSIM(img_vi, img_fused)
    metrics['SSIM'] = (ssim_ir + ssim_vi) / 2.0
    metrics['SSIM_IR'] = ssim_ir
    metrics['SSIM_VI'] = ssim_vi
    
    # 6. AG (平均梯度)
    metrics['AG'] = calculate_AG(img_fused)
    
    return metrics


if __name__ == '__main__':
    # 测试代码
    import cv2
    
    # 创建测试图像
    img1 = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    img_f = (img1.astype(np.float32) + img2.astype(np.float32)) / 2.0
    
    # 计算指标
    metrics = calculate_all_metrics(img1, img2, img_f)
    
    print("评估指标结果:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

