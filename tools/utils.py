# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: hui_li_jnu@163.com
# @Project : vision_transformer_pytorch
# @File : tools.py
# @Time : 2021/5/12 20:22

import random
import jittor as jt
from jittor import nn
import numpy as np
import cv2
from os import listdir
from os.path import join, isdir
import os
import seaborn as sns
import logging
import math
import matplotlib.pyplot as plt
from args_trans import Args as args

EPSILON = 1e-6


def list_images_datasets(directory, num, filter_type=None):
    """
    递归遍历目录，收集所有图像文件
    支持 KAIST 数据集结构：set00/V000/lwir/, set00/V001/lwir/ 等
    
    Args:
        directory: 目录列表，例如 ['./database/']
        num: 最大图像数量
        filter_type: 'ir' 只收集 lwir 目录下的图像，'vi' 只收集 visible 目录下的图像，None 收集所有
    """
    images = []
    names = []
    n = len(directory)
    
    def collect_images_recursive(root_dir, max_count=None, filter_keyword=None):
        """递归收集图像文件"""
        collected = []
        for root, dirs, files in os.walk(root_dir):
            # 根据 filter_keyword 过滤路径
            if filter_keyword == 'ir':
                if 'lwir' not in root:
                    continue
            elif filter_keyword == 'vi':
                if 'visible' not in root:
                    continue
            
            # 只处理包含 lwir 或 visible 的路径（KAIST 数据集结构）
            if 'lwir' in root or 'visible' in root:
                for file in sorted(files):
                    if max_count is not None and len(collected) >= max_count:
                        return collected
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                        collected.append(join(root, file))
        return collected
    
    for i in range(n):
        dir_temp = directory[i]
        # 递归收集所有图像，根据 filter_type 过滤
        filter_keyword = filter_type  # 'ir' -> 只收集 lwir, 'vi' -> 只收集 visible
        collected = collect_images_recursive(dir_temp, num if i == 0 else None, filter_keyword)
        
        for img_path in collected:
            if i == 0 and len(images) >= num:
                break
            images.append(img_path)
            names.append(os.path.basename(img_path))
        
        if i == 0 and len(images) >= num:
            break
    
    return images, names


def list_images_test(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        # name = file.lower()
        name = file
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        elif name.endswith('.tif'):
            images.append(join(directory, file))
        # name1 = name.split('.')
        names.append(name)
    
    return images, names


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 4 epochs"""
    # 对齐原始 PyTorch 逻辑：从第 2 个 epoch 开始衰减 10 倍
    # PyTorch: if epoch-1 > 0: lr *= 0.1
    if epoch - 1 > 0:
        lr *= 0.1

    # Jittor optimizers expose lr 属性；如果未来自定义优化器带 param_groups，则同时兼容
    if hasattr(optimizer, "param_groups"):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    if hasattr(optimizer, "lr"):
        optimizer.lr = lr
    return lr


def getBinaryTensor(tensor, boundary):
    one = jt.ones_like(tensor)
    zero = jt.zeros_like(tensor)
    return jt.where(tensor > boundary, one, zero)


class UpsampleReshape_eval(nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def execute(self, x1, x2):
        # x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 == 0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 == 0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        # Jittor pad 格式为 (pad_left, pad_right, pad_top, pad_bottom)
        x2 = nn.pad(x2, reflection_padding, mode="reflect")
        return x2


def recons_midle_feature(x_in, start=0):
    b, c, h, w = x_in.shape
    # x = x[:, 0:64, :, :]
    # x = x.view(b, 1, h*8, w*8)
    h_patches = w_patches = 8
    x = jt.abs(x_in)
    # x = x_in
    x = normalize_tensor(x)
    x = x * 255

    patch_matrix = None
    for i in range(h_patches):
        raw_img = None
        for j in range(h_patches):
            patch_one = x[:, i * w_patches + j + start, :, :]
            patch_one = patch_one.reshape(b, 1, h, w)
            # patch_one = torch.abs(patch_one)
            # patch_one = normalize_tensor(patch_one) * 255
            # patch_one = np.reshape(patch_one, [1, c, c_h, c_w])
            if j == 0:
                raw_img = patch_one
            else:
                raw_img = jt.concat((raw_img, patch_one), 2)
        if i == 0:
            patch_matrix = raw_img
        else:
            patch_matrix = jt.concat((patch_matrix, raw_img), 3)
    # for i in range(6,7):
    #     raw_img = None
    #     for j in range(2,3):
    #         patch_one = x[:, i * w_patches + j + start, :, :]
    #         patch_one = patch_one.view(b, 1, h, w)
    #         # raw_img = patch_one
    #         patch_matrix = patch_one

    return patch_matrix


def recons_midle_feature_two(x_in, y_in, start=0):
    b, c, h, w = x_in.shape
    # x = x[:, 0:64, :, :]
    # x = x.view(b, 1, h*8, w*8)
    h_patches = w_patches = 8
    # x = (normalize_tensor(torch.abs(x_in))) * 255
    # y = (normalize_tensor(torch.abs(y_in))) * 255
    x = x_in
    y = y_in

    patch_matrix_x = None
    patch_matrix_y = None
    for i in range(3,4):
        raw_img = None
        for j in range(3,4):
            patch_one = x[:, i * w_patches + j + start, :, :]
            patch_one = patch_one.reshape(b, 1, h, w)
            # raw_img = patch_one
            patch_matrix_x = patch_one
            
    for i in range(6,7):
        raw_img = None
        for j in range(2,3):
            patch_ = y[:, i * w_patches + j + start, :, :]
            patch_ = patch_.reshape(b, 1, h, w)
            # raw_img = patch_one
            patch_matrix_y = patch_
    z = patch_matrix_x + patch_matrix_y
    z = (normalize_tensor(jt.abs(z))) * 255

    return z


def save_image_heat_map_two(x, y, output_path):
    
    img_fusion = recons_midle_feature_two(x, y)
    
    # img_fusion = normalize_tensor(torch.mean(torch.abs(img_fusion), dim=1, keepdim=True)) * 255
    # img_fusion = normalize_tensor(torch.mean(img_fusion, dim=1, keepdim=True)) * 255
    img_fusion = img_fusion.numpy()[0]

    if len(img_fusion.shape) > 2:
        img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    else:
        img_fusion = img_fusion.astype('uint8')
    
    img_fusion=cv2.applyColorMap(img_fusion, cv2.COLORMAP_JET) # for heat map
    cv2.imwrite(output_path, img_fusion)
    # return img_fusion


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def img_padding(x, c_h, c_w):
    c, h, w = x.shape
    h_patches = int(np.ceil(h / c_h))
    w_patches = int(np.ceil(w / c_w))

    h_padding = h_patches * c_h - h
    w_padding = w_patches * c_w - w
    # reflect, symmetric, wrap, edge, linear_ramp, maximum, mean, median, minimum
    x = np.pad(x, ((0, 0), (0, h_padding), (0, w_padding)), 'reflect')
    return x, [h_patches, w_patches, h_padding, w_padding]


def crop_op(img, c_h, c_w):
    img, pad_para = img_padding(img, c_h, c_w)
    c, h, w = img.shape
    h_patches = pad_para[0]
    w_patches = pad_para[1]
    # -------------------------------------------
    patch_matrix = None
    for i in range(h_patches):
        for j in range(w_patches):
            patch_one = img[:, i * c_h: (i + 1) * c_h, j * c_w: (j + 1) * c_w]
            patch_one = np.reshape(patch_one, [1, c, c_h, c_w])
            if i == 0 and j == 0:
                patch_matrix = patch_one
            else:
                patch_matrix = np.concatenate((patch_matrix, patch_one), 0)

    return patch_matrix, pad_para


# load images
def get_image(path, height=256, width=256, flag=False):
    if flag is True:
        mode = cv2.IMREAD_COLOR
    else:
        mode = cv2.IMREAD_GRAYSCALE
    image = cv2.imread(path, mode)
    # -----------------------------------------------------
    assert image is not None, \
        f"The type of image ({path}) is None."
    # -----------------------------------------------------
    if height is not None and width is not None:
        image = cv2.resize(image, (height, width))
    return image


# get training images
def get_train_images(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        images.append(image)
    images = np.stack(images, axis=0)
    images = jt.array(images, dtype=jt.float32)
    return images


#  --------------------------------------------------------------------
def get_test_images_color(paths, height=256, width=256, flag=True):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    images_cb = []
    images_cr = []
    for path in paths:
        image = get_image(path, height, width, flag)
        image_ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        image_y = image_ycbcr[:, :, 0]
        image_cb = image_ycbcr[:, :, 1]
        image_cr = image_ycbcr[:, :, 2]

        image_y = np.reshape(image_y, [1, image_y.shape[0], image_y.shape[1]])
        image_cb = np.reshape(image_cb, [image.shape[0], image.shape[1], 1])
        image_cr = np.reshape(image_cr, [image.shape[0], image.shape[1], 1])
        
        images.append(image_y)
        images_cb.append(image_cb)
        images_cr.append(image_cr)
        
    images = np.stack(images, axis=0)
    images = jt.array(images, dtype=jt.float32)
    return images, images_cb, images_cr


# get testing images
def get_test_images(paths, crop_h=256, crop_w=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    pad_para = None
    for path in paths:
        image = get_image(path, None, None, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        image_crop, pad_para = crop_op(image, crop_h, crop_w)

        images.append(image_crop)
    images = np.stack(images, axis=0)
    images = jt.array(images, dtype=jt.float32)
    return images, pad_para


def save_image(img_fusion, output_path):
    img_fusion = img_fusion.numpy()[0]

    # Ensure channel number is compatible with OpenCV (1, 3, or 4)
    if len(img_fusion.shape) > 2:
        # img_fusion shape here is (C, H, W)
        c, h, w = img_fusion.shape
        if c not in (1, 3, 4):
            # reduce to single channel for visualization (keep generation logic unchanged)
            img_fusion = img_fusion[0:1, :, :]
        img_fusion = img_fusion.transpose(1, 2, 0)
    
    # Normalize to 0-255 if values are small (feature maps)
    img_max = np.max(img_fusion)
    img_min = np.min(img_fusion)
    if img_max - img_min > EPSILON:
        img_fusion = (img_fusion - img_min) / (img_max - img_min) * 255
    img_fusion = img_fusion.astype('uint8')
    
    cv2.imwrite(output_path, img_fusion)
    

# def save_image_heat(img_fusion, output_path):
#     img_fusion = img_fusion.cpu().data[0].numpy()

#     if len(img_fusion.shape) > 2:
#         img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
#     else:
#         img_fusion = img_fusion.astype('uint8')
#     heat_img = cv2.applyColorMap(img_fusion, cv2.COLORMAP_JET)
#     cv2.imwrite(output_path, heat_img)


def save_image_heat_map_list(fea_list, output_path):
    fea_all = []
    for i, fea in enumerate(fea_list):
        fea_array = save_image_heat_map(fea, output_path)
        if i == 0:
            fea_all = fea_array
        else:
            fea_all = np.concatenate((fea_all, fea_array), 1)
    # fea_all = np.abs(fea_all)
    # fea_all = (fea_all - np.min(fea_all)) / (np.max(fea_all) - np.min(fea_all) + EPSILON)
    # fea_all = fea_all * 255
    fea_all = fea_all.astype('uint8')
    
    fea_all=cv2.applyColorMap(fea_all, cv2.COLORMAP_JET) # for heat map
    cv2.imwrite(output_path, fea_all)
    

def save_image_heat_map(img_fusion, output_path):
    
    # img_fusion = recons_midle_feature(img_fusion)
    
    img_fusion = normalize_tensor(jt.mean(jt.abs(img_fusion), dim=1, keepdims=True)) * 255
    img_fusion = img_fusion.numpy()[0]

    if len(img_fusion.shape) > 2:
        img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    else:
        img_fusion = img_fusion.astype('uint8')
    
    img_fusion=cv2.applyColorMap(img_fusion, cv2.COLORMAP_JET) # for heat map
    cv2.imwrite(output_path, img_fusion)
    # return img_fusion
    

def save_image_color(img_fusion, vi_cb, vi_cr, output_path):
    img_fusion = img_fusion.numpy()[0]

    if len(img_fusion.shape) > 2:
        img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    else:
        img_fusion = img_fusion.astype('uint8')
    img_fusion = np.reshape(img_fusion, [img_fusion.shape[0], img_fusion.shape[1], 1])
    img = [img_fusion, vi_cb[0], vi_cr[0]]
    img = np.squeeze(np.stack(img, axis=2))
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    cv2.imwrite(output_path, img)


def save_image_with_pad(outputs, output_path, pad_para):
    h_patches = pad_para[0]
    w_patches = pad_para[1]
    h_paddings = pad_para[2]
    w_paddings = pad_para[3]
    assert len(outputs) == h_patches * w_patches , \
        f"The number of output patches ({len(outputs)}) doesn't match the crop operation ({h_patches}*{w_patches})."
    final_img = None
    for i in range(h_patches):
        raw_img = None
        for j in range(w_patches):
            patch = outputs[i * w_patches + j]
            patch = patch.numpy()[0]
            if j == 0:
                raw_img = patch
            else:
                raw_img = np.concatenate((raw_img, patch), 2)
        if i == 0:
            final_img = raw_img
        else:
            final_img = np.concatenate((final_img, raw_img), 1)

    if h_paddings == 0 and w_paddings != 0:
        final_img = final_img[:, :, :-w_paddings]
    elif h_paddings != 0 and w_paddings == 0:
        final_img = final_img[:, :-h_paddings, :]
    elif h_paddings == 0 and w_paddings == 0:
        final_img = final_img[:, :, :]
    else:
        final_img = final_img[:, :-h_paddings, :-w_paddings]

    if len(final_img.shape) > 2:
        img_fusion = final_img.transpose(1, 2, 0).astype('uint8')
    else:
        img_fusion = final_img.astype('uint8')
    cv2.imwrite(output_path, img_fusion)


def normalize_tensor(tensor):
    """
    Per-channel min-max normalize tensor to [0,1].
    对齐 PyTorch 版本：使用 view + repeat 而不是 broadcast，确保完全一致。
    遇到非 4D 或元素数量不匹配时，降级为全局归一化以避免 reshape 异常。
    """
    # 如果维度不足 4（如标量/向量），退化为全局归一化，避免 reshape 失败
    if tensor.ndim < 4:
        t_min = jt.min(tensor)
        t_max = jt.max(tensor)
        return (tensor - t_min) / (t_max - t_min + EPSILON)

    b, ch, h, w = tensor.shape
    # flatten spatial dims: (b, ch, h*w)
    tensor_v = tensor.reshape(b, ch, h * w)
    # reduce over last dim，对齐 PyTorch: torch.min(tensor_v, 2)[0]
    # PyTorch: torch.min(tensor_v, 2) 返回 (values, indices) 元组，values 形状为 (b, ch)
    # Jittor: jt.min 对多维张量可能行为不同，需要手动实现
    # 方法：reshape 为 (b*ch, h*w)，然后对 dim=1 求 min/max，再 reshape 回 (b, ch)
    tensor_v_flat = tensor_v.reshape(b * ch, h * w)  # (b*ch, h*w)
    t_min_result = jt.min(tensor_v_flat, dim=1)  # 返回 (values, indices) 或 values
    t_max_result = jt.max(tensor_v_flat, dim=1)
    
    # 处理返回值
    if isinstance(t_min_result, tuple):
        t_min = t_min_result[0]  # (b*ch,)
    else:
        t_min = t_min_result  # (b*ch,)
    
    if isinstance(t_max_result, tuple):
        t_max = t_max_result[0]  # (b*ch,)
    else:
        t_max = t_max_result  # (b*ch,)
    
    # reshape 回 (b, ch)
    t_min = t_min.reshape(b, ch)  # (b*ch,) -> (b, ch)
    t_max = t_max.reshape(b, ch)  # (b*ch,) -> (b, ch)

    # 对齐 PyTorch: view(b, ch, 1, 1).repeat(1, 1, h, w)
    t_min = t_min.reshape(b, ch, 1, 1).repeat(1, 1, h, w)  # (b, ch, h, w)
    t_max = t_max.reshape(b, ch, 1, 1).repeat(1, 1, h, w)  # (b, ch, h, w)

    tensor = (tensor - t_min) / (t_max - t_min + EPSILON)
    return tensor


def vision_features(features, img_type, fea_type):
    file_name = 'feature_maps_' + img_type + '_' + fea_type + '.png'
    output_path = './output/feature_maps/' + file_name

    h = w = int(np.sqrt(features.shape[1]))
    map_all = None
    for idx_h in range(h):
        map_raw = None
        for idx_w in range(w):
            index = idx_h * w + idx_w
            map = features[0, index, :, :].reshape(1, 1, features.shape[2], features.shape[3])
            map = normalize_tensor(map)
            if idx_w == 0:
                map_raw = map
            else:
                map_raw = jt.concat((map_raw, map), 3)
        if idx_h == 0:
            map_all = map_raw
        else:
            map_all = jt.concat((map_all, map_raw), 2)

    # map_all = map_all * 255
    # save images
    # save_image(map_all, output_path)
    show_heatmap(map_all, output_path)


def show_heatmap(feature, output_path):
    sns.set()
    feature = feature.float32()
    feature = feature.clamp(0, 255).numpy()[0]

    feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature) + EPSILON)
    feature = feature * 255
    feature = feature.transpose(1, 2, 0).astype('uint8')
    if feature.shape[2] == 1:
        feature = feature.reshape([feature.shape[0], feature.shape[1]])

    img_fusion=cv2.applyColorMap(feature, cv2.COLORMAP_JET) # for heat map
    cv2.imwrite(output_path, img_fusion)
    
    # fig = plt.figure()
    # # sns.heatmap(feature, cmap='YlGnBu', xticklabels=50, yticklabels=50)
    # sns.heatmap(feature, xticklabels=50, yticklabels=50)
    # fig.savefig(output_path, bbox_inches='tight')
    
