#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量评估融合结果的脚本

计算六个评估指标：EN, SD, MI, Qabf, SSIM, AG
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tools.fusion_metrics import calculate_all_metrics

def load_image(image_path, is_gray=True):
    """加载图像"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    if not is_gray and len(img.shape) == 3:
        # 如果是彩色图像，转换为灰度
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img


def evaluate_single_pair(ir_path, vi_path, fused_path, verbose=True):
    """
    评估单对图像的融合结果
    
    Args:
        ir_path: 红外图像路径
        vi_path: 可见光图像路径
        fused_path: 融合图像路径
        verbose: 是否打印详细信息
    
    Returns:
        dict: 包含所有指标值的字典
    """
    # 加载图像
    img_ir = load_image(ir_path, is_gray=True)
    img_vi = load_image(vi_path, is_gray=True)
    img_fused = load_image(fused_path, is_gray=True)
    
    # 确保图像大小一致
    h_min = min(img_ir.shape[0], img_vi.shape[0], img_fused.shape[0])
    w_min = min(img_ir.shape[1], img_vi.shape[1], img_fused.shape[1])
    
    img_ir = img_ir[:h_min, :w_min]
    img_vi = img_vi[:h_min, :w_min]
    img_fused = img_fused[:h_min, :w_min]
    
    # 计算指标
    metrics = calculate_all_metrics(img_ir, img_vi, img_fused)
    
    if verbose:
        print(f"\n评估结果: {os.path.basename(fused_path)}")
        print("-" * 60)
        print(f"EN (信息熵):      {metrics['EN']:.4f}")
        print(f"SD (标准差):      {metrics['SD']:.4f}")
        print(f"MI (互信息):      {metrics['MI']:.4f}")
        print(f"Qabf (梯度质量):  {metrics['Qabf']:.4f}")
        print(f"SSIM (结构相似):  {metrics['SSIM']:.4f}")
        print(f"AG (平均梯度):    {metrics['AG']:.4f}")
    
    return metrics


def match_source_images(fused_file, ir_dir, vi_dir):
    """
    根据融合图像文件名匹配对应的IR和VI源图像
    
    支持的融合图像文件名格式：
    - results_transfuse_IR1.png -> IR1.png (IR) 和 VIS1.png (VI)
    - results_crossfuse_IR1.png -> IR1.png (IR) 和 VIS1.png (VI)
    - IR1.png -> IR1.png (IR) 和 VIS1.png (VI)
    - 其他格式...
    """
    import re
    
    base_name = os.path.splitext(fused_file)[0]
    ext = os.path.splitext(fused_file)[1]
    
    # 尝试从融合图像文件名中提取源图像名称
    # 模式1: results_transfuse_IR1.png -> IR1
    match = re.search(r'IR(\d+)', base_name)
    if match:
        ir_num = match.group(1)
        ir_name = f'IR{ir_num}{ext}'
        vi_name = f'VIS{ir_num}{ext}'
    else:
        # 模式2: 直接使用文件名
        ir_name = fused_file
        vi_name = fused_file.replace('IR', 'VIS')
    
    ir_path = os.path.join(ir_dir, ir_name)
    vi_path = os.path.join(vi_dir, vi_name)
    
    # 如果找不到，尝试其他扩展名
    if not os.path.exists(ir_path):
        for alt_ext in ['.png', '.jpg', '.bmp', '.PNG', '.JPG', '.BMP']:
            alt_ir_path = os.path.join(ir_dir, os.path.splitext(ir_name)[0] + alt_ext)
            if os.path.exists(alt_ir_path):
                ir_path = alt_ir_path
                ir_name = os.path.basename(alt_ir_path)
                break
    
    if not os.path.exists(vi_path):
        for alt_ext in ['.png', '.jpg', '.bmp', '.PNG', '.JPG', '.BMP']:
            alt_vi_path = os.path.join(vi_dir, os.path.splitext(vi_name)[0] + alt_ext)
            if os.path.exists(alt_vi_path):
                vi_path = alt_vi_path
                vi_name = os.path.basename(alt_vi_path)
                break
    
    return ir_path, vi_path


def evaluate_dataset(ir_dir, vi_dir, fused_dir, output_csv=None):
    """
    批量评估数据集中的所有融合结果
    
    Args:
        ir_dir: 红外图像目录
        vi_dir: 可见光图像目录
        fused_dir: 融合图像目录
        output_csv: 输出CSV文件路径（可选）
    
    Returns:
        pandas.DataFrame: 包含所有评估结果的DataFrame
    """
    # 获取融合图像列表
    fused_files = sorted([f for f in os.listdir(fused_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
    
    if not fused_files:
        print(f"警告：在 {fused_dir} 中未找到融合图像")
        return None
    
    results = []
    
    print(f"开始评估，共 {len(fused_files)} 张图像...")
    print("=" * 80)
    
    for fused_file in fused_files:
        fused_path = os.path.join(fused_dir, fused_file)
        
        # 使用新的匹配函数
        ir_path, vi_path = match_source_images(fused_file, ir_dir, vi_dir)
        
        if not os.path.exists(ir_path):
            print(f"警告：无法找到 {fused_file} 对应的IR图像 ({os.path.basename(ir_path)})，跳过")
            continue
        
        if not os.path.exists(vi_path):
            print(f"警告：无法找到 {fused_file} 对应的VI图像 ({os.path.basename(vi_path)})，跳过")
            continue
        
        try:
            metrics = evaluate_single_pair(ir_path, vi_path, fused_path, verbose=False)
            metrics['image_name'] = fused_file
            results.append(metrics)
        except Exception as e:
            print(f"错误：评估 {fused_file} 时出错: {str(e)}")
            continue
    
    if not results:
        print("错误：没有成功评估任何图像")
        return None
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 重新排列列的顺序
    columns_order = ['image_name', 'EN', 'SD', 'MI', 'Qabf', 'SSIM', 'SSIM_IR', 'SSIM_VI', 'AG']
    df = df[columns_order]
    
    # 计算平均值
    avg_row = {
        'image_name': 'Average',
        'EN': df['EN'].mean(),
        'SD': df['SD'].mean(),
        'MI': df['MI'].mean(),
        'Qabf': df['Qabf'].mean(),
        'SSIM': df['SSIM'].mean(),
        'SSIM_IR': df['SSIM_IR'].mean(),
        'SSIM_VI': df['SSIM_VI'].mean(),
        'AG': df['AG'].mean(),
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("评估结果汇总")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # 保存到CSV
    if output_csv:
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存至: {output_csv}")
    
    return df


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='评估图像融合结果')
    parser.add_argument('--ir_dir', type=str, required=True, help='红外图像目录')
    parser.add_argument('--vi_dir', type=str, required=True, help='可见光图像目录')
    parser.add_argument('--fused_dir', type=str, required=True, help='融合图像目录')
    parser.add_argument('--output_csv', type=str, default=None, help='输出CSV文件路径（可选）')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.ir_dir):
        print(f"错误：红外图像目录不存在: {args.ir_dir}")
        return
    
    if not os.path.exists(args.vi_dir):
        print(f"错误：可见光图像目录不存在: {args.vi_dir}")
        return
    
    if not os.path.exists(args.fused_dir):
        print(f"错误：融合图像目录不存在: {args.fused_dir}")
        return
    
    # 执行评估
    df = evaluate_dataset(args.ir_dir, args.vi_dir, args.fused_dir, args.output_csv)
    
    if df is not None:
        print("\n评估完成！")


if __name__ == '__main__':
    # 如果没有命令行参数，可以在这里直接指定路径进行测试
    import sys
    
    if len(sys.argv) == 1:
        # 示例用法
        print("使用示例:")
        print("python evaluate_fusion_results.py --ir_dir <IR_DIR> --vi_dir <VI_DIR> --fused_dir <FUSED_DIR> [--output_csv <OUTPUT.csv>]")
        print("\n或者直接在代码中调用 evaluate_dataset() 函数")
    else:
        main()

