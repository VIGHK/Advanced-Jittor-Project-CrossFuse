#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制融合网络训练损失对比图：PyTorch vs Jittor
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from pathlib import Path

# 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def parse_fusion_log(log_path):
    """
    从融合网络训练日志文件中解析损失数据
    
    Returns:
        dict: 包含各损失项的列表和epoch平均值
    """
    losses = {
        'epoch': [],
        'batch': [],
        'pix_loss': [],
        'gra_loss': [],
        'total_loss': [],
        'epoch_avg_pix': {},
        'epoch_avg_gra': {},
        'epoch_avg_total': {}
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_epoch = 0
    epoch_pix_losses = []
    epoch_gra_losses = []
    epoch_total_losses = []
    
    for i, line in enumerate(lines):
        # 解析epoch和batch信息
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
        batch_match = re.search(r'Batch (\d+)/(\d+)', line)
        
        # 解析损失值
        pix_match = re.search(r'pix loss:\s*([\d.]+)', line)
        gra_match = re.search(r'gra loss:\s*([\d.]+)', line)
        # total loss 可能在当前行或下一行
        total_match = re.search(r'total loss:\s*([\d.]+)', line)
        
        # 如果这一行包含epoch信息
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            if epoch_num != current_epoch:
                # 新epoch开始，保存上一个epoch的平均值
                if current_epoch > 0 and epoch_pix_losses:
                    losses['epoch_avg_pix'][current_epoch] = np.mean(epoch_pix_losses)
                    losses['epoch_avg_gra'][current_epoch] = np.mean(epoch_gra_losses)
                    losses['epoch_avg_total'][current_epoch] = np.mean(epoch_total_losses)
                current_epoch = epoch_num
                epoch_pix_losses = []
                epoch_gra_losses = []
                epoch_total_losses = []
        
        # 如果这一行包含 pix loss 和 gra loss
        if pix_match and gra_match:
            epoch_num = int(epoch_match.group(1)) if epoch_match else current_epoch
            batch_num = int(batch_match.group(1)) if batch_match else 0
            
            pix_loss = float(pix_match.group(1))
            gra_loss = float(gra_match.group(1))
            
            # 尝试在当前行找 total loss，如果找不到，检查下一行
            total_loss = None
            if total_match:
                total_loss = float(total_match.group(1))
            elif i + 1 < len(lines):
                next_line = lines[i + 1]
                next_total_match = re.search(r'total loss:\s*([\d.]+)', next_line)
                if next_total_match:
                    total_loss = float(next_total_match.group(1))
            
            # 如果找到了所有损失值
            if total_loss is not None and epoch_num > 0:
                losses['epoch'].append(epoch_num)
                losses['batch'].append(batch_num)
                losses['pix_loss'].append(pix_loss)
                losses['gra_loss'].append(gra_loss)
                losses['total_loss'].append(total_loss)
                
                epoch_pix_losses.append(pix_loss)
                epoch_gra_losses.append(gra_loss)
                epoch_total_losses.append(total_loss)
    
    # 保存最后一个epoch的平均值
    if current_epoch > 0 and epoch_pix_losses:
        losses['epoch_avg_pix'][current_epoch] = np.mean(epoch_pix_losses)
        losses['epoch_avg_gra'][current_epoch] = np.mean(epoch_gra_losses)
        losses['epoch_avg_total'][current_epoch] = np.mean(epoch_total_losses)
    
    return losses


def plot_fusion_comparison(pytorch_losses, jittor_losses, output_dir):
    """
    绘制PyTorch vs Jittor的融合网络训练损失对比图
    
    Args:
        pytorch_losses: PyTorch融合网络损失数据
        jittor_losses: Jittor融合网络损失数据
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 颜色和样式定义
    colors = {
        'pytorch': '#1f77b4',  # 蓝色
        'jittor': '#2ca02c'    # 绿色
    }
    linestyles = {
        'pytorch': '-',
        'jittor': '--'
    }
    markers = {
        'pytorch': 'o',
        'jittor': 's'
    }
    
    # 获取epoch列表
    pytorch_epochs = sorted([e for e in pytorch_losses['epoch_avg_total'].keys() if e > 0])
    jittor_epochs = sorted([e for e in jittor_losses['epoch_avg_total'].keys() if e > 0])
    
    # ============================================================
    # 图1：三个损失对比（总损失、像素损失、梯度损失）
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Fusion Network Training Loss Comparison: PyTorch vs Jittor', 
                 fontsize=16, fontweight='bold')
    
    # 总损失
    if pytorch_epochs:
        pytorch_total = [pytorch_losses['epoch_avg_total'][e] for e in pytorch_epochs]
        axes[0].plot(pytorch_epochs, pytorch_total, 
                    label='PyTorch', color=colors['pytorch'], 
                    linestyle=linestyles['pytorch'], linewidth=2, 
                    marker=markers['pytorch'], markersize=6, alpha=0.8)
    
    if jittor_epochs:
        jittor_total = [jittor_losses['epoch_avg_total'][e] for e in jittor_epochs]
        axes[0].plot(jittor_epochs, jittor_total, 
                    label='Jittor', color=colors['jittor'], 
                    linestyle=linestyles['jittor'], linewidth=2, 
                    marker=markers['jittor'], markersize=6, alpha=0.8)
    
    axes[0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # 像素损失
    if pytorch_epochs:
        pytorch_pix = [pytorch_losses['epoch_avg_pix'][e] for e in pytorch_epochs]
        axes[1].plot(pytorch_epochs, pytorch_pix, 
                    label='PyTorch', color=colors['pytorch'], 
                    linestyle=linestyles['pytorch'], linewidth=2, 
                    marker=markers['pytorch'], markersize=6, alpha=0.8)
    
    if jittor_epochs:
        jittor_pix = [jittor_losses['epoch_avg_pix'][e] for e in jittor_epochs]
        axes[1].plot(jittor_epochs, jittor_pix, 
                    label='Jittor', color=colors['jittor'], 
                    linestyle=linestyles['jittor'], linewidth=2, 
                    marker=markers['jittor'], markersize=6, alpha=0.8)
    
    axes[1].set_title('Pixel Loss (L_pix)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    # 梯度损失
    if pytorch_epochs:
        pytorch_gra = [pytorch_losses['epoch_avg_gra'][e] for e in pytorch_epochs]
        axes[2].plot(pytorch_epochs, pytorch_gra, 
                    label='PyTorch', color=colors['pytorch'], 
                    linestyle=linestyles['pytorch'], linewidth=2, 
                    marker=markers['pytorch'], markersize=6, alpha=0.8)
    
    if jittor_epochs:
        jittor_gra = [jittor_losses['epoch_avg_gra'][e] for e in jittor_epochs]
        axes[2].plot(jittor_epochs, jittor_gra, 
                    label='Jittor', color=colors['jittor'], 
                    linestyle=linestyles['jittor'], linewidth=2, 
                    marker=markers['jittor'], markersize=6, alpha=0.8)
    
    axes[2].set_title('Gradient Loss (L_gra)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Loss', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fusion_network_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"融合网络对比图已保存至: {output_path}")
    plt.close()
    
    # ============================================================
    # 图2：总损失对比（单独大图）
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if pytorch_epochs:
        pytorch_total = [pytorch_losses['epoch_avg_total'][e] for e in pytorch_epochs]
        ax.plot(pytorch_epochs, pytorch_total, 
               label='PyTorch', color=colors['pytorch'], 
               linestyle=linestyles['pytorch'], linewidth=2, 
               marker=markers['pytorch'], markersize=6, alpha=0.8)
    
    if jittor_epochs:
        jittor_total = [jittor_losses['epoch_avg_total'][e] for e in jittor_epochs]
        ax.plot(jittor_epochs, jittor_total, 
               label='Jittor', color=colors['jittor'], 
               linestyle=linestyles['jittor'], linewidth=2, 
               marker=markers['jittor'], markersize=6, alpha=0.8)
    
    ax.set_title('Fusion Network Total Loss Comparison: PyTorch vs Jittor', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fusion_network_total_loss_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"融合网络总损失对比图已保存至: {output_path}")
    plt.close()
    
    # ============================================================
    # 图3：像素损失和梯度损失对比（上下布局）
    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('Fusion Network Loss Components Comparison: PyTorch vs Jittor', 
                 fontsize=16, fontweight='bold')
    
    # 像素损失
    if pytorch_epochs:
        pytorch_pix = [pytorch_losses['epoch_avg_pix'][e] for e in pytorch_epochs]
        axes[0].plot(pytorch_epochs, pytorch_pix, 
                    label='PyTorch', color=colors['pytorch'], 
                    linestyle=linestyles['pytorch'], linewidth=2, 
                    marker=markers['pytorch'], markersize=6, alpha=0.8)
    
    if jittor_epochs:
        jittor_pix = [jittor_losses['epoch_avg_pix'][e] for e in jittor_epochs]
        axes[0].plot(jittor_epochs, jittor_pix, 
                    label='Jittor', color=colors['jittor'], 
                    linestyle=linestyles['jittor'], linewidth=2, 
                    marker=markers['jittor'], markersize=6, alpha=0.8)
    
    axes[0].set_title('Pixel Loss (L_pix)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # 梯度损失
    if pytorch_epochs:
        pytorch_gra = [pytorch_losses['epoch_avg_gra'][e] for e in pytorch_epochs]
        axes[1].plot(pytorch_epochs, pytorch_gra, 
                    label='PyTorch', color=colors['pytorch'], 
                    linestyle=linestyles['pytorch'], linewidth=2, 
                    marker=markers['pytorch'], markersize=6, alpha=0.8)
    
    if jittor_epochs:
        jittor_gra = [jittor_losses['epoch_avg_gra'][e] for e in jittor_epochs]
        axes[1].plot(jittor_epochs, jittor_gra, 
                    label='Jittor', color=colors['jittor'], 
                    linestyle=linestyles['jittor'], linewidth=2, 
                    marker=markers['jittor'], markersize=6, alpha=0.8)
    
    axes[1].set_title('Gradient Loss (L_gra)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fusion_network_loss_components_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"融合网络损失组件对比图已保存至: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("绘制融合网络训练损失对比图：PyTorch vs Jittor")
    print("=" * 80)
    
    # 日志文件路径
    pytorch_log = "./logs/my_trained_models/transfuse/train_transfuse_20251220_140533.log"
    jittor_log = "./logs/my_trained_models/transfuse/jittor/train_transfuse_20260220_021031.log"
    
    # 检查文件是否存在
    if not os.path.exists(pytorch_log):
        print(f"错误：PyTorch日志文件不存在: {pytorch_log}")
        return
    
    if not os.path.exists(jittor_log):
        print(f"错误：Jittor日志文件不存在: {jittor_log}")
        return
    
    print(f"\n正在解析日志文件...")
    print(f"  PyTorch: {pytorch_log}")
    print(f"  Jittor: {jittor_log}")
    
    # 解析日志
    print("\n解析中...")
    pytorch_losses = parse_fusion_log(pytorch_log)
    jittor_losses = parse_fusion_log(jittor_log)
    
    # 检查数据
    pytorch_epochs = sorted([e for e in pytorch_losses['epoch_avg_total'].keys() if e > 0])
    jittor_epochs = sorted([e for e in jittor_losses['epoch_avg_total'].keys() if e > 0])
    
    print(f"\n解析完成：")
    print(f"  PyTorch: {len(pytorch_epochs)} epochs")
    print(f"  Jittor: {len(jittor_epochs)} epochs")
    
    if not pytorch_epochs:
        print("警告：PyTorch日志中没有解析到有效的epoch数据")
        return
    
    if not jittor_epochs:
        print("警告：Jittor日志中没有解析到有效的epoch数据")
        return
    
    # 绘制对比图
    print(f"\n正在绘制对比图...")
    output_dir = './output/loss_curves'
    plot_fusion_comparison(pytorch_losses, jittor_losses, output_dir)
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"所有图表已保存至: {output_dir}/")
    print("  - fusion_network_comparison.png (三个损失对比)")
    print("  - fusion_network_total_loss_comparison.png (总损失对比)")
    print("  - fusion_network_loss_components_comparison.png (损失组件对比)")
    
    # 打印关键指标对比
    print("\n" + "=" * 80)
    print("关键指标对比（Epoch平均值）")
    print("=" * 80)
    print(f"{'Epoch':<8} {'PyTorch Total':<18} {'Jittor Total':<18} {'差异率':<15}")
    print("-" * 80)
    max_epochs = max(len(pytorch_epochs), len(jittor_epochs))
    for i in range(max_epochs):
        epoch = i + 1
        pytorch_total = pytorch_losses['epoch_avg_total'].get(epoch, None)
        jittor_total = jittor_losses['epoch_avg_total'].get(epoch, None)
        
        if pytorch_total and jittor_total:
            diff_pct = ((jittor_total - pytorch_total) / pytorch_total) * 100
            print(f"{epoch:<8} {pytorch_total:<18.2f} {jittor_total:<18.2f} {diff_pct:+.1f}%")
        elif pytorch_total:
            print(f"{epoch:<8} {pytorch_total:<18.2f} {'N/A':<18}")
        elif jittor_total:
            print(f"{epoch:<8} {'N/A':<18} {jittor_total:<18.2f}")
    
    # 打印像素损失和梯度损失对比
    print("\n" + "=" * 80)
    print("像素损失对比")
    print("=" * 80)
    print(f"{'Epoch':<8} {'PyTorch Pix':<18} {'Jittor Pix':<18} {'差异率':<15}")
    print("-" * 80)
    for i in range(max_epochs):
        epoch = i + 1
        pytorch_pix = pytorch_losses['epoch_avg_pix'].get(epoch, None)
        jittor_pix = jittor_losses['epoch_avg_pix'].get(epoch, None)
        
        if pytorch_pix and jittor_pix:
            diff_pct = ((jittor_pix - pytorch_pix) / pytorch_pix) * 100
            print(f"{epoch:<8} {pytorch_pix:<18.2f} {jittor_pix:<18.2f} {diff_pct:+.1f}%")
    
    print("\n" + "=" * 80)
    print("梯度损失对比")
    print("=" * 80)
    print(f"{'Epoch':<8} {'PyTorch Gra':<18} {'Jittor Gra':<18} {'差异率':<15}")
    print("-" * 80)
    for i in range(max_epochs):
        epoch = i + 1
        pytorch_gra = pytorch_losses['epoch_avg_gra'].get(epoch, None)
        jittor_gra = jittor_losses['epoch_avg_gra'].get(epoch, None)
        
        if pytorch_gra and jittor_gra:
            diff_pct = ((jittor_gra - pytorch_gra) / pytorch_gra) * 100
            print(f"{epoch:<8} {pytorch_gra:<18.2f} {jittor_gra:<18.2f} {diff_pct:+.1f}%")


if __name__ == '__main__':
    main()
