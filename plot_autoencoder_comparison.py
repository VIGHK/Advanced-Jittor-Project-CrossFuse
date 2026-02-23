#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制自编码器训练损失对比图：PyTorch vs Jittor
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


def parse_autoencoder_log(log_path):
    """
    从自编码器训练日志文件中解析损失数据
    
    Returns:
        dict: 包含各损失项的列表和epoch平均值
    """
    losses = {
        'epoch': [],
        'batch': [],
        'recon_loss': [],
        'ssim_loss': [],
        'total_loss': [],
        'epoch_avg_recon': {},
        'epoch_avg_ssim': {},
        'epoch_avg_total': {}
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_epoch = 0
    epoch_recon_losses = []
    epoch_ssim_losses = []
    epoch_total_losses = []
    
    for line in lines:
        # 解析epoch和batch信息
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
        batch_match = re.search(r'Batch (\d+)/(\d+)', line)
        
        # 解析损失值
        recon_match = re.search(r'recon loss:\s*([\d.]+)', line)
        ssim_match = re.search(r'ssim loss:\s*([\d.]+)', line)
        total_match = re.search(r'total loss:\s*([\d.]+)', line)
        
        # 如果这一行包含epoch信息
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            if epoch_num != current_epoch:
                # 新epoch开始，保存上一个epoch的平均值
                if current_epoch > 0 and epoch_recon_losses:
                    losses['epoch_avg_recon'][current_epoch] = np.mean(epoch_recon_losses)
                    losses['epoch_avg_ssim'][current_epoch] = np.mean(epoch_ssim_losses)
                    losses['epoch_avg_total'][current_epoch] = np.mean(epoch_total_losses)
                current_epoch = epoch_num
                epoch_recon_losses = []
                epoch_ssim_losses = []
                epoch_total_losses = []
        
        # 如果这一行包含所有损失值
        if recon_match and ssim_match and total_match:
            epoch_num = int(epoch_match.group(1)) if epoch_match else current_epoch
            batch_num = int(batch_match.group(1)) if batch_match else 0
            
            recon_loss = float(recon_match.group(1))
            ssim_loss = float(ssim_match.group(1))
            total_loss = float(total_match.group(1))
            
            if epoch_num > 0:
                losses['epoch'].append(epoch_num)
                losses['batch'].append(batch_num)
                losses['recon_loss'].append(recon_loss)
                losses['ssim_loss'].append(ssim_loss)
                losses['total_loss'].append(total_loss)
                
                epoch_recon_losses.append(recon_loss)
                epoch_ssim_losses.append(ssim_loss)
                epoch_total_losses.append(total_loss)
    
    # 保存最后一个epoch的平均值
    if current_epoch > 0 and epoch_recon_losses:
        losses['epoch_avg_recon'][current_epoch] = np.mean(epoch_recon_losses)
        losses['epoch_avg_ssim'][current_epoch] = np.mean(epoch_ssim_losses)
        losses['epoch_avg_total'][current_epoch] = np.mean(epoch_total_losses)
    
    return losses


def plot_comparison(pytorch_ir_losses, pytorch_vi_losses, 
                    jittor_ir_losses, jittor_vi_losses, output_dir):
    """
    绘制PyTorch vs Jittor的对比图
    
    Args:
        pytorch_ir_losses: PyTorch IR自编码器损失数据
        pytorch_vi_losses: PyTorch VI自编码器损失数据
        jittor_ir_losses: Jittor IR自编码器损失数据
        jittor_vi_losses: Jittor VI自编码器损失数据
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 颜色和样式定义
    colors = {
        'pytorch': {'ir': '#1f77b4', 'vi': '#ff7f0e'},  # 蓝色和橙色
        'jittor': {'ir': '#2ca02c', 'vi': '#d62728'}   # 绿色和红色
    }
    linestyles = {
        'pytorch': '-',
        'jittor': '--'
    }
    markers = {
        'pytorch': 'o',
        'jittor': 's'
    }
    
    # ============================================================
    # 图1：IR自编码器对比（三个损失）
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('IR Autoencoder Training Loss Comparison: PyTorch vs Jittor', 
                 fontsize=16, fontweight='bold')
    
    # 获取epoch列表
    pytorch_ir_epochs = sorted([e for e in pytorch_ir_losses['epoch_avg_total'].keys() if e > 0])
    jittor_ir_epochs = sorted([e for e in jittor_ir_losses['epoch_avg_total'].keys() if e > 0])
    
    # 总损失
    if pytorch_ir_epochs:
        pytorch_total = [pytorch_ir_losses['epoch_avg_total'][e] for e in pytorch_ir_epochs]
        axes[0].plot(pytorch_ir_epochs, pytorch_total, 
                    label='PyTorch', color=colors['pytorch']['ir'], 
                    linestyle=linestyles['pytorch'], linewidth=2, 
                    marker=markers['pytorch'], markersize=6, alpha=0.8)
    
    if jittor_ir_epochs:
        jittor_total = [jittor_ir_losses['epoch_avg_total'][e] for e in jittor_ir_epochs]
        axes[0].plot(jittor_ir_epochs, jittor_total, 
                    label='Jittor', color=colors['jittor']['ir'], 
                    linestyle=linestyles['jittor'], linewidth=2, 
                    marker=markers['jittor'], markersize=6, alpha=0.8)
    
    axes[0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # 重建损失
    if pytorch_ir_epochs:
        pytorch_recon = [pytorch_ir_losses['epoch_avg_recon'][e] for e in pytorch_ir_epochs]
        axes[1].plot(pytorch_ir_epochs, pytorch_recon, 
                    label='PyTorch', color=colors['pytorch']['ir'], 
                    linestyle=linestyles['pytorch'], linewidth=2, 
                    marker=markers['pytorch'], markersize=6, alpha=0.8)
    
    if jittor_ir_epochs:
        jittor_recon = [jittor_ir_losses['epoch_avg_recon'][e] for e in jittor_ir_epochs]
        axes[1].plot(jittor_ir_epochs, jittor_recon, 
                    label='Jittor', color=colors['jittor']['ir'], 
                    linestyle=linestyles['jittor'], linewidth=2, 
                    marker=markers['jittor'], markersize=6, alpha=0.8)
    
    axes[1].set_title('Reconstruction Loss (L_recon)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    # SSIM损失
    if pytorch_ir_epochs:
        pytorch_ssim = [pytorch_ir_losses['epoch_avg_ssim'][e] for e in pytorch_ir_epochs]
        axes[2].plot(pytorch_ir_epochs, pytorch_ssim, 
                    label='PyTorch', color=colors['pytorch']['ir'], 
                    linestyle=linestyles['pytorch'], linewidth=2, 
                    marker=markers['pytorch'], markersize=6, alpha=0.8)
    
    if jittor_ir_epochs:
        jittor_ssim = [jittor_ir_losses['epoch_avg_ssim'][e] for e in jittor_ir_epochs]
        axes[2].plot(jittor_ir_epochs, jittor_ssim, 
                    label='Jittor', color=colors['jittor']['ir'], 
                    linestyle=linestyles['jittor'], linewidth=2, 
                    marker=markers['jittor'], markersize=6, alpha=0.8)
    
    axes[2].set_title('SSIM Loss (L_ssim)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Loss', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ir_autoencoder_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"IR自编码器对比图已保存至: {output_path}")
    plt.close()
    
    # ============================================================
    # 图2：VI自编码器对比（三个损失）
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('VI Autoencoder Training Loss Comparison: PyTorch vs Jittor', 
                 fontsize=16, fontweight='bold')
    
    # 获取epoch列表
    pytorch_vi_epochs = sorted([e for e in pytorch_vi_losses['epoch_avg_total'].keys() if e > 0])
    jittor_vi_epochs = sorted([e for e in jittor_vi_losses['epoch_avg_total'].keys() if e > 0])
    
    # 总损失
    if pytorch_vi_epochs:
        pytorch_total = [pytorch_vi_losses['epoch_avg_total'][e] for e in pytorch_vi_epochs]
        axes[0].plot(pytorch_vi_epochs, pytorch_total, 
                    label='PyTorch', color=colors['pytorch']['vi'], 
                    linestyle=linestyles['pytorch'], linewidth=2, 
                    marker=markers['pytorch'], markersize=6, alpha=0.8)
    
    if jittor_vi_epochs:
        jittor_total = [jittor_vi_losses['epoch_avg_total'][e] for e in jittor_vi_epochs]
        axes[0].plot(jittor_vi_epochs, jittor_total, 
                    label='Jittor', color=colors['jittor']['vi'], 
                    linestyle=linestyles['jittor'], linewidth=2, 
                    marker=markers['jittor'], markersize=6, alpha=0.8)
    
    axes[0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # 重建损失
    if pytorch_vi_epochs:
        pytorch_recon = [pytorch_vi_losses['epoch_avg_recon'][e] for e in pytorch_vi_epochs]
        axes[1].plot(pytorch_vi_epochs, pytorch_recon, 
                    label='PyTorch', color=colors['pytorch']['vi'], 
                    linestyle=linestyles['pytorch'], linewidth=2, 
                    marker=markers['pytorch'], markersize=6, alpha=0.8)
    
    if jittor_vi_epochs:
        jittor_recon = [jittor_vi_losses['epoch_avg_recon'][e] for e in jittor_vi_epochs]
        axes[1].plot(jittor_vi_epochs, jittor_recon, 
                    label='Jittor', color=colors['jittor']['vi'], 
                    linestyle=linestyles['jittor'], linewidth=2, 
                    marker=markers['jittor'], markersize=6, alpha=0.8)
    
    axes[1].set_title('Reconstruction Loss (L_recon)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    # SSIM损失
    if pytorch_vi_epochs:
        pytorch_ssim = [pytorch_vi_losses['epoch_avg_ssim'][e] for e in pytorch_vi_epochs]
        axes[2].plot(pytorch_vi_epochs, pytorch_ssim, 
                    label='PyTorch', color=colors['pytorch']['vi'], 
                    linestyle=linestyles['pytorch'], linewidth=2, 
                    marker=markers['pytorch'], markersize=6, alpha=0.8)
    
    if jittor_vi_epochs:
        jittor_ssim = [jittor_vi_losses['epoch_avg_ssim'][e] for e in jittor_vi_epochs]
        axes[2].plot(jittor_vi_epochs, jittor_ssim, 
                    label='Jittor', color=colors['jittor']['vi'], 
                    linestyle=linestyles['jittor'], linewidth=2, 
                    marker=markers['jittor'], markersize=6, alpha=0.8)
    
    axes[2].set_title('SSIM Loss (L_ssim)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Loss', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'vi_autoencoder_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"VI自编码器对比图已保存至: {output_path}")
    plt.close()
    
    # ============================================================
    # 图3：总损失对比（IR和VI一起）
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Autoencoder Total Loss Comparison: PyTorch vs Jittor', 
                 fontsize=16, fontweight='bold')
    
    # IR自编码器总损失
    if pytorch_ir_epochs:
        pytorch_ir_total = [pytorch_ir_losses['epoch_avg_total'][e] for e in pytorch_ir_epochs]
        axes[0].plot(pytorch_ir_epochs, pytorch_ir_total, 
                    label='PyTorch', color=colors['pytorch']['ir'], 
                    linestyle=linestyles['pytorch'], linewidth=2, 
                    marker=markers['pytorch'], markersize=6, alpha=0.8)
    
    if jittor_ir_epochs:
        jittor_ir_total = [jittor_ir_losses['epoch_avg_total'][e] for e in jittor_ir_epochs]
        axes[0].plot(jittor_ir_epochs, jittor_ir_total, 
                    label='Jittor', color=colors['jittor']['ir'], 
                    linestyle=linestyles['jittor'], linewidth=2, 
                    marker=markers['jittor'], markersize=6, alpha=0.8)
    
    axes[0].set_title('IR Autoencoder Total Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Total Loss', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # VI自编码器总损失
    if pytorch_vi_epochs:
        pytorch_vi_total = [pytorch_vi_losses['epoch_avg_total'][e] for e in pytorch_vi_epochs]
        axes[1].plot(pytorch_vi_epochs, pytorch_vi_total, 
                    label='PyTorch', color=colors['pytorch']['vi'], 
                    linestyle=linestyles['pytorch'], linewidth=2, 
                    marker=markers['pytorch'], markersize=6, alpha=0.8)
    
    if jittor_vi_epochs:
        jittor_vi_total = [jittor_vi_losses['epoch_avg_total'][e] for e in jittor_vi_epochs]
        axes[1].plot(jittor_vi_epochs, jittor_vi_total, 
                    label='Jittor', color=colors['jittor']['vi'], 
                    linestyle=linestyles['jittor'], linewidth=2, 
                    marker=markers['jittor'], markersize=6, alpha=0.8)
    
    axes[1].set_title('VI Autoencoder Total Loss', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Total Loss', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'autoencoder_total_loss_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"总损失对比图已保存至: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("绘制自编码器训练损失对比图：PyTorch vs Jittor")
    print("=" * 80)
    
    # 日志文件路径
    pytorch_ir_log = "./logs/my_trained_models/autoencoder/train_autoencoder_ir_20251219_194808.log"
    pytorch_vi_log = "./logs/my_trained_models/autoencoder/train_autoencoder_vi_20251219_204107.log"
    jittor_ir_log = "./logs/my_trained_models/autoencoder/jittor/train_autoencoder_ir_20260219_164733.log"
    jittor_vi_log = "./logs/my_trained_models/autoencoder/jittor/train_autoencoder_vi_20260219_181054.log"
    
    # 检查文件是否存在
    log_files = {
        'PyTorch IR': pytorch_ir_log,
        'PyTorch VI': pytorch_vi_log,
        'Jittor IR': jittor_ir_log,
        'Jittor VI': jittor_vi_log
    }
    
    for name, path in log_files.items():
        if not os.path.exists(path):
            print(f"错误：{name} 日志文件不存在: {path}")
            return
    
    print(f"\n正在解析日志文件...")
    for name, path in log_files.items():
        print(f"  {name}: {path}")
    
    # 解析日志
    print("\n解析中...")
    pytorch_ir_losses = parse_autoencoder_log(pytorch_ir_log)
    pytorch_vi_losses = parse_autoencoder_log(pytorch_vi_log)
    jittor_ir_losses = parse_autoencoder_log(jittor_ir_log)
    jittor_vi_losses = parse_autoencoder_log(jittor_vi_log)
    
    # 检查数据
    pytorch_ir_epochs = sorted([e for e in pytorch_ir_losses['epoch_avg_total'].keys() if e > 0])
    pytorch_vi_epochs = sorted([e for e in pytorch_vi_losses['epoch_avg_total'].keys() if e > 0])
    jittor_ir_epochs = sorted([e for e in jittor_ir_losses['epoch_avg_total'].keys() if e > 0])
    jittor_vi_epochs = sorted([e for e in jittor_vi_losses['epoch_avg_total'].keys() if e > 0])
    
    print(f"\n解析完成：")
    print(f"  PyTorch IR: {len(pytorch_ir_epochs)} epochs")
    print(f"  PyTorch VI: {len(pytorch_vi_epochs)} epochs")
    print(f"  Jittor IR: {len(jittor_ir_epochs)} epochs")
    print(f"  Jittor VI: {len(jittor_vi_epochs)} epochs")
    
    # 绘制对比图
    print(f"\n正在绘制对比图...")
    output_dir = './output/loss_curves'
    plot_comparison(pytorch_ir_losses, pytorch_vi_losses, 
                    jittor_ir_losses, jittor_vi_losses, output_dir)
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"所有图表已保存至: {output_dir}/")
    print("  - ir_autoencoder_comparison.png (IR自编码器对比)")
    print("  - vi_autoencoder_comparison.png (VI自编码器对比)")
    print("  - autoencoder_total_loss_comparison.png (总损失对比)")
    
    # 打印关键指标对比
    print("\n" + "=" * 80)
    print("关键指标对比（Epoch平均值）")
    print("=" * 80)
    
    # IR自编码器对比
    print("\n【IR自编码器】")
    print(f"{'Epoch':<8} {'PyTorch Total':<18} {'Jittor Total':<18} {'差异率':<15}")
    print("-" * 80)
    max_epochs = max(len(pytorch_ir_epochs), len(jittor_ir_epochs))
    for i in range(max_epochs):
        epoch = i + 1
        pytorch_total = pytorch_ir_losses['epoch_avg_total'].get(epoch, None)
        jittor_total = jittor_ir_losses['epoch_avg_total'].get(epoch, None)
        
        if pytorch_total and jittor_total:
            diff_pct = ((jittor_total - pytorch_total) / pytorch_total) * 100
            print(f"{epoch:<8} {pytorch_total:<18.2f} {jittor_total:<18.2f} {diff_pct:+.1f}%")
        elif pytorch_total:
            print(f"{epoch:<8} {pytorch_total:<18.2f} {'N/A':<18}")
        elif jittor_total:
            print(f"{epoch:<8} {'N/A':<18} {jittor_total:<18.2f}")
    
    # VI自编码器对比
    print("\n【VI自编码器】")
    print(f"{'Epoch':<8} {'PyTorch Total':<18} {'Jittor Total':<18} {'差异率':<15}")
    print("-" * 80)
    max_epochs = max(len(pytorch_vi_epochs), len(jittor_vi_epochs))
    for i in range(max_epochs):
        epoch = i + 1
        pytorch_total = pytorch_vi_losses['epoch_avg_total'].get(epoch, None)
        jittor_total = jittor_vi_losses['epoch_avg_total'].get(epoch, None)
        
        if pytorch_total and jittor_total:
            diff_pct = ((jittor_total - pytorch_total) / pytorch_total) * 100
            print(f"{epoch:<8} {pytorch_total:<18.2f} {jittor_total:<18.2f} {diff_pct:+.1f}%")
        elif pytorch_total:
            print(f"{epoch:<8} {pytorch_total:<18.2f} {'N/A':<18}")
        elif jittor_total:
            print(f"{epoch:<8} {'N/A':<18} {jittor_total:<18.2f}")


if __name__ == '__main__':
    main()
