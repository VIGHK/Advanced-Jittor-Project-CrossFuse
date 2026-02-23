# -*- coding:utf-8 -*-
"""
损失曲线绘制工具
支持从训练日志文件(.log)中提取损失数据并绘制曲线图
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def parse_log_file(log_path):
    """
    从日志文件中解析损失数据
    
    Returns:
        dict: 包含各损失项的列表
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
    
    # 缓存上一行的信息
    prev_line_epoch = 0
    prev_line_batch = 0
    prev_pix_loss = None
    prev_gra_loss = None
    
    for i, line in enumerate(lines):
        # 解析epoch和batch信息
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
        batch_match = re.search(r'Batch (\d+)/(\d+)', line)
        
        # 解析损失值（pix loss 和 gra loss 在第一行）
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
        
        # 如果这一行包含 pix loss 和 gra loss（第一行）
        if pix_match and gra_match:
            # 确定当前epoch和batch
            epoch_num = int(epoch_match.group(1)) if epoch_match else current_epoch
            batch_num = int(batch_match.group(1)) if batch_match else 0
            
            pix_loss = float(pix_match.group(1))
            gra_loss = float(gra_match.group(1))
            
            # 尝试在当前行找 total loss，如果找不到，检查下一行
            total_loss = None
            if total_match:
                total_loss = float(total_match.group(1))
            elif i + 1 < len(lines):
                # 检查下一行是否有 total loss
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


def plot_loss_curves(loss_dicts, labels, output_path, plot_type='epoch_avg'):
    """
    绘制损失曲线
    
    Args:
        loss_dicts: 损失数据字典列表
        labels: 实验标签列表
        output_path: 输出图片路径
        plot_type: 'epoch_avg' 绘制每个epoch的平均损失, 'batch' 绘制所有batch的损失
    """
    title_suffix = " (Epoch Average)" if plot_type == 'epoch_avg' else " (Batch Level)"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Loss Curves{title_suffix}', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = ['-', '--', '-.', ':']
    
    for idx, (losses, label) in enumerate(zip(loss_dicts, labels)):
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        
        if plot_type == 'epoch_avg':
            # 绘制每个epoch的平均损失
            epochs = sorted([e for e in losses['epoch_avg_total'].keys() if e > 0])
            if not epochs:
                print(f"警告：{label} 没有解析到有效的epoch数据")
                continue
            total_vals = [losses['epoch_avg_total'][e] for e in epochs]
            pix_vals = [losses['epoch_avg_pix'][e] for e in epochs]
            gra_vals = [losses['epoch_avg_gra'][e] for e in epochs]
            x_data = epochs
        else:
            # 绘制所有batch的损失（可选：每N个batch采样一次以减少点数）
            step = max(1, len(losses['total_loss']) // 2000)  # 最多显示2000个点
            x_data = np.arange(0, len(losses['total_loss']), step)
            total_vals = losses['total_loss'][::step]
            pix_vals = losses['pix_loss'][::step]
            gra_vals = losses['gra_loss'][::step]
        
        # 总损失
        axes[0, 0].plot(x_data, total_vals, label=label, color=color, 
                       linestyle=linestyle, linewidth=1.5, alpha=0.8)
        
        # 像素损失
        axes[0, 1].plot(x_data, pix_vals, label=label, color=color, 
                       linestyle=linestyle, linewidth=1.5, alpha=0.8)
        
        # 梯度损失
        axes[1, 0].plot(x_data, gra_vals, label=label, color=color, 
                       linestyle=linestyle, linewidth=1.5, alpha=0.8)
        
        # 组合图（总损失 + 像素损失）
        axes[1, 1].plot(x_data, total_vals, label=f'{label} (Total)', 
                       color=color, linestyle=linestyle, linewidth=1.5, alpha=0.8)
        axes[1, 1].plot(x_data, pix_vals, label=f'{label} (Pixel)', 
                       color=color, linestyle=linestyle, linewidth=1.2, alpha=0.6)
    
    # 设置标签和标题
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch' if plot_type == 'epoch_avg' else 'Batch', fontsize=10)
    axes[0, 0].set_ylabel('Loss', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    
    axes[0, 1].set_title('Pixel Loss (L_pix)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch' if plot_type == 'epoch_avg' else 'Batch', fontsize=10)
    axes[0, 1].set_ylabel('Loss', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=9)
    
    axes[1, 0].set_title('Gradient Loss (L_gra)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch' if plot_type == 'epoch_avg' else 'Batch', fontsize=10)
    axes[1, 0].set_ylabel('Loss', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=9)
    
    axes[1, 1].set_title('Total Loss vs Pixel Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch' if plot_type == 'epoch_avg' else 'Batch', fontsize=10)
    axes[1, 1].set_ylabel('Loss', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"损失曲线已保存至: {output_path}")
    plt.close()


def plot_comparison(loss_dicts, labels, output_path):
    """
    绘制对比图：只显示总损失，用于对比不同实验（按epoch均值）
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = ['-', '--', '-.', ':']
    
    for idx, (losses, label) in enumerate(zip(loss_dicts, labels)):
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        
        epochs = sorted([e for e in losses['epoch_avg_total'].keys() if e > 0])
        if not epochs:
            print(f"警告：{label} 没有解析到有效的epoch数据")
            continue
        total_vals = [losses['epoch_avg_total'][e] for e in epochs]
        
        ax.plot(epochs, total_vals, label=label, color=color, 
               linestyle=linestyle, linewidth=2, marker='o', markersize=5, alpha=0.8)
    
    ax.set_title('Total Loss Comparison (Epoch Average)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss (Epoch Average)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    
    # 添加数值标注（在关键点显示数值）
    for idx, (losses, label) in enumerate(zip(loss_dicts, labels)):
        epochs = sorted([e for e in losses['epoch_avg_total'].keys() if e > 0])
        if epochs:
            total_vals = [losses['epoch_avg_total'][e] for e in epochs]
            # 在第一个和最后一个epoch标注数值
            if len(epochs) > 0:
                ax.annotate(f'{total_vals[0]:.1f}', 
                           xy=(epochs[0], total_vals[0]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.7)
            if len(epochs) > 1:
                ax.annotate(f'{total_vals[-1]:.1f}', 
                           xy=(epochs[-1], total_vals[-1]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {output_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    import os
    
    # ============================================================
    # 配置：可以传入日志文件路径作为参数，或使用默认路径
    # ============================================================
    
    # 默认日志文件（Jittor训练日志）
    default_log = "./logs/my_trained_models/transfuse/jittor/train_transfuse_20260221_031544.log"
    
    # 如果命令行提供了日志文件路径，使用它；否则使用默认路径
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = default_log
    
    # 输出目录
    output_dir = "./output/loss_curves"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # 解析日志文件
    # ============================================================
    print("=" * 80)
    print("训练损失曲线绘制工具")
    print("=" * 80)
    print(f"\n正在解析日志文件: {log_file}")
    
    if not Path(log_file).exists():
        print(f"错误：日志文件不存在: {log_file}")
        exit(1)
    
    losses = parse_log_file(log_file)
    
    # 检查解析结果
    epochs = sorted([e for e in losses['epoch_avg_total'].keys() if e > 0])
    if not epochs:
        print("错误：未能解析到有效的训练数据！")
        exit(1)
    
    print(f"解析完成：")
    print(f"  - 总epoch数: {len(epochs)}")
    print(f"  - 总batch数: {len(losses['total_loss'])}")
    print(f"  - Epoch范围: {min(epochs)} - {max(epochs)}")
    
    # ============================================================
    # 打印每个epoch的详细统计信息（按epoch均值，消掉噪声）
    # ============================================================
    print("\n" + "=" * 80)
    print("每个Epoch的平均损失统计（按epoch均值，已消噪）")
    print("=" * 80)
    print(f"{'Epoch':<8} {'Total Loss':<15} {'Pixel Loss':<15} {'Gradient Loss':<15} {'Batch数':<10}")
    print("-" * 80)
    for e in epochs:
        total_avg = losses['epoch_avg_total'][e]
        pix_avg = losses['epoch_avg_pix'][e]
        gra_avg = losses['epoch_avg_gra'][e]
        # 计算该epoch的batch数
        epoch_batches = sum(1 for ep in losses['epoch'] if ep == e)
        print(f"{e:<8} {total_avg:<15.2f} {pix_avg:<15.2f} {gra_avg:<15.2f} {epoch_batches:<10}")
    
    # ============================================================
    # 绘制损失曲线（主要使用epoch均值，消掉噪声）
    # ============================================================
    print("\n正在绘制损失曲线（按epoch均值，已消噪）...")
    
    # 生成输出文件名（基于日志文件名）
    log_name = Path(log_file).stem
    output_prefix = f"{output_dir}/{log_name}"
    
    # 1. 绘制详细的损失曲线（每个epoch的平均值）- 主要图表
    plot_loss_curves([losses], ["Jittor Training (Epoch Avg)"], 
                    f"{output_prefix}_epoch_avg.png", 
                    plot_type='epoch_avg')
    
    # 2. 绘制总损失对比图（按epoch均值）- 用于量级对比
    plot_comparison([losses], ["Jittor Training (Epoch Avg)"], 
                   f"{output_prefix}_total_loss_epoch_avg.png")
    
    # 3. 可选：绘制batch级别的损失曲线（采样显示，避免点太多）- 仅作参考
    plot_loss_curves([losses], ["Jittor Training (Batch)"], 
                    f"{output_prefix}_batch.png", 
                    plot_type='batch')
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"所有图表已保存至: {output_dir}/")
    print(f"  - {log_name}_epoch_avg.png           (每个epoch的平均损失 - 主要图表)")
    print(f"  - {log_name}_total_loss_epoch_avg.png (总损失曲线 - 按epoch均值，用于量级对比)")
    print(f"  - {log_name}_batch.png               (所有batch的损失 - 仅作参考)")
    
    # 打印关键统计信息
    print("\n关键统计信息（按epoch均值）：")
    print(f"  第一轮平均loss: {losses['epoch_avg_total'][1]:.2f}")
    if len(epochs) > 1:
        print(f"  最后一轮平均loss: {losses['epoch_avg_total'][epochs[-1]]:.2f}")
        print(f"  损失下降: {losses['epoch_avg_total'][1] - losses['epoch_avg_total'][epochs[-1]]:.2f}")
        print(f"  损失下降比例: {((losses['epoch_avg_total'][1] - losses['epoch_avg_total'][epochs[-1]]) / losses['epoch_avg_total'][1] * 100):.1f}%")

