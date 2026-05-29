import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def visualize_save_and_clear_cache(cache_dict, target_layer_name, save_dir="activation_plots"):
#     if target_layer_name not in cache_dict:
#         return

#     print(f">>> 正在使用曲面图 (plot_surface) 高效渲染层: {target_layer_name}...")
    
#     # 1. 获取并格式化激活值数据
#     activation = cache_dict[target_layer_name].float().numpy()
#     if activation.ndim == 3:
#         activation = activation.squeeze(0) # [SeqLen, HiddenDim]

#     num_tokens, num_channels = activation.shape

#     # 💡 保护机制：如果通道数依然极大（比如 4096x2048 = 800万点），
#     # plot_surface 虽然不卡，但生成的图片文件会非常大。
#     # 我们可以通过步长切片 (stride) 稍微稀疏化一下，不仅画得飞快，视觉起伏感甚至更好。
#     stride_x = 1 if num_channels <= 1024 else 4
#     stride_y = 1 if num_tokens <= 512 else 4
    
#     activation_sampled = activation[::stride_y, ::stride_x]
#     y_sampled, x_sampled = activation_sampled.shape

#     # 2. 准备 3D 网格数据
#     x = np.arange(0, num_channels, stride_x)
#     y = np.arange(0, num_tokens, stride_y)
#     X, Y = np.meshgrid(x, y)

#     # 3. 创建画布和 3D 坐标系
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # 4. 🚀 核心替换：使用你的 plot_surface 方法
#     # cmap='coolwarm' 可以完美还原你之前想要的那种 "蓝->白->红" 的异常值预警效果
#     surf = ax.plot_surface(X, Y, activation_sampled, cmap='coolwarm', edgecolor='none', alpha=0.9)

#     # 5. 设置坐标轴和视觉效果
#     ax.set_title(f"3D Activation Surface: {target_layer_name}", fontsize=14)
#     ax.set_xlabel('Channel', fontsize=12)
#     ax.set_ylabel('Token', fontsize=12)
#     ax.set_zlabel('Activation Magnitude', fontsize=12)
    
#     # 强制固定 Z 轴高度 (0到16)，把那些远超正常值的异常点逼到图表顶端
#     ax.set_zlim(-150, 150)
    
#     # 添加 Colorbar
#     fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Magnitude')

#     # 6. 保存并清理
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
        
#     safe_name = target_layer_name.replace(".", "_")
#     save_path = os.path.join(save_dir, f"{safe_name}_surface.png")
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close(fig)
#     del cache_dict[target_layer_name]
    
#     print(f">>> [成功] 3D 曲面图已极速保存至: {save_path}")

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_save_and_clear_cache(cache_dict, target_layer_name, save_dir="activation_plots_Rot"):
    if target_layer_name not in cache_dict:
        return

    print(f">>> 正在同时渲染 {target_layer_name} 的 3D曲面图 和 小提琴统计图...")
    
    # 1. 获取并格式化激活值数据
    activation = cache_dict[target_layer_name].float().numpy()
    if activation.ndim == 3:
        activation = activation.squeeze(0) # [SeqLen, HiddenDim]

    num_tokens, num_channels = activation.shape
    safe_name = target_layer_name.replace(".", "_")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ==========================================================
    # 🎨 第一部分：绘制并保存原始的 3D 曲面图
    # ==========================================================
    stride_x = 1 if num_channels <= 1024 else 4
    stride_y = 1 if num_tokens <= 512 else 4
    activation_sampled = activation[::stride_y, ::stride_x]
    
    x = np.arange(0, num_channels, stride_x)
    y = np.arange(0, num_tokens, stride_y)
    X, Y = np.meshgrid(x, y)

    fig_3d = plt.figure(figsize=(12, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    surf = ax_3d.plot_surface(X, Y, activation_sampled, cmap='coolwarm', edgecolor='none', alpha=0.9)

    ax_3d.set_title(f"3D Activation Surface: {target_layer_name}", fontsize=14)
    ax_3d.set_xlabel('Channel', fontsize=12)
    ax_3d.set_ylabel('Token', fontsize=12)
    ax_3d.set_zlabel('Activation Magnitude', fontsize=12)
    ax_3d.set_zlim(-150, 150)
    
    fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=10, label='Magnitude')

    save_path_3d = os.path.join(save_dir, f"{safe_name}_surface.png")
    plt.tight_layout()
    plt.savefig(save_path_3d, dpi=300)
    plt.close(fig_3d)

    # ==========================================================
    # 📊 第二部分：绘制并保存 小提琴图 + 统计表格
    # ==========================================================
    activation_flat = activation.flatten()
    
    stats = {
        "max": np.max(activation_flat),
        "p95": np.percentile(activation_flat, 95),
        "p75": np.percentile(activation_flat, 75),
        "p50": np.percentile(activation_flat, 50),
        "p25": np.percentile(activation_flat, 25),
        "p5":  np.percentile(activation_flat, 5),
        "min": np.min(activation_flat)
    }

    fig_v, (ax_violin, ax_table) = plt.subplots(
        1, 2, 
        figsize=(12, 7), 
        gridspec_kw={'width_ratios': [3, 1]}
    )

    parts = ax_violin.violinplot(activation_flat, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    # 绘制箱线图内核
    ax_violin.vlines(1, stats['p25'], stats['p75'], color='black', linestyle='-', lw=5)
    ax_violin.vlines(1, stats['p5'], stats['p95'], color='black', linestyle='-', lw=1.5)
    ax_violin.scatter(1, stats['p50'], marker='o', color='white', s=30, zorder=3)

    ax_violin.set_title(f"Activation Distribution: {target_layer_name}", fontsize=14)
    ax_violin.set_ylabel("Activation Magnitude", fontsize=12)
    ax_violin.set_xticks([])
    ax_violin.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 绘制右侧表格
    ax_table.axis('off')
    cell_text = [[f"{v:.4f}"] for v in stats.values()]
    row_labels = list(stats.keys())
    
    table = ax_table.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=["Value"],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)

    save_path_v = os.path.join(save_dir, f"{safe_name}_violin.png")
    plt.tight_layout()
    plt.savefig(save_path_v, dpi=300)
    plt.close(fig_v)

    # ==========================================================
    # 🧹 第三部分：清理缓存，防止爆内存
    # ==========================================================
    del cache_dict[target_layer_name]
    
    print(f">>> [成功] 已生成两组可视化文件：\n    1) {save_path_3d}\n    2) {save_path_v}")