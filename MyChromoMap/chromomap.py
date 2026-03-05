import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
from io import BytesIO
import re
import math
import numpy as np

# --- 页面配置 ---
st.set_page_config(page_title="染色体图谱 v12.8 (精度修正版)", layout="wide")

# --- 基础样式 ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    html, body, [class*="css"] { font-family: 'Times New Roman', serif; }
    .paper-text { background-color: #f8f9fa; border-left: 4px solid #2c3e50; padding: 15px; font-family: 'Times New Roman', serif; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 注册自定义色系 (含黑白印刷灰阶)
# ==========================================
mako_cmap = mcolors.LinearSegmentedColormap.from_list("Mako", ['#e3f7e3', '#BAE6E1', '#91D5DE'])
gray_print_cmap = mcolors.LinearSegmentedColormap.from_list("Grayscale_Print", ['#F8F8F8', '#969696', '#000000'])

for name, cmap in [("Mako", mako_cmap), ("Grayscale_Print", gray_print_cmap)]:
    try:
        cm.get_cmap(name)
    except ValueError:
        if hasattr(plt.colormaps, 'register'):
            plt.colormaps.register(cmap=cmap, name=name)
        else:
            cm.register_cmap(name=name, cmap=cmap)

# ==========================================
# 2. 核心计算函数
# ==========================================
def calculate_windowed_density(gff_file_obj, len_dict_bp, window_size_bp):
    gff_cols = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
    try:
        df_gff = pd.read_csv(gff_file_obj, sep='\t', comment='#', header=None, names=gff_cols, dtype={'seqid': str}, on_bad_lines='skip')
        df_genes = df_gff[df_gff['type'] == 'gene'].copy()
        df_genes['midpoint'] = (df_genes['start'] + df_genes['end']) / 2
        density_profile_map = {}
        global_max_density = 0.0
        
        for chr_name, chr_len in len_dict_bp.items():
            chr_sub_df = df_genes[df_genes['seqid'] == chr_name]
            bin_edges = np.arange(0, chr_len + window_size_bp, window_size_bp)
            if bin_edges[-1] < chr_len:
                bin_edges = np.append(bin_edges, chr_len)
            
            binned_data = pd.cut(chr_sub_df['midpoint'], bins=bin_edges, include_lowest=True)
            counts_series = binned_data.value_counts(sort=False)
            window_size_mb = window_size_bp / 1_000_000
            density_values = counts_series.values / window_size_mb
            
            density_profile_map[chr_name] = density_values
            if len(density_values) > 0:
                global_max_density = max(global_max_density, density_values.max())
                
        return density_profile_map, global_max_density
    except Exception as e:
        st.sidebar.error(f"GFF 解析失败: {e}")
        return {}, 0.0

def avoid_collisions(positions, min_spacing):
    n = len(positions)
    if n <= 1: return positions
    new_pos = np.array(positions, dtype=float)
    order = np.argsort(new_pos)
    new_pos_sorted = new_pos[order]
    for _ in range(100):
        diffs = np.diff(new_pos_sorted)
        overlaps = np.where(diffs < min_spacing)[0]
        if len(overlaps) == 0: break
        for i in overlaps:
            push = (min_spacing - diffs[i]) / 2 + 1e-5
            new_pos_sorted[i] -= push
            new_pos_sorted[i+1] += push
    final_pos = np.zeros_like(new_pos)
    final_pos[order] = new_pos_sorted
    return final_pos.tolist()

# ==========================================
# 3. 侧边栏设置 (包含你要求的两个新增功能)
# ==========================================
st.sidebar.header("🎨 布局与比例尺设置")
chrs_per_row = st.sidebar.number_input("每行染色体数量", 1, 50, 10)
fig_width = st.sidebar.slider("画布宽度 (inch)", 5.0, 30.0, 12.0)
row_height = st.sidebar.slider("行高度 (inch)", 2.0, 15.0, 7.0)

st.sidebar.subheader("📏 比例尺精修")
ruler_offset = st.sidebar.slider("↔️ 标尺-染色体间距", 0.1, 3.0, 0.8)
major_tick_interval = st.sidebar.number_input("主刻度间隔 (Mb)", 1, 100, 10)
tick_line_length = st.sidebar.slider("刻度线长度", 0.05, 0.5, 0.2)

st.sidebar.subheader("🔥 密度热力图")
uploaded_gff = st.sidebar.file_uploader("上传 GFF3 文件", type=["gff3"])
window_size_mb = st.sidebar.selectbox("窗口大小 (Mb)", [0.1, 0.5, 1.0, 2.0], index=2)
use_density_color = st.sidebar.checkbox("启用密度着色", value=False)
colormap_name = st.sidebar.selectbox("色系选择", ['Mako', 'Grayscale_Print', 'Reds', 'viridis'], index=1)

# ==========================================
# 4. 绘图函数 (核心修正位置)
# ==========================================
def plot_ideogram_v12_8(genes, len_dict, d_profile_map, d_norm, d_cmap_obj):
    sorted_chrs = sorted(len_dict.keys())
    total_chrs = len(sorted_chrs)
    num_rows = math.ceil(total_chrs / chrs_per_row)
    global_max_len_mb = max(len_dict.values()) / 1_000_000
    
    fig, axes = plt.subplots(num_rows, 1, figsize=(fig_width, row_height * num_rows))
    if num_rows == 1: axes = [axes]

    for r in range(num_rows):
        ax = axes[r]
        start_idx = r * chrs_per_row
        end_idx = min((r + 1) * chrs_per_row, total_chrs)
        current_row_chrs = sorted_chrs[start_idx:end_idx]
        
        ax.set_xlim(-ruler_offset - 0.5, chrs_per_row * 1.5)
        ax.set_ylim(global_max_len_mb * 1.05, -global_max_len_mb * 0.05)
        ax.axis('off')

        # --- 绘制带刻度线的标尺 ---
        ruler_x = -ruler_offset
        ax.plot([ruler_x, ruler_x], [0, global_max_len_mb], color='black', lw=1.2)
        for t in range(0, int(global_max_len_mb) + 1, major_tick_interval):
            ax.plot([ruler_x - tick_line_length, ruler_x], [t, t], color='black', lw=1)
            ax.text(ruler_x - tick_line_length - 0.1, t, str(t), ha='right', va='center', fontname='Times New Roman', fontsize=10)
        ax.text(ruler_x, -2, "Mb", ha='center', va='bottom', fontname='Times New Roman', fontweight='bold')

        for i, chr_name in enumerate(current_row_chrs):
            x_pos = i * 1.2 
            length_mb = len_dict[chr_name] / 1_000_000
            chr_w = 0.4

            # --- 密度条带 (高度修正逻辑) ---
            if use_density_color and d_norm and chr_name in d_profile_map:
                densities = d_profile_map[chr_name]
                for idx, val in enumerate(densities):
                    start_y = idx * window_size_mb
                    # 关键修正：确保最后一个方块不超出染色体实际长度
                    this_bin_height = min(window_size_mb, length_mb - start_y)
                    
                    if this_bin_height > 0:
                        rect = patches.Rectangle(
                            (x_pos - chr_w/2, start_y), chr_w, this_bin_height,
                            facecolor=d_cmap_obj(d_norm(val)), lw=0, zorder=0
                        )
                        ax.add_patch(rect)

            # 染色体轮廓 (圆角)
            box = patches.FancyBboxPatch(
                (x_pos - chr_w/2, 0), chr_w, length_mb,
                boxstyle=f"round,pad=0,rounding_size={chr_w/2}", 
                ec='black', fc='none', lw=1.2, zorder=1
            )
            ax.add_patch(box)
            ax.text(x_pos, -2, chr_name, ha='center', fontname='Times New Roman', fontweight='bold')

            # 基因标记 (逻辑保持)
            chr_genes = genes[genes['Chr'] == chr_name].copy()
            if not chr_genes.empty:
                chr_genes['y'] = chr_genes['Start'] / 1_000_000
                chr_genes['ly'] = avoid_collisions(chr_genes['y'].tolist(), label_space)
                for _, row in chr_genes.iterrows():
                    ax.plot([x_pos - chr_w/2, x_pos + chr_w/2], [row['y'], row['y']], color=row['Color'] if row['Color'] else '#FF0000', lw=2, zorder=2)
                    ax.plot([x_pos + chr_w/2, x_pos + chr_w/2 + 0.1], [row['y'], row['ly']], color='black', lw=0.5)
                    ax.text(x_pos + chr_w/2 + 0.15, row['ly'], row['Gene'], va='center', fontname='Times New Roman', style='italic', fontsize=font_sz)

    return fig

# ==========================================
# 5. 主运行区
# ==========================================
if st.button("🚀 生成物理图谱", type="primary"):
    d_map, d_max = {}, 0.0
    if use_density_color and uploaded_gff:
        d_map, d_max = calculate_windowed_density(uploaded_gff, len_dict, window_size_mb * 1e6)
    
    norm = mcolors.Normalize(vmin=0, vmax=d_max) if d_max > 0 else None
    cmap = cm.get_cmap(colormap_name)
    
    fig = plot_ideogram_v12_8(df_genes, len_dict, d_map, norm, cmap)
    st.pyplot(fig)
