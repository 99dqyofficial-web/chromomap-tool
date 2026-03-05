import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import os
from io import BytesIO
import re
import math
import numpy as np

# --- 页面配置 ---
st.set_page_config(page_title="染色体图谱 v12.7 (精准刻度版)", layout="wide")

# --- 样式设置 (保持学术风) ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    html, body, [class*="css"] { font-family: 'Times New Roman', serif; }
    h1, h2, h3, .stMarkdown, .stText, .stButton button { font-family: 'Times New Roman', serif !important; color: #000; }
    .paper-text { background-color: #f8f9fa; border-left: 4px solid #2c3e50; padding: 15px; font-family: 'Times New Roman', serif; white-space: pre-wrap; line-height: 1.6; color: #333; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 字体与全局配置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none' 

# ==========================================
# 注册色系 (Mako & 灰阶)
# ==========================================
mako_cmap = mcolors.LinearSegmentedColormap.from_list("Mako", ['#e3f7e3', '#BAE6E1', '#91D5DE'])
gray_print_cmap = mcolors.LinearSegmentedColormap.from_list("Grayscale_Print", ['#F0F0F0', '#969696', '#000000'])

for name, cmap in [("Mako", mako_cmap), ("Grayscale_Print", gray_print_cmap)]:
    try:
        cm.get_cmap(name)
    except ValueError:
        if hasattr(plt.colormaps, 'register'):
            plt.colormaps.register(cmap=cmap, name=name)
        else:
            cm.register_cmap(name=name, cmap=cmap)

# ==========================================
# 算法逻辑 (密度计算与标签避让)
# ==========================================
def calculate_windowed_density(gff_file_obj, len_dict_bp, window_size_bp):
    gff_cols = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
    try:
        df_gff = pd.read_csv(gff_file_obj, sep='\t', comment='#', header=None, names=gff_cols, dtype={'seqid': str}, on_bad_lines='skip')
        df_genes = df_gff[df_gff['type'] == 'gene'].copy()
        df_genes['midpoint'] = (df_genes['start'] + df_genes['end']) / 2
        density_map, max_d = {}, 0.0
        for chr_name, chr_len in len_dict_bp.items():
            chr_sub = df_genes[df_genes['seqid'] == chr_name]
            bin_edges = np.arange(0, chr_len + window_size_bp, window_size_bp)
            if bin_edges[-1] < chr_len: bin_edges = np.append(bin_edges, chr_len)
            binned = pd.cut(chr_sub['midpoint'], bins=bin_edges, include_lowest=True)
            counts = binned.value_counts(sort=False).values / (window_size_bp / 1e6)
            density_map[chr_name] = counts
            if len(counts) > 0: max_d = max(max_d, counts.max())
        return density_map, max_d
    except: return {}, 0.0

def avoid_collisions(positions, min_spacing):
    n = len(positions)
    if n <= 1: return positions
    new_pos = np.array(positions, dtype=float)
    order = np.argsort(new_pos)
    sorted_p = new_pos[order]
    for _ in range(100):
        diffs = np.diff(sorted_p)
        overlaps = np.where(diffs < min_spacing)[0]
        if len(overlaps) == 0: break
        for i in overlaps:
            push = (min_spacing - diffs[i]) / 2 + 1e-5
            sorted_p[i] -= push
            sorted_p[i+1] += push
    final_p = np.zeros_like(new_pos)
    final_p[order] = sorted_p
    return final_p.tolist()

# ==========================================
# 侧边栏设置 (新增标尺控制)
# ==========================================
st.sidebar.header("🎨 布局精修")
chrs_per_row = st.sidebar.number_input("每行染色体数", 1, 30, 10)
fig_width = st.sidebar.slider("画布宽度", 5.0, 30.0, 12.0)
row_height = st.sidebar.slider("行高度", 2.0, 15.0, 7.0)

st.sidebar.subheader("📏 标尺设置 (新)")
ruler_offset = st.sidebar.slider("标尺与染色体间距", 0.1, 5.0, 0.8, help="调整左侧标尺距离第一条染色体的横向距离")
major_tick = st.sidebar.number_input("主刻度间隔 (Mb)", 1, 100, 10)
show_minor_tick = st.sidebar.checkbox("显示次刻度 (5 Mb)", value=True)
tick_width = st.sidebar.slider("刻度线长度", 0.05, 0.5, 0.2)

st.sidebar.subheader("🧬 基因密度与颜色")
uploaded_gff = st.sidebar.file_uploader("上传 GFF3", type=["gff3"])
use_density = st.sidebar.checkbox("启用密度热力图", value=False)
colormap_name = st.sidebar.selectbox("选择色系", ['Mako', 'Grayscale_Print', 'YlOrRd', 'viridis'], index=1)
chr_w = st.sidebar.slider("染色体宽度", 0.1, 1.0, 0.4)

st.sidebar.subheader("🏷️ 标签设置")
label_space = st.sidebar.slider("标签间距 (Mb)", 0.1, 5.0, 1.2)
font_sz = st.sidebar.slider("字体大小", 6, 20, 10)

# ==========================================
# 主界面输入
# ==========================================
st.title("📍 染色体图谱 v12.7")
col1, col2 = st.columns(2)
with col1:
    gene_input = st.text_area("1. 基因位置 (ID Start End Chr [Color])", "Gene1 5000000 5500000 Chr01 #CC0000", height=150)
with col2:
    len_input = st.text_area("2. 染色体长度 (Chr Length_bp)", "Chr01 50000000\nChr02 45000000", height=150)

# 解析数据
len_dict = {l.split()[0]: float(l.split()[1]) for l in len_input.strip().split('\n') if len(l.split()) >= 2}
genes_data = []
for l in gene_input.strip().split('\n'):
    p = l.split()
    if len(p) >= 4: genes_data.append({'Gene': p[0], 'Start': float(p[1]), 'End': float(p[2]), 'Chr': p[3], 'Color': p[4] if len(p) > 4 else '#FF0000'})
df_genes = pd.DataFrame(genes_data)

# ==========================================
# 绘图逻辑
# ==========================================
def draw_plot():
    sorted_chrs = sorted(len_dict.keys())
    num_rows = math.ceil(len(sorted_chrs) / chrs_per_row)
    max_mb = max(len_dict.values()) / 1e6
    
    fig, axes = plt.subplots(num_rows, 1, figsize=(fig_width, row_height * num_rows))
    if num_rows == 1: axes = [axes]

    # 计算密度
    d_map, d_max = {}, 0.0
    if use_density and uploaded_gff:
        d_map, d_max = calculate_windowed_density(uploaded_gff, len_dict, 1e6)
    norm = mcolors.Normalize(0, d_max) if d_max > 0 else None
    cmap = cm.get_cmap(colormap_name)

    for r in range(num_rows):
        ax = axes[r]
        row_chrs = sorted_chrs[r*chrs_per_row : (r+1)*chrs_per_row]
        ax.set_xlim(-ruler_offset - 1, chrs_per_row * 2)
        ax.set_ylim(max_mb * 1.05, -max_mb * 0.05)
        ax.axis('off')

        # --- 标尺绘制逻辑 (核心改动) ---
        ruler_x = -ruler_offset
        ax.plot([ruler_x, ruler_x], [0, max_mb], color='black', lw=1.5, zorder=3)
        # 主刻度
        for t in range(0, int(max_mb) + 1, major_tick):
            ax.plot([ruler_x - tick_width, ruler_x], [t, t], color='black', lw=1.2)
            ax.text(ruler_x - tick_width - 0.1, t, str(t), ha='right', va='center', fontsize=font_sz)
        # 次刻度
        if show_minor_tick:
            for t in range(0, int(max_mb) + 1, 5):
                if t % major_tick != 0:
                    ax.plot([ruler_x - tick_width/2, ruler_x], [t, t], color='black', lw=0.8)
        ax.text(ruler_x, -max_mb*0.02, "Mb", ha='center', fontweight='bold', fontsize=font_sz)

        # --- 染色体绘制 ---
        for i, name in enumerate(row_chrs):
            x = i * 2
            c_len = len_dict[name] / 1e6
            
            # 密度填充
            if use_density and norm and name in d_map:
                for idx, v in enumerate(d_map[name]):
                    rect = patches.Rectangle((x-chr_w/2, idx), chr_w, 1.0, facecolor=cmap(norm(v)), lw=0, zorder=0)
                    ax.add_patch(rect)
            
            # 轮廓
            box = patches.FancyBboxPatch((x-chr_w/2, 0), chr_w, c_len, boxstyle=f"round,pad=0,rounding_size={chr_w/2}", 
                                         ec='black', fc='none', lw=1.2, zorder=1)
            ax.add_patch(box)
            ax.text(x, -max_mb*0.02, name, ha='center', va='bottom', fontweight='bold', fontsize=font_sz+2)

            # 基因标记
            c_genes = df_genes[df_genes['Chr'] == name].copy()
            if not c_genes.empty:
                c_genes['y'] = (c_genes['Start'] + c_genes['End']) / 2e6
                c_genes['ly'] = avoid_collisions(c_genes['y'].tolist(), label_space)
                for _, row in c_genes.iterrows():
                    ax.plot([x-chr_w/2, x+chr_w/2], [row['y'], row['y']], color=row['Color'], lw=2, zorder=2)
                    ax.plot([x+chr_w/2, x+chr_w/2+0.15], [row['y'], row['ly']], color='black', lw=0.6)
                    ax.text(x+chr_w/2+0.2, row['ly'], row['Gene'], va='center', fontsize=font_sz, style='italic')

    return fig

if st.button("🚀 生成物理图谱"):
    if not len_dict: st.error("请先输入染色体长度")
    else:
        fig = draw_plot()
        st.pyplot(fig)
        buf = BytesIO()
        fig.savefig(buf, format="svg", bbox_inches='tight')
        st.download_button("📥 下载 SVG (AI可编辑)", buf.getvalue(), "chrom_map.svg", "image/svg+xml")
