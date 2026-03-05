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
st.set_page_config(page_title="染色体图谱 v12.6 (支持黑白印刷灰阶)", layout="wide")

# --- 样式设置 ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    html, body, [class*="css"] { font-family: 'Times New Roman', serif; }
    h1, h2, h3, .stMarkdown, .stText, .stButton button { font-family: 'Times New Roman', serif !important; color: #000; }
    .stDataFrame { font-family: 'Times New Roman', serif; }
    .stTextArea textarea { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 12px; }
    .paper-text { background-color: #f8f9fa; border-left: 4px solid #2c3e50; padding: 15px; font-family: 'Times New Roman', serif; white-space: pre-wrap; line-height: 1.6; color: #333; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 字体与导出设置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
font_dir = os.path.join(current_dir, 'fonts')

if os.path.exists(font_dir):
    font_files = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith('.ttf')]
    for font_file in font_files:
        fm.fontManager.addfont(font_file)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none' 

# ==========================================
# 注册自定义 Colormaps (Mako & Grayscale)
# ==========================================
# 1. Mako 配色
mako_colors = ['#e3f7e3', '#BAE6E1', '#91D5DE']
mako_cmap = mcolors.LinearSegmentedColormap.from_list("Mako", mako_colors)

# 2. 灰阶配色 (用于黑白印刷: 浅灰 -> 中灰 -> 纯黑)
gray_print_colors = ['#F0F0F0', '#969696', '#000000']
gray_print_cmap = mcolors.LinearSegmentedColormap.from_list("Grayscale_Print", gray_print_colors)

# 注册到系统
for name, cmap in [("Mako", mako_cmap), ("Grayscale_Print", gray_print_cmap)]:
    try:
        cm.get_cmap(name)
    except ValueError:
        if hasattr(plt.colormaps, 'register'):
            plt.colormaps.register(cmap=cmap, name=name)
        else:
            cm.register_cmap(name=name, cmap=cmap)

# ==========================================
# 核心计算函数
# ==========================================
def calculate_windowed_density(gff_file_obj, len_dict_bp, window_size_bp):
    gff_cols = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
    try:
        df_gff = pd.read_csv(gff_file_obj, sep='\t', comment='#', header=None, names=gff_cols, dtype={'seqid': str}, on_bad_lines='skip')
        df_genes = df_gff[df_gff['type'] == 'gene'].copy()
        if df_genes.empty: return {}, 0.0
        df_genes['midpoint'] = (df_genes['start'] + df_genes['end']) / 2
        density_profile_map = {}
        global_max_density = 0.0
        for chr_name, chr_len in len_dict_bp.items():
            chr_sub_df = df_genes[df_genes['seqid'] == chr_name]
            bin_edges = np.arange(0, chr_len + window_size_bp, window_size_bp)
            if bin_edges[-1] < chr_len: bin_edges = np.append(bin_edges, chr_len)
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

def avoid_collisions(positions, min_spacing, max_iterations=100):
    n = len(positions)
    if n <= 1: return positions
    new_pos = np.array(positions, dtype=float)
    order = np.argsort(new_pos)
    new_pos_sorted = new_pos[order]
    for _ in range(max_iterations):
        moved = False
        diffs = np.diff(new_pos_sorted)
        overlaps = np.where(diffs < min_spacing)[0]
        if len(overlaps) == 0: break
        moved = True
        for i in overlaps:
            push = (min_spacing - diffs[i]) / 2 + 1e-5
            new_pos_sorted[i] -= push
            new_pos_sorted[i+1] += push
    final_pos = np.zeros_like(new_pos)
    final_pos[order] = new_pos_sorted
    return final_pos.tolist()

# ==========================================
# 侧边栏 UI
# ==========================================
st.sidebar.header("🎨 绘图与布局设置")
chrs_per_row = st.sidebar.number_input("每行染色体数量", 1, 50, 10)
fig_width = st.sidebar.slider("画布宽度 (inch)", 4.0, 40.0, 15.0)
row_height = st.sidebar.slider("单行高度 (inch)", 2.0, 20.0, 8.0)

st.sidebar.subheader("热力图设置")
uploaded_gff = st.sidebar.file_uploader("上传 GFF3 计算密度", type=["gff", "gff3"])
window_size_mb_select = st.sidebar.selectbox("窗口大小 (Mb)", [0.1, 0.5, 1.0, 2.0, 5.0], index=2)
use_density_color = st.sidebar.checkbox("启用密度热力图", value=False)
colormap_name = st.sidebar.selectbox("选择色系", ['Mako', 'Grayscale_Print', 'YlOrRd', 'viridis', 'gray'], index=0)

st.sidebar.subheader("染色体与标签")
chr_width = st.sidebar.slider("染色体宽度", 0.1, 1.5, 0.5)
label_spacing = st.sidebar.slider("标签最小间距 (Mb)", 0.1, 10.0, 1.5)
label_color = st.sidebar.color_picker("标签文字颜色", "#000000")
font_size = st.sidebar.slider("标签字号", 6, 20, 10)

# ==========================================
# 主界面数据输入
# ==========================================
st.title("📍 染色体物理图谱可视化 v12.6")
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 目标基因位置")
    default_genes = "Glyma.01G000100 27355 28320 Chr01 #FF0000"
    gene_text = st.text_area("格式: ID Start End Chr [Color]", value=default_genes, height=150)
    data_list = []
    for line in gene_text.strip().split('\n'):
        p = re.split(r'\s+', line.strip())
        if len(p) >= 4:
            data_list.append({'Gene': p[0], 'Start': float(p[1]), 'End': float(p[2]), 'Chr': p[3], 'Color': p[4] if len(p) > 4 else ''})
    df_genes = pd.DataFrame(data_list)

with col2:
    st.subheader("2. 染色体长度 (bp)")
    default_len = "Chr01 55000000\nChr02 50000000"
    chr_len_input = st.text_area("格式: Chr Length", value=default_len, height=150)
    chr_len_dict = {}
    for line in chr_len_input.strip().split('\n'):
        p = re.split(r'\s+', line.strip())
        if len(p) >= 2: chr_len_dict[p[0]] = float(p[1])

# ==========================================
# 绘图核心逻辑
# ==========================================
def plot_ideogram(genes, len_dict, d_profile, d_norm, d_cmap):
    sorted_chrs = sorted(len_dict.keys())
    num_rows = math.ceil(len(sorted_chrs) / chrs_per_row)
    max_len_mb = max(len_dict.values()) / 1e6
    
    fig, axes = plt.subplots(num_rows, 1, figsize=(fig_width, row_height * num_rows))
    if num_rows == 1: axes = [axes]

    for r in range(num_rows):
        ax = axes[r]
        row_chrs = sorted_chrs[r*chrs_per_row : (r+1)*chrs_per_row]
        ax.set_xlim(-1, chrs_per_row * 2)
        ax.set_ylim(max_len_mb * 1.1, -max_len_mb * 0.1)
        ax.axis('off')

        # 比例尺
        ax.plot([-0.5, -0.5], [0, max_len_mb], color='black', lw=1.5)
        for t in range(0, int(max_len_mb)+1, 10):
            ax.text(-0.7, t, str(t), ha='right', va='center', fontsize=font_size)
        ax.text(-0.7, -2, "Mb", fontweight='bold', fontsize=font_size)

        for i, chr_name in enumerate(row_chrs):
            x = i * 2
            curr_len = len_dict[chr_name] / 1e6
            
            # 绘制密度热力图
            if use_density_color and d_norm and chr_name in d_profile:
                vals = d_profile[chr_name]
                for idx, v in enumerate(vals):
                    b_start = idx * window_size_mb_select
                    b_h = min(window_size_mb_select, curr_len - b_start)
                    if b_h > 0:
                        rect = patches.Rectangle((x - chr_width/2, b_start), chr_width, b_h, facecolor=d_cmap(d_norm(v)), zorder=0)
                        ax.add_patch(rect)

            # 染色体轮廓
            box = patches.FancyBboxPatch((x-chr_width/2, 0), chr_width, curr_len, boxstyle=f"round,pad=0,rounding_size={chr_width/2}", 
                                         edgecolor='black', facecolor='none', lw=1.2, zorder=1)
            ax.add_patch(box)
            ax.text(x, -2, chr_name, ha='center', fontweight='bold', fontsize=font_size+2)

            # 基因标记
            c_genes = genes[genes['Chr'] == chr_name].copy()
            if not c_genes.empty:
                c_genes['y'] = (c_genes['Start'] + c_genes['End']) / 2e6
                c_genes['label_y'] = avoid_collisions(c_genes['y'].tolist(), label_spacing)
                for _, row in c_genes.iterrows():
                    color = row['Color'] if row['Color'] else "#FF0000"
                    ax.plot([x - chr_width/2, x + chr_width/2], [row['y'], row['y']], color=color, lw=2, zorder=2)
                    ax.plot([x + chr_width/2, x + chr_width/2 + 0.2], [row['y'], row['label_y']], color='black', lw=0.5)
                    ax.text(x + chr_width/2 + 0.25, row['label_y'], row['Gene'], va='center', fontsize=font_size, color=label_color, style='italic')

    if use_density_color and d_norm:
        cbar_ax = fig.add_axes([0.35, 0.05, 0.3, 0.02])
        fig.colorbar(cm.ScalarMappable(norm=d_norm, cmap=d_cmap), cax=cbar_ax, orientation='horizontal', label='Genes / Mb')
    
    return fig

# ==========================================
# 运行与输出
# ==========================================
if st.button("🚀 生成图谱"):
    d_map, d_max = {}, 0.0
    if use_density_color and uploaded_gff:
        d_map, d_max = calculate_windowed_density(uploaded_gff, chr_len_dict, window_size_mb_select*1e6)
    
    norm = mcolors.Normalize(vmin=0, vmax=d_max) if d_max > 0 else None
    cmap = cm.get_cmap(colormap_name)
    
    fig = plot_ideogram(df_genes, chr_len_dict, d_map, norm, cmap)
    st.pyplot(fig)
    
    # 下载
    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches='tight')
    st.download_button("下载 SVG (可编辑)", buf.getvalue(), "map.svg", "image/svg+xml")
