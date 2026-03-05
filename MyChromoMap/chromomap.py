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
st.set_page_config(page_title="染色体图谱 v12.9 (全功能修正版)", layout="wide")

# --- 样式设置 (学术论文风) ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    html, body, [class*="css"] { font-family: 'Times New Roman', serif; }
    h1, h2, h3, .stMarkdown, .stText, .stButton button { font-family: 'Times New Roman', serif !important; color: #000; }
    .stTextArea textarea { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 13px; }
    .paper-text { background-color: #f8f9fa; border-left: 4px solid #2c3e50; padding: 15px; font-family: 'Times New Roman', serif; white-space: pre-wrap; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 字体与全局配置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none' 

# ==========================================
# 2. 注册自定义色系 (含黑白印刷灰阶)
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
# 3. 核心计算函数 (含高度溢出修正)
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
            if chr_name not in df_genes['seqid'].unique(): continue
            chr_sub_df = df_genes[df_genes['seqid'] == chr_name]
            bin_edges = np.arange(0, chr_len + window_size_bp, window_size_bp)
            if bin_edges[-1] < chr_len: bin_edges = np.append(bin_edges, chr_len)
            binned_data = pd.cut(chr_sub_df['midpoint'], bins=bin_edges, include_lowest=True)
            counts_series = binned_data.value_counts(sort=False)
            density_values = counts_series.values / (window_size_bp / 1_000_000)
            density_profile_map[chr_name] = density_values
            if len(density_values) > 0: global_max_density = max(global_max_density, density_values.max())
        return density_profile_map, global_max_density
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
# 4. 侧边栏 UI
# ==========================================
st.sidebar.header("🎨 图谱细节调优")
chrs_per_row = st.sidebar.number_input("每行染色体数量", 1, 30, 10)
fig_width = st.sidebar.slider("画布总宽度 (inch)", 5.0, 40.0, 15.0)
row_height = st.sidebar.slider("单行高度 (inch)", 2.0, 20.0, 8.0)

st.sidebar.subheader("📏 标尺与刻度")
ruler_offset = st.sidebar.slider("↔️ 标尺-染色体间距", 0.1, 4.0, 0.8)
major_tick_int = st.sidebar.number_input("主刻度间隔 (Mb)", 1, 200, 10)
tick_line_len = st.sidebar.slider("刻度线突出长度", 0.05, 0.5, 0.2)
show_minor = st.sidebar.checkbox("显示 5Mb 次刻度", value=True)

st.sidebar.subheader("🔥 密度热力图")
uploaded_gff = st.sidebar.file_uploader("上传 GFF3 文件", type=["gff3"])
use_density_color = st.sidebar.checkbox("启用密度着色", value=False)
window_size_mb = st.sidebar.selectbox("窗口大小 (Mb)", [0.1, 0.5, 1.0, 2.0], index=2)
colormap_name = st.sidebar.selectbox("色系选择", ['Mako', 'Grayscale_Print', 'Reds', 'viridis'], index=1)

st.sidebar.subheader("🏷️ 标签与样式")
chr_width = st.sidebar.slider("染色体体宽", 0.1, 1.0, 0.4)
label_spacing = st.sidebar.slider("标签垂直间距 (Mb)", 0.1, 5.0, 1.2)
font_size = st.sidebar.slider("字体大小", 6, 20, 10)
label_color = st.sidebar.color_picker("标签文字颜色", "#000000")

# ==========================================
# 5. 主界面数据输入区 (Tabs 已找回)
# ==========================================
st.title("📍 染色体图谱 v12.9")
col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.subheader("1. 目标基因数据")
    input_tab1, input_tab2 = st.tabs(["📋 文本粘贴", "📂 Excel 上传"])
    df_genes = pd.DataFrame()
    with input_tab1:
        default_paste = "Glyma.04G235900.Wm82.a2.v1 27355 28320 Chr04 #FF0000"
        text_data = st.text_area("格式: ID Start End Chr [Color]", value=default_paste, height=180)
        if text_data.strip():
            lines = [re.split(r'\s+', l.strip()) for l in text_data.strip().split('\n') if l.strip()]
            df_genes = pd.DataFrame(lines).iloc[:, :5]
            df_genes.columns = ['Gene', 'Start', 'End', 'Chr', 'Color'][:len(df_genes.columns)]
            df_genes[['Start', 'End']] = df_genes[['Start', 'End']].apply(pd.to_numeric)
    with input_tab2:
        up_file = st.file_uploader("上传 Excel", type=["xlsx", "xls"])
        if up_file:
            df_genes = pd.read_excel(up_file)

with col2:
    st.subheader("2. 染色体长度 (bp)")
    len_text = st.text_area("格式: Chr Length", "Chr04 52000000\nChr05 48000000\nChr17 42000000", height=205)
    chr_len_dict = {l.split()[0]: float(l.split()[1]) for l in len_text.strip().split('\n') if len(l.split()) >= 2}

# ==========================================
# 6. 绘图执行
# ==========================================
def plot_v12_9(genes, l_dict, d_map, d_norm, d_cmap):
    sorted_chrs = sorted(l_dict.keys())
    num_rows = math.ceil(len(sorted_chrs) / chrs_per_row)
    max_mb = max(l_dict.values()) / 1e6
    
    fig, axes = plt.subplots(num_rows, 1, figsize=(fig_width, row_height * num_rows))
    if num_rows == 1: axes = [axes]

    for r in range(num_rows):
        ax = axes[r]
        row_chrs = sorted_chrs[r*chrs_per_row : (r+1)*chrs_per_row]
        ax.set_xlim(-ruler_offset - 0.5, chrs_per_row * 1.5)
        ax.set_ylim(max_mb * 1.05, -max_mb * 0.05)
        ax.axis('off')

        # 标尺与刻度线
        rx = -ruler_offset
        ax.plot([rx, rx], [0, max_mb], color='black', lw=1.2)
        for t in range(0, int(max_mb) + 1, major_tick_int):
            ax.plot([rx - tick_line_len, rx], [t, t], color='black', lw=1)
            ax.text(rx - tick_line_len - 0.1, t, str(t), ha='right', va='center', fontsize=font_size)
        if show_minor:
            for t in range(0, int(max_mb) + 1, 5):
                if t % major_tick_int != 0: ax.plot([rx - tick_line_len/2, rx], [t, t], color='black', lw=0.7)
        ax.text(rx, -2, "Mb", ha='center', fontweight='bold', fontsize=font_size)

        for i, name in enumerate(row_chrs):
            x = i * 1.2
            c_len_mb = l_dict[name] / 1e6
            
            # 密度条带 (高度溢出修正)
            if use_density_color and d_norm and name in d_map:
                dens = d_map[name]
                for idx, v in enumerate(dens):
                    sy = idx * window_size_mb
                    bh = min(window_size_mb, c_len_mb - sy) # 关键修正
                    if bh > 0:
                        ax.add_patch(patches.Rectangle((x-chr_width/2, sy), chr_width, bh, facecolor=d_cmap(d_norm(v)), lw=0, zorder=0))

            # 染色体圆角外框
            ax.add_patch(patches.FancyBboxPatch((x-chr_width/2, 0), chr_width, c_len_mb, boxstyle=f"round,pad=0,rounding_size={chr_width/2}", ec='black', fc='none', lw=1.2, zorder=1))
            ax.text(x, -2, name, ha='center', fontweight='bold', fontsize=font_size+1)

            # 基因标记
            c_gs = genes[genes['Chr'] == name].copy()
            if not c_gs.empty:
                c_gs['y'] = c_gs['Start'] / 1e6
                c_gs['ly'] = avoid_collisions(c_gs['y'].tolist(), label_spacing)
                for _, row in c_gs.iterrows():
                    color = row['Color'] if 'Color' in row and pd.notna(row['Color']) else "#FF0000"
                    ax.plot([x-chr_width/2, x+chr_width/2], [row['y'], row['y']], color=color, lw=2, zorder=2)
                    ax.plot([x+chr_width/2, x+chr_width/2+0.1], [row['y'], row['ly']], color='black', lw=0.5)
                    ax.text(x+chr_width/2+0.12, row['ly'], row['Gene'], va='center', fontsize=font_size, color=label_color, style='italic')

    return fig

if st.button("🚀 生成并预览图谱", type="primary"):
    if not chr_len_dict or df_genes.empty:
        st.error("请输入完整的染色体长度和基因位置数据。")
    else:
        dm, dmax = {}, 0.0
        if use_density_color and uploaded_gff:
            dm, dmax = calculate_windowed_density(uploaded_gff, chr_len_dict, window_size_mb * 1e6)
        
        fig = plot_v12_9(df_genes, chr_len_dict, dm, mcolors.Normalize(0, dmax) if dmax > 0 else None, cm.get_cmap(colormap_name))
        st.pyplot(fig)
        
        # 导出选项
        buf = BytesIO()
        fig.savefig(buf, format="svg", bbox_inches='tight')
        st.download_button("💾 下载 SVG (学术出版级)", buf.getvalue(), "soybean_map.svg", "image/svg+xml")
