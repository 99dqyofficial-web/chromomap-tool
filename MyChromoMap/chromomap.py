import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.cm as cm
# --- 导入字体管理器和操作系统路径库 ---
import matplotlib.font_manager as fm
import os
from io import BytesIO
import re
import math
import numpy as np

# --- 页面配置 ---
st.set_page_config(page_title="染色体图谱 v12.4 (SVG字体修复)", layout="wide")

# --- 样式设置 (保持不变) ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    /* 网页界面的字体设置，与 Matplotlib 无关，但也建议设置 */
    html, body, [class*="css"] { font-family: 'Times New Roman', serif; }
    h1, h2, h3, .stMarkdown, .stText, .stButton button { font-family: 'Times New Roman', serif !important; color: #000; }
    .stDataFrame { font-family: 'Times New Roman', serif; }
    .stTextArea textarea { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 12px; }
    .paper-text { background-color: #f8f9fa; border-left: 4px solid #2c3e50; padding: 15px; font-family: 'Times New Roman', serif; white-space: pre-wrap; line-height: 1.6; color: #333; }
    [data-testid='stFileUploader'] { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# Matplotlib 字体全局强制加载与配置
# ==========================================
# 1. 加载本地字体文件
current_dir = os.path.dirname(os.path.abspath(__file__))
font_dir = os.path.join(current_dir, 'fonts')

if os.path.exists(font_dir):
    font_files = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith('.ttf')]
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
else:
    st.error("⚠️ 未找到 'fonts' 文件夹！请确保在项目根目录下创建 'fonts' 文件夹并放入 times.ttf, timesbd.ttf, timesi.ttf 文件。")

# 2. 设置全局字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# --- 【核心修改】设置 SVG 导出时不转曲 ---
# 'none' 表示不将字体转换为路径，保留文本元素
plt.rcParams['svg.fonttype'] = 'none' 


# ==========================================
# 以下代码保持不变...
# ==========================================
# 核心新函数：计算窗口化基因密度 (Window-based Density)
def calculate_windowed_density(gff_file_obj, len_dict_bp, window_size_bp):
    gff_cols = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
    try:
        # 读取 GFF
        df_gff = pd.read_csv(gff_file_obj, sep='\t', comment='#', header=None, names=gff_cols, dtype={'seqid': str}, on_bad_lines='skip')
        # 筛选 gene
        df_genes = df_gff[df_gff['type'] == 'gene'].copy()
        
        if df_genes.empty: return {}, 0.0
        
        # 计算基因中心点，用于简化落入窗口的判定
        df_genes['midpoint'] = (df_genes['start'] + df_genes['end']) / 2
        
        density_profile_map = {}
        global_max_density = 0.0
        
        # 遍历有长度记录的染色体
        for chr_name, chr_len in len_dict_bp.items():
            if chr_name not in df_genes['seqid'].unique():
                continue
                
            chr_sub_df = df_genes[df_genes['seqid'] == chr_name]
            
            # --- 核心：创建窗口切片 ---
            bin_edges = np.arange(0, chr_len + window_size_bp, window_size_bp)
            if bin_edges[-1] < chr_len:
                 bin_edges = np.append(bin_edges, chr_len)
                 
            # 使用 pandas.cut 将基因中心点分箱统计
            binned_data = pd.cut(chr_sub_df['midpoint'], bins=bin_edges, include_lowest=True)
            counts_series = binned_data.value_counts(sort=False)
            
            # 将计数转换为密度 (Genes / Mb)
            window_size_mb = window_size_bp / 1_000_000
            density_values = counts_series.values / window_size_mb
            
            density_profile_map[chr_name] = density_values
            if len(density_values) > 0:
                global_max_density = max(global_max_density, density_values.max())
                
        return density_profile_map, global_max_density
        
    except Exception as e:
        st.sidebar.error(f"GFF 解析或计算失败: {e}")
        return {}, 0.0

# ==========================================
# 辅助函数：标签避让算法
# ==========================================
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

# --- 侧边栏设置 ---
st.sidebar.header("🎨 绘图设置")

# Tab 1: 基础布局
st.sidebar.subheader("1. 画布与网格")
chrs_per_row = st.sidebar.number_input("每行染色体数量", 1, 50, 10)
fig_width = st.sidebar.slider("每行图片宽度 (inch)", 4.0, 30.0, 12.0, 0.5)
row_height = st.sidebar.slider("单行图片高度 (inch)", 2.0, 15.0, 6.0, 0.5)

# Tab 2: 精细布局
st.sidebar.subheader("2. 精细间距调整")
ruler_gap = st.sidebar.slider("↔️ 比例尺-染色体间距", 0.2, 2.0, 0.8, 0.1)
chr_spacing = st.sidebar.slider("↔️ 染色体间横向间距", 0.0, 3.0, 0.8, 0.1)
y_pad_top = st.sidebar.slider("↕️ 顶部留白比例", 0.01, 0.2, 0.05, 0.01)
y_pad_bottom = st.sidebar.slider("↕️ 底部留白比例", 0.01, 0.2, 0.05, 0.01)

# Tab 3: 比例尺样式
st.sidebar.subheader("3. 比例尺样式")
show_ruler = st.sidebar.checkbox("显示比例尺", value=True)
tick_interval = st.sidebar.number_input("刻度间隔 (Mb)", 1, 500, 10)
ruler_fs = st.sidebar.slider("刻度字号", 8, 20, 12)
arrow_dist = st.sidebar.slider("↓ 箭头垂直间距", 0.0, 3.0, 0.8)

# Tab 4: 窗口化密度分布
st.sidebar.subheader("4. 窗口化基因密度分布 (Windowed Density)")
st.sidebar.markdown("**上传 GFF3 计算沿染色体的密度分布**")
uploaded_gff = st.sidebar.file_uploader("上传 GFF3 注释文件", type=["gff", "gff3"], key="gff_uploader")

# 新增：窗口大小选择
window_size_mb_select = st.sidebar.selectbox("选择计算窗口大小 (Mb)", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0], index=3, help="例如选择 1.0，则计算每 1Mb 区域内的基因密度")
window_size_bp = int(window_size_mb_select * 1_000_000)

use_density_color = st.sidebar.checkbox("启用窗口密度热力图", value=False, disabled=uploaded_gff is None)
colormap_name = st.sidebar.selectbox("选择热力图色系", ['YlOrRd', 'Reds', 'Blues', 'viridis', 'plasma', 'inferno'], index=0)

chr_width = st.sidebar.slider("染色体宽窄", 0.1, 1.5, 0.6, 0.05)
chr_fill_color = st.sidebar.color_picker("单一填充颜色", "#E0E0E0", disabled=use_density_color)
chr_edge_color = st.sidebar.color_picker("边框颜色", "#000000")

# Tab 5: 基因标记
st.sidebar.subheader("5. 基因标记 (智能防重叠)")
enable_avoidance = st.sidebar.checkbox("启用智能防重叠", value=True)
label_spacing = st.sidebar.slider("标签最小垂直间距 (Mb)", 0.1, 5.0, 1.5, 0.1)

# 新增标签颜色选择器
label_color = st.sidebar.color_picker("标签文字颜色", "#000000", help="设置基因名称标签的字体颜色")

font_size = st.sidebar.slider("标签字号", 8, 24, 11)
label_offset_x = st.sidebar.slider("标签引线横向长度", 0.0, 2.0, 0.3)
min_marker_mb = st.sidebar.slider("最小显示高度 (Mb)", 0.1, 10.0, 1.0, 0.1)
default_marker_color = st.sidebar.color_picker("默认基因颜色", "#FF0000")

# --- 主界面 ---
st.title("📍 染色体物理图谱 v12.4")
st.markdown("*(特性：基于固定窗口的基因密度分布热力图 + 标签颜色自定义 + 全局Times New Roman字体 + SVG可编辑)*")

col1, col2 = st.columns([1, 1])

# ==========================================
# 数据输入
# ==========================================
with col1:
    st.subheader("1. 输入目标基因数据")
    input_tab1, input_tab2 = st.tabs(["📋 文本粘贴", "📂 Excel 上传"])
    df_genes = pd.DataFrame()
    with input_tab1:
        default_paste = """gene1 5000000 5100000 Chr01 red
gene2 15000000 15500000 Chr01
gene3 45000000 48000000 Chr02 blue
gene4 20000000 22000000 Chr01 green"""
        text_data = st.text_area("格式: Gene Start End Chr [Color]", value=default_paste, height=200)
        if text_data.strip():
            try:
                lines = text_data.strip().split('\n')
                data_list = []
                for line in lines:
                    parts = re.split(r'\s+', line.strip())
                    if len(parts) >= 4:
                        data_list.append({'Gene': parts[0], 'Start': float(parts[1]), 'End': float(parts[2]), 'Chr': parts[3], 'Color': parts[4] if len(parts) > 4 else ''})
                df_genes = pd.DataFrame(data_list)
            except: pass
    with input_tab2:
        uploaded_file = st.file_uploader("上传 Excel", type=["xlsx", "xls"])
        if uploaded_file:
            try:
                df_temp = pd.read_excel(uploaded_file)
                df_temp.columns = [c.strip().lower() for c in df_temp.columns]
                col_map = {c: 'Gene' if 'gene' in c else 'Start' if 'start' in c else 'End' if 'end' in c else 'Chr' if 'chr' in c else 'Color' if 'color' in c else c for c in df_temp.columns}
                df_genes = df_temp.rename(columns=col_map)
            except: pass
    if not df_genes.empty and {'Gene', 'Start', 'End', 'Chr'}.issubset(set(df_genes.columns)):
        st.success(f"✅ 已加载 {len(df_genes)} 个目标基因")

with col2:
    st.subheader("2. 定义染色体长度 (必需)")
    st.caption("准确的长度对于密度窗口计算至关重要")
    default_len_text = """Chr01 40000000\nChr02 30000000"""
    chr_len_input = st.text_area("格式: `Chr Length (bp)`", value=default_len_text, height=200)
    chr_len_dict = {}
    try:
        for line in chr_len_input.strip().split('\n'):
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 2: chr_len_dict[parts[0]] = float(parts[1])
    except: pass

is_bp_unit = False
if chr_len_dict and max(chr_len_dict.values()) > 5000: is_bp_unit = True
def convert_unit(val): return val / 1_000_000 if is_bp_unit else val

# --- 窗口密度计算逻辑调用 ---
density_profile_map = {}
density_norm = None
density_cmap_obj = None
max_density_val = 0.0

if uploaded_gff and chr_len_dict and use_density_color:
    with st.spinner(f"正在以 {window_size_mb_select}Mb 为窗口计算密度分布..."):
        uploaded_gff.seek(0)
        # 调用新的计算函数
        density_profile_map, max_density_val = calculate_windowed_density(uploaded_gff, chr_len_dict, window_size_bp)
        
        if density_profile_map and max_density_val > 0:
            st.sidebar.success(f"计算完成! 最大密度: {max_density_val:.1f} Genes/Mb (窗口: {window_size_mb_select}Mb)")
            # 创建颜色标准化对象 (从0到最大密度)
            density_norm = mcolors.Normalize(vmin=0, vmax=max_density_val)
            density_cmap_obj = cm.get_cmap(colormap_name)
        elif max_density_val == 0:
             st.sidebar.warning("密度计算结果全为0，请检查 GFF 文件内容或染色体名称匹配。")


# ==========================================
# 核心绘图逻辑 v12 (支持绘制密度色带)
# ==========================================
def plot_ideogram_v12(genes, len_dict, 
                     max_col, row_h, fig_w, 
                     c_width, default_fill, edge_col, 
                     f_size, min_h_mb, label_off_x, def_col, lbl_color,
                     is_ruler, tick_int, r_fs, arr_dist,
                     r_gap, c_spacing, y_pad_t, y_pad_b,
                     do_avoid, lbl_spacing,
                     d_profile_map, d_norm, d_cmap_obj, win_size_mb
                     ):
    
    sorted_chrs = sorted(len_dict.keys())
    total_chrs = len(sorted_chrs)
    num_rows = math.ceil(total_chrs / max_col)
    global_max_len_mb = convert_unit(max(len_dict.values())) if len_dict else 100
    
    y_top_limit = global_max_len_mb * (1 + y_pad_t) + arr_dist
    y_bottom_limit = -global_max_len_mb * y_pad_b
    
    fig, axes = plt.subplots(num_rows, 1, figsize=(fig_w, row_h * num_rows))
    if num_rows == 1: axes = [axes]
    
    for r in range(num_rows):
        ax = axes[r]
        start_idx = r * max_col
        end_idx = min((r + 1) * max_col, total_chrs)
        current_row_chrs = sorted_chrs[start_idx:end_idx]
        num_in_this_row = len(current_row_chrs)

        base_x = 1.0
        final_chr_x = base_x + (num_in_this_row - 1) * (1.0 + c_spacing) if num_in_this_row > 0 else base_x
        
        ax.set_xlim(base_x - r_gap - 0.5, final_chr_x + 2.5)
        ax.set_ylim(y_top_limit, y_bottom_limit)
        ax.axis('off')

        if is_ruler:
            ruler_x = base_x - r_gap
            line = mlines.Line2D([ruler_x, ruler_x], [0, global_max_len_mb], color='black', linewidth=1.2)
            ax.add_line(line)
            ticks = list(range(0, int(global_max_len_mb) + 1, int(tick_int)))
            for t in ticks:
                ax.plot([ruler_x, ruler_x + 0.1], [t, t], color='black', linewidth=1)
                ax.text(ruler_x + 0.2, t, str(t), ha='left', va='center', fontname='Times New Roman', fontsize=r_fs)
            ax.text(ruler_x, y_bottom_limit * 0.5, "Mb", ha='center', va='bottom', fontname='Times New Roman', fontsize=r_fs, fontweight='bold')
            ax.plot(ruler_x, global_max_len_mb + arr_dist, marker='v', color='black', markersize=6, clip_on=False)

        for i, chr_name in enumerate(current_row_chrs):
            x_pos = base_x + i * (1.0 + c_spacing)
            length_mb = convert_unit(len_dict[chr_name])
            
            # --- 核心更新：绘制密度色带 (Heatmap Bins) ---
            # 如果启用了密度着色且该染色体有数据
            if d_norm and d_cmap_obj and chr_name in d_profile_map:
                density_values = d_profile_map[chr_name]
                # 遍历每个窗口的密度值
                for bin_idx, density_val in enumerate(density_values):
                    # 计算当前窗口的起始和结束位置 (Mb)
                    bin_start_mb = bin_idx * win_size_mb
                    # 高度就是窗口大小，但不能超过染色体总长
                    bin_height = min(win_size_mb, length_mb - bin_start_mb)
                    
                    # 获取颜色
                    bin_color = d_cmap_obj(d_norm(density_val))
                    
                    # 绘制一个小矩形色块，zorder设为0，在最底层
                    # 注意：这里画的是直角矩形
                    rect = patches.Rectangle(
                        (x_pos - c_width/2, bin_start_mb), 
                        c_width, bin_height, 
                        linewidth=0, facecolor=bin_color, zorder=0
                    )
                    ax.add_patch(rect)
                    
                # --- 绘制染色体轮廓 (Frame) ---
                # 在色带上面画一个透明填充、带边框的圆角矩形，作为轮廓，盖住锯齿边缘
                box = patches.FancyBboxPatch(
                    (x_pos - c_width/2, 0), c_width, length_mb,
                    boxstyle=f"round,pad=0.02,rounding_size={c_width/2}", 
                    linewidth=1.5, edgecolor=edge_col, 
                    facecolor='none', # 透明填充
                    zorder=1 # 在色带之上
                )
                ax.add_patch(box)
                
            else:
                # 如果没有密度数据，回退到绘制单一颜色的实心染色体
                box = patches.FancyBboxPatch(
                    (x_pos - c_width/2, 0), c_width, length_mb,
                    boxstyle=f"round,pad=0.02,rounding_size={c_width/2}", 
                    linewidth=1.5, edgecolor=edge_col, 
                    facecolor=default_fill, zorder=1
                )
                ax.add_patch(box)

            # 绘制名称
            ax.text(x_pos, -global_max_len_mb * y_pad_b * 0.5, chr_name, ha='center', va='bottom', 
                    fontname='Times New Roman', fontsize=f_size+2, fontweight='bold')
            
            # 绘制基因 (逻辑不变)
            chr_genes = genes[genes['Chr'] == chr_name].copy()
            if chr_genes.empty: continue
            chr_genes['start_mb'] = chr_genes['Start'].apply(convert_unit)
            chr_genes['end_mb'] = chr_genes['End'].apply(convert_unit)
            chr_genes['center'] = (chr_genes['start_mb'] + chr_genes['end_mb']) / 2
            
            if do_avoid:
                chr_genes['label_y'] = avoid_collisions(chr_genes['center'].tolist(), lbl_spacing)
            else:
                chr_genes['label_y'] = chr_genes['center']

            for _, row in chr_genes.iterrows():
                name, color = str(row['Gene']), def_col
                if 'Color' in row and pd.notna(row['Color']) and str(row['Color']).strip(): color = str(row['Color']).strip()
                draw_h = max(row['end_mb'] - row['start_mb'], min_h_mb)
                rect = patches.Rectangle((x_pos - c_width/2, row['center'] - draw_h/2), c_width, draw_h, linewidth=0, facecolor=color, zorder=2)
                ax.add_patch(rect)
                line_end_x = x_pos + c_width/2 + label_off_x
                ax.plot([x_pos + c_width/2, line_end_x], [row['center'], row['label_y']], color='black', lw=0.5, zorder=1)
                # --- 使用 lbl_color 设置文本颜色 ---
                ax.text(line_end_x + 0.05, row['label_y'], name, ha='left', va='center', 
                        fontname='Times New Roman', style='italic', fontsize=f_size, color=lbl_color)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    # --- 绘制 Colorbar ---
    if d_norm and d_cmap_obj:
        cbar_ax = fig.add_axes([0.3, 0.02, 0.4, 0.015])
        cb = fig.colorbar(cm.ScalarMappable(norm=d_norm, cmap=d_cmap_obj), 
                          cax=cbar_ax, orientation='horizontal')
        # 更新 Label，注明窗口大小
        cb.set_label(f'Gene Density (Genes / {win_size_mb} Mb Window)', fontname='Times New Roman', fontsize=f_size)
        cb.ax.tick_params(labelsize=f_size*0.9)
        for l in cb.ax.get_xticklabels(): l.set_fontname('Times New Roman')
        plt.subplots_adjust(bottom=0.1) 

    return fig

# --- 论文生成 (保持不变) ---
def generate_paper_text(genes, len_dict, do_avoid, use_density, win_size_mb):
    total_genes = len(genes)
    counts = genes['Chr'].value_counts()
    if counts.empty: return "", "", "", ""
    max_chr, max_count = counts.idxmax(), counts.max()
    min_chr, min_count = counts.idxmin(), counts.min()
    
    avoid_cn = "，并采用力导向算法优化了密集区域的标签位置以防止重叠" if do_avoid else ""
    avoid_en = ", and a force-directed algorithm was applied to optimize label placement in dense regions to prevent overlapping" if do_avoid else ""
    
    # 更新密度描述，指明窗口大小
    density_cn = f"。此外，基于 GFF3 注释文件，以 {win_size_mb} Mb 为窗口大小计算了基因密度，并通过热力图颜色映射直观展示了染色体沿线的基因分布模式。" if use_density else "。"
    density_en = f". Furthermore, gene density was calculated based on GFF3 annotations using a {win_size_mb} Mb window size and visualized via heatmap color mapping to illustrate gene distribution patterns along the chromosomes." if use_density else "."

    cn_m = f"""【材料与方法】\n基因组物理位置可视化基于 Python 编程环境实现。Pandas 库用于数据的预处理与分箱计算。核心图谱调用 Matplotlib 库绘制，染色体及基因位置均严格按实际物理距离（Mb）按比例展示，左侧配有垂直比例尺{avoid_cn}{density_cn}"""
    cn_r = f"""【结果与分析】\n物理图谱显示（图1），{total_genes} 个目标基因不均匀地分布在 {len(counts)} 条染色体上（{max_chr} 最多，{min_chr} 最少）。""" + ("染色体热力图色带反映了不同区域基因密度的变化趋势。" if use_density else "")
    en_m = f"""[Materials and Methods]\The visualization was implemented in Python using Pandas for data preprocessing and binning. The core ideogram was generated using Matplotlib, where chromosomes and gene positions were drawn strictly in proportion to their actual physical distances (Mb), with a vertical scale bar on the left{avoid_en}{density_en}"""
    en_r = f"""[Results]\nThe physical map (Fig. 1) showed an uneven distribution of {total_genes} target genes across {len(counts)} chromosomes (highest on {max_chr}, lowest on {min_chr}).""" + (" The heatmap bands along chromosomes reflect trends in gene density variation." if use_density else "")
    
    return cn_m, cn_r, en_m, en_r

# ==========================================
# 主运行区
# ==========================================
st.markdown("---")
if st.button("🚀 生成图谱与论文文本", type="primary"):
    if not chr_len_dict: st.error("❌ 请先输入染色体长度（必需）！")
    elif df_genes.empty: st.error("❌ 请输入目标基因数据！")
    else:
        if use_density_color and not density_profile_map:
            st.warning("⚠️ 已启用密度着色但计算失败（请检查 GFF 和染色体名称）。将使用单一填充色。")

        # 调用绘图函数，传入新的 label_color 参数
        fig = plot_ideogram_v12(
            df_genes, chr_len_dict, chrs_per_row, row_height, fig_width, 
            chr_width, chr_fill_color, chr_edge_color, font_size, min_marker_mb, label_offset_x, default_marker_color, label_color, # 传入新参数
            show_ruler, tick_interval, ruler_fs, arrow_dist,
            ruler_gap, chr_spacing, y_pad_top, y_pad_bottom,
            enable_avoidance, label_spacing,
            density_profile_map, density_norm, density_cmap_obj, window_size_mb_select
        )
        st.pyplot(fig)
        
        c1, c2, c3 = st.columns(3)
        buf_svg = BytesIO()
        fig.savefig(buf_svg, format="svg", bbox_inches='tight')
        c3.download_button("🎨 下载 SVG", buf_svg.getvalue(), "chromomap.svg", "image/svg+xml")
        buf_pdf = BytesIO()
        fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
        c2.download_button("📄 下载 PDF", buf_pdf.getvalue(), "chromomap.pdf", "application/pdf")
        buf_png = BytesIO()
        fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
        c1.download_button("💾 下载 PNG", buf_png.getvalue(), "chromomap.png", "image/png")
        
        st.markdown("---")
        st.header("📝 论文写作助手")
        cn_m, cn_r, en_m, en_r = generate_paper_text(df_genes, chr_len_dict, enable_avoidance, use_density_color and density_profile_map, window_size_mb_select)
        t1, t2 = st.tabs(["🇨🇳 中文", "🇺🇸 English"])
        with t1: st.markdown(f"<div class='paper-text'>{cn_m}\n\n{cn_r}</div>", unsafe_allow_html=True)
        with t2: st.markdown(f"<div class='paper-text'>{en_m}\n\n{en_r}</div>", unsafe_allow_html=True)
