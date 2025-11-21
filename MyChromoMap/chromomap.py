import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from io import BytesIO
import re
import math

# --- 页面配置 ---
st.set_page_config(page_title="染色体图谱 v9.1 (论文版)", layout="wide")

# --- 样式设置 ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    /* 强制修改 Streamlit 组件字体为 Times New Roman */
    html, body, [class*="css"] {
        font-family: 'Times New Roman', serif;
    }
    h1, h2, h3, .stMarkdown, .stText, .stButton button { font-family: 'Times New Roman', serif !important; color: #000; }
    .stDataFrame { font-family: 'Times New Roman', serif; }
    /* 文本域保持等宽字体以便输入 */
    .stTextArea textarea { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 12px; }
    
    /* 论文文本区域 */
    .paper-text {
        background-color: #f8f9fa;
        border-left: 4px solid #2c3e50;
        padding: 15px;
        font-family: 'Times New Roman', serif;
        white-space: pre-wrap;
        line-height: 1.6;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 字体全局配置 (Matplotlib) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# --- 侧边栏设置 ---
st.sidebar.header("🎨 绘图设置 (Plot Settings)")

# Tab 1: 基础布局
st.sidebar.subheader("1. 画布与网格布局")
chrs_per_row = st.sidebar.number_input("每行染色体数量", 1, 50, 10)
fig_width = st.sidebar.slider("每行图片宽度 (inch)", 4.0, 30.0, 12.0, 0.5)
row_height = st.sidebar.slider("单行图片高度 (inch)", 2.0, 15.0, 5.0, 0.5)

# Tab 2: 精细布局
st.sidebar.subheader("2. 精细间距调整 (Layout Tuning)")
st.sidebar.info("在此微调各元素间的距离")
ruler_gap = st.sidebar.slider("↔️ 比例尺-染色体间距", 0.2, 2.0, 0.8, 0.1)
chr_spacing = st.sidebar.slider("↔️ 染色体间横向间距", 0.0, 3.0, 0.5, 0.1)
y_pad_top = st.sidebar.slider("↕️ 顶部留白比例", 0.01, 0.2, 0.05, 0.01)
y_pad_bottom = st.sidebar.slider("↕️ 底部留白比例 (用于名称)", 0.01, 0.2, 0.05, 0.01)

# Tab 3: 比例尺样式
st.sidebar.subheader("3. 左侧比例尺样式")
show_ruler = st.sidebar.checkbox("显示比例尺", value=True)
tick_interval = st.sidebar.number_input("刻度间隔 (Mb)", 1, 500, 10)
ruler_fs = st.sidebar.slider("刻度字号", 8, 20, 12)
arrow_dist = st.sidebar.slider("↓ 箭头垂直间距", 0.0, 3.0, 0.8)

# Tab 4: 染色体外观
st.sidebar.subheader("4. 染色体外观")
chr_width = st.sidebar.slider("染色体宽窄 (相对宽度)", 0.1, 1.5, 0.4, 0.05)
chr_fill_color = st.sidebar.color_picker("填充颜色", "#E0E0E0") 
chr_edge_color = st.sidebar.color_picker("边框颜色", "#000000")

# Tab 5: 基因标记
st.sidebar.subheader("5. 基因标记")
font_size = st.sidebar.slider("标签字号", 8, 24, 12)
label_offset = st.sidebar.slider("标签引线长度", 0.0, 1.5, 0.2)
min_marker_mb = st.sidebar.slider("最小显示高度 (Mb)", 0.1, 10.0, 1.0, 0.1)
default_marker_color = st.sidebar.color_picker("默认基因颜色", "#FF0000")

# --- 主界面 ---
st.title("📍 染色体物理图谱 v9.1")
st.markdown("*(特性：更新了符合学术规范的论文写作助手)*")

col1, col2 = st.columns([1, 1])

# ==========================================
# 数据输入 (保持不变)
# ==========================================
with col1:
    st.subheader("1. 输入基因数据")
    input_tab1, input_tab2 = st.tabs(["📋 文本粘贴", "📂 Excel 上传"])
    df_genes = pd.DataFrame()
    with input_tab1:
        default_paste = """gsample1 5000000 6000000 Chr01 red
gsample2 15000000 15500000 Chr01
gsample3 45000000 48000000 Chr02 blue
gsample4 10000000 12000000 Chr11 green"""
        text_data = st.text_area("格式: Gene Start End Chr [Color]", value=default_paste, height=200)
        if text_data.strip():
            try:
                lines = text_data.strip().split('\n')
                data_list = []
                for line in lines:
                    parts = re.split(r'\s+', line.strip())
                    if len(parts) >= 4:
                        row = {'Gene': parts[0], 'Start': float(parts[1]), 'End': float(parts[2]), 'Chr': parts[3], 'Color': parts[4] if len(parts) > 4 else ''}
                        data_list.append(row)
                df_genes = pd.DataFrame(data_list)
            except: pass
    with input_tab2:
        uploaded_file = st.file_uploader("上传 Excel", type=["xlsx", "xls"])
        if uploaded_file:
            try:
                df_temp = pd.read_excel(uploaded_file)
                df_temp.columns = [c.strip().lower() for c in df_temp.columns]
                col_map = {}
                for c in df_temp.columns:
                    if 'gene' in c: col_map[c] = 'Gene'
                    if 'start' in c: col_map[c] = 'Start'
                    if 'end' in c: col_map[c] = 'End'
                    if 'chr' in c: col_map[c] = 'Chr'
                    if 'color' in c: col_map[c] = 'Color'
                df_genes = df_temp.rename(columns=col_map)
            except: pass
    if not df_genes.empty and {'Gene', 'Start', 'End', 'Chr'}.issubset(set(df_genes.columns)):
        st.success(f"✅ 已加载 {len(df_genes)} 个基因")

with col2:
    st.subheader("2. 定义染色体长度")
    default_len_text = """Chr01 57932355
Chr02 50400358
Chr03 46951866
Chr04 51203389
Chr11 55000000"""
    chr_len_input = st.text_area("格式: `Chr Length`", value=default_len_text, height=200)
    chr_len_dict = {}
    try:
        for line in chr_len_input.strip().split('\n'):
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 2: chr_len_dict[parts[0]] = float(parts[1])
    except: pass

is_bp_unit = False
if chr_len_dict and max(chr_len_dict.values()) > 5000: is_bp_unit = True
def convert_unit(val): return val / 1_000_000 if is_bp_unit else val

# ==========================================
# 核心绘图逻辑 v9 (保持不变)
# ==========================================
def plot_ideogram_v9(genes, len_dict, 
                     max_col, row_h, fig_w, 
                     c_width, fill_col, edge_col, 
                     f_size, min_h_mb, label_off, def_col,
                     is_ruler, tick_int, r_fs, arr_dist,
                     r_gap, c_spacing, y_pad_t, y_pad_b
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
        
        ax.set_xlim(base_x - r_gap - 0.5, final_chr_x + 1.5)
        ax.set_ylim(y_top_limit, y_bottom_limit)
        ax.axis('off')

        if is_ruler:
            ruler_x = base_x - r_gap
            line = mlines.Line2D([ruler_x, ruler_x], [0, global_max_len_mb], color='black', linewidth=1.2)
            ax.add_line(line)
            ticks = list(range(0, int(global_max_len_mb) + 1, int(tick_int)))
            tick_width = 0.1
            for t in ticks:
                ax.plot([ruler_x, ruler_x + tick_width], [t, t], color='black', linewidth=1)
                ax.text(ruler_x + tick_width + 0.1, t, str(t), 
                        ha='left', va='center', fontname='Times New Roman', fontsize=r_fs)
            ax.text(ruler_x, y_bottom_limit * 0.5, "Mb", ha='center', va='bottom',
                    fontname='Times New Roman', fontsize=r_fs, fontweight='bold')
            arrow_y = global_max_len_mb + arr_dist
            ax.plot(ruler_x, arrow_y, marker='v', color='black', markersize=6, clip_on=False)

        for i, chr_name in enumerate(current_row_chrs):
            x_pos = base_x + i * (1.0 + c_spacing)
            length_mb = convert_unit(len_dict[chr_name])
            
            box = patches.FancyBboxPatch(
                (x_pos - c_width/2, 0), c_width, length_mb,
                boxstyle=f"round,pad=0.02,rounding_size={c_width/2}", 
                linewidth=1.5, edgecolor=edge_col, facecolor=fill_col, zorder=1
            )
            ax.add_patch(box)
            ax.text(x_pos, -global_max_len_mb * y_pad_b * 0.5, chr_name, ha='center', va='bottom', 
                    fontname='Times New Roman', fontsize=f_size+2, fontweight='bold')
            
            chr_genes = genes[genes['Chr'] == chr_name]
            for _, row in chr_genes.iterrows():
                start_mb, end_mb = convert_unit(row['Start']), convert_unit(row['End'])
                name = str(row['Gene'])
                color = def_col
                if 'Color' in row and pd.notna(row['Color']) and str(row['Color']).strip():
                    color = str(row['Color']).strip()
                
                draw_height = max(end_mb - start_mb, min_h_mb)
                center = (start_mb + end_mb) / 2
                draw_start = center - (draw_height / 2)
                
                rect = patches.Rectangle((x_pos - c_width/2, draw_start), c_width, draw_height, linewidth=0, facecolor=color, zorder=2)
                ax.add_patch(rect)
                
                label_x = x_pos + c_width/2 + label_off
                ax.plot([x_pos + c_width/2, label_x], [center, center], color='black', lw=0.5, zorder=1)
                ax.text(label_x + 0.05, center, name, ha='left', va='center', 
                        fontname='Times New Roman', style='italic', fontsize=f_size)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3) 
    return fig

# --- 论文生成 (已更新) ---
def generate_paper_text(genes, len_dict):
    total_genes = len(genes)
    counts = genes['Chr'].value_counts()
    if counts.empty: return "", "", "", ""
    max_chr, max_count = counts.idxmax(), counts.max()
    min_chr, min_count = counts.idxmin(), counts.min()
    
    # --- 核心修改部分 ---
    cn_m = f"""【材料与方法】\n基因组物理位置可视化基于 Python 编程环境实现。其中，Pandas 库用于基因组位置数据的预处理与格式化。核心图谱调用 Matplotlib 绘图库进行绘制，所有染色体长度及基因分布位置均严格按实际物理距离（单位：Mb）成比例展示，并在图谱左侧设置垂直比例尺以指示物理距离。"""
    
    cn_r = f"""【结果与分析】\n物理图谱显示（图1），{total_genes} 个目标基因分布在 {len(counts)} 条染色体上。基因在基因组中的分布呈现不均匀性，其中 {max_chr} 包含的基因数量最多，达到 {max_count} 个；而 {min_chr} 分布最少，仅有 {min_count} 个基因。"""
    
    # --- 核心修改部分 (英文版) ---
    en_m = f"""[Materials and Methods]\nThe visualization of genomic physical positions was implemented in the Python programming environment. The Pandas library was used for preprocessing and formatting genomic location data. The core ideogram was generated using the Matplotlib plotting library, where all chromosome lengths and gene distribution positions were drawn strictly in proportion to their actual physical distances (Mb). A vertical scale bar was included on the left side of the ideogram to indicate physical distances."""
    
    en_r = f"""[Results]\nThe physical map (Fig. 1) revealed that {total_genes} target genes were distributed across {len(counts)} chromosomes. The distribution pattern in the genome was uneven, with Chromosome {max_chr} harboring the highest number of genes ({max_count}), whereas Chromosome {min_chr} contained the fewest ({min_count})."""
    
    return cn_m, cn_r, en_m, en_r

# ==========================================
# 主运行区
# ==========================================
st.markdown("---")
if st.button("🚀 生成图谱与论文文本", type="primary"):
    if not chr_len_dict: st.error("❌ 请输入染色体长度！")
    elif df_genes.empty: st.error("❌ 请输入基因数据！")
    else:
        fig = plot_ideogram_v9(
            df_genes, chr_len_dict, chrs_per_row, row_height, fig_width, 
            chr_width, chr_fill_color, chr_edge_color, font_size, min_marker_mb, label_offset, default_marker_color,
            show_ruler, tick_interval, ruler_fs, arrow_dist,
            ruler_gap, chr_spacing, y_pad_top, y_pad_bottom
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
        cn_m, cn_r, en_m, en_r = generate_paper_text(df_genes, chr_len_dict)
        t1, t2 = st.tabs(["🇨🇳 中文", "🇺🇸 English"])
        with t1: st.markdown(f"<div class='paper-text'>{cn_m}\n\n{cn_r}</div>", unsafe_allow_html=True)
        with t2: st.markdown(f"<div class='paper-text'>{en_m}\n\n{en_r}</div>", unsafe_allow_html=True)
