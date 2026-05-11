import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EDA Pro · Exploración de Datos",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
h1, h2, h3, h4                     { color: #e6edf3 !important; }
p, li, label                       { color: #8b949e !important; }
[data-testid="metric-container"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 16px 20px;
}
.section-title {
    font-size: 1.25rem; font-weight: 700; color: #58a6ff !important;
    border-left: 4px solid #58a6ff; padding-left: 12px; margin: 28px 0 16px 0;
}
.info-box {
    background: #1c2230; border: 1px solid #30363d; border-radius: 10px;
    padding: 16px 20px; margin: 8px 0; color: #c9d1d9 !important;
}
.hero {
    background: linear-gradient(135deg,#1a2744 0%,#0d1117 60%,#1a1a2e 100%);
    border:1px solid #30363d; border-radius:16px; padding:36px 40px;
    margin-bottom:24px; text-align:center;
}
.hero h1 { font-size:2.4rem !important; color:#58a6ff !important; margin:0 0 8px 0; }
.hero p  { font-size:1.05rem !important; color:#8b949e !important; margin:0; }
.pill {
    display:inline-block; background:#21262d; border:1px solid #30363d;
    border-radius:20px; padding:4px 14px; font-size:0.82rem;
    color:#58a6ff !important; margin:3px;
}
hr { border-color:#21262d !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  THEME PALETTE
# ══════════════════════════════════════════════════════════════════════════════
DARK    = "#0d1117"
CARD    = "#161b22"
BORDER  = "#30363d"
BLUE    = "#58a6ff"
GREEN   = "#3fb950"
ORANGE  = "#f0883e"
RED     = "#f85149"
PURPLE  = "#bc8cff"
TEAL    = "#39d353"
TEXT    = "#c9d1d9"
DIM     = "#8b949e"
PALETTE = [BLUE, GREEN, ORANGE, PURPLE, TEAL, RED, "#f78166", "#79c0ff", "#56d364", "#ffa657"]

def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.tick_params(colors=DIM)
    return fig, ax

def dark_figs(rows, cols, w=14, h=5, **kw):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h), **kw)
    fig.patch.set_facecolor(DARK)
    for ax in np.array(axes).flatten():
        ax.set_facecolor(CARD)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.tick_params(colors=DIM)
    return fig, axes

def title_ax(ax, t, fs=11):
    ax.set_title(t, fontsize=fs, fontweight="bold", color=TEXT, pad=8)

def xlabel(ax, t): ax.set_xlabel(t, fontsize=8, color=DIM)
def ylabel(ax, t): ax.set_ylabel(t, fontsize=8, color=DIM)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuración general")
    sep = st.selectbox("Separador CSV",
        [",",";","\t","|"],
        format_func=lambda x: {",":",  Coma",";":";  Punto y coma","\t":"⇥  Tab","|":"|  Pipe"}[x])
    bins_n = st.slider("Bins histograma", 10, 80, 30)
    st.markdown("---")
    st.markdown("### 🔗 Correlación")
    corr_method  = st.selectbox("Método correlación", ["pearson","spearman","kendall"])
    cmap_corr    = st.selectbox("Paleta mapa calor", ["coolwarm","RdBu_r","viridis","plasma","magma"])
    annot_corr   = st.checkbox("Valores en celdas", True)
    fmt_dec      = st.slider("Decimales", 1, 3, 2)
    st.markdown("---")
    st.markdown("### 📡 Scatter plot")
    st.markdown("*(configura abajo)*")
    st.markdown("---")
    st.caption("EDA Pro · Streamlit + Matplotlib")

# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🔬 EDA Pro</h1>
  <p>Exploración de Datos Avanzada · Sube tu CSV y descubre patrones al instante</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
uploaded = st.file_uploader("📂 Arrastra o selecciona tu archivo CSV", type=["csv"])
if uploaded is None:
    st.markdown('<div class="info-box">👆 Sube un archivo <b>CSV</b> para comenzar el análisis.</div>', unsafe_allow_html=True)
    st.stop()

try:
    df = pd.read_csv(uploaded, sep=sep)
except Exception as e:
    st.error(f"❌ Error al leer el archivo: {e}"); st.stop()

num_df = df.select_dtypes(include="number")
cat_df = df.select_dtypes(include=["object","category"])
n_rows, n_cols = df.shape
n_null = int(df.isnull().sum().sum())
n_dup  = int(df.duplicated().sum())
mem_kb = df.memory_usage(deep=True).sum() / 1024

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📋 Resumen",
    "📈 Distribuciones",
    "📦 Boxplot & Violin",
    "🔗 Correlación",
    "🔵 Scatter Plot",
    "🧩 Pair Plot",
    "🥧 Categóricas",
    "🕳️ Nulos",
    "📉 Tendencias",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 0 · RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="section-title">📋 Resumen del Dataset</p>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🗂️ Filas",      f"{n_rows:,}")
    c2.metric("📐 Columnas",   f"{n_cols}")
    c3.metric("🔢 Numéricas",  f"{len(num_df.columns)}")
    c4.metric("🔤 Categ.",     f"{len(cat_df.columns)}")
    c5.metric("⚠️ Nulos",      f"{n_null:,}")

    st.markdown(f"""
    <div class="info-box">
    <span class="pill">🧬 Duplicados: {n_dup}</span>
    <span class="pill">💾 Memoria: {mem_kb:.1f} KB</span>
    <span class="pill">📄 Archivo: {uploaded.name}</span>
    </div>""", unsafe_allow_html=True)

    n_prev = st.slider("Filas a mostrar", 3, 30, 8, key="prev")
    st.dataframe(df.head(n_prev), use_container_width=True, height=320)

    with st.expander("📊 Estadísticas descriptivas"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with st.expander("🔍 Tipos, nulos y únicos por columna"):
        meta = pd.DataFrame({
            "Tipo"   : df.dtypes.astype(str),
            "No Nulos": df.notnull().sum(),
            "Nulos"  : df.isnull().sum(),
            "% Nulos": (df.isnull().sum()/n_rows*100).round(2),
            "Únicos" : df.nunique(),
        })
        st.dataframe(meta, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 · DISTRIBUCIONES (Histogramas + KDE)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="section-title">📈 Histogramas con curva de densidad (KDE)</p>', unsafe_allow_html=True)
    if len(num_df.columns) == 0:
        st.warning("No hay columnas numéricas."); st.stop()

    sel_hist = st.multiselect("Columnas:", num_df.columns.tolist(),
                               default=num_df.columns.tolist()[:min(6,len(num_df.columns))],
                               key="sel_hist")
    if not sel_hist:
        st.info("Selecciona al menos una columna.")
    else:
        nc = min(3, len(sel_hist))
        nr = (len(sel_hist)+nc-1)//nc
        fig, axes = plt.subplots(nr, nc, figsize=(6*nc, 3.8*nr))
        fig.patch.set_facecolor(DARK)
        fig.subplots_adjust(hspace=0.55, wspace=0.35)
        axf = np.array(axes).flatten()
        for i, col in enumerate(sel_hist):
            ax = axf[i]; ax.set_facecolor(CARD)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            s = df[col].dropna()
            color = PALETTE[i % len(PALETTE)]
            n_b, edges, _ = ax.hist(s, bins=bins_n, color=color, alpha=0.8,
                                     edgecolor=DARK, linewidth=0.4)
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(s, bw_method=0.4)
                xs  = np.linspace(s.min(), s.max(), 300)
                ax.plot(xs, kde(xs)*len(s)*(edges[1]-edges[0]),
                        color="white", lw=1.8, alpha=0.9, label="KDE")
            except Exception: pass
            ax.axvline(s.mean(),   color=ORANGE, lw=1.5, ls="--", label=f"μ={s.mean():.2f}")
            ax.axvline(s.median(), color=GREEN,  lw=1.5, ls=":",  label=f"Md={s.median():.2f}")
            title_ax(ax, col)
            xlabel(ax,"Valor"); ylabel(ax,"Frecuencia")
            ax.tick_params(colors=DIM, labelsize=7)
            ax.legend(fontsize=6, framealpha=0.15, labelcolor=TEXT)
        for j in range(i+1, len(axf)): axf[j].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 · BOXPLOT + VIOLIN
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-title">📦 Boxplot & 🎻 Violin Plot</p>', unsafe_allow_html=True)
    if len(num_df.columns) == 0:
        st.warning("No hay columnas numéricas.")
    else:
        sel_bv = st.multiselect("Columnas:", num_df.columns.tolist(),
                                 default=num_df.columns.tolist()[:min(6,len(num_df.columns))],
                                 key="sel_bv")
        if not sel_bv:
            st.info("Selecciona al menos una columna.")
        else:
            data_bv = [df[c].dropna().values for c in sel_bv]

            # ── Boxplot ──
            st.markdown("#### 📦 Boxplot — outliers como puntos rojos")
            fig_b, ax_b = dark_fig(w=max(8, len(sel_bv)*1.4), h=4.5)
            bp = ax_b.boxplot(data_bv, patch_artist=True,
                medianprops=dict(color=ORANGE,linewidth=2),
                whiskerprops=dict(color=DIM),
                capprops=dict(color=DIM),
                flierprops=dict(marker="o",color=RED,alpha=0.5,markersize=3))
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color); patch.set_alpha(0.55)
            ax_b.set_xticks(range(1,len(sel_bv)+1))
            ax_b.set_xticklabels(sel_bv, rotation=30, ha="right", fontsize=9, color=DIM)
            ax_b.yaxis.grid(True, ls="--", alpha=0.25, color=BORDER)
            title_ax(ax_b,"Distribución · Outliers en rojo",12)
            plt.tight_layout(); st.pyplot(fig_b, use_container_width=True)

            # ── Violin ──
            st.markdown("#### 🎻 Violin Plot — forma de la distribución")
            fig_v, ax_v = dark_fig(w=max(8, len(sel_bv)*1.4), h=4.5)
            parts = ax_v.violinplot(data_bv, showmedians=True, showextrema=True)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(PALETTE[i % len(PALETTE)])
                pc.set_edgecolor("white"); pc.set_alpha(0.6)
            parts["cmedians"].set_color(ORANGE); parts["cmedians"].set_linewidth(2)
            parts["cmins"].set_color(DIM);   parts["cmaxes"].set_color(DIM)
            parts["cbars"].set_color(DIM)
            ax_v.set_xticks(range(1,len(sel_bv)+1))
            ax_v.set_xticklabels(sel_bv, rotation=30, ha="right", fontsize=9, color=DIM)
            ax_v.yaxis.grid(True, ls="--", alpha=0.25, color=BORDER)
            title_ax(ax_v,"Forma de la distribución · Mediana en naranja",12)
            plt.tight_layout(); st.pyplot(fig_v, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 · CORRELACIÓN
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="section-title">🔗 Matriz de Correlación</p>', unsafe_allow_html=True)
    num_cols = num_df.columns.tolist()
    if len(num_cols) < 2:
        st.warning("Se necesitan al menos 2 columnas numéricas.")
    else:
        sel_corr = st.multiselect("Columnas:", num_cols,
                                   default=num_cols[:min(8,len(num_cols))], key="sel_corr")
        if len(sel_corr) < 2:
            st.info("Selecciona al menos 2 columnas.")
        else:
            corr = df[sel_corr].corr(method=corr_method)
            data_c = corr.values; n = len(sel_corr)

            fig_c, ax_c = plt.subplots(figsize=(max(6,n*1.2), max(5,n*1.1)))
            fig_c.patch.set_facecolor(DARK); ax_c.set_facecolor(CARD)
            im = ax_c.imshow(data_c, cmap=cmap_corr, vmin=-1, vmax=1, aspect="auto")

            ax_c.set_xticks(np.arange(n+1)-0.5, minor=True)
            ax_c.set_yticks(np.arange(n+1)-0.5, minor=True)
            ax_c.grid(which="minor", color=BORDER, linewidth=0.8)
            ax_c.tick_params(which="minor", bottom=False, left=False)
            ax_c.set_xticks(range(n)); ax_c.set_yticks(range(n))
            ax_c.set_xticklabels(sel_corr, rotation=38, ha="right", fontsize=9, color=DIM)
            ax_c.set_yticklabels(sel_corr, fontsize=9, color=DIM)

            if annot_corr:
                for i in range(n):
                    for j in range(n):
                        v = data_c[i,j]
                        fc = "white" if abs(v)>0.55 else DIM
                        ax_c.text(j,i,f"{v:.{fmt_dec}f}", ha="center", va="center",
                                  fontsize=8.5, color=fc,
                                  fontweight="bold" if abs(v)>0.7 else "normal")
            for k in range(n):
                ax_c.add_patch(plt.Rectangle((k-.5,k-.5),1,1,fill=False,edgecolor=BLUE,lw=2))
            cbar = fig_c.colorbar(im, ax=ax_c, shrink=0.75)
            cbar.ax.tick_params(colors=DIM, labelsize=8)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=DIM)
            cbar.set_label("Coeficiente de correlación", color=DIM, fontsize=9)
            ax_c.set_title(f"Correlación · {corr_method.capitalize()}",
                           fontsize=13, fontweight="bold", color=TEXT, pad=14)
            plt.tight_layout(); st.pyplot(fig_c, use_container_width=True)

            with st.expander("📋 Ranking de pares"):
                pairs = [{"A":sel_corr[i],"B":sel_corr[j],
                          "r":round(corr.iloc[i,j],4),
                          "│r│":abs(round(corr.iloc[i,j],4))}
                         for i in range(len(sel_corr)) for j in range(i+1,len(sel_corr))]
                pdf = pd.DataFrame(pairs).sort_values("│r│",ascending=False)
                pdf["Tipo"] = pdf["r"].apply(
                    lambda v:"🟢 Pos. fuerte" if v>0.7 else
                             "🔵 Pos. mod."   if v>0.4 else
                             "🔴 Neg. fuerte" if v<-0.7 else
                             "🟠 Neg. mod."   if v<-0.4 else "⚪ Débil")
                st.dataframe(pdf.drop(columns="│r│"), use_container_width=True)

            with st.expander("ℹ️ Guía de interpretación"):
                st.markdown("""
| Rango | Tipo |
|:---:|:---|
| 0.9–1.0 | 🟢 Positiva muy fuerte |
| 0.7–0.9 | 🟢 Positiva fuerte |
| 0.4–0.7 | 🔵 Positiva moderada |
| 0–0.4   | ⚪ Débil |
| -0.4–-0.7 | 🟠 Negativa moderada |
| -0.7–-1.0 | 🔴 Negativa fuerte |
                """)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 · SCATTER PLOT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="section-title">🔵 Scatter Plot — Relación entre dos variables</p>', unsafe_allow_html=True)
    num_cols = num_df.columns.tolist()
    if len(num_cols) < 2:
        st.warning("Se necesitan al menos 2 columnas numéricas.")
    else:
        col_sc1, col_sc2, col_sc3 = st.columns(3)
        x_col = col_sc1.selectbox("Eje X", num_cols, index=0, key="sc_x")
        y_col = col_sc2.selectbox("Eje Y", num_cols, index=min(1,len(num_cols)-1), key="sc_y")
        color_col = col_sc3.selectbox("Color por (opcional)", ["— ninguno —"]+cat_df.columns.tolist(), key="sc_col")
        show_reg = st.checkbox("Mostrar línea de regresión", True, key="sc_reg")

        sx = df[x_col].dropna(); sy = df[y_col].dropna()
        idx = sx.index.intersection(sy.index)
        sx, sy = df.loc[idx, x_col], df.loc[idx, y_col]

        fig_sc, ax_sc = dark_fig(w=9, h=5.5)
        if color_col != "— ninguno —" and color_col in df.columns:
            groups = df.loc[idx, color_col].astype(str)
            uniq   = groups.unique()
            for k, g in enumerate(uniq):
                mask = groups == g
                ax_sc.scatter(sx[mask], sy[mask], color=PALETTE[k%len(PALETTE)],
                               alpha=0.65, s=28, label=g, edgecolors="none")
            ax_sc.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT,
                         loc="best", ncol=min(3,len(uniq)))
        else:
            # density-colored scatter
            from numpy.random import default_rng
            ax_sc.scatter(sx, sy, color=BLUE, alpha=0.55, s=22, edgecolors="none")

        if show_reg:
            try:
                m, b = np.polyfit(sx, sy, 1)
                xs_r = np.linspace(sx.min(), sx.max(), 200)
                ax_sc.plot(xs_r, m*xs_r+b, color=ORANGE, lw=2, ls="--", label=f"y={m:.2f}x+{b:.2f}")
                # Pearson r
                r = np.corrcoef(sx, sy)[0,1]
                ax_sc.text(0.04, 0.95, f"r = {r:.3f}", transform=ax_sc.transAxes,
                           fontsize=10, color=ORANGE, va="top",
                           bbox=dict(boxstyle="round,pad=0.3", fc=CARD, ec=BORDER))
                ax_sc.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT)
            except Exception: pass

        title_ax(ax_sc, f"{x_col}  vs  {y_col}", 12)
        xlabel(ax_sc, x_col); ylabel(ax_sc, y_col)
        ax_sc.grid(True, ls="--", alpha=0.2, color=BORDER)
        plt.tight_layout(); st.pyplot(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 · PAIR PLOT (scatter matrix)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="section-title">🧩 Pair Plot — Matriz de dispersión</p>', unsafe_allow_html=True)
    num_cols = num_df.columns.tolist()
    if len(num_cols) < 2:
        st.warning("Se necesitan al menos 2 columnas numéricas.")
    else:
        sel_pair = st.multiselect("Columnas (máx. 5 recomendado):", num_cols,
                                   default=num_cols[:min(4,len(num_cols))], key="sel_pair")
        if len(sel_pair) < 2:
            st.info("Selecciona al menos 2 columnas.")
        else:
            n_p = len(sel_pair)
            fig_p, axes_p = plt.subplots(n_p, n_p, figsize=(3*n_p, 3*n_p))
            fig_p.patch.set_facecolor(DARK)
            fig_p.subplots_adjust(hspace=0.08, wspace=0.08)

            with st.spinner("Generando pair plot…"):
                for i, ci in enumerate(sel_pair):
                    for j, cj in enumerate(sel_pair):
                        ax = axes_p[i][j]; ax.set_facecolor(CARD)
                        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
                        ax.tick_params(colors=DIM, labelsize=6)

                        if i == j:
                            # diagonal → histogram
                            s = df[ci].dropna()
                            ax.hist(s, bins=25, color=PALETTE[i%len(PALETTE)],
                                    alpha=0.8, edgecolor=DARK, linewidth=0.3)
                            ax.set_facecolor("#1a2030")
                        else:
                            sx2 = df[cj].dropna(); sy2 = df[ci].dropna()
                            idx2 = sx2.index.intersection(sy2.index)
                            ax.scatter(df.loc[idx2,cj], df.loc[idx2,ci],
                                       s=6, alpha=0.45, color=BLUE, edgecolors="none")
                            try:
                                m2,b2 = np.polyfit(df.loc[idx2,cj], df.loc[idx2,ci],1)
                                xs2   = np.linspace(df.loc[idx2,cj].min(), df.loc[idx2,cj].max(),100)
                                ax.plot(xs2, m2*xs2+b2, color=ORANGE, lw=1, alpha=0.8)
                            except Exception: pass

                        if i == n_p-1: ax.set_xlabel(cj, fontsize=7, color=DIM)
                        if j == 0:     ax.set_ylabel(ci, fontsize=7, color=DIM)
                        if i != n_p-1: ax.set_xticklabels([])
                        if j != 0:     ax.set_yticklabels([])

            fig_p.suptitle("Pair Plot — Diagonal: distribución · Off-diagonal: scatter + regresión",
                           fontsize=11, color=TEXT, y=1.01)
            st.pyplot(fig_p, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 · CATEGÓRICAS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<p class="section-title">🥧 Variables Categóricas</p>', unsafe_allow_html=True)
    if len(cat_df.columns) == 0:
        st.info("No hay columnas categóricas en el dataset.")
    else:
        sel_cat = st.multiselect("Columnas categóricas:",
                                  cat_df.columns.tolist(),
                                  default=cat_df.columns.tolist()[:min(4,len(cat_df.columns))],
                                  key="sel_cat")
        top_n = st.slider("Top N valores", 5, 20, 10, key="top_n")
        chart_type = st.radio("Tipo de gráfico", ["Barras horizontales","Donut/Pie"], horizontal=True)

        if not sel_cat:
            st.info("Selecciona al menos una columna.")
        else:
            for col in sel_cat:
                vc = df[col].value_counts().head(top_n)
                st.markdown(f"##### 🏷️ `{col}`  — {df[col].nunique()} valores únicos")

                if chart_type == "Barras horizontales":
                    fig_cat, ax_cat = dark_fig(w=9, h=max(3, len(vc)*0.45))
                    cols_cat = [PALETTE[i%len(PALETTE)] for i in range(len(vc))]
                    bars = ax_cat.barh(vc.index.astype(str)[::-1],
                                       vc.values[::-1], color=cols_cat[::-1],
                                       edgecolor=DARK, linewidth=0.4)
                    for bar, val in zip(bars, vc.values[::-1]):
                        ax_cat.text(val+vc.max()*0.01, bar.get_y()+bar.get_height()/2,
                                    f"{val} ({val/n_rows*100:.1f}%)",
                                    va="center", fontsize=8, color=TEXT)
                    xlabel(ax_cat,"Frecuencia"); title_ax(ax_cat,col)
                    ax_cat.tick_params(colors=DIM, labelsize=8)
                    ax_cat.xaxis.grid(True, ls="--", alpha=0.2, color=BORDER)
                    plt.tight_layout(); st.pyplot(fig_cat, use_container_width=True)

                else:
                    fig_d, ax_d = plt.subplots(figsize=(7,4.5))
                    fig_d.patch.set_facecolor(DARK); ax_d.set_facecolor(DARK)
                    wedges, texts, autotexts = ax_d.pie(
                        vc.values, labels=None,
                        colors=[PALETTE[i%len(PALETTE)] for i in range(len(vc))],
                        autopct="%1.1f%%", startangle=140,
                        wedgeprops=dict(width=0.55, edgecolor=DARK, linewidth=1.5),
                        pctdistance=0.78)
                    for at in autotexts: at.set_color("white"); at.set_fontsize(8)
                    ax_d.legend(wedges, vc.index.astype(str), fontsize=8,
                                loc="center left", bbox_to_anchor=(1,0.5),
                                framealpha=0.1, labelcolor=TEXT)
                    ax_d.set_title(col, fontsize=12, fontweight="bold", color=TEXT)
                    plt.tight_layout(); st.pyplot(fig_d, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 7 · MAPA DE NULOS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<p class="section-title">🕳️ Análisis de Valores Nulos</p>', unsafe_allow_html=True)
    null_pct = (df.isnull().sum()/n_rows*100).sort_values(ascending=False)

    # Barra de nulos por columna
    fig_n, ax_n = dark_fig(w=10, h=max(3, len(null_pct)*0.38))
    cols_null = [RED if p>30 else ORANGE if p>10 else BLUE for p in null_pct]
    bars_n = ax_n.barh(null_pct.index[::-1], null_pct.values[::-1],
                        color=cols_null[::-1], edgecolor=DARK, linewidth=0.4)
    for bar, val in zip(bars_n, null_pct.values[::-1]):
        ax_n.text(val+0.3, bar.get_y()+bar.get_height()/2,
                  f"{val:.1f}%", va="center", fontsize=8, color=TEXT)
    xlabel(ax_n,"% de nulos"); title_ax(ax_n,"Nulos por columna",12)
    ax_n.tick_params(colors=DIM, labelsize=8)
    ax_n.xaxis.grid(True, ls="--", alpha=0.2, color=BORDER)
    legend_el = [mpatches.Patch(color=RED,label=">30%"),
                 mpatches.Patch(color=ORANGE,label="10-30%"),
                 mpatches.Patch(color=BLUE,label="<10%")]
    ax_n.legend(handles=legend_el, fontsize=8, framealpha=0.2, labelcolor=TEXT)
    plt.tight_layout(); st.pyplot(fig_n, use_container_width=True)

    # Heatmap visual de nulos (mapa de bits)
    st.markdown("#### 🗺️ Mapa de bits — presencia/ausencia de datos")
    sample_size = min(500, n_rows)
    df_sample   = df.sample(sample_size, random_state=42) if n_rows > sample_size else df
    null_matrix = df_sample.isnull().astype(int).values

    fig_nm, ax_nm = plt.subplots(figsize=(max(8,n_cols*0.8), 4.5))
    fig_nm.patch.set_facecolor(DARK); ax_nm.set_facecolor(CARD)
    ax_nm.imshow(null_matrix.T, cmap="RdYlGn_r", aspect="auto",
                 interpolation="nearest", vmin=0, vmax=1)
    ax_nm.set_yticks(range(n_cols))
    ax_nm.set_yticklabels(df.columns, fontsize=8, color=DIM)
    ax_nm.set_xlabel(f"Filas (muestra de {sample_size})", fontsize=8, color=DIM)
    ax_nm.set_title("Verde = dato presente · Rojo = nulo", fontsize=11, color=TEXT, pad=8)
    plt.tight_layout(); st.pyplot(fig_nm, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 8 · TENDENCIAS (Line / Bar acumulado)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown('<p class="section-title">📉 Tendencias & Series</p>', unsafe_allow_html=True)
    num_cols = num_df.columns.tolist()
    if len(num_cols) == 0:
        st.warning("No hay columnas numéricas.")
    else:
        col_t1, col_t2 = st.columns(2)
        y_trend = col_t1.multiselect("Variables Y:",num_cols,
                                      default=num_cols[:min(2,len(num_cols))], key="t_y")
        x_trend = col_t2.selectbox("Eje X (índice o columna):",
                                    ["— índice de fila —"]+df.columns.tolist(), key="t_x")
        roll_w  = st.slider("Ventana media móvil (0 = sin media móvil)", 0, 50, 0, key="roll")
        chart_kind = st.radio("Tipo", ["Líneas","Área apilada","Barras agrupadas"], horizontal=True)

        if not y_trend:
            st.info("Selecciona al menos una variable Y.")
        else:
            x_vals = df.index if x_trend == "— índice de fila —" else df[x_trend]
            fig_t, ax_t = dark_fig(w=11, h=4.8)

            for k, col in enumerate(y_trend):
                ys = df[col].values.astype(float)
                color = PALETTE[k%len(PALETTE)]
                if chart_kind == "Líneas":
                    ax_t.plot(x_vals, ys, color=color, lw=1.8, alpha=0.9, label=col)
                    if roll_w > 1:
                        roll = pd.Series(ys).rolling(roll_w, min_periods=1).mean()
                        ax_t.plot(x_vals, roll, color="white", lw=1.2, ls="--",
                                  alpha=0.7, label=f"{col} MA{roll_w}")
                    ax_t.fill_between(x_vals, ys, alpha=0.08, color=color)
                elif chart_kind == "Área apilada":
                    ax_t.fill_between(x_vals, ys, alpha=0.4, color=color, label=col)
                    ax_t.plot(x_vals, ys, color=color, lw=1, alpha=0.6)
                else:
                    w_bar = 0.8/len(y_trend)
                    offset = (k - len(y_trend)/2)*w_bar + w_bar/2
                    ax_t.bar(np.arange(len(ys))+offset, ys, width=w_bar*0.9,
                             color=color, alpha=0.8, label=col, edgecolor=DARK, linewidth=0.3)

            ax_t.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT)
            ax_t.grid(True, ls="--", alpha=0.2, color=BORDER)
            xlabel(ax_t, x_trend if x_trend != "— índice de fila —" else "Índice")
            ylabel(ax_t, "Valor")
            title_ax(ax_t, f"Tendencia: {', '.join(y_trend)}", 12)
            plt.tight_layout(); st.pyplot(fig_t, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<p class="section-title">⬇️ Descargas</p>', unsafe_allow_html=True)
d1,d2,d3 = st.columns(3)
num_cols_dl = num_df.columns.tolist()
with d1:
    st.download_button("📥 Dataset completo (.csv)",
        df.to_csv(index=False).encode(), "dataset.csv","text/csv",use_container_width=True)
with d2:
    st.download_button("📥 Estadísticas descriptivas (.csv)",
        df.describe(include="all").T.to_csv().encode(), "estadisticas.csv","text/csv",use_container_width=True)
with d3:
    if len(num_cols_dl) >= 2:
        corr_dl = df[num_cols_dl].corr(method=corr_method)
        st.download_button("📥 Matriz correlación (.csv)",
            corr_dl.to_csv().encode(), "correlacion.csv","text/csv",use_container_width=True)

st.markdown("<div style='text-align:center;color:#30363d;font-size:0.8rem;margin-top:20px;'>EDA Pro · Streamlit + Matplotlib</div>",
            unsafe_allow_html=True)
