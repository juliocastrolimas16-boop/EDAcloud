import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
/* ── General ── */
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
h1, h2, h3, h4                     { color: #e6edf3 !important; }
p, li, label                       { color: #8b949e !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 20px;
}

/* ── Section headers ── */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #58a6ff !important;
    border-left: 4px solid #58a6ff;
    padding-left: 12px;
    margin: 28px 0 16px 0;
}

/* ── Info boxes ── */
.info-box {
    background: #1c2230;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    color: #c9d1d9 !important;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a2744 0%, #0d1117 60%, #1a1a2e 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 24px;
    text-align: center;
}
.hero h1 { font-size: 2.4rem !important; color: #58a6ff !important; margin: 0 0 8px 0; }
.hero p  { font-size: 1.05rem !important; color: #8b949e !important; margin: 0; }

/* ── Stat pill ── */
.pill {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: #58a6ff !important;
    margin: 3px;
}

/* ── Divider ── */
hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
BORDER    = "#30363d"
BLUE      = "#58a6ff"
GREEN     = "#3fb950"
ORANGE    = "#f0883e"
RED       = "#f85149"
TEXT      = "#c9d1d9"
TEXT_DIM  = "#8b949e"

def styled_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT_DIM)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_color(TEXT_DIM)
    ax.title.set_color(TEXT)
    return fig, ax

def multi_fig(rows, cols, w=14, h=5):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat:
        ax.set_facecolor(CARD_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(colors=TEXT_DIM)
    return fig, axes

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
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")
    sep = st.selectbox("🔤 Separador CSV", [",", ";", "\t", "|"],
                       format_func=lambda x: {"," : "Coma (,)",
                                              ";" : "Punto y coma (;)",
                                              "\t": "Tabulación",
                                              "|" : "Pipe (|)"}[x])
    st.markdown("---")
    st.markdown("### 🎨 Correlación")
    method = st.selectbox("Método", ["pearson", "spearman", "kendall"],
                          help="Pearson=lineal · Spearman/Kendall=rangos")
    cmap_choice = st.selectbox("Paleta", ["coolwarm","RdBu_r","viridis","plasma","magma"])
    annot_vals  = st.checkbox("Mostrar valores", value=True)
    fmt_dec     = st.slider("Decimales", 1, 3, 2)
    st.markdown("---")
    st.markdown("### 📊 Distribuciones")
    bins_n = st.slider("Bins del histograma", 10, 80, 30)
    st.markdown("---")
    st.caption("EDA Pro · Hecho con Streamlit")

# ══════════════════════════════════════════════════════════════════════════════
#  FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
uploaded = st.file_uploader("📂  Arrastra o selecciona tu archivo CSV",
                            type=["csv"],
                            help="Máx. 200 MB · Formato CSV")

if uploaded is None:
    st.markdown("""
    <div class="info-box">
    ℹ️  Sube un archivo <b>CSV</b> usando el botón de arriba para comenzar el análisis.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════════════════════
try:
    df = pd.read_csv(uploaded, sep=sep)
except Exception as e:
    st.error(f"❌ Error al leer el archivo: {e}")
    st.stop()

num_df  = df.select_dtypes(include="number")
cat_df  = df.select_dtypes(include=["object", "category"])
n_rows, n_cols = df.shape
n_null  = int(df.isnull().sum().sum())
n_dup   = int(df.duplicated().sum())
mem_kb  = df.memory_usage(deep=True).sum() / 1024

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 · RESUMEN GENERAL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">📋 Resumen del Dataset</p>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🗂️ Filas",       f"{n_rows:,}")
c2.metric("📐 Columnas",    f"{n_cols}")
c3.metric("🔢 Numéricas",   f"{len(num_df.columns)}")
c4.metric("🔤 Categóricas", f"{len(cat_df.columns)}")
c5.metric("⚠️ Nulos",       f"{n_null:,}")

st.markdown(f"""
<div class="info-box">
<span class="pill">🧬 Filas duplicadas: {n_dup}</span>
<span class="pill">💾 Memoria: {mem_kb:.1f} KB</span>
<span class="pill">📄 Archivo: {uploaded.name}</span>
</div>
""", unsafe_allow_html=True)

# ── Vista previa ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">👁️ Vista Previa de Datos</p>', unsafe_allow_html=True)
n_preview = st.slider("Filas a mostrar", 3, 25, 8)
st.dataframe(df.head(n_preview), use_container_width=True, height=300)

# ── Estadísticas descriptivas ─────────────────────────────────────────────────
with st.expander("📊 Estadísticas Descriptivas completas"):
    st.dataframe(df.describe(include="all").T, use_container_width=True)

# ── Tipos y nulos ─────────────────────────────────────────────────────────────
with st.expander("🔍 Tipos de datos y valores nulos por columna"):
    meta = pd.DataFrame({
        "Tipo"     : df.dtypes.astype(str),
        "No Nulos" : df.notnull().sum(),
        "Nulos"    : df.isnull().sum(),
        "% Nulos"  : (df.isnull().sum() / n_rows * 100).round(2),
        "Únicos"   : df.nunique(),
    })
    st.dataframe(meta, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 · DISTRIBUCIONES
# ══════════════════════════════════════════════════════════════════════════════
if len(num_df.columns) > 0:
    st.markdown('<p class="section-title">📈 Distribuciones de Variables Numéricas</p>',
                unsafe_allow_html=True)

    cols_to_plot = num_df.columns.tolist()
    n_plots      = len(cols_to_plot)
    ncols_grid   = min(3, n_plots)
    nrows_grid   = (n_plots + ncols_grid - 1) // ncols_grid

    fig_dist, axes_dist = plt.subplots(
        nrows_grid, ncols_grid,
        figsize=(6 * ncols_grid, 3.8 * nrows_grid)
    )
    fig_dist.patch.set_facecolor(DARK_BG)
    fig_dist.subplots_adjust(hspace=0.55, wspace=0.35)

    axes_flat = np.array(axes_dist).flatten()

    GRAD_COLORS = [BLUE, GREEN, ORANGE, "#bc8cff", "#39d353", "#f78166"]

    for i, col in enumerate(cols_to_plot):
        ax = axes_flat[i]
        ax.set_facecolor(CARD_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

        series = df[col].dropna()
        color  = GRAD_COLORS[i % len(GRAD_COLORS)]

        # histogram
        n_b, edges, patches = ax.hist(series, bins=bins_n, color=color,
                                       alpha=0.85, edgecolor=DARK_BG, linewidth=0.4)

        # KDE overlay (manual)
        if len(series) > 5:
            from scipy.stats import gaussian_kde  # optional, skip if missing
            try:
                kde  = gaussian_kde(series, bw_method=0.4)
                xs   = np.linspace(series.min(), series.max(), 300)
                ys   = kde(xs) * len(series) * (edges[1] - edges[0])
                ax.plot(xs, ys, color="white", linewidth=1.6, alpha=0.9)
            except Exception:
                pass

        # mean / median lines
        ax.axvline(series.mean(),   color="#f0883e", linewidth=1.4,
                   linestyle="--", label=f"Media: {series.mean():.2f}")
        ax.axvline(series.median(), color="#3fb950", linewidth=1.4,
                   linestyle=":",  label=f"Mediana: {series.median():.2f}")

        ax.set_title(col, fontsize=10, fontweight="bold", color=TEXT, pad=6)
        ax.set_xlabel("Valor", fontsize=8, color=TEXT_DIM)
        ax.set_ylabel("Frecuencia", fontsize=8, color=TEXT_DIM)
        ax.tick_params(colors=TEXT_DIM, labelsize=7)
        ax.legend(fontsize=6.5, framealpha=0.2, labelcolor=TEXT)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    st.pyplot(fig_dist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 · BOXPLOTS
# ══════════════════════════════════════════════════════════════════════════════
if len(num_df.columns) > 0:
    st.markdown('<p class="section-title">📦 Boxplots · Detección de Outliers</p>',
                unsafe_allow_html=True)

    fig_box, ax_box = styled_fig(w=max(8, len(num_df.columns) * 1.4), h=5)

    bp = ax_box.boxplot(
        [num_df[c].dropna().values for c in num_df.columns],
        patch_artist=True,
        medianprops=dict(color=ORANGE, linewidth=2),
        whiskerprops=dict(color=TEXT_DIM),
        capprops=dict(color=TEXT_DIM),
        flierprops=dict(marker="o", color=RED, alpha=0.5, markersize=3),
    )
    box_colors = [BLUE, GREEN, ORANGE, "#bc8cff", "#39d353", "#f78166"]
    for patch, color in zip(bp["boxes"], box_colors * 10):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax_box.set_xticks(range(1, len(num_df.columns) + 1))
    ax_box.set_xticklabels(num_df.columns, rotation=30, ha="right",
                            fontsize=9, color=TEXT_DIM)
    ax_box.set_title("Distribución por columna · Outliers en rojo",
                     fontsize=12, color=TEXT, pad=10)
    ax_box.yaxis.grid(True, linestyle="--", alpha=0.3, color=BORDER)
    plt.tight_layout()
    st.pyplot(fig_box, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 · MAPA DE NULOS
# ══════════════════════════════════════════════════════════════════════════════
null_counts = df.isnull().sum()
cols_with_nulls = null_counts[null_counts > 0]

if len(cols_with_nulls) > 0:
    st.markdown('<p class="section-title">🕳️ Mapa de Valores Nulos</p>',
                unsafe_allow_html=True)

    fig_null, ax_null = styled_fig(w=10, h=3.5)
    pcts = (cols_with_nulls / n_rows * 100).sort_values(ascending=True)
    colors_null = [RED if p > 30 else ORANGE if p > 10 else BLUE for p in pcts]
    bars = ax_null.barh(pcts.index, pcts.values, color=colors_null,
                         edgecolor=DARK_BG, linewidth=0.5)
    for bar, val in zip(bars, pcts.values):
        ax_null.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=8, color=TEXT)
    ax_null.set_xlabel("% de nulos", color=TEXT_DIM)
    ax_null.set_title("Columnas con valores faltantes", color=TEXT, fontsize=12)
    ax_null.tick_params(colors=TEXT_DIM)
    legend_elements = [
        mpatches.Patch(color=RED,    label="> 30% nulos"),
        mpatches.Patch(color=ORANGE, label="10-30% nulos"),
        mpatches.Patch(color=BLUE,   label="< 10% nulos"),
    ]
    ax_null.legend(handles=legend_elements, fontsize=8,
                   framealpha=0.2, labelcolor=TEXT, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig_null, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 · CORRELACIÓN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">🔗 Matriz de Correlación</p>',
            unsafe_allow_html=True)

num_cols = num_df.columns.tolist()

if len(num_cols) < 2:
    st.warning("⚠️ Se necesitan al menos 2 columnas numéricas para calcular correlaciones.")
    st.stop()

selected = st.multiselect(
    "Selecciona las columnas a incluir:",
    options=num_cols,
    default=num_cols[:min(8, len(num_cols))],
)

if len(selected) < 2:
    st.warning("⚠️ Selecciona al menos 2 columnas.")
    st.stop()

corr = df[selected].corr(method=method)
data = corr.values
n    = len(selected)

fig_corr, ax_corr = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.1)))
fig_corr.patch.set_facecolor(DARK_BG)
ax_corr.set_facecolor(CARD_BG)

im = ax_corr.imshow(data, cmap=cmap_choice, vmin=-1, vmax=1, aspect="auto")

# grid lines
ax_corr.set_xticks(np.arange(n + 1) - 0.5, minor=True)
ax_corr.set_yticks(np.arange(n + 1) - 0.5, minor=True)
ax_corr.grid(which="minor", color=BORDER, linewidth=0.8)
ax_corr.tick_params(which="minor", bottom=False, left=False)

# labels
ax_corr.set_xticks(range(n))
ax_corr.set_yticks(range(n))
ax_corr.set_xticklabels(selected, rotation=38, ha="right", fontsize=9, color=TEXT_DIM)
ax_corr.set_yticklabels(selected, fontsize=9, color=TEXT_DIM)

# annotations
if annot_vals:
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            # background circle for emphasis
            circle = plt.Circle((j, i), 0.38, color="white", alpha=0.05)
            ax_corr.add_patch(circle)
            fc = "white" if abs(val) > 0.55 else TEXT_DIM
            weight = "bold" if abs(val) > 0.7 else "normal"
            ax_corr.text(j, i, f"{val:.{fmt_dec}f}",
                         ha="center", va="center",
                         fontsize=8.5, color=fc, fontweight=weight)

# diagonal highlight
for k in range(n):
    rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                          fill=False, edgecolor=BLUE, linewidth=2)
    ax_corr.add_patch(rect)

# colorbar
cbar = fig_corr.colorbar(im, ax=ax_corr, shrink=0.75, pad=0.02)
cbar.ax.tick_params(colors=TEXT_DIM, labelsize=8)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_DIM)
cbar.set_label("Coeficiente de correlación", color=TEXT_DIM, fontsize=9)
cbar.ax.set_facecolor(CARD_BG)

ax_corr.set_title(f"Matriz de Correlación · Método: {method.capitalize()}",
                  fontsize=13, fontweight="bold", color=TEXT, pad=14)
plt.tight_layout()
st.pyplot(fig_corr, use_container_width=True)

# ── Tabla ranking de correlaciones ────────────────────────────────────────────
with st.expander("📋 Ranking de pares más correlacionados"):
    pairs = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            pairs.append({
                "Variable A": selected[i],
                "Variable B": selected[j],
                "Correlación": round(corr.iloc[i, j], 4),
                "Fuerza": abs(round(corr.iloc[i, j], 4)),
            })
    pairs_df = pd.DataFrame(pairs).sort_values("Fuerza", ascending=False).drop(columns="Fuerza")
    pairs_df["Tipo"] = pairs_df["Correlación"].apply(
        lambda v: "🟢 Positiva fuerte" if v > 0.7
             else "🔵 Positiva moderada" if v > 0.4
             else "🔴 Negativa fuerte" if v < -0.7
             else "🟠 Negativa moderada" if v < -0.4
             else "⚪ Débil / Sin correlación"
    )
    st.dataframe(pairs_df, use_container_width=True)

# ── Guía de interpretación ────────────────────────────────────────────────────
with st.expander("ℹ️ Guía de interpretación"):
    st.markdown("""
| Rango | Tipo | Descripción |
|:---:|:---:|:---|
| **0.9 – 1.0** | 🟢 Positiva muy fuerte | Las variables crecen juntas casi perfectamente |
| **0.7 – 0.9** | 🟢 Positiva fuerte | Relación directa clara |
| **0.4 – 0.7** | 🔵 Positiva moderada | Tendencia positiva notable |
| **0.1 – 0.4** | ⚪ Débil | Poca relación lineal |
| **~ 0**        | ⚪ Nula | Sin relación lineal aparente |
| **-0.1 – -0.4**| ⚪ Negativa débil | Leve tendencia inversa |
| **-0.4 – -0.7**| 🟠 Negativa moderada | Relación inversa notable |
| **-0.7 – -1.0**| 🔴 Negativa fuerte | A medida que sube una, baja la otra |

**Métodos:**
- **Pearson**: mide relaciones *lineales*. Ideal para datos normalmente distribuidos.
- **Spearman**: basado en rangos, robusto ante outliers y no linealidades.
- **Kendall**: más conservador; útil con muestras pequeñas.
    """)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 · VARIABLES CATEGÓRICAS
# ══════════════════════════════════════════════════════════════════════════════
if len(cat_df.columns) > 0:
    st.markdown('<p class="section-title">🏷️ Variables Categóricas · Top Valores</p>',
                unsafe_allow_html=True)

    cat_cols_show = cat_df.columns.tolist()[:6]   # limit to 6
    ncols_cat = min(2, len(cat_cols_show))
    nrows_cat = (len(cat_cols_show) + 1) // 2

    fig_cat, axes_cat = plt.subplots(nrows_cat, ncols_cat,
                                     figsize=(7 * ncols_cat, 3.5 * nrows_cat))
    fig_cat.patch.set_facecolor(DARK_BG)
    fig_cat.subplots_adjust(hspace=0.6, wspace=0.4)
    axes_cat_flat = np.array(axes_cat).flatten()

    for i, col in enumerate(cat_cols_show):
        ax = axes_cat_flat[i]
        ax.set_facecolor(CARD_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        top10 = df[col].value_counts().head(10)
        colors_cat = plt.cm.get_cmap("cool")(np.linspace(0.3, 0.9, len(top10)))
        bars = ax.barh(top10.index.astype(str)[::-1],
                       top10.values[::-1],
                       color=colors_cat[::-1], edgecolor=DARK_BG, linewidth=0.4)
        for bar, val in zip(bars, top10.values[::-1]):
            ax.text(val + top10.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=7.5, color=TEXT_DIM)
        ax.set_title(f"{col}  ({df[col].nunique()} únicos)",
                     fontsize=9.5, fontweight="bold", color=TEXT, pad=6)
        ax.tick_params(colors=TEXT_DIM, labelsize=7.5)

    for j in range(i + 1, len(axes_cat_flat)):
        axes_cat_flat[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig_cat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER · DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<p class="section-title">⬇️ Descargas</p>', unsafe_allow_html=True)

d1, d2, d3 = st.columns(3)
with d1:
    st.download_button(
        "📥 Matriz de correlación (.csv)",
        data=corr.to_csv().encode("utf-8"),
        file_name="correlacion.csv",
        mime="text/csv",
        use_container_width=True,
    )
with d2:
    st.download_button(
        "📥 Estadísticas descriptivas (.csv)",
        data=df.describe(include="all").T.to_csv().encode("utf-8"),
        file_name="estadisticas.csv",
        mime="text/csv",
        use_container_width=True,
    )
with d3:
    st.download_button(
        "📥 Dataset limpio (.csv)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("""
<div style='text-align:center; color:#30363d; font-size:0.8rem; margin-top:20px;'>
EDA Pro · Construido con Streamlit & Matplotlib
</div>
""", unsafe_allow_html=True)
