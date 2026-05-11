import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Exploración de Datos CSV",
    page_icon="📊",
    layout="wide",
)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📊 Exploración de Datos CSV")
st.markdown(
    "Sube un archivo **CSV**, revisa los datos y genera un "
    "**diagrama de correlación** personalizado entre las columnas que elijas."
)
st.divider()

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Sube tu archivo CSV",
    type=["csv"],
    help="Solo se aceptan archivos con extensión .csv",
)

if uploaded_file is None:
    st.info("👆 Sube un archivo CSV para comenzar.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    sep = st.session_state.get("sep", ",")
    return pd.read_csv(file, sep=sep)

with st.sidebar:
    st.header("⚙️ Opciones de carga")
    sep = st.selectbox("Separador", [",", ";", "\t", "|"], index=0)
    st.session_state["sep"] = sep

try:
    df = pd.read_csv(uploaded_file, sep=sep)
except Exception as e:
    st.error(f"❌ Error al leer el archivo: {e}")
    st.stop()

# ── Section 1 · Data preview ──────────────────────────────────────────────────
st.subheader("✅ Datos cargados correctamente")

col1, col2, col3 = st.columns(3)
col1.metric("Filas", f"{df.shape[0]:,}")
col2.metric("Columnas", df.shape[1])
col3.metric("Valores nulos", int(df.isnull().sum().sum()))

st.markdown("**Vista previa — `df.head()`**")
n_rows = st.slider("Número de filas a mostrar", 3, 20, 5)
st.dataframe(df.head(n_rows), use_container_width=True)

with st.expander("📋 Tipos de datos por columna"):
    dtype_df = pd.DataFrame(
        {"Tipo": df.dtypes.astype(str), "Nulos": df.isnull().sum()}
    )
    st.dataframe(dtype_df, use_container_width=True)

st.divider()

# ── Section 2 · Correlation diagram ──────────────────────────────────────────
st.subheader("🔗 Diagrama de Correlación")

# Only numeric columns are valid for correlation
numeric_cols = df.select_dtypes(include="number").columns.tolist()

if len(numeric_cols) < 2:
    st.warning(
        "⚠️ El archivo debe tener al menos **2 columnas numéricas** "
        "para calcular correlaciones."
    )
    st.stop()

with st.sidebar:
    st.header("🎛️ Configuración de correlación")
    selected_cols = st.multiselect(
        "Columnas a incluir",
        options=numeric_cols,
        default=numeric_cols[:min(6, len(numeric_cols))],
        help="Selecciona 2 o más columnas numéricas.",
    )
    method = st.selectbox(
        "Método", ["pearson", "spearman", "kendall"], index=0,
        help="pearson = lineal · spearman/kendall = rangos (no lineal)"
    )
    cmap_choice = st.selectbox(
        "Paleta de colores",
        ["coolwarm", "RdBu_r", "viridis", "plasma", "YlGnBu"],
        index=0,
    )
    annot = st.checkbox("Mostrar valores en celdas", value=True)
    fmt_decimals = st.slider("Decimales", 1, 3, 2)

if len(selected_cols) < 2:
    st.warning("⚠️ Selecciona al menos **2 columnas** en el panel lateral.")
    st.stop()

corr_matrix = df[selected_cols].corr(method=method)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(
    figsize=(max(6, len(selected_cols) * 1.1), max(4, len(selected_cols) * 1.0))
)

mask = np.zeros_like(corr_matrix, dtype=bool)
# Uncomment next line to show only the lower triangle:
# mask[np.triu_indices_from(mask, k=1)] = True

sns.heatmap(
    corr_matrix,
    annot=annot,
    fmt=f".{fmt_decimals}f",
    cmap=cmap_choice,
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor="white",
    square=True,
    ax=ax,
    cbar_kws={"shrink": 0.8, "label": "Correlación"},
)

ax.set_title(
    f"Matriz de correlación ({method.capitalize()})",
    fontsize=14,
    fontweight="bold",
    pad=14,
)
plt.xticks(rotation=35, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

st.pyplot(fig, use_container_width=True)

# ── Interpretation guide ──────────────────────────────────────────────────────
with st.expander("ℹ️ ¿Cómo interpretar el diagrama?"):
    st.markdown(
        """
| Valor | Interpretación |
|---|---|
| **1.0** | Correlación positiva perfecta |
| **0.7 – 0.9** | Correlación positiva fuerte |
| **0.4 – 0.6** | Correlación positiva moderada |
| **0.1 – 0.3** | Correlación positiva débil |
| **0** | Sin correlación lineal |
| **-0.1 – -0.3** | Correlación negativa débil |
| **-0.4 – -0.6** | Correlación negativa moderada |
| **-0.7 – -1.0** | Correlación negativa fuerte |

- **Pearson**: mide relaciones *lineales*. Sensible a outliers.  
- **Spearman / Kendall**: basados en rangos, robustos ante outliers y no linealidades.
        """
    )

# ── Download correlation matrix ───────────────────────────────────────────────
st.divider()
csv_corr = corr_matrix.to_csv().encode("utf-8")
st.download_button(
    label="⬇️ Descargar matriz de correlación (.csv)",
    data=csv_corr,
    file_name="correlacion.csv",
    mime="text/csv",
)
