from pathlib import Path
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Mantenciones Minería",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAYOR_KEYWORDS = {
    "mayor",
    "mayor cambio de barra",
    "mayor cambio de cuerpo",
    "recuperación mayor con cambio de barra",
    "recuperacion mayor con cambio de barra",
}

MAPA_MESES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

ORDEN_MESES_CALENDARIO = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# ============================================================
# ESTILO
# ============================================================
def aplicar_estilo():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1220 0%, #111827 40%, #0f172a 100%);
        color: #e5e7eb;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
        letter-spacing: 0.5px;
    }

    .sub-title {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }

    .glass-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.28);
        backdrop-filter: blur(8px);
        margin-bottom: 14px;
    }

    .kpi-box {
        background: linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.30);
        min-height: 118px;
    }

    .kpi-label {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-bottom: 8px;
        font-weight: 600;
    }

    .kpi-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .kpi-note {
        color: #60a5fa;
        font-size: 0.82rem;
        margin-top: 10px;
    }

    .ok-banner {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.40);
        color: #d1fae5;
        border-radius: 14px;
        padding: 12px 16px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .warn-banner {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.40);
        color: #fde68a;
        border-radius: 14px;
        padding: 12px 16px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .error-banner {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.40);
        color: #fecaca;
        border-radius: 14px;
        padding: 12px 16px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 6px;
        margin-bottom: 8px;
    }

    .section-note {
        color: #94a3b8;
        font-size: 0.93rem;
        margin-bottom: 14px;
    }

    .filter-panel {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148,163,184,0.15);
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 16px;
    }

    .filter-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 10px;
        border-radius: 14px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(15,23,42,0.70);
        border-radius: 12px 12px 0 0;
        color: #cbd5e1;
        padding: 10px 16px;
        border: 1px solid rgba(148,163,184,0.10);
    }

    .stTabs [aria-selected="true"] {
        background: rgba(30,41,59,0.95) !important;
        color: #f8fafc !important;
    }
    </style>
    """, unsafe_allow_html=True)


aplicar_estilo()


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def fmt_num(n):
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return str(n)


def normalizar_texto(valor):
    if pd.isna(valor):
        return ""
    return str(valor).strip()


def limpiar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def guardar_archivo_local(uploaded_file) -> Path:
    ruta = UPLOAD_DIR / uploaded_file.name
    with open(ruta, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return ruta


@st.cache_data(show_spinner=False)
def cargar_archivo(ruta: str) -> pd.DataFrame:
    ruta = Path(ruta)
    if ruta.suffix.lower() in [".xlsx", ".xls", ".xlsm"]:
        return limpiar_columnas(pd.read_excel(ruta))
    if ruta.suffix.lower() == ".csv":
        try:
            return limpiar_columnas(pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8"))
        except Exception:
            return limpiar_columnas(pd.read_csv(ruta, sep=";", encoding="latin-1"))
    raise ValueError("Formato no soportado. Usa Excel o CSV.")


def validar_columnas(df: pd.DataFrame, col_fecha: str, col_tag: str, col_tipo: str):
    faltantes = [c for c in [col_fecha, col_tag, col_tipo] if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")


def normalizar_tipo_mantenimiento(texto: str) -> str:
    t = normalizar_texto(texto).lower()
    if t in MAYOR_KEYWORDS:
        return "Mantenimiento Mayor"
    return normalizar_texto(texto)


def preparar_datos(df: pd.DataFrame, col_fecha: str, col_tag: str, col_tipo: str) -> pd.DataFrame:
    datos = df.copy()

    datos[col_tag] = datos[col_tag].apply(normalizar_texto)
    datos[col_tipo] = datos[col_tipo].apply(normalizar_texto)
    datos[col_fecha] = pd.to_datetime(datos[col_fecha], errors="coerce")

    datos = datos.dropna(subset=[col_fecha])
    datos = datos[datos[col_tag] != ""].copy()

    datos["MesN"] = datos[col_fecha].dt.month
    datos["Mes"] = datos["MesN"].map(MAPA_MESES)
    datos["Año"] = datos[col_fecha].dt.year
    datos["AñoMes"] = datos[col_fecha].dt.to_period("M").astype(str)
    datos["TipoNormalizado"] = datos[col_tipo].apply(normalizar_tipo_mantenimiento)
    datos["EsMayor"] = datos["TipoNormalizado"].eq("Mantenimiento Mayor")

    return datos.sort_values([col_tag, col_fecha]).reset_index(drop=True)


def filtrar_por_rango(df: pd.DataFrame, col_fecha: str, fecha_inicio: pd.Timestamp, fecha_fin: pd.Timestamp) -> pd.DataFrame:
    fecha_inicio = pd.Timestamp(fecha_inicio).normalize()
    fecha_fin = pd.Timestamp(fecha_fin).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(df[col_fecha] >= fecha_inicio) & (df[col_fecha] <= fecha_fin)].copy()


def aplicar_regla_menores(df_menores: pd.DataFrame, col_fecha: str, col_tag: str, umbral_dias: int):
    if df_menores.empty:
        vacio = df_menores.copy()
        vacio["FechaAnteriorMenor"] = pd.NaT
        vacio["DiasDesdeMenorAnterior"] = np.nan
        vacio["DescartarPorReglaMenor"] = False
        return vacio.copy(), vacio.copy()

    datos = df_menores.sort_values([col_tag, col_fecha]).copy()
    datos["FechaAnteriorMenor"] = datos.groupby(col_tag)[col_fecha].shift(1)
    datos["DiasDesdeMenorAnterior"] = (datos[col_fecha] - datos["FechaAnteriorMenor"]).dt.days
    datos["DescartarPorReglaMenor"] = datos["DiasDesdeMenorAnterior"].fillna(999999) < umbral_dias

    validos = datos[~datos["DescartarPorReglaMenor"]].copy()
    descartados = datos[datos["DescartarPorReglaMenor"]].copy()
    return validos, descartados


def aplicar_bloqueo_post_mayor(df: pd.DataFrame, col_fecha: str, col_tag: str, dias_bloqueo: int):
    if df.empty:
        vacio = df.copy()
        vacio["FechaUltimoMayorPrevio"] = pd.NaT
        vacio["DiasDesdeUltimoMayor"] = np.nan
        vacio["ExcluirPorPostMayor"] = False
        return vacio.copy(), vacio.copy()

    datos = df.sort_values([col_tag, col_fecha]).copy()

    datos["FechaMayorSolo"] = np.where(datos["EsMayor"], datos[col_fecha], pd.NaT)
    datos["FechaMayorSolo"] = pd.to_datetime(datos["FechaMayorSolo"], errors="coerce")

    datos["FechaUltimoMayorPrevio"] = (
        datos.groupby(col_tag)["FechaMayorSolo"].ffill()
    )

    datos.loc[datos["EsMayor"], "FechaUltimoMayorPrevio"] = (
        datos.groupby(col_tag)["FechaMayorSolo"].shift(1)
    )

    datos["DiasDesdeUltimoMayor"] = (
        datos[col_fecha] - datos["FechaUltimoMayorPrevio"]
    ).dt.days

    datos["ExcluirPorPostMayor"] = (
        (~datos["EsMayor"]) &
        (datos["FechaUltimoMayorPrevio"].notna()) &
        (datos["DiasDesdeUltimoMayor"] < dias_bloqueo)
    )

    excluidos = datos[datos["ExcluirPorPostMayor"]].copy()
    validos = datos[~datos["ExcluirPorPostMayor"]].copy()

    return validos, excluidos


def construir_dataset_analitico(
    df: pd.DataFrame,
    col_fecha: str,
    col_tag: str,
    umbral_dias: int,
    excluir_menores_regla: bool,
    aplicar_filtro_post_mayor: bool = True,
    dias_bloqueo_post_mayor: int = 60
):
    mayores = df[df["EsMayor"]].copy()
    menores = df[~df["EsMayor"]].copy()

    menores_validos_9d, menores_descartados_9d = aplicar_regla_menores(
        menores, col_fecha, col_tag, umbral_dias
    )

    menores_base = menores_validos_9d.copy() if excluir_menores_regla else menores.copy()

    temporal = pd.concat([mayores, menores_base], ignore_index=True)
    temporal = temporal.sort_values([col_tag, col_fecha]).reset_index(drop=True)

    if aplicar_filtro_post_mayor:
        validos_finales, menores_excluidos_post_mayor = aplicar_bloqueo_post_mayor(
            temporal,
            col_fecha=col_fecha,
            col_tag=col_tag,
            dias_bloqueo=dias_bloqueo_post_mayor
        )
    else:
        validos_finales = temporal.copy()
        menores_excluidos_post_mayor = temporal.iloc[0:0].copy()
        menores_excluidos_post_mayor["FechaUltimoMayorPrevio"] = pd.NaT
        menores_excluidos_post_mayor["DiasDesdeUltimoMayor"] = np.nan
        menores_excluidos_post_mayor["ExcluirPorPostMayor"] = False

    mayores_finales = validos_finales[validos_finales["EsMayor"]].copy()
    menores_finales = validos_finales[~validos_finales["EsMayor"]].copy()

    return {
        "mayores": mayores.sort_values([col_tag, col_fecha]).reset_index(drop=True),
        "menores": menores.sort_values([col_tag, col_fecha]).reset_index(drop=True),
        "menores_validos": menores_finales.sort_values([col_tag, col_fecha]).reset_index(drop=True),
        "menores_descartados": menores_descartados_9d.sort_values([col_tag, col_fecha]).reset_index(drop=True),
        "menores_excluidos_post_mayor": menores_excluidos_post_mayor.sort_values([col_tag, col_fecha]).reset_index(drop=True),
        "validos": validos_finales.sort_values([col_tag, col_fecha]).reset_index(drop=True),
        "mayores_finales": mayores_finales.sort_values([col_tag, col_fecha]).reset_index(drop=True),
    }


def clasificar_recurrencia(cantidad: int) -> str:
    if cantidad == 1:
        return "1 vez"
    if cantidad == 2:
        return "2 veces"
    if cantidad == 3:
        return "3 veces"
    if cantidad >= 4:
        return "4 o más"
    return "Sin clasificar"


def tabla_recurrencias_mensuales(df: pd.DataFrame, col_tag: str, col_fecha: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "FinalTAG", "AñoMes", "MesN", "Mes",
            "CantidadMantenciones", "Recurrencia", "PrimeraFecha", "UltimaFecha"
        ])

    tabla = (
        df.groupby([col_tag, "AñoMes", "MesN", "Mes"], as_index=False)
        .agg(
            CantidadMantenciones=(col_tag, "size"),
            PrimeraFecha=(col_fecha, "min"),
            UltimaFecha=(col_fecha, "max"),
        )
    )
    tabla["Recurrencia"] = tabla["CantidadMantenciones"].apply(clasificar_recurrencia)
    return tabla.sort_values(["AñoMes", "CantidadMantenciones", col_tag], ascending=[True, False, True])


def tabla_recurrencias_periodo(df: pd.DataFrame, col_tag: str, col_fecha: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "FinalTAG", "CantidadMantenciones", "Recurrencia",
            "PrimeraFecha", "UltimaFecha", "DiasEntrePrimeraYUltima"
        ])

    tabla = (
        df.groupby(col_tag, as_index=False)
        .agg(
            CantidadMantenciones=(col_tag, "size"),
            PrimeraFecha=(col_fecha, "min"),
            UltimaFecha=(col_fecha, "max"),
        )
    )
    tabla["Recurrencia"] = tabla["CantidadMantenciones"].apply(clasificar_recurrencia)
    tabla["DiasEntrePrimeraYUltima"] = (tabla["UltimaFecha"] - tabla["PrimeraFecha"]).dt.days
    return tabla.sort_values(["CantidadMantenciones", "UltimaFecha", col_tag], ascending=[False, False, True])


def resumen_recurrencias_por_mes(tabla_rec: pd.DataFrame) -> pd.DataFrame:
    if tabla_rec.empty:
        return pd.DataFrame(columns=["Mes", "MesN", "TAG repetidos", "2 veces", "3 veces", "4 o más"])

    base = tabla_rec[tabla_rec["CantidadMantenciones"] >= 2].copy()
    if base.empty:
        return pd.DataFrame(columns=["Mes", "MesN", "TAG repetidos", "2 veces", "3 veces", "4 o más"])

    resumen = (
        base.groupby(["MesN", "Mes"], as_index=False)
        .agg(
            TAG_repetidos=("CantidadMantenciones", "size"),
            Rec_2=("CantidadMantenciones", lambda s: int((s == 2).sum())),
            Rec_3=("CantidadMantenciones", lambda s: int((s == 3).sum())),
            Rec_4mas=("CantidadMantenciones", lambda s: int((s >= 4).sum())),
        )
    )

    resumen = resumen.rename(columns={
        "TAG_repetidos": "TAG repetidos",
        "Rec_2": "2 veces",
        "Rec_3": "3 veces",
        "Rec_4mas": "4 o más",
    })

    resumen["Orden"] = resumen["MesN"].map({m: i for i, m in enumerate(ORDEN_MESES_CALENDARIO)})
    return resumen.sort_values(["Orden", "MesN"]).drop(columns="Orden")


def resumen_recurrencias_por_categoria(tabla_rec: pd.DataFrame) -> pd.DataFrame:
    if tabla_rec.empty:
        return pd.DataFrame(columns=["Recurrencia", "Total TAG"])

    base = tabla_rec[tabla_rec["Recurrencia"].isin(["2 veces", "3 veces", "4 o más"])].copy()
    if base.empty:
        return pd.DataFrame(columns=["Recurrencia", "Total TAG"])

    orden = ["2 veces", "3 veces", "4 o más"]
    resumen = base.groupby("Recurrencia", as_index=False).agg(**{"Total TAG": ("Recurrencia", "size")})
    resumen["Orden"] = resumen["Recurrencia"].map({v: i for i, v in enumerate(orden)})
    return resumen.sort_values("Orden").drop(columns="Orden")


def tabla_mayores_periodo(df_mayores: pd.DataFrame, col_tag: str, col_fecha: str) -> pd.DataFrame:
    if df_mayores.empty:
        return pd.DataFrame(columns=["FinalTAG", "Fecha", "TipoNormalizado", "AñoMes"])
    tabla = df_mayores.rename(columns={col_tag: "FinalTAG", col_fecha: "Fecha"}).copy()
    return tabla[["FinalTAG", "Fecha", "TipoNormalizado", "AñoMes"]].sort_values(["FinalTAG", "Fecha"])


def resumen_mayores_por_mes(df_mayores: pd.DataFrame) -> pd.DataFrame:
    if df_mayores.empty:
        return pd.DataFrame(columns=["Mes", "MesN", "TotalMayores"])
    tabla = df_mayores.groupby(["Mes", "MesN"], as_index=False).agg(TotalMayores=("EsMayor", "size"))
    tabla["Orden"] = tabla["MesN"].map({m: i for i, m in enumerate(ORDEN_MESES_CALENDARIO)})
    return tabla.sort_values(["Orden", "MesN"]).drop(columns="Orden")


def analizar_historial_post_mayor(df: pd.DataFrame, col_tag: str, col_fecha: str) -> pd.DataFrame:
    filas = []

    for tag, g in df.sort_values([col_tag, col_fecha]).groupby(col_tag):
        g = g.reset_index(drop=True)
        idx_mayor = g.index[g["EsMayor"]].tolist()

        if not idx_mayor:
            continue

        ultimo_mayor_idx = idx_mayor[-1]
        ultimo_mayor = g.loc[ultimo_mayor_idx]
        posteriores = g.loc[ultimo_mayor_idx + 1:].copy()

        if posteriores.empty:
            filas.append({
                "FinalTAG": tag,
                "FechaUltimoMayor": ultimo_mayor[col_fecha],
                "TipoUltimoMayor": ultimo_mayor["TipoNormalizado"],
                "TieneReingreso": "No",
                "FechaReingreso": pd.NaT,
                "TipoReingresoPostMayor": "Sin reingreso",
                "DiasAlReingreso": np.nan,
            })
            continue

        primer_reingreso = posteriores.iloc[0]
        filas.append({
            "FinalTAG": tag,
            "FechaUltimoMayor": ultimo_mayor[col_fecha],
            "TipoUltimoMayor": ultimo_mayor["TipoNormalizado"],
            "TieneReingreso": "Sí",
            "FechaReingreso": primer_reingreso[col_fecha],
            "TipoReingresoPostMayor": primer_reingreso["TipoNormalizado"],
            "DiasAlReingreso": int((primer_reingreso[col_fecha] - ultimo_mayor[col_fecha]).days),
        })

    if not filas:
        return pd.DataFrame(columns=[
            "FinalTAG", "FechaUltimoMayor", "TipoUltimoMayor", "TieneReingreso",
            "FechaReingreso", "TipoReingresoPostMayor", "DiasAlReingreso"
        ])

    return pd.DataFrame(filas).sort_values(
        ["TieneReingreso", "DiasAlReingreso", "FinalTAG"],
        ascending=[True, True, True]
    )


def filtrar_por_tipo(df: pd.DataFrame, columna_tipo_normalizado: str, tipos_seleccionados):
    if not tipos_seleccionados or "Todos" in tipos_seleccionados:
        return df.copy()
    return df[df[columna_tipo_normalizado].isin(tipos_seleccionados)].copy()


def filtrar_por_recurrencia_periodo(tabla_periodo: pd.DataFrame, recurrencias_seleccionadas, col_tag: str):
    if tabla_periodo.empty or not recurrencias_seleccionadas or "Todas" in recurrencias_seleccionadas:
        return set(tabla_periodo[col_tag].tolist()) if col_tag in tabla_periodo.columns else set()
    return set(tabla_periodo.loc[tabla_periodo["Recurrencia"].isin(recurrencias_seleccionadas), col_tag].tolist())


def grafico_barras(df: pd.DataFrame, x: str, y: str, titulo: str, orden_meses=None, color=None):
    if df.empty:
        return None

    fig = px.bar(
        df,
        x=x,
        y=y,
        text=y,
        title=titulo,
        color=color,
        category_orders={x: orden_meses} if orden_meses else None,
        template="plotly_dark"
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=470,
        margin=dict(l=20, r=20, t=70, b=30),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.75)",
        title_font=dict(size=18),
        legend_title_text=""
    )
    fig.update_xaxes(tickangle=-25, automargin=True, gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(automargin=True, gridcolor="rgba(148,163,184,0.12)")
    return fig


def descargar_excel(dic_hojas: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for nombre_hoja, df_hoja in dic_hojas.items():
            hoja = nombre_hoja[:31] if nombre_hoja else "Hoja1"
            df_hoja.to_excel(writer, sheet_name=hoja, index=False)
    output.seek(0)
    return output.read()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Datos") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    output.seek(0)
    return output.read()


def renombrar_columnas(df: pd.DataFrame, mapa: dict) -> pd.DataFrame:
    return df.rename(columns=mapa).copy()


def seleccionar_columnas_seguras(df: pd.DataFrame, columnas):
    existentes = [c for c in columnas if c in df.columns]
    return df[existentes].copy()


def render_kpi(label, value, note=""):
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def render_tabla_descargable(
    titulo: str,
    df: pd.DataFrame,
    nombre_base: str,
    mostrar_excel: bool = True,
    altura: int = 420
):
    st.markdown(f"#### {titulo}")

    if df is None or df.empty:
        st.info("No hay datos para mostrar o descargar.")
        return

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            label="⬇️ Descargar CSV",
            data=dataframe_to_csv_bytes(df),
            file_name=f"{nombre_base}.csv",
            mime="text/csv",
            key=f"csv_{nombre_base}"
        )

    with c2:
        if mostrar_excel:
            try:
                st.download_button(
                    label="⬇️ Descargar Excel",
                    data=dataframe_to_excel_bytes(df, sheet_name=nombre_base[:31]),
                    file_name=f"{nombre_base}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"xlsx_{nombre_base}"
                )
            except Exception as e:
                st.warning(f"No se pudo generar Excel para esta tabla: {e}")

    st.dataframe(df, use_container_width=True, hide_index=True, height=altura)


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">⛏️ Dashboard de Mantenciones Minería 4.0</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Lógica corregida: Mayores fijos, regla de 9 días solo para menores y exclusión de menores demasiado cercanos a un Mayor.</div>',
    unsafe_allow_html=True
)


# ============================================================
# CARGA DE ARCHIVO EN LA VISTA
# ============================================================
with st.container():
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.markdown('<div class="filter-title">📂 Carga de archivo y configuración base</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1, 1])

    with f1:
        uploaded_file = st.file_uploader(
            "Carga tu archivo Excel o CSV",
            type=["xlsx", "xls", "xlsm", "csv"],
            key="uploader_principal"
        )
    with f2:
        ruta_manual = st.text_input("O escribe una ruta local", value="")
    with f3:
        col_fecha = st.text_input("Columna fecha", value="Fecha_Ingreso")
    with f4:
        col_tag = st.text_input("Columna TAG", value="FinalTAG")

    f5, f6 = st.columns(2)
    with f5:
        col_tipo = st.text_input("Columna tipo/estado", value="Estado")
    with f6:
        st.markdown("""
        **Reglas aplicadas**
        - Mayores siempre se conservan
        - Regla de días solo para menores
        - Mayor / Mayor barra / Mayor cuerpo → Mantenimiento Mayor
        - Menor posterior a Mayor se excluye si cae antes del umbral
        """)

    st.markdown('</div>', unsafe_allow_html=True)


ruta_carga = None
if uploaded_file is not None:
    ruta_carga = guardar_archivo_local(uploaded_file)
elif ruta_manual.strip():
    ruta_carga = Path(ruta_manual.strip())

if ruta_carga is None:
    st.info("Carga un archivo Excel o CSV en la parte superior para comenzar.")
    st.stop()

try:
    df_raw = cargar_archivo(str(ruta_carga))
    validar_columnas(df_raw, col_fecha, col_tag, col_tipo)
    datos_base = preparar_datos(df_raw, col_fecha, col_tag, col_tipo)
except Exception as e:
    st.error(f"Error al cargar/procesar el archivo: {e}")
    st.stop()

if datos_base.empty:
    st.warning("El archivo no contiene registros válidos.")
    st.stop()


# ============================================================
# FILTROS EN LA MISMA VISTA
# ============================================================
with st.container():
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.markdown('<div class="filter-title">🎛️ Filtros del análisis</div>', unsafe_allow_html=True)

    ff1, ff2, ff3, ff4 = st.columns(4)

    with ff1:
        fecha_inicio = st.date_input("Fecha inicio", value=pd.Timestamp(2025, 7, 1), key="fecha_inicio")
    with ff2:
        fecha_fin = st.date_input("Fecha fin", value=pd.Timestamp(2025, 12, 31), key="fecha_fin")
    with ff3:
        umbral_dias = st.number_input(
            "Regla menores: días mínimos",
            min_value=1,
            max_value=365,
            value=9,
            step=1,
            key="umbral_dias"
        )
    with ff4:
        dias_bloqueo_post_mayor = st.number_input(
            "Mínimo días entre Mayor y Menor",
            min_value=1,
            max_value=365,
            value=60,
            step=1,
            key="dias_bloqueo_post_mayor"
        )

    datos_rango = filtrar_por_rango(
        datos_base,
        col_fecha,
        pd.Timestamp(fecha_inicio),
        pd.Timestamp(fecha_fin)
    )

    tipos_disponibles = sorted(datos_rango["TipoNormalizado"].dropna().astype(str).unique().tolist())
    opciones_tipo = ["Todos"] + tipos_disponibles if tipos_disponibles else ["Todos"]

    ff5, ff6, ff7, ff8 = st.columns([2.0, 1.5, 1.2, 1.2])

    with ff5:
        seleccion_tipos = st.multiselect(
            "Filtrar por tipo agrupado",
            options=opciones_tipo,
            default=["Todos"],
            key="seleccion_tipos"
        )
    with ff6:
        seleccion_recurrencia = st.multiselect(
            "Filtrar por recurrencia del período",
            options=["Todas", "2 veces", "3 veces", "4 o más"],
            default=["Todas"],
            key="seleccion_recurrencia"
        )
    with ff7:
        excluir_menores_regla = st.checkbox(
            "Excluir menores por regla 9 días",
            value=True,
            key="excluir_menores_regla"
        )
    with ff8:
        aplicar_filtro_post_mayor = st.checkbox(
            "Excluir menores cercanos a Mayor",
            value=True,
            key="aplicar_filtro_post_mayor"
        )

    st.markdown('</div>', unsafe_allow_html=True)

if datos_rango.empty:
    st.warning("No hay registros dentro del rango seleccionado.")
    st.stop()

filtrado = filtrar_por_tipo(datos_rango, "TipoNormalizado", seleccion_tipos)

paquete = construir_dataset_analitico(
    filtrado,
    col_fecha=col_fecha,
    col_tag=col_tag,
    umbral_dias=int(umbral_dias),
    excluir_menores_regla=bool(excluir_menores_regla),
    aplicar_filtro_post_mayor=bool(aplicar_filtro_post_mayor),
    dias_bloqueo_post_mayor=int(dias_bloqueo_post_mayor)
)

mayores = paquete["mayores"]
menores = paquete["menores"]
menores_validos = paquete["menores_validos"]
menores_descartados = paquete["menores_descartados"]
menores_excluidos_post_mayor = paquete["menores_excluidos_post_mayor"]
validos = paquete["validos"]

recurrencias_periodo_base = tabla_recurrencias_periodo(validos, col_tag, col_fecha)

if "Todas" not in seleccion_recurrencia:
    tags_validos = filtrar_por_recurrencia_periodo(recurrencias_periodo_base, seleccion_recurrencia, col_tag)

    if tags_validos:
        validos = validos[validos[col_tag].isin(tags_validos)].copy()
        mayores = mayores[mayores[col_tag].isin(tags_validos)].copy()
        menores = menores[menores[col_tag].isin(tags_validos)].copy()
        menores_validos = menores_validos[menores_validos[col_tag].isin(tags_validos)].copy()
        menores_descartados = menores_descartados[menores_descartados[col_tag].isin(tags_validos)].copy()
        menores_excluidos_post_mayor = menores_excluidos_post_mayor[
            menores_excluidos_post_mayor[col_tag].isin(tags_validos)
        ].copy()
    else:
        validos = validos.iloc[0:0].copy()
        mayores = mayores.iloc[0:0].copy()
        menores = menores.iloc[0:0].copy()
        menores_validos = menores_validos.iloc[0:0].copy()
        menores_descartados = menores_descartados.iloc[0:0].copy()
        menores_excluidos_post_mayor = menores_excluidos_post_mayor.iloc[0:0].copy()

recurrencias_mensuales = tabla_recurrencias_mensuales(validos, col_tag, col_fecha)
recurrencias_periodo = tabla_recurrencias_periodo(validos, col_tag, col_fecha)
resumen_mes = resumen_recurrencias_por_mes(recurrencias_mensuales)
resumen_categoria = resumen_recurrencias_por_categoria(recurrencias_periodo)
tabla_mayores = tabla_mayores_periodo(mayores, col_tag, col_fecha)
mayores_mes = resumen_mayores_por_mes(mayores)
post_mayor = analizar_historial_post_mayor(validos, col_tag, col_fecha)

orden_meses_graf = [MAPA_MESES[m] for m in ORDEN_MESES_CALENDARIO if MAPA_MESES[m] in validos["Mes"].unique()]


# ============================================================
# KPIS
# ============================================================
mayores_brutos = int(len(mayores))
mayores_en_validos = int(validos["EsMayor"].sum()) if not validos.empty else 0
validacion_mayores_ok = mayores_brutos == mayores_en_validos

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    render_kpi("Registros filtrados", fmt_num(len(filtrado)), "Base del período y filtros")
with c2:
    render_kpi("Mantenimiento Mayor", fmt_num(mayores_brutos), "Debe mantenerse constante")
with c3:
    render_kpi("Menores válidos", fmt_num(len(menores_validos)), "Después de todas las reglas")
with c4:
    render_kpi("Descartados regla días", fmt_num(len(menores_descartados)), "Solo menores")
with c5:
    render_kpi("Excluidos post-mayor", fmt_num(len(menores_excluidos_post_mayor)), f"Menores antes de {int(dias_bloqueo_post_mayor)} días")
with c6:
    render_kpi("TAG únicos válidos", fmt_num(validos[col_tag].nunique() if not validos.empty else 0), "Universo final analítico")

if validacion_mayores_ok:
    st.markdown(
        f'<div class="ok-banner">✅ Validación OK: los mayores se mantienen constantes en {fmt_num(mayores_brutos)} registros.</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="error-banner">❌ Alerta: la cantidad de mayores cambió. Base={fmt_num(mayores_brutos)} | Final={fmt_num(mayores_en_validos)}</div>',
        unsafe_allow_html=True
    )


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📌 Resumen Ejecutivo",
    "🔁 Recurrencias",
    "🟡 Menores descartados",
    "⛔ Post-mayor excluidos",
    "🔴 Mayores",
    "📈 Mayor vs Reingreso",
    "📤 Exportación"
])

with tab1:
    st.markdown('<div class="section-title">Resumen Ejecutivo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Aquí no se usa “Total eventos” como eje principal. El foco está en recurrencias, reincidencia y estabilidad de mantenimiento mayor.</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:
        if resumen_categoria.empty:
            st.info("No hay recurrencias 2, 3 o 4+ en el período actual.")
        else:
            fig = grafico_barras(
                resumen_categoria,
                x="Recurrencia",
                y="Total TAG",
                titulo="Distribución de recurrencias del período"
            )
            st.plotly_chart(fig, use_container_width=True)

    with b:
        if resumen_mes.empty:
            st.info("No hay recurrencias mensuales para mostrar.")
        else:
            base_plot = resumen_mes.copy()
            base_plot["Total recurrencias"] = base_plot[["2 veces", "3 veces", "4 o más"]].sum(axis=1)
            fig = grafico_barras(
                base_plot,
                x="Mes",
                y="Total recurrencias",
                titulo="Recurrencias por mes",
                orden_meses=orden_meses_graf
            )
            st.plotly_chart(fig, use_container_width=True)

    render_tabla_descargable(
        "Resumen mensual de recurrencias",
        resumen_mes,
        "resumen_mensual_recurrencias"
    )

with tab2:
    st.markdown('<div class="section-title">Recurrencias por FinalTAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">A la izquierda se muestra la clasificación general del período. A la derecha, los meses con más TAG repetidos.</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:
        if resumen_categoria.empty:
            st.info("No hay recurrencias para mostrar.")
        else:
            fig = grafico_barras(
                resumen_categoria,
                x="Recurrencia",
                y="Total TAG",
                titulo="Clasificación general de recurrencias"
            )
            st.plotly_chart(fig, use_container_width=True)

    with b:
        if resumen_mes.empty:
            st.info("No hay resumen mensual para mostrar.")
        else:
            fig = grafico_barras(
                resumen_mes,
                x="Mes",
                y="TAG repetidos",
                titulo="TAG repetidos por mes",
                orden_meses=orden_meses_graf
            )
            st.plotly_chart(fig, use_container_width=True)

    if recurrencias_mensuales.empty:
        st.info("No hay detalle mensual con la recurrencia seleccionada.")
    else:
        rec_mostrar = renombrar_columnas(
            recurrencias_mensuales,
            {
                col_tag: "FinalTAG",
                "CantidadMantenciones": "Cantidad",
                "PrimeraFecha": "Primera Fecha",
                "UltimaFecha": "Última Fecha",
            }
        )
        columnas_rec = ["FinalTAG", "AñoMes", "Mes", "Cantidad", "Recurrencia", "Primera Fecha", "Última Fecha"]
        df_rec_detalle = seleccionar_columnas_seguras(rec_mostrar, columnas_rec)

        render_tabla_descargable(
            "Detalle mensual",
            df_rec_detalle,
            "detalle_mensual_recurrencias"
        )

    if recurrencias_periodo.empty:
        st.info("No hay detalle del período con la recurrencia seleccionada.")
    else:
        rec_periodo_mostrar = renombrar_columnas(
            recurrencias_periodo,
            {
                col_tag: "FinalTAG",
                "CantidadMantenciones": "Cantidad",
                "PrimeraFecha": "Primera Fecha",
                "UltimaFecha": "Última Fecha",
                "DiasEntrePrimeraYUltima": "Días entre primera y última",
            }
        )
        columnas_rec_periodo = ["FinalTAG", "Cantidad", "Recurrencia", "Primera Fecha", "Última Fecha", "Días entre primera y última"]
        df_rec_periodo = seleccionar_columnas_seguras(rec_periodo_mostrar, columnas_rec_periodo)

        render_tabla_descargable(
            "Detalle del período",
            df_rec_periodo,
            "detalle_periodo_recurrencias"
        )

with tab3:
    st.markdown(f'<div class="section-title">Menores descartados por regla de {umbral_dias} días</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Esta regla aplica solo a menores. Los mayores no participan en esta validación.</div>',
        unsafe_allow_html=True
    )

    if menores_descartados.empty:
        st.markdown(
            f'<div class="ok-banner">✅ No se detectaron menores descartados por la regla de {umbral_dias} días.</div>',
            unsafe_allow_html=True
        )
    else:
        resumen_desc = (
            menores_descartados.groupby(["Mes", "MesN"], as_index=False)
            .agg(TotalDescartados=(col_tag, "size"))
        )

        fig = grafico_barras(
            resumen_desc,
            x="Mes",
            y="TotalDescartados",
            titulo=f"Menores descartados por mes (< {umbral_dias} días)",
            orden_meses=orden_meses_graf
        )
        st.plotly_chart(fig, use_container_width=True)

        desc_mostrar = renombrar_columnas(
            menores_descartados,
            {
                col_tag: "FinalTAG",
                col_fecha: "Fecha",
                col_tipo: "EstadoOriginal",
                "TipoNormalizado": "TipoAgrupado",
                "FechaAnteriorMenor": "Fecha Menor Anterior",
                "DiasDesdeMenorAnterior": "Días desde menor anterior",
            }
        )
        columnas_desc = [
            "FinalTAG", "Fecha", "Fecha Menor Anterior", "Días desde menor anterior",
            "EstadoOriginal", "TipoAgrupado", "AñoMes"
        ]
        df_desc = seleccionar_columnas_seguras(desc_mostrar, columnas_desc)

        render_tabla_descargable(
            "Detalle de menores descartados",
            df_desc,
            "menores_descartados_regla_dias"
        )

with tab4:
    st.markdown(
        f'<div class="section-title">Menores excluidos por cercanía a Mantenimiento Mayor (&lt; {int(dias_bloqueo_post_mayor)} días)</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-note">Aquí se muestran los casos donde un TAG tuvo un Mayor y luego volvió como Menor demasiado pronto. Estos registros se excluyen del análisis final.</div>',
        unsafe_allow_html=True
    )

    if menores_excluidos_post_mayor.empty:
        st.markdown(
            f'<div class="ok-banner">✅ No se detectaron menores demasiado cercanos a un Mayor usando un umbral de {int(dias_bloqueo_post_mayor)} días.</div>',
            unsafe_allow_html=True
        )
    else:
        resumen_post = (
            menores_excluidos_post_mayor
            .groupby(["Mes", "MesN"], as_index=False)
            .agg(TotalExcluidos=(col_tag, "size"))
        )

        fig = grafico_barras(
            resumen_post,
            x="Mes",
            y="TotalExcluidos",
            titulo=f"Menores excluidos por cercanía a Mayor (< {int(dias_bloqueo_post_mayor)} días)",
            orden_meses=orden_meses_graf
        )
        st.plotly_chart(fig, use_container_width=True)

        post_mostrar = renombrar_columnas(
            menores_excluidos_post_mayor,
            {
                col_tag: "FinalTAG",
                col_fecha: "Fecha",
                col_tipo: "EstadoOriginal",
                "TipoNormalizado": "TipoAgrupado",
                "FechaUltimoMayorPrevio": "Fecha Último Mayor",
                "DiasDesdeUltimoMayor": "Días desde último Mayor",
            }
        )

        columnas_post = [
            "FinalTAG",
            "Fecha",
            "Fecha Último Mayor",
            "Días desde último Mayor",
            "EstadoOriginal",
            "TipoAgrupado",
            "AñoMes"
        ]

        df_post = seleccionar_columnas_seguras(post_mostrar, columnas_post)

        render_tabla_descargable(
            "Detalle de excluidos post-mayor",
            df_post,
            "excluidos_post_mayor"
        )

with tab5:
    st.markdown('<div class="section-title">Mantenimiento Mayor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Todos los tipos mayores se muestran agrupados como una sola categoría: Mantenimiento Mayor.</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:
        if mayores_mes.empty:
            st.info("No hay mayores en el rango seleccionado.")
        else:
            fig = grafico_barras(
                mayores_mes,
                x="Mes",
                y="TotalMayores",
                titulo="Mantenimiento Mayor por mes",
                orden_meses=orden_meses_graf
            )
            st.plotly_chart(fig, use_container_width=True)

    with b:
        if post_mayor.empty:
            st.info("No hay histórico post mayor para mostrar.")
        else:
            resumen_reing_tipo = post_mayor.groupby("TipoReingresoPostMayor", as_index=False).agg(TotalTAG=("FinalTAG", "size"))
            fig = grafico_barras(
                resumen_reing_tipo,
                x="TipoReingresoPostMayor",
                y="TotalTAG",
                titulo="Reingreso posterior al último mayor"
            )
            st.plotly_chart(fig, use_container_width=True)

    render_tabla_descargable(
        "Detalle de eventos mayores",
        tabla_mayores,
        "detalle_eventos_mayores"
    )

    if not post_mayor.empty:
        render_tabla_descargable(
            "Detalle de historial post mayor",
            post_mayor,
            "historial_post_mayor"
        )

with tab6:
    st.markdown('<div class="section-title">Mayor vs Reingreso como menor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Permite evaluar si un TAG que tuvo Mantenimiento Mayor vuelve después como menor u otro tipo.</div>',
        unsafe_allow_html=True
    )

    if validos.empty:
        st.info("No hay datos suficientes.")
    else:
        data = validos.sort_values([col_tag, col_fecha]).copy()
        data["FueMayorAntes"] = data.groupby(col_tag)["EsMayor"].shift(1).fillna(False)
        data["ReingresoMenor"] = (data["FueMayorAntes"] == True) & (data["EsMayor"] == False)

        mayores_mes_comp = data[data["EsMayor"]].groupby(["Mes", "MesN"], as_index=False).agg(Mayores=(col_tag, "size"))
        reingreso_mes = data[data["ReingresoMenor"]].groupby(["Mes", "MesN"], as_index=False).agg(Reingresos=(col_tag, "size"))

        tabla = pd.merge(mayores_mes_comp, reingreso_mes, on=["Mes", "MesN"], how="outer").fillna(0)

        if not tabla.empty:
            tabla["Orden"] = tabla["MesN"].map({m: i for i, m in enumerate(ORDEN_MESES_CALENDARIO)})
            tabla = tabla.sort_values(["Orden", "MesN"]).drop(columns="Orden")

        total_mayores = int(tabla["Mayores"].sum()) if not tabla.empty else 0
        total_reingresos = int(tabla["Reingresos"].sum()) if not tabla.empty else 0
        tasa_reingreso = (total_reingresos / total_mayores * 100) if total_mayores > 0 else 0

        x1, x2, x3 = st.columns(3)
        with x1:
            render_kpi("Total Mantenimiento Mayor", fmt_num(total_mayores), "Eventos mayores del filtro actual")
        with x2:
            render_kpi("Reingresos como menor", fmt_num(total_reingresos), "Después de haber tenido mayor")
        with x3:
            render_kpi("Tasa de reingreso", f"{tasa_reingreso:.1f}%", "Indicador de reincidencia")

        if tabla.empty:
            st.info("No hay datos de mantenimiento mayor o reingreso como menor en el filtro actual.")
        else:
            df_plot = tabla.melt(
                id_vars=["Mes", "MesN"],
                value_vars=["Mayores", "Reingresos"],
                var_name="Tipo",
                value_name="Cantidad"
            ).sort_values("MesN")

            fig = px.bar(
                df_plot,
                x="Mes",
                y="Cantidad",
                color="Tipo",
                barmode="group",
                text="Cantidad",
                title="Mantenimiento Mayor vs Reingresos como Menor",
                category_orders={"Mes": orden_meses_graf},
                template="plotly_dark"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                height=520,
                uniformtext_minsize=8,
                uniformtext_mode="hide",
                margin=dict(l=20, r=20, t=70, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.75)",
                legend_title_text=""
            )
            fig.update_xaxes(tickangle=-25, automargin=True, gridcolor="rgba(148,163,184,0.12)")
            fig.update_yaxes(automargin=True, gridcolor="rgba(148,163,184,0.12)")
            st.plotly_chart(fig, use_container_width=True)

            tabla_mostrar = tabla.copy()
            tabla_mostrar["Mayores"] = tabla_mostrar["Mayores"].astype(int)
            tabla_mostrar["Reingresos"] = tabla_mostrar["Reingresos"].astype(int)

            df_mayor_vs = tabla_mostrar[["Mes", "Mayores", "Reingresos"]].copy()

            render_tabla_descargable(
                "Detalle Mayor vs Reingresos",
                df_mayor_vs,
                "mayor_vs_reingresos"
            )

            if tasa_reingreso > 40:
                st.markdown(
                    f'<div class="error-banner">❌ Alta tasa de reingreso: {tasa_reingreso:.1f}%</div>',
                    unsafe_allow_html=True
                )
            elif tasa_reingreso > 20:
                st.markdown(
                    f'<div class="warn-banner">⚠️ Tasa moderada de reingreso: {tasa_reingreso:.1f}%</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="ok-banner">✅ Baja tasa de reingreso: {tasa_reingreso:.1f}%</div>',
                    unsafe_allow_html=True
                )

with tab7:
    st.markdown('<div class="section-title">Exportación de resultados</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Descarga el análisis completo en Excel, incluyendo mayores, menores, descartados, excluidos post-mayor, recurrencias y post-mayor.</div>',
        unsafe_allow_html=True
    )

    try:
        paquete_excel = descargar_excel({
            "DatosFiltrados": filtrado,
            "Mayores": mayores,
            "Menores": menores,
            "MenoresValidos": menores_validos,
            "MenoresDescartados": menores_descartados,
            "ExcluidosPostMayor": menores_excluidos_post_mayor,
            "ValidosFinales": validos,
            "RecMensuales": recurrencias_mensuales,
            "RecPeriodo": recurrencias_periodo,
            "ResumenMensual": resumen_mes,
            "ResumenCategorias": resumen_categoria,
            "MayoresPeriodo": tabla_mayores,
            "PostMayor": post_mayor,
        })

        st.download_button(
            label="📥 Descargar resultados en Excel",
            data=paquete_excel,
            file_name=f"analisis_mantenciones_mineria40_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.error(f"No se pudo generar el Excel general: {e}")

    render_tabla_descargable(
        "Datos filtrados finales",
        validos,
        "datos_filtrados_finales",
        altura=500
    )

    with st.expander("Ver glosario y reglas del modelo"):
        st.markdown(f"""
        - **Mantenimiento Mayor**: agrupación única de todas las variantes mayores.
        - **Menores válidos**: eventos menores que sobreviven a todas las reglas activas.
        - **Menores descartados**: eventos menores cuyo menor anterior del mismo TAG ocurrió hace menos de {umbral_dias} días.
        - **Excluidos post-mayor**: menores que ocurrieron antes de {int(dias_bloqueo_post_mayor)} días desde el último mayor del mismo TAG.
        - **Recurrencia mensual**: veces que un mismo FinalTAG aparece dentro del mismo mes.
        - **Recurrencia del período**: veces que un mismo FinalTAG aparece en todo el rango filtrado.
        - **TAG repetidos**: cantidad de TAG con recurrencia 2, 3 o 4+ en un mes.
        - **Validación de mayores**: el sistema verifica que la cantidad de mayores no cambie por las reglas.
        """)

st.markdown("---")
st.caption(
    "Dashboard analítico de mantenciones | Lógica corregida: mayores fijos, menores con regla de días, "
    "exclusión post-mayor y filtros visibles en la misma vista."
)
