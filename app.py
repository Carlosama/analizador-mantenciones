import os
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
    page_title="Analizador de Mantenciones",
    page_icon="📊",
    layout="wide",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAYOR_KEYWORDS = {
    "recuperación mayor con cambio de barra",
    "mayor",
    "mayor cambio de barra",
    "mayor cambio de cuerpo",
}

MAPA_MESES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

ORDEN_MESES_OPERACIONALES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
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


def clasificar_grupo_mantenimiento(texto: str) -> str:
    t = normalizar_texto(texto).lower()
    if t in MAYOR_KEYWORDS:
        return "Mantenimiento Mayor"
    return normalizar_texto(texto)


def obtener_anio_operacional(fecha: pd.Timestamp) -> int:
    # Abr-Mar. Abril 2025 a Marzo 2026 = Año Operacional 2026
    if fecha.month >= 4:
        return fecha.year + 1
    return fecha.year


def obtener_rango_operacional(anio_operacional: int):
    fecha_inicio = pd.Timestamp(year=anio_operacional - 1, month=4, day=1)
    fecha_fin = pd.Timestamp(year=anio_operacional, month=3, day=31, hour=23, minute=59, second=59)
    return fecha_inicio, fecha_fin


def preparar_datos(df: pd.DataFrame, col_fecha: str, col_tag: str, col_tipo: str) -> pd.DataFrame:
    datos = df.copy()
    datos[col_tag] = datos[col_tag].apply(normalizar_texto)
    datos[col_tipo] = datos[col_tipo].apply(normalizar_texto)
    datos[col_fecha] = pd.to_datetime(datos[col_fecha], errors="coerce")

    datos = datos.dropna(subset=[col_fecha])
    datos = datos[datos[col_tag] != ""].copy()

    datos["Año"] = datos[col_fecha].dt.year
    datos["MesN"] = datos[col_fecha].dt.month
    datos["Mes"] = datos["MesN"].map(MAPA_MESES)
    datos["AñoMes"] = datos[col_fecha].dt.to_period("M").astype(str)
    datos["AñoOperacional"] = datos[col_fecha].apply(obtener_anio_operacional)
    datos["GrupoMantenimiento"] = datos[col_tipo].apply(clasificar_grupo_mantenimiento)
    datos["EsMayor"] = datos["GrupoMantenimiento"].eq("Mantenimiento Mayor")

    datos = datos.sort_values([col_tag, col_fecha]).reset_index(drop=True)
    return datos


def filtrar_periodo_operacional(df: pd.DataFrame, col_fecha: str, anio_operacional: int) -> pd.DataFrame:
    fecha_inicio, fecha_fin = obtener_rango_operacional(anio_operacional)
    return df[(df[col_fecha] >= fecha_inicio) & (df[col_fecha] <= fecha_fin)].copy()


def calcular_descartes_por_umbral(df: pd.DataFrame, col_fecha: str, col_tag: str, umbral_dias: int):
    datos = df.sort_values([col_tag, col_fecha]).copy()
    datos["FechaAnterior"] = datos.groupby(col_tag)[col_fecha].shift(1)
    datos["DiasDesdeMantAnterior"] = (datos[col_fecha] - datos["FechaAnterior"]).dt.days
    nombre_flag = f"DescartarMenor{umbral_dias}Dias"
    datos[nombre_flag] = datos["DiasDesdeMantAnterior"].fillna(999999) < umbral_dias
    descartados = datos[datos[nombre_flag]].copy()
    validos = datos[~datos[nombre_flag]].copy()
    return validos, descartados, nombre_flag


def tabla_recurrencias_mensuales(df: pd.DataFrame, col_tag: str, col_fecha: str) -> pd.DataFrame:
    tabla = (
        df.groupby([col_tag, "AñoMes", "AñoOperacional", "MesN", "Mes"], as_index=False)
        .agg(
            CantidadMantenciones=(col_tag, "size"),
            PrimeraFecha=(col_fecha, "min"),
            UltimaFecha=(col_fecha, "max"),
        )
    )

    tabla["Recurrencia"] = np.select(
        [
            tabla["CantidadMantenciones"] == 1,
            tabla["CantidadMantenciones"] == 2,
            tabla["CantidadMantenciones"] == 3,
            tabla["CantidadMantenciones"] >= 4,
        ],
        ["1 vez", "2 veces", "3 veces", "4 o más"],
        default="Sin clasificar",
    )
    return tabla.sort_values(["AñoMes", "CantidadMantenciones", col_tag], ascending=[True, False, True])


def resumen_recurrencias_por_mes(tabla_rec: pd.DataFrame, col_tag_nombre: str) -> pd.DataFrame:
    if tabla_rec.empty:
        return pd.DataFrame(columns=["Mes", "MesN", f"{col_tag_nombre} únicos", "2 veces", "3 veces", "4 o más", "Total registros"])

    base = tabla_rec.copy()
    resumen = (
        base.groupby(["MesN", "Mes"], as_index=False)
        .agg(
            TAG_unicos=(base.columns[0], "size"),
            Total_registros=("CantidadMantenciones", "sum"),
            Rec_2=("CantidadMantenciones", lambda s: int((s == 2).sum())),
            Rec_3=("CantidadMantenciones", lambda s: int((s == 3).sum())),
            Rec_4mas=("CantidadMantenciones", lambda s: int((s >= 4).sum())),
        )
    )
    resumen = resumen.rename(columns={
        "TAG_unicos": f"{col_tag_nombre} únicos",
        "Total_registros": "Total registros",
        "Rec_2": "2 veces",
        "Rec_3": "3 veces",
        "Rec_4mas": "4 o más",
    })
    resumen["OrdenOperacional"] = resumen["MesN"].map({m: i for i, m in enumerate(ORDEN_MESES_OPERACIONALES)})
    return resumen.sort_values("OrdenOperacional").drop(columns="OrdenOperacional")


def resumen_recurrencias_por_categoria(tabla_rec: pd.DataFrame) -> pd.DataFrame:
    if tabla_rec.empty:
        return pd.DataFrame(columns=["Recurrencia", "Total TAG-Mes"])
    orden = ["1 vez", "2 veces", "3 veces", "4 o más"]
    resumen = tabla_rec.groupby("Recurrencia", as_index=False).agg(**{"Total TAG-Mes": ("Recurrencia", "size")})
    resumen["Orden"] = resumen["Recurrencia"].map({v: i for i, v in enumerate(orden)})
    return resumen.sort_values("Orden").drop(columns="Orden")


def analizar_historial_post_mayor(df: pd.DataFrame, col_tag: str, col_fecha: str, col_tipo: str) -> pd.DataFrame:
    filas = []

    for tag, g in df.sort_values([col_tag, col_fecha]).groupby(col_tag):
        g = g.reset_index(drop=True)
        idx_mayor = g.index[g["EsMayor"]].tolist()
        if not idx_mayor:
            continue

        ultimo_idx_mayor = idx_mayor[-1]
        fila_mayor = g.loc[ultimo_idx_mayor]
        posteriores = g[g.index > ultimo_idx_mayor].copy()

        reingresa = not posteriores.empty
        tipo_reingreso = "No reingresa"
        fecha_reingreso = pd.NaT
        dias_reingreso = np.nan
        estado_reingreso = ""
        mes_reingreso = ""
        mesn_reingreso = np.nan
        anio_mes_reingreso = ""

        if reingresa:
            primera_vuelta = posteriores.iloc[0]
            fecha_reingreso = primera_vuelta[col_fecha]
            dias_reingreso = (fecha_reingreso - fila_mayor[col_fecha]).days
            estado_reingreso = primera_vuelta[col_tipo]
            mes_reingreso = MAPA_MESES.get(int(fecha_reingreso.month), "")
            mesn_reingreso = int(fecha_reingreso.month)
            anio_mes_reingreso = str(fecha_reingreso.to_period("M"))
            tipo_reingreso = "Reingresa como mayor" if primera_vuelta["EsMayor"] else "Reingresa como menor/otro"

        filas.append({
            "FinalTAG": tag,
            "FechaÚltimoMayor": fila_mayor[col_fecha],
            "EstadoÚltimoMayor": fila_mayor[col_tipo],
            "MesÚltimoMayor": MAPA_MESES.get(int(fila_mayor[col_fecha].month), ""),
            "MesNÚltimoMayor": int(fila_mayor[col_fecha].month),
            "AñoMesÚltimoMayor": str(fila_mayor[col_fecha].to_period("M")),
            "TieneMayorHistórico": "Sí",
            "ReingresaPostMayor": "Sí" if reingresa else "No",
            "TipoReingresoPostMayor": tipo_reingreso,
            "FechaReingreso": fecha_reingreso,
            "MesReingreso": mes_reingreso,
            "MesNReingreso": mesn_reingreso,
            "AñoMesReingreso": anio_mes_reingreso,
            "DíasHastaReingreso": dias_reingreso,
            "EstadoReingreso": estado_reingreso,
            "TotalEventosTAG": len(g),
            "TotalMayoresTAG": int(g["EsMayor"].sum()),
        })

    if not filas:
        return pd.DataFrame(columns=[
            "FinalTAG", "FechaÚltimoMayor", "EstadoÚltimoMayor", "MesÚltimoMayor", "MesNÚltimoMayor",
            "AñoMesÚltimoMayor", "TieneMayorHistórico", "ReingresaPostMayor", "TipoReingresoPostMayor",
            "FechaReingreso", "MesReingreso", "MesNReingreso", "AñoMesReingreso", "DíasHastaReingreso",
            "EstadoReingreso", "TotalEventosTAG", "TotalMayoresTAG"
        ])

    return pd.DataFrame(filas).sort_values(
        ["ReingresaPostMayor", "DíasHastaReingreso", "FinalTAG"],
        ascending=[False, True, True]
    )


def resumen_reingresos_por_mes(post_mayor: pd.DataFrame) -> pd.DataFrame:
    if post_mayor.empty:
        return pd.DataFrame(columns=["Mes", "Reingresa como menor/otro", "Reingresa como mayor", "No reingresa"])

    base = post_mayor.copy()
    base["MesGraf"] = base["MesReingreso"].replace("", np.nan)
    base.loc[base["TipoReingresoPostMayor"].eq("No reingresa"), "MesGraf"] = "Sin reingreso"

    resumen = (
        base.groupby(["MesGraf", "TipoReingresoPostMayor"], as_index=False)
        .agg(TotalTAG=("FinalTAG", "size"))
        .pivot(index="MesGraf", columns="TipoReingresoPostMayor", values="TotalTAG")
        .fillna(0)
        .reset_index()
        .rename(columns={"MesGraf": "Mes"})
    )

    for col in ["Reingresa como menor/otro", "Reingresa como mayor", "No reingresa"]:
        if col not in resumen.columns:
            resumen[col] = 0

    orden_meses = {MAPA_MESES[m]: i for i, m in enumerate(ORDEN_MESES_OPERACIONALES)}
    orden_meses["Sin reingreso"] = 99
    resumen["OrdenOperacional"] = resumen["Mes"].map(orden_meses)
    resumen = resumen.sort_values("OrdenOperacional").drop(columns="OrdenOperacional")
    return resumen


def tabla_mayores_periodo(df: pd.DataFrame, col_tag: str, col_fecha: str, col_tipo: str) -> pd.DataFrame:
    mayores = df[df["EsMayor"]].copy()
    if mayores.empty:
        return pd.DataFrame(columns=["FinalTAG", "Fecha", "Estado", "AñoMes"])
    return mayores.rename(
        columns={col_tag: "FinalTAG", col_fecha: "Fecha", col_tipo: "Estado"}
    )[["FinalTAG", "Fecha", "Estado", "AñoMes"]]


def renombrar_columnas(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    cols_existentes = {k: v for k, v in mapping.items() if k in df.columns}
    df2 = df.rename(columns=cols_existentes).copy()
    nuevos_nombres = []
    conteo = {}
    for col in df2.columns:
        if col not in conteo:
            conteo[col] = 1
            nuevos_nombres.append(col)
        else:
            conteo[col] += 1
            nuevos_nombres.append(f"{col}_{conteo[col]}")
    df2.columns = nuevos_nombres
    return df2


def seleccionar_columnas_seguras(df: pd.DataFrame, columnas_deseadas: list[str]) -> pd.DataFrame:
    columnas_existentes = [c for c in columnas_deseadas if c in df.columns]
    columnas_unicas = list(dict.fromkeys(columnas_existentes))
    return df[columnas_unicas].copy()


def grafico_barras(df_plot: pd.DataFrame, x: str, y: str, titulo: str, text_auto=True):
    fig = px.bar(df_plot, x=x, y=y, text_auto=text_auto, title=titulo)
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=70, b=30),
        xaxis_title=None,
        yaxis_title=None
    )
    fig.update_xaxes(tickangle=-25, automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def grafico_barras_agrupadas(df_plot: pd.DataFrame, x: str, ys: list[str], titulo: str):
    largo = df_plot.melt(id_vars=[x], value_vars=ys, var_name="Serie", value_name="Valor")
    fig = px.bar(
        largo,
        x=x,
        y="Valor",
        color="Serie",
        barmode="group",
        text_auto=True,
        title=titulo
    )
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=70, b=30),
        xaxis_title=None,
        yaxis_title=None,
        legend_title_text=""
    )
    fig.update_xaxes(tickangle=-25, automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def descargar_excel(df_dict: dict[str, pd.DataFrame]) -> bytes:
    salida = BytesIO()
    with pd.ExcelWriter(salida, engine="openpyxl") as writer:
        for nombre, df_hoja in df_dict.items():
            df_hoja.to_excel(writer, index=False, sheet_name=nombre[:31])
    salida.seek(0)
    return salida.getvalue()


# ============================================================
# INTERFAZ
# ============================================================
st.title("📊 Analizador Web de Mantenciones por FinalTAG")
st.caption(
    "Análisis usando la columna Fecha_Ingreso, período operacional abril-marzo, "
    "recurrencias, descartes configurables y seguimiento post mantenimiento mayor."
)

st.subheader("1) Carga de archivo")
archivo = st.file_uploader("Sube tu archivo Excel o CSV", type=["xlsx", "xls", "xlsm", "csv"])

if not archivo:
    st.info("Sube un archivo para comenzar el análisis.")
    st.stop()

ruta_local = guardar_archivo_local(archivo)
df_raw = cargar_archivo(str(ruta_local))
st.success(f"Archivo guardado localmente en: {ruta_local}")

with st.expander("Vista previa del archivo", expanded=False):
    st.dataframe(df_raw.head(20), use_container_width=True, hide_index=True)

st.subheader("2) Configuración de columnas")
columnas = list(df_raw.columns)
candidatos_fecha = [c for c in columnas if "fecha_ingreso" in c.lower()] or [c for c in columnas if "fecha" in c.lower()]
candidatos_tag = [c for c in columnas if "finaltag" in c.lower()] or [c for c in columnas if c.lower() == "tag"]
candidatos_tipo = [c for c in columnas if "estado" in c.lower()] or [c for c in columnas if "mant" in c.lower() or "tipo" in c.lower() or "repar" in c.lower()]

c1, c2, c3 = st.columns(3)
with c1:
    col_fecha = st.selectbox("Columna de fecha", columnas, index=columnas.index(candidatos_fecha[0]) if candidatos_fecha else 0)
with c2:
    col_tag = st.selectbox("Columna FinalTAG", columnas, index=columnas.index(candidatos_tag[0]) if candidatos_tag else 0)
with c3:
    col_tipo = st.selectbox("Columna Estado / Tipo mantención", columnas, index=columnas.index(candidatos_tipo[0]) if candidatos_tipo else 0)

validar_columnas(df_raw, col_fecha, col_tag, col_tipo)
df = preparar_datos(df_raw, col_fecha, col_tag, col_tipo)

if df.empty:
    st.warning("No hay datos válidos luego de limpiar el archivo.")
    st.stop()

anios_operacionales = sorted(df["AñoOperacional"].dropna().unique().tolist())
if not anios_operacionales:
    st.warning("No fue posible calcular años operacionales.")
    st.stop()

st.subheader("3) Filtros interactivos")
fecha_hoy = pd.Timestamp.today()
anio_operacional_actual = obtener_anio_operacional(fecha_hoy)
idx_default = anios_operacionales.index(anio_operacional_actual) if anio_operacional_actual in anios_operacionales else len(anios_operacionales) - 1

f1, f2, f3, f4 = st.columns(4)
with f1:
    anio_operacional_sel = st.selectbox(
        "Año operacional",
        options=anios_operacionales,
        index=idx_default,
        help="Ejemplo: 2026 significa desde abril 2025 hasta marzo 2026.",
    )
with f2:
    fecha_ini_op, fecha_fin_op = obtener_rango_operacional(anio_operacional_sel)
    st.text_input(
        "Rango aplicado",
        value=f"{fecha_ini_op.strftime('%d-%m-%Y')} a {fecha_fin_op.strftime('%d-%m-%Y')}",
        disabled=True
    )
with f3:
    estados_disp = sorted([x for x in df[col_tipo].dropna().astype(str).unique().tolist() if str(x).strip() != ""])
    estados_sel = st.multiselect("Estado / tipo de mantención", options=estados_disp, default=estados_disp)
with f4:
    filtro_recurrencia = st.multiselect(
        "Recurrencia",
        options=["1 vez", "2 veces", "3 veces", "4 o más"],
        default=["1 vez", "2 veces", "3 veces", "4 o más"]
    )

g1, g2, g3 = st.columns(3)
with g1:
    solo_mayores = st.checkbox("Solo mantenimientos mayores", value=False)
with g2:
    umbral_dias = st.number_input("Regla mínima de días", min_value=1, max_value=60, value=9, step=1)
with g3:
    excluir_menor_umbral = st.checkbox(f"Excluir eventos menores a {umbral_dias} días", value=True)

filtrado = filtrar_periodo_operacional(df, col_fecha, anio_operacional_sel)
filtrado = filtrado[filtrado[col_tipo].astype(str).isin(estados_sel)].copy()
if solo_mayores:
    filtrado = filtrado[filtrado["EsMayor"]].copy()

validos_base, descartados, nombre_flag = calcular_descartes_por_umbral(filtrado, col_fecha, col_tag, umbral_dias)
validos = validos_base.copy() if excluir_menor_umbral else filtrado.copy()

recurrencias = tabla_recurrencias_mensuales(validos, col_tag, col_fecha)
if filtro_recurrencia:
    recurrencias = recurrencias[recurrencias["Recurrencia"].isin(filtro_recurrencia)].copy()

resumen_mes = resumen_recurrencias_por_mes(recurrencias, "TAG")
resumen_categoria = resumen_recurrencias_por_categoria(recurrencias)
post_mayor = analizar_historial_post_mayor(validos, col_tag, col_fecha, col_tipo)
tabla_mayores = tabla_mayores_periodo(validos, col_tag, col_fecha, col_tipo)

# ============================================================
# KPIS
# ============================================================
st.subheader("4) Resumen ejecutivo")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Registros cargados", f"{len(df_raw):,}".replace(",", "."))
k2.metric("Registros filtrados", f"{len(filtrado):,}".replace(",", "."))
k3.metric("Registros válidos", f"{len(validos):,}".replace(",", "."))
k4.metric(f"Descartados < {umbral_dias} días", f"{len(descartados):,}".replace(",", "."))
k5.metric("Eventos mayores", f"{int(validos['EsMayor'].sum()):,}".replace(",", "."))
k6.metric("TAG con mayor histórico", f"{len(post_mayor):,}".replace(",", "."))

st.info(
    f"La lógica de fechas usada en esta app es operacional: si eliges 2026, se analiza desde abril 2025 hasta marzo 2026. "
    f"Además, el seguimiento de mantenimiento mayor revisa si un FinalTAG que tuvo un mayor vuelve a ingresar después, "
    f"incluyendo reingresos como menor. La regla mínima actual es > {umbral_dias} días."
)

# ============================================================
# PESTAÑAS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Resumen mensual",
    "Recurrencias",
    f"Descartados < {umbral_dias} días",
    "Mantenimiento mayor",
    "Exportación",
    "📈 Mayor vs Reingreso (Crítico)"
])

with tab1:
    st.markdown("### Meses con más recurrencia")
    st.caption("Aquí se muestra qué mes del período operacional tuvo más repeticiones de TAG, no solo cuántas veces se repitió cada categoría.")

    if resumen_mes.empty:
        st.info("No hay datos para el período seleccionado.")
    else:
        graf_mes = grafico_barras_agrupadas(
            resumen_mes,
            x="Mes",
            ys=["Total registros", "2 veces", "3 veces", "4 o más"],
            titulo="Resumen mensual de recurrencias por mes del período operacional"
        )
        st.plotly_chart(graf_mes, use_container_width=True)

        top_mes = resumen_mes.sort_values(["4 o más", "3 veces", "2 veces", "Total registros"], ascending=False).head(1)
        if not top_mes.empty:
            fila = top_mes.iloc[0]
            st.success(
                f"El mes con mayor concentración de recurrencias del período fue **{fila['Mes']}**, "
                f"con {int(fila['Total registros'])} registros analizados, "
                f"{int(fila['2 veces'])} TAG con recurrencia 2, "
                f"{int(fila['3 veces'])} TAG con recurrencia 3 y "
                f"{int(fila['4 o más'])} TAG con recurrencia 4 o más."
            )

        st.dataframe(resumen_mes, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### Recurrencias por FinalTAG")
    st.caption(
        "'1 vez', '2 veces', '3 veces' y '4 o más' significan cuántas veces se repitió el mismo "
        "FinalTAG dentro del mismo mes calendario, calculado desde la columna de fecha seleccionada."
    )

    c21, c22 = st.columns(2)
    with c21:
        if resumen_categoria.empty:
            st.info("No hay recurrencias para mostrar.")
        else:
            st.plotly_chart(
                grafico_barras(
                    resumen_categoria,
                    x="Recurrencia",
                    y="Total TAG-Mes",
                    titulo="Clasificación general de recurrencias"
                ),
                use_container_width=True,
            )
    with c22:
        if resumen_mes.empty:
            st.info("No hay resumen mensual para mostrar.")
        else:
            base_mes_simple = resumen_mes[["Mes", "2 veces", "3 veces", "4 o más"]].copy()
            base_mes_simple["Total recurrencias >1"] = base_mes_simple[["2 veces", "3 veces", "4 o más"]].sum(axis=1)
            st.plotly_chart(
                grafico_barras(
                    base_mes_simple,
                    x="Mes",
                    y="Total recurrencias >1",
                    titulo="Meses con más TAG repetidos"
                ),
                use_container_width=True,
            )

    if recurrencias.empty:
        st.info("No hay detalle de recurrencias para el período seleccionado.")
    else:
        rec_mostrar = renombrar_columnas(
            recurrencias,
            {
                col_tag: "FinalTAG",
                "CantidadMantenciones": "Cantidad",
                "PrimeraFecha": "Primera Fecha",
                "UltimaFecha": "Última Fecha",
            }
        )
        columnas_rec = ["FinalTAG", "AñoMes", "Mes", "Cantidad", "Recurrencia", "Primera Fecha", "Última Fecha"]
        st.dataframe(seleccionar_columnas_seguras(rec_mostrar, columnas_rec), use_container_width=True, hide_index=True)

with tab3:
    st.markdown(f"### Registros descartados por menos de {umbral_dias} días")
    st.caption(
        f"Estos registros se marcan cuando el mismo FinalTAG vuelve a ingresar con una diferencia "
        f"menor a {umbral_dias} días respecto de su evento anterior."
    )

    if descartados.empty:
        st.success(f"No se detectaron descartes por la regla de {umbral_dias} días.")
    else:
        resumen_desc = descartados.groupby("Mes", as_index=False).agg(TotalDescartados=(col_tag, "size"))
        resumen_desc["OrdenOperacional"] = resumen_desc["Mes"].map({MAPA_MESES[m]: i for i, m in enumerate(ORDEN_MESES_OPERACIONALES)})
        resumen_desc = resumen_desc.sort_values("OrdenOperacional").drop(columns="OrdenOperacional")

        st.plotly_chart(
            grafico_barras(
                resumen_desc,
                x="Mes",
                y="TotalDescartados",
                titulo=f"Descartes por mes operacional (< {umbral_dias} días)"
            ),
            use_container_width=True,
        )

        desc_mostrar = renombrar_columnas(
            descartados,
            {
                col_tag: "FinalTAG",
                col_fecha: "Fecha",
                col_tipo: "Estado",
                "FechaAnterior": "Fecha Anterior",
                "DiasDesdeMantAnterior": "Días desde anterior",
            }
        )
        columnas_desc = ["FinalTAG", "Fecha", "Fecha Anterior", "Días desde anterior", "Estado", "GrupoMantenimiento", "AñoMes"]
        st.dataframe(seleccionar_columnas_seguras(desc_mostrar, columnas_desc), use_container_width=True, hide_index=True)

with tab4:
    st.markdown("### Seguimiento de mantenimiento mayor y reingresos")
    st.caption(
        "Aquí se muestra cuántos TAG salieron de mantenimiento mayor y luego reingresaron como menor/otro "
        "o como mayor, además del mes en que ocurrió ese reingreso."
    )

    if post_mayor.empty:
        st.info("No se encontraron TAG con histórico mayor para el período filtrado.")
    else:
        resumen_reing_mes = resumen_reingresos_por_mes(post_mayor)
        resumen_reing_tipo = post_mayor.groupby("TipoReingresoPostMayor", as_index=False).agg(TotalTAG=("FinalTAG", "size"))

        c41, c42 = st.columns(2)
        with c41:
            if tabla_mayores.empty:
                st.info("No hay eventos mayores en el período seleccionado.")
            else:
                mayores_mes = tabla_mayores.copy()
                mayores_mes["Mes"] = mayores_mes["Fecha"].dt.month.map(MAPA_MESES)
                mayores_mes = mayores_mes.groupby("Mes", as_index=False).agg(TotalMayores=("FinalTAG", "size"))
                mayores_mes["OrdenOperacional"] = mayores_mes["Mes"].map({MAPA_MESES[m]: i for i, m in enumerate(ORDEN_MESES_OPERACIONALES)})
                mayores_mes = mayores_mes.sort_values("OrdenOperacional").drop(columns="OrdenOperacional")
                st.plotly_chart(
                    grafico_barras(mayores_mes, x="Mes", y="TotalMayores", titulo="Eventos de mantenimiento mayor por mes"),
                    use_container_width=True,
                )
        with c42:
            st.plotly_chart(
                grafico_barras(
                    resumen_reing_tipo,
                    x="TipoReingresoPostMayor",
                    y="TotalTAG",
                    titulo="Cantidad total de TAG según tipo de reingreso"
                ),
                use_container_width=True,
            )

        st.markdown("#### Reingresos por mes después de un mantenimiento mayor")
        st.plotly_chart(
            grafico_barras_agrupadas(
                resumen_reing_mes,
                x="Mes",
                ys=["Reingresa como menor/otro", "Reingresa como mayor", "No reingresa"],
                titulo="Mes en que reingresan los TAG después de un mayor"
            ),
            use_container_width=True,
        )

        solo_menor = post_mayor[post_mayor["TipoReingresoPostMayor"].eq("Reingresa como menor/otro")].copy()
        if solo_menor.empty:
            st.info("No hay TAG que hayan reingresado desde mayor hacia menor/otro con el filtro aplicado.")
        else:
            resumen_menor_mes = (
                solo_menor.groupby(["MesReingreso", "MesNReingreso"], as_index=False)
                .agg(TotalTAG=("FinalTAG", "size"))
            )
            resumen_menor_mes["OrdenOperacional"] = resumen_menor_mes["MesNReingreso"].map({m: i for i, m in enumerate(ORDEN_MESES_OPERACIONALES)})
            resumen_menor_mes = resumen_menor_mes.sort_values("OrdenOperacional").drop(columns="OrdenOperacional")

            st.markdown("#### TAG que reingresaron de mayor a menor/otro por mes")
            st.plotly_chart(
                grafico_barras(
                    resumen_menor_mes,
                    x="MesReingreso",
                    y="TotalTAG",
                    titulo="Cantidad de TAG que reingresan de mayor a menor/otro por mes"
                ),
                use_container_width=True,
            )

            top_mes_menor = resumen_menor_mes.sort_values("TotalTAG", ascending=False).head(1)
            if not top_mes_menor.empty:
                fila_top = top_mes_menor.iloc[0]
                st.success(
                    f"El mes con más reingresos desde mantenimiento mayor hacia menor/otro fue "
                    f"**{fila_top['MesReingreso']}**, con **{int(fila_top['TotalTAG'])} TAG**."
                )

        st.markdown("**Detalle de TAG con histórico de mantenimiento mayor**")
        st.dataframe(post_mayor, use_container_width=True, hide_index=True)

        st.markdown("**Detalle de eventos clasificados como mantenimiento mayor dentro del período**")
        st.dataframe(tabla_mayores, use_container_width=True, hide_index=True)

with tab5:
    st.markdown("### Exportación")
    paquete = descargar_excel({
        "DatosFiltrados": filtrado,
        "Validos": validos,
        "DescartadosUmbral": descartados,
        "RecurrenciasDetalle": recurrencias,
        "ResumenMensual": resumen_mes,
        "ResumenCategorias": resumen_categoria,
        "MayoresPeriodo": tabla_mayores,
        "PostMayor": post_mayor,
    })

    st.download_button(
        label="📥 Descargar resultados en Excel",
        data=paquete,
        file_name=f"analisis_mantenciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    with st.expander("Glosario"):
        st.markdown(
            f"""
            - **Año operacional**: período entre abril de un año y marzo del siguiente.
            - **Recurrencia**: cantidad de veces que un mismo FinalTAG aparece dentro del mismo mes.
            - **2 veces / 3 veces / 4 o más**: cantidad de TAG-mes que tuvieron esa recurrencia.
            - **Descartado < {umbral_dias} días**: evento del mismo FinalTAG cuya diferencia con el anterior es menor a {umbral_dias} días.
            - **Mantenimiento Mayor**: agrupación de Recuperación Mayor con cambio de Barra, Mayor, Mayor Cambio de barra y Mayor Cambio de cuerpo.
            - **Reingreso post mayor**: primer evento posterior al último mayor del TAG, aunque vuelva como menor.
            """
        )

with tab6:
    st.markdown("### 📈 Análisis crítico: Mayor vs Reingreso como menor")
    st.caption(
        "Este análisis muestra cuántos cátodos intervenidos con mantenimiento mayor vuelven a ingresar posteriormente "
        "como mantenimiento menor, permitiendo evaluar la efectividad real de la reparación."
    )

    if validos.empty:
        st.info("No hay datos suficientes.")
    else:
        data = validos.sort_values([col_tag, col_fecha]).copy()
        data["FueMayorAntes"] = data.groupby(col_tag)["EsMayor"].shift(1).fillna(False)
        data["ReingresoMenor"] = (data["FueMayorAntes"] == True) & (data["EsMayor"] == False)

        data["MesOrden"] = data["MesN"].map({m: i for i, m in enumerate(ORDEN_MESES_OPERACIONALES)})

        mayores_mes = data[data["EsMayor"]].groupby(["Mes", "MesN"], as_index=False).agg(Mayores=(col_tag, "size"))
        reingreso_mes = data[data["ReingresoMenor"]].groupby(["Mes", "MesN"], as_index=False).agg(Reingresos=(col_tag, "size"))

        tabla = pd.merge(mayores_mes, reingreso_mes, on=["Mes", "MesN"], how="outer").fillna(0)
        tabla["Orden"] = tabla["MesN"].map({m: i for i, m in enumerate(ORDEN_MESES_OPERACIONALES)})
        tabla = tabla.sort_values(["Orden", "MesN"]).drop(columns="Orden")

        total_mayores = int(tabla["Mayores"].sum()) if not tabla.empty else 0
        total_reingresos = int(tabla["Reingresos"].sum()) if not tabla.empty else 0
        tasa_reingreso = (total_reingresos / total_mayores * 100) if total_mayores > 0 else 0

        x1, x2, x3 = st.columns(3)
        x1.metric("Total Mantenciones Mayores", f"{total_mayores:,}".replace(",", "."))
        x2.metric("Reingresos como menor", f"{total_reingresos:,}".replace(",", "."))
        x3.metric("Tasa de reingreso", f"{tasa_reingreso:.1f}%")

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
                title="Mantenciones Mayores vs Reingresos como Menor"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                height=520,
                uniformtext_minsize=8,
                uniformtext_mode="hide",
                margin=dict(l=20, r=20, t=70, b=30),
                legend_title_text=""
            )
            fig.update_xaxes(tickangle=-25, automargin=True)
            fig.update_yaxes(automargin=True)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Tabla resumen")
            tabla_mostrar = tabla.copy()
            tabla_mostrar["Mayores"] = tabla_mostrar["Mayores"].astype(int)
            tabla_mostrar["Reingresos"] = tabla_mostrar["Reingresos"].astype(int)
            st.dataframe(tabla_mostrar[["Mes", "Mayores", "Reingresos"]], use_container_width=True, hide_index=True)

            st.markdown("#### Interpretación automática")
            if tasa_reingreso > 40:
                st.error(
                    f"Alta tasa de reingreso ({tasa_reingreso:.1f}%). "
                    f"Las mantenciones mayores presentan baja efectividad en el período seleccionado."
                )
            elif tasa_reingreso > 20:
                st.warning(
                    f"Tasa moderada de reingreso ({tasa_reingreso:.1f}%). "
                    f"Se recomienda revisar la calidad de la reparación y las condiciones operacionales."
                )
            else:
                st.success(
                    f"Baja tasa de reingreso ({tasa_reingreso:.1f}%). "
                    f"El desempeño de las mantenciones mayores es favorable en el filtro actual."
                )

            top_mes_reing = tabla.sort_values("Reingresos", ascending=False).head(1)
            if not top_mes_reing.empty and int(top_mes_reing.iloc[0]["Reingresos"]) > 0:
                fila = top_mes_reing.iloc[0]
                st.info(
                    f"El mes con mayor reingreso como menor fue **{fila['Mes']}**, "
                    f"con **{int(fila['Reingresos'])}** eventos, frente a "
                    f"**{int(fila['Mayores'])}** mantenciones mayores."
                )