import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Configuración general ────────────────────────────────────────────────────
st.set_page_config(
    page_title="PM2.5 Fontibón — Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# ── Carga de datos ───────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv("data/iboca_fontibon.csv", encoding="utf-8-sig", parse_dates=["fecha_hora"])
    return df

df = cargar_datos()

# ── Cálculo del desfase ──────────────────────────────────────────────────────
df["desfase_relativo"] = (df["pm25"] - df["NowCast"]).abs() / df["pm25"]
df_desfasados = df[df["desfase_relativo"] > 0.10]

total_registros     = len(df)
total_desfasados    = len(df_desfasados)
pct_desfasados      = total_desfasados / total_registros * 100
media_desfase       = (df_desfasados["pm25"] - df_desfasados["NowCast"]).mean()
desviacion_desfase  = (df_desfasados["pm25"] - df_desfasados["NowCast"]).std()

# ── Encabezado ───────────────────────────────────────────────────────────────
st.title("🌫️ Monitoreo PM2.5 — Localidad de Fontibón, Bogotá")
st.markdown(
    "Comparación entre el método **NowCast (IBOCA)** y modelos de Machine Learning "
    "(**Random Forest / XGBoost**) | Periodo: Enero 2024 – Diciembre 2025"
)
st.divider()

# ── Sección 1: Métricas del desfase ─────────────────────────────────────────
st.subheader("📊 Desfase NowCast vs. Concentración Real de PM2.5")
st.markdown(
    "El siguiente análisis cuantifica la diferencia entre las estimaciones del método "
    "NowCast y las mediciones reales registradas en la estación de Fontibón."
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Total de registros",
    value=f"{total_registros:,}"
)
col2.metric(
    label="Registros con desfase > 10%",
    value=f"{total_desfasados:,}",
    delta=f"{pct_desfasados:.1f}% del total",
    delta_color="inverse"
)
col3.metric(
    label="Diferencia promedio (desfasados)",
    value=f"{media_desfase:.2f} µg/m³"
)
col4.metric(
    label="Desviación estándar (desfasados)",
    value=f"{desviacion_desfase:.2f} µg/m³"
)

st.divider()
# ── Sección 2: Serie temporal PM2.5 real vs. NowCast ────────────────────────
st.subheader("📈 Serie Temporal — PM2.5 Real vs. NowCast")
st.markdown("Selecciona un rango de fechas para explorar el comportamiento hora a hora.")

# Filtros
col_f1, col_f2 = st.columns(2)

with col_f1:
    anios_disponibles = sorted(df["año"].unique())
    anio_sel = st.selectbox("Año", anios_disponibles)

with col_f2:
    meses = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    meses_disponibles = sorted(df[df["año"] == anio_sel]["mes"].unique())
    mes_sel = st.selectbox("Mes", meses_disponibles, format_func=lambda x: meses[x])

# Filtrar datos
df_filtrado = df[(df["año"] == anio_sel) & (df["mes"] == mes_sel)].copy()

# Gráfico
fig_serie = go.Figure()

fig_serie.add_trace(go.Scatter(
    x=df_filtrado["fecha_hora"],
    y=df_filtrado["pm25"],
    name="PM2.5 Real",
    line=dict(color="#1f77b4", width=1.5),
    hovertemplate="<b>Real</b>: %{y:.2f} µg/m³<br>%{x}<extra></extra>"
))

fig_serie.add_trace(go.Scatter(
    x=df_filtrado["fecha_hora"],
    y=df_filtrado["NowCast"],
    name="NowCast",
    line=dict(color="#ff7f0e", width=1.5, dash="dot"),
    hovertemplate="<b>NowCast</b>: %{y:.2f} µg/m³<br>%{x}<extra></extra>"
))

# Línea de límite normativo (Resolución 2254 de 2017: 37 µg/m³ promedio 24h)
fig_serie.add_hline(
    y=37,
    line_dash="dash",
    line_color="red",
    annotation_text="Límite Res. 2254/2017 (37 µg/m³)",
    annotation_position="top left"
)

fig_serie.update_layout(
    title=f"PM2.5 Real vs. NowCast — {meses[mes_sel]} {anio_sel}",
    xaxis_title="Fecha y hora",
    yaxis_title="Concentración PM2.5 (µg/m³)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=450
)

st.plotly_chart(fig_serie, use_container_width=True)

# Estadísticas del mes seleccionado
st.markdown(f"**Estadísticas del periodo seleccionado — {meses[mes_sel]} {anio_sel}**")
col_e1, col_e2, col_e3, col_e4 = st.columns(4)
col_e1.metric("PM2.5 real promedio", f"{df_filtrado['pm25'].mean():.2f} µg/m³")
col_e2.metric("NowCast promedio", f"{df_filtrado['NowCast'].mean():.2f} µg/m³")
col_e3.metric("PM2.5 real máximo", f"{df_filtrado['pm25'].max():.2f} µg/m³")
col_e4.metric("Datos faltantes", f"{df_filtrado['pm25'].isna().sum()} horas")

st.divider()
# ── Sección 3: Análisis del desfase ─────────────────────────────────────────
st.subheader("🔍 Análisis del Desfase — NowCast vs. PM2.5 Real")
st.markdown(
    "Distribución de la diferencia entre la concentración real medida y la estimación "
    "generada por NowCast, hora a hora durante el periodo completo de estudio."
)

# Calcular diferencia
df["diferencia"] = df["pm25"] - df["NowCast"]

col_g1, col_g2 = st.columns(2)

# ── Gráfico 1: Histograma de la diferencia ───────────────────────────────────
with col_g1:
    fig_hist = px.histogram(
        df.dropna(subset=["diferencia"]),
        x="diferencia",
        nbins=80,
        color_discrete_sequence=["#1f77b4"],
        title="Distribución de la diferencia (Real − NowCast)",
        labels={"diferencia": "Diferencia (µg/m³)", "count": "Frecuencia"}
    )
    fig_hist.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Sin desfase",
        annotation_position="top right"
    )
    fig_hist.add_vline(
        x=df["diferencia"].mean(),
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Media: {df['diferencia'].mean():.2f} µg/m³",
        annotation_position="top left"
    )
    fig_hist.update_layout(height=400, yaxis_title="Frecuencia")
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Gráfico 2: Desfase promedio por hora del día ─────────────────────────────
with col_g2:
    desfase_por_hora = (
        df.drop(subset=["diferencia"])
        .groupby("hora")["diferencia"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "media", "std": "desviacion"})
    )

    fig_hora = go.Figure()

    fig_hora.add_trace(go.Scatter(
        x=desfase_por_hora["hora"],
        y=desfase_por_hora["media"],
        mode="lines+markers",
        name="Diferencia promedio",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="Hora %{x}:00 — Media: %{y:.2f} µg/m³<extra></extra>"
    ))

    fig_hora.add_trace(go.Scatter(
        x=pd.concat([desfase_por_hora["hora"], desfase_por_hora["hora"][::-1]]),
        y=pd.concat([
            desfase_por_hora["media"] + desfase_por_hora["desviacion"],
            (desfase_por_hora["media"] - desfase_por_hora["desviacion"])[::-1]
        ]),
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="±1 Desviación estándar",
        hoverinfo="skip"
    ))

    fig_hora.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Sin desfase"
    )

    fig_hora.update_layout(
        title="Desfase promedio por hora del día",
        xaxis_title="Hora del día",
        yaxis_title="Diferencia promedio (µg/m³)",
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_hora, use_container_width=True)

# ── Tabla resumen del desfase ────────────────────────────────────────────────
st.markdown("**Resumen estadístico del desfase por año**")
resumen = (
    df.dropna(subset=["diferencia"])
    .groupby("año")["diferencia"]
    .agg(
        Registros="count",
        Media="mean",
        Desviacion_Std="std",
        MAE=lambda x: x.abs().mean(),
        Desfase_mayor_10pct=lambda x: (
            (x.abs() / df.loc[x.index, "pm25"]) > 0.10
        ).sum()
    )
    .reset_index()
    .rename(columns={"año": "Año"})
)
resumen["% Desfase > 10%"] = (resumen["Desfase_mayor_10pct"] / resumen["Registros"] * 100).round(1)
resumen["Media"] = resumen["Media"].round(2)
resumen["Desviacion_Std"] = resumen["Desviacion_Std"].round(2)
resumen["MAE"] = resumen["MAE"].round(2)

st.dataframe(resumen, use_container_width=True, hide_index=True)

st.divider()
