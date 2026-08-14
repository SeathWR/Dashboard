import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


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
        df.dropna(subset=["diferencia"])
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

# ── Sección 4: Comparación de métricas de modelos ───────────────────────────
st.subheader("🤖 Comparación de Modelos — Random Forest vs. XGBoost vs. NowCast")
st.markdown(
    "Métricas de desempeño calculadas sobre el periodo de estudio completo "
    "(Enero 2024 – Diciembre 2025). "
    "**Nota:** la comparación con NowCast es estructuralmente asimétrica — "
    "NowCast utiliza lecturas de estación en tiempo real, mientras que los modelos "
    "ML predicen sin acceso a ese dato inmediato."
)

# ── Datos de métricas ────────────────────────────────────────────────────────
metricas = pd.DataFrame({
    "Modelo": ["Random Forest Optimizado", "XGBoost", "NowCast IBOCA"],
    "RMSE": [6.7050, 6.6824, 3.9885],
    "MAE": [4.8114, 4.7461, 2.8479],
    "R²": [0.5772, 0.5801, 0.8505]
})

colores_modelos = {
    "Random Forest Optimizado": "#1f77b4",
    "XGBoost": "#2ca02c",
    "NowCast IBOCA": "#ff7f0e"
}

col_m1, col_m2 = st.columns([1, 2])

# ── Tabla de métricas ────────────────────────────────────────────────────────
with col_m1:
    st.markdown("**Tabla resumen**")

    # Agregar columna de color como emoji para identificar cada modelo
    metricas_display = metricas.copy()
    metricas_display.insert(0, "Color", ["🔵", "🟢", "🟠"])

    st.dataframe(
        metricas_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Color": st.column_config.TextColumn("", width="small"),
            "Modelo": st.column_config.TextColumn("Modelo"),
            "RMSE": st.column_config.NumberColumn("RMSE", format="%.4f"),
            "MAE": st.column_config.NumberColumn("MAE", format="%.4f"),
            "R²": st.column_config.NumberColumn("R²", format="%.4f")
        }
    )

    st.caption(
        "RMSE y MAE en µg/m³ — valores más bajos indican mejor desempeño. "
        "R² más cercano a 1 indica mejor ajuste."
    )

# ── Gráfico de barras agrupadas ──────────────────────────────────────────────
with col_m2:
    metricas_melted = metricas.melt(
        id_vars="Modelo",
        value_vars=["RMSE", "MAE", "R²"],
        var_name="Métrica",
        value_name="Valor"
    )

    fig_metricas = px.bar(
        metricas_melted,
        x="Métrica",
        y="Valor",
        color="Modelo",
        barmode="group",
        color_discrete_map=colores_modelos,
        title="Comparación visual de métricas por modelo",
        text_auto=".4f"
    )

    fig_metricas.update_traces(textposition="outside", textfont_size=11)
    fig_metricas.update_layout(
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Valor",
        xaxis_title=""
    )

    st.plotly_chart(fig_metricas, use_container_width=True)

st.divider()

# ── Sección 5: Importancia de variables (Feature Importance) ─────────────────
st.subheader("📌 Importancia de Variables — Random Forest Optimizado")
st.markdown(
    "Contribución relativa de cada variable predictora en el modelo Random Forest. "
    "Un valor más alto indica mayor influencia en la predicción de PM2.5."
)

@st.cache_data
def cargar_importancia():
    return pd.read_csv("data/feature_importance_rf.csv", encoding="utf-8-sig")

df_imp = cargar_importancia()

# Etiquetas legibles para cada variable
etiquetas = {
    'pm25_lag1':              'PM2.5 hora anterior (lag 1)',
    'pm25_lag2':              'PM2.5 hace 2 horas (lag 2)',
    'pm25_lag3':              'PM2.5 hace 3 horas (lag 3)',
    'pm25_lag6':              'PM2.5 hace 6 horas (lag 6)',
    'pm25_lag24':             'PM2.5 hace 24 horas (lag 24)',
    'hora':                   'Hora del día',
    'mes':                    'Mes del año',
    'temperature_2m':         'Temperatura (°C)',
    'relative_humidity_2m':   'Humedad relativa (%)',
    'wind_speed_10m':         'Velocidad del viento (km/h)',
    'direct_radiation':       'Radiación solar directa (W/m²)',
    'surface_pressure':       'Presión superficial (hPa)'
}

df_imp['variable_label'] = df_imp['variable'].map(etiquetas).fillna(df_imp['variable'])
df_imp['porcentaje'] = (df_imp['importancia'] * 100).round(2)
df_imp = df_imp.sort_values('importancia', ascending=True)

col_i1, col_i2 = st.columns([2, 1])

with col_i1:
    fig_imp = go.Figure()

    fig_imp.add_trace(go.Bar(
        x=df_imp['importancia'],
        y=df_imp['variable_label'],
        orientation='h',
        marker=dict(
            color=df_imp['importancia'],
            colorscale='Teal',
            showscale=False
        ),
        text=df_imp['porcentaje'].apply(lambda x: f"{x:.2f}%"),
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>"
    ))

    fig_imp.update_layout(
        title="Importancia relativa de variables predictoras",
        xaxis_title="Importancia relativa",
        yaxis_title="",
        height=480,
        margin=dict(l=10, r=80, t=50, b=40),
        xaxis=dict(range=[0, df_imp['importancia'].max() * 1.25])
    )

    st.plotly_chart(fig_imp, use_container_width=True)

with col_i2:
    st.markdown("**Tabla de importancia**")
    st.dataframe(
        df_imp[['variable_label', 'porcentaje']]
        .sort_values('porcentaje', ascending=False)
        .rename(columns={
            'variable_label': 'Variable',
            'porcentaje': 'Importancia (%)'
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            'Importancia (%)': st.column_config.NumberColumn(
                'Importancia (%)', format="%.2f%%"
            )
        }
    )

    st.caption(
        "pm25_lag1 concentra aproximadamente el 73% del poder predictivo, "
        "confirmando la alta autocorrelación temporal del PM2.5 en Fontibón."
    )

st.divider()

# ── Sección 6: PM2.5 Real vs. Predicho ──────────────────────────────────────
st.subheader("📉 PM2.5 Real vs. Predicho — Conjunto de Prueba")
st.markdown(
    "Comparación hora a hora entre la concentración real de PM2.5 y las predicciones "
    "del modelo Random Forest, XGBoost y el método NowCast durante el periodo de prueba."
)

@st.cache_data
def cargar_resultados():
    df = pd.read_csv("data/resultados_prediccion.csv",
                     encoding="utf-8-sig",
                     parse_dates=["fecha_hora"])
    return df

df_res = cargar_resultados()

# ── Filtros ──────────────────────────────────────────────────────────────────
col_r1, col_r2, col_r3 = st.columns(3)

with col_r1:
    anios_res = sorted(df_res["fecha_hora"].dt.year.unique())
    anio_res = st.selectbox("Año ", anios_res, key="anio_res")

with col_r2:
    meses_res = sorted(df_res[df_res["fecha_hora"].dt.year == anio_res]["fecha_hora"].dt.month.unique())
    mes_res = st.selectbox("Mes ", meses_res,
                           format_func=lambda x: meses[x],
                           key="mes_res")

with col_r3:
    modelos_sel = st.multiselect(
        "Modelos a mostrar",
        options=["Random Forest", "XGBoost", "NowCast"],
        default=["Random Forest", "XGBoost", "NowCast"]
    )

# ── Filtrar datos ─────────────────────────────────────────────────────────────
df_res_filtrado = df_res[
    (df_res["fecha_hora"].dt.year == anio_res) &
    (df_res["fecha_hora"].dt.month == mes_res)
].copy()

if df_res_filtrado.empty:
    st.warning("No hay datos de predicción para el periodo seleccionado. "
               "Recuerda que los resultados corresponden solo al conjunto de prueba (último 20% cronológico).")
else:
    # ── Gráfico serie temporal ────────────────────────────────────────────────
    fig_pred = go.Figure()

    fig_pred.add_trace(go.Scatter(
        x=df_res_filtrado["fecha_hora"],
        y=df_res_filtrado["pm25_real"],
        name="PM2.5 Real",
        line=dict(color="#2C3E50", width=2),
        hovertemplate="<b>Real</b>: %{y:.2f} µg/m³<br>%{x}<extra></extra>"
    ))

    if "Random Forest" in modelos_sel:
        fig_pred.add_trace(go.Scatter(
            x=df_res_filtrado["fecha_hora"],
            y=df_res_filtrado["pm25_predicho_rf"],
            name="Random Forest",
            line=dict(color="#1f77b4", width=1.5, dash="dot"),
            hovertemplate="<b>RF</b>: %{y:.2f} µg/m³<br>%{x}<extra></extra>"
        ))

    if "XGBoost" in modelos_sel:
        fig_pred.add_trace(go.Scatter(
            x=df_res_filtrado["fecha_hora"],
            y=df_res_filtrado["pm25_predicho_xgb"],
            name="XGBoost",
            line=dict(color="#2ca02c", width=1.5, dash="dash"),
            hovertemplate="<b>XGBoost</b>: %{y:.2f} µg/m³<br>%{x}<extra></extra>"
        ))

    if "NowCast" in modelos_sel:
        fig_pred.add_trace(go.Scatter(
            x=df_res_filtrado["fecha_hora"],
            y=df_res_filtrado["pm25_nowcast"],
            name="NowCast",
            line=dict(color="#ff7f0e", width=1.5, dash="longdash"),
            hovertemplate="<b>NowCast</b>: %{y:.2f} µg/m³<br>%{x}<extra></extra>"
        ))

    fig_pred.add_hline(
        y=37,
        line_dash="dash",
        line_color="red",
        annotation_text="Límite Res. 2254/2017 (37 µg/m³)",
        annotation_position="top left"
    )

    fig_pred.update_layout(
        title=f"PM2.5 Real vs. Predicho — {meses[mes_res]} {anio_res}",
        xaxis_title="Fecha y hora",
        yaxis_title="Concentración PM2.5 (µg/m³)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=460
    )

    st.plotly_chart(fig_pred, use_container_width=True)

# ── Métricas del periodo seleccionado ────────────────────────────────────
    st.markdown(f"**Métricas del periodo — {meses[mes_res]} {anio_res}**")
    col_p1, col_p2, col_p3 = st.columns(3)

    real = df_res_filtrado["pm25_real"].dropna()

    for col_met, nombre, col_pred in zip(
        [col_p1, col_p2, col_p3],
        ["Random Forest", "XGBoost", "NowCast"],
        ["pm25_predicho_rf", "pm25_predicho_xgb", "pm25_nowcast"]
    ):
        if nombre in modelos_sel:
            pred = df_res_filtrado[col_pred].dropna()
            idx  = real.index.intersection(pred.index)
            if len(idx) > 0:
                r = real[idx].values
                p = pred[idx].values
                rmse = np.sqrt(np.mean((r - p) ** 2))
                ss_res = np.sum((r - p) ** 2)
                ss_tot = np.sum((r - np.mean(r)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                col_met.metric(
                    label=nombre,
                    value=f"RMSE: {rmse:.2f} µg/m³",
                    delta=f"R²: {r2:.3f}",
                    delta_color="off"
                )

st.divider()
