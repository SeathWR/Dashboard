import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime

# 1. Configuración Responsiva
st.set_page_config(
    page_title="Aire Fontibón - PM2.5",
    page_icon="🍃",
    layout="wide", # Permite expansión en escritorio y apilamiento en móvil
    initial_sidebar_state="collapsed"
)

# 2. Inyección de CSS (Color de fuentes oscuro explícito para evitar fallos en Modo Oscuro)
st.markdown("""
<style>
    .circle-container { display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 10px auto; }
    .circle-badge { width: 220px; height: 220px; border-radius: 50%; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #FFFFFF; box-shadow: 0 8px 24px rgba(0,0,0,0.12); transition: transform 0.3s ease; }
    .circle-badge:hover { transform: scale(1.02); }
    .pm-val { font-size: 3.8rem; font-weight: 800; line-height: 1; }
    .pm-unit { font-size: 1rem; font-weight: 400; margin-top: 5px; opacity: 0.9; }
    .risk-card { border-radius: 12px; padding: 16px 20px; margin: 15px auto 25px auto; text-align: center; max-width: 600px; }
    .metric-pill { background-color: #F8F9FA; border-radius: 10px; padding: 15px; text-align: center; border: 1px solid #E9ECEF; height: 100%; }
    
    @media (max-width: 768px) {
        .pm-val { font-size: 3.2rem; }
        .circle-badge { width: 190px; height: 190px; }
    }
</style>
""", unsafe_allow_html=True)

# 3. Lógica Semiótica IBOCA
def clasificar_iboca(pm25):
    if pm25 <= 12.0: return "Bajo", "#2ECC71", "#E8F8F0", "Aire limpio. Condiciones óptimas para toda actividad."
    elif pm25 <= 35.4: return "Moderado", "#F1C40F", "#FEF9E7", "Aceptable. Personas muy sensibles deben considerar pausas."
    elif pm25 <= 55.4: return "Regular", "#E67E22", "#FBEEE6", "Alerta Fase 1: Grupos sensibles reduzcan esfuerzo al aire libre."
    elif pm25 <= 151.2: return "Alto", "#E74C3C", "#FDEDEC", "Alerta Fase 2: Riesgo alto. Evite ejercicio en exteriores."
    else: return "Peligroso", "#8E44AD", "#F4ECF7", "Emergencia ambiental: Permanezca en espacios cerrados."

# 4. Extracción de Datos (ETL)
@st.cache_data
def load_data():
    pred_df = pd.read_csv("data/resultados_prediccion.csv")
    meteo_df = pd.read_csv("data/data_openmeteo.csv")
    metricas_df = pd.read_csv("data/metricas_modelos.csv")
    feat_imp_df = pd.read_csv("data/feature_importance_rf.csv")
    
    pred_df['fecha_hora'] = pd.to_datetime(pred_df['fecha_hora'])
    meteo_df['fecha_hora'] = pd.to_datetime(meteo_df['fecha_hora'])
    
    pred_df = pred_df.sort_values('fecha_hora')
    meteo_df = meteo_df.sort_values('fecha_hora')
    
    return pred_df, meteo_df, metricas_df, feat_imp_df

pred_df, meteo_df, metricas_df, feat_imp_df = load_data()

min_date = pred_df['fecha_hora'].min().date()
max_date = pred_df['fecha_hora'].max().date()
default_start = max_date - datetime.timedelta(days=3)

# 5. GESTIÓN DEL ESTADO (Para el Reset y Filtros)
if 'fechas' not in st.session_state:
    st.session_state.fechas = (default_start, max_date)
if 'horas' not in st.session_state:
    st.session_state.horas = (0, 23)
if 'mostrar_actual' not in st.session_state:
    st.session_state.mostrar_actual = True

# Callbacks para actualizar el estado
def reset_actual():
    st.session_state.mostrar_actual = True
    st.session_state.fechas = (default_start, max_date)
    st.session_state.horas = (0, 23)

def on_filter_change():
    st.session_state.mostrar_actual = False


# ----------------- CABECERA Y FILTROS -----------------
st.markdown("<h3 style='text-align: center; margin-bottom: 0px;'>Fontibón | Calidad del Aire</h3>", unsafe_allow_html=True)
st.write("---")

col_date, col_time, col_reset = st.columns([2, 2, 1])

with col_date:
    rango_fechas = st.date_input(
        "📅 Rango de Fechas:",
        value=st.session_state.fechas,
        min_value=min_date,
        max_value=max_date,
        key="fechas",
        on_change=on_filter_change
    )

with col_time:
    rango_horas = st.slider(
        "⏱️ Franja Horaria:",
        min_value=0, max_value=23,
        value=st.session_state.horas,
        format="%02d:00",
        key="horas",
        on_change=on_filter_change
    )

with col_reset:
    st.write("") # Espaciador para alinear el botón verticalmente
    st.write("") 
    st.button("🔄 Ver Actual", on_click=reset_actual, use_container_width=True, help="Borrar filtros y ver el estado más reciente")


# ----------------- LÓGICA DE SINCRONIZACIÓN DE DATOS -----------------
if st.session_state.mostrar_actual:
    # MODO 1: Sin filtros (Reset activo) -> Muestra el último dato exacto
    df_plot = pred_df.tail(24).copy() # Gráfico de las últimas 24h
    
    pm_mostrar = round(pred_df.iloc[-1]['pm25_real'], 1)
    viento_mostrar = round(meteo_df.iloc[-1]['wind_speed_10m'], 1)
    humedad_mostrar = round(meteo_df.iloc[-1]['relative_humidity_2m'], 1)
    temp_mostrar = round(meteo_df.iloc[-1]['temperature_2m'], 1)
    
    subtitulo = f"Estación Tráfico Carrera 104 • Actualizado: {pred_df.iloc[-1]['fecha_hora'].strftime('%Y-%m-%d %H:00')}"
    
else:
    # MODO 2: Filtros aplicados -> Muestra los PROMEDIOS del rango seleccionado
    if len(rango_fechas) == 2:
        f_ini, f_fin = rango_fechas
        h_ini, h_fin = rango_horas
        
        # Filtrar Predicciones
        mask_fecha = (pred_df['fecha_hora'].dt.date >= f_ini) & (pred_df['fecha_hora'].dt.date <= f_fin)
        mask_hora = (pred_df['fecha_hora'].dt.hour >= h_ini) & (pred_df['fecha_hora'].dt.hour <= h_fin)
        df_plot = pred_df.loc[mask_fecha & mask_hora].copy()
        
        # Filtrar Meteorología
        mask_meteo_f = (meteo_df['fecha_hora'].dt.date >= f_ini) & (meteo_df['fecha_hora'].dt.date <= f_fin)
        mask_meteo_h = (meteo_df['fecha_hora'].dt.hour >= h_ini) & (meteo_df['fecha_hora'].dt.hour <= h_fin)
        meteo_plot = meteo_df.loc[mask_meteo_f & mask_meteo_h].copy()
        
        if not df_plot.empty:
            pm_mostrar = round(df_plot['pm25_real'].mean(), 1)
            viento_mostrar = round(meteo_plot['wind_speed_10m'].mean(), 1)
            humedad_mostrar = round(meteo_plot['relative_humidity_2m'].mean(), 1)
            temp_mostrar = round(meteo_plot['temperature_2m'].mean(), 1)
            subtitulo = f"Promedio del periodo: {f_ini} al {f_fin} (Franja: {h_ini:02d}:00 - {h_fin:02d}:00)"
        else:
            pm_mostrar, viento_mostrar, humedad_mostrar, temp_mostrar = 0, 0, 0, 0
            subtitulo = "Sin datos en el rango seleccionado"
    else:
        df_plot = pd.DataFrame()
        pm_mostrar, viento_mostrar, humedad_mostrar, temp_mostrar = 0, 0, 0, 0
        subtitulo = "Seleccione un rango de fechas válido"

estado, color_hex, fondo_hex, recomendacion = clasificar_iboca(pm_mostrar)

# ----------------- BLOQUE 1: DIAGNÓSTICO (CÍRCULO MUNARI) -----------------
st.markdown(f"<p style='text-align: center; color: gray;'>{subtitulo}</p>", unsafe_allow_html=True)

if pm_mostrar > 0:
    st.markdown(f"""
    <div class="circle-container">
        <div class="circle-badge" style="background-color: {color_hex};">
            <span class="pm-val">{pm_mostrar}</span>
            <span class="pm-unit">µg/m³ PM₂.₅</span>
        </div>
    </div>
    <div class="risk-card" style="background-color: {fondo_hex}; color: {color_hex}; border: 1px solid {color_hex};">
        <span style="font-size: 1.2rem; font-weight: 700;">Nivel {estado.upper()}</span><br>
        <span style="font-size: 1rem; color: #333333;">{recomendacion}</span>
    </div>
    """, unsafe_allow_html=True)

    # ----------------- BLOQUE 2: DISPERSIÓN ATMOSFÉRICA SINCRONIZADA -----------------
    # Se añade etiqueta "(Promedio)" si hay filtros aplicados, para no confundir al usuario
    label_suffix = "" if st.session_state.mostrar_actual else " (Promedio)"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-pill">
            <span style="font-size:0.9rem; color:gray;">Viento{label_suffix}</span><br>
            <span style="font-size:1.4rem; font-weight:bold; color:#2C3E50;">{viento_mostrar}</span> 
            <span style="font-size:0.8rem; color:#2C3E50;">km/h</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-pill">
            <span style="font-size:0.9rem; color:gray;">Humedad{label_suffix}</span><br>
            <span style="font-size:1.4rem; font-weight:bold; color:#2C3E50;">{humedad_mostrar}</span> 
            <span style="font-size:0.8rem; color:#2C3E50;">%</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-pill">
            <span style="font-size:0.9rem; color:gray;">Temp.{label_suffix}</span><br>
            <span style="font-size:1.4rem; font-weight:bold; color:#2C3E50;">{temp_mostrar}</span> 
            <span style="font-size:0.8rem; color:#2C3E50;">°C</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ----------------- BLOQUE 3: GRÁFICO PLOTLY -----------------
    st.subheader("Comportamiento y Predicción")
    fig = go.Figure()
    
    # 1. Línea Real (Gris oscuro/Azul marino)
    fig.add_trace(go.Scatter(
        x=df_plot['fecha_hora'], y=df_plot['pm25_real'], 
        mode='lines+markers', name='Concentración Real', 
        line=dict(color='#2C3E50', width=2), marker=dict(size=4)
    ))
    
    # 2. Línea NowCast IBOCA (Roja punteada)
    fig.add_trace(go.Scatter(
        x=df_plot['fecha_hora'], y=df_plot['pm25_nowcast'], 
        mode='lines', name='NowCast (IBOCA)', 
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    # 3. Línea Random Forest (Verde)
    fig.add_trace(go.Scatter(
        x=df_plot['fecha_hora'], y=df_plot['pm25_predicho_rf'], 
        mode='lines', name='Random Forest', 
        line=dict(color='#27AE60', width=2.5)
    ))

    # 4. Línea XGBoost (Azul) - ¡NUEVA!
    fig.add_trace(go.Scatter(
        x=df_plot['fecha_hora'], y=df_plot['pm25_predicho_xgb'], 
        mode='lines', name='XGBoost', 
        line=dict(color='#2980B9', width=2.5)
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        # Se ajusta la leyenda para que se adapte si hay muchos elementos
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(showgrid=False, title="Fecha y Hora"),
        yaxis=dict(showgrid=True, title="µg/m³"),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------- BLOQUE 4: MÓDULO DIDÁCTICO -----------------
with st.expander("📊 Profundización Técnica: Rendimiento del Modelo"):
    st.write("### Desempeño Estadístico (Métricas)")
    st.dataframe(metricas_df, use_container_width=True, hide_index=True)
    st.write("### Variables Clave de Predicción (Feature Importance)")
    st.bar_chart(feat_imp_df.set_index("variable")["importancia"], horizontal=True)