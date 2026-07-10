"""
╔══════════════════════════════════════════════════════════════════╗
║        PREDICTLINE 5 PET — ARCA CONTINENTAL LINDLEY             ║
║        Predicción de Paradas Operativas · Línea 5 PET           ║
║        Curso  : Applied Machine Learning                        ║
║        Año    : 2026                                            ║
╚══════════════════════════════════════════════════════════════════╝

CÓMO AGREGAR TU MODELO ENTRENADO
─────────────────────────────────
1. En tu notebook de Colab, ejecuta al final:

       import joblib
       joblib.dump(mlp_img,  "modelo_linea5.pkl")
       joblib.dump(scaler,   "scaler_linea5.pkl")
       # guarda también las columnas que usó el modelo:
       import json
       json.dump(list(X_train_scaled.columns), open("columnas_modelo.json","w"))

       from google.colab import files
       files.download("modelo_linea5.pkl")
       files.download("scaler_linea5.pkl")
       files.download("columnas_modelo.json")

2. Pon los 3 archivos en la MISMA carpeta que este .py:
       mi_proyecto/
       ├── app_linea5_pet.py
       ├── modelo_linea5.pkl
       ├── scaler_linea5.pkl
       └── columnas_modelo.json

3. Ejecuta:  streamlit run app_linea5_pet.py
   La app detecta automáticamente si el modelo existe y lo usa.
   Si no existe, corre en modo simulado.
"""

# ══════════════════════════════════════════════════════════════════
# 0. IMPORTACIONES
# ══════════════════════════════════════════════════════════════════

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import time
import random
import os
import json

# ══════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN GENERAL DE LA PÁGINA
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PredictLine 5 PET | Arca Continental Lindley",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

with st.sidebar:

    st.title("🏭 PredictLine")

    st.markdown("---")

    st.markdown("### Navegación")

    st.markdown("""
    - 🏠 Inicio
    - 📘 Sistema
    - 📋 Variables
    - ⚙️ Ingreso de datos
    - 🤖 Procesamiento
    - 📊 Resultado
    - 🛡️ Recomendaciones
    - 📈 Dashboard
    - 🎓 Proyecto
    """)

    st.markdown("---")

    st.info("Modelo de predicción de paradas operativas mediante Machine Learning.")

# ══════════════════════════════════════════════════════════════════
# 2. CARGA DEL MODELO ENTRENADO (si existe en la carpeta)
# ══════════════════════════════════════════════════════════════════

# ── Variables categóricas y sus valores reales del dataset ────────
TURNOS          = ["T1", "T2", "T3"]
BEBIDAS         = ["Coca-Cola Reducida", "Coca-Cola Sin azúcar KAIZEN",
                   "Inca Kola ReducidaV2", "Inca Kola Sin azúcar", "San Luis CG"]
TAMANOS         = [1.0, 1.5, 2.25, 2.5, 3.0]
TAMANOS_LABEL   = {"1.0 L": 1.0, "1.5 L": 1.5, "2.25 L": 2.25,
                   "2.5 L": 2.5, "3.0 L": 3.0}
NIVEL2_OPCIONES = [
    "AJUSTES OPERACIONALES", "EXCESO DE TIEMPOS",
    "FALTA DE PERSONAL", "MINUTOS SIN DEFINIR",
]
NIVEL3_OPCIONES = [
    "CORRECION OPERACIONAL", "EXCESO DE TIEMPOS", "FALTA DE PERSONAL",
    "LIMPIEZA", "MINUTOS SIN DEFINIR", "PARAMETROS DEL PROCESO",
    "REGULACION DEL EQUIPO OPERACIONAL", "REGULACION DEL EQUIPO POR CALIDAD",
    "REGULACION DEL EQUIPO POR MANTENIMIENTO",
]
NIVEL4_OPCIONES = [
    "MINUTOS SIN DEFINIR", "ACUMULADOR HELIX", "ALIMENTADOR DE TAPAS",
    "APLICADOR DE TAPAS", "CAPSULADORA", "CODIFICADOR", "DESPALETIZADORA",
    "ENCAJONADORA", "ENVOLVEDORA", "ETIQUETADORA", "FALTA DE PERSONAL",
    "INSPECTOR ELECTRÓNICO DE BOTELLAS", "INSPECTOR ELECTRÓNICO DE NIVEL",
    "INSPECTOR OLORES CONTAMINANTES", "LLENADORA (ENVASADORA)", "PALETIZADORA",
    "PANTALLA DE INSPECCIÓN LLENAS", "SETUP", "SISTEMA DE QUASI JARABE",
    "SISTEMA DE REFRIGERACIÓN", "TERMOENCOGIBLE DE PAQUETES",
    "TRANSPORTADOR DE BOTELLAS", "TRANSPORTADOR DE PALETAS",
    "TRANSPORTADOR DE PAQUETES", "TRANSPORTADOR NEUMÁTICO",
]
DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# ── Columnas numéricas que se escalaron con StandardScaler ────────
COLUMNAS_A_ESCALAR = [
    "hora_dia", "mes",
    "paradas_operativas_ult_30min", "paradas_operativas_ult_60min",
    "paradas_operativas_ult_120min", "min_operativo_ult_120min",
    "tiempo_desde_ultima_parada_operativa_min",
    "paradas_operativas_ult_8h", "min_operativo_ult_8h",
    "fallas_mismo_nivel4_ult_8h", "min_mismo_nivel4_ult_8h",
]

# ── Intentar cargar el modelo real ───────────────────────────────
MODELO_PATH   = "modelo_stacking_final.pkl"
SCALER_PATH   = "scaler_linea5.pkl"          # opcional, puede no existir
COLUMNAS_PATH = "columnas_modelo.pkl"        # guardado con joblib, no json

modelo_cargado  = None
scaler_cargado  = None
columnas_modelo = None
MODO_REAL       = False

try:
    import joblib
    if os.path.exists(MODELO_PATH):
        modelo_cargado = joblib.load(MODELO_PATH)
        if os.path.exists(SCALER_PATH):
            scaler_cargado = joblib.load(SCALER_PATH)
        if os.path.exists(COLUMNAS_PATH):
            columnas_modelo = joblib.load(COLUMNAS_PATH)
            # por si se guardó como Index de pandas en vez de lista
            columnas_modelo = list(columnas_modelo)
        MODO_REAL = True
except Exception as e:
    import traceback
    MODO_REAL = False
    _error_carga_modelo = traceback.format_exc()
else:
    _error_carga_modelo = None

# ── Panel de diagnóstico oculto ──────────────────────────────────
# Solo aparece si abres la app con ?debug=true al final de la URL.
# Ejemplo: http://localhost:8501/?debug=true
if st.query_params.get("debug") == "true":
    with st.sidebar:
        with st.expander("🔧 Diagnóstico de carga del modelo"):
            st.write(f"**{MODELO_PATH}** existe: {os.path.exists(MODELO_PATH)}")
            st.write(f"**{SCALER_PATH}** existe: {os.path.exists(SCALER_PATH)}")
            st.write(f"**{COLUMNAS_PATH}** existe: {os.path.exists(COLUMNAS_PATH)}")
            st.write(f"**MODO_REAL:** {MODO_REAL}")
            if os.path.exists(MODELO_PATH):
                tam_mb = os.path.getsize(MODELO_PATH) / (1024 * 1024)
                st.write(f"Tamaño de {MODELO_PATH}: {tam_mb:.1f} MB")
                with open(MODELO_PATH, "rb") as f:
                    primeros_bytes = f.read(16)
                st.write(f"Primeros bytes (hex): {primeros_bytes.hex()}")
            if _error_carga_modelo:
                st.error("Error al cargar el modelo:")
                st.code(_error_carga_modelo, language="python")

# ══════════════════════════════════════════════════════════════════
# 3. ESTILOS CSS PERSONALIZADOS
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  /* ── Fuente ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

 /* ── Fondo general ── */
.stApp {
    background: #111827;
    color: #F9FAFB;
}

/* ── Ocultar elementos Streamlit por defecto ── */
#MainMenu, footer {
    visibility: hidden;
}
/* Mantener visible el header solo para el botón de abrir/cerrar la barra lateral */
header {
    background: transparent !important;
}
header [data-testid="stToolbar"] {
    visibility: hidden;
}
/* Ocultar la franja decorativa de colores que Streamlit pone arriba del todo */
[data-testid="stDecoration"] {
    display: none !important;
}

/* Contenedor principal centrado */
.block-container{
    max-width: 1400px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

  /* ────────────────────────────────────────────
     HERO SECTION
  ──────────────────────────────────────────── */
  .hero-section {
      background: linear-gradient(135deg, #0d2137 0%, #1a3a5c 50%, #0d2137 100%);
      border-radius: 20px;
      padding: 3.5rem 3rem;
      text-align: center;
      border: 1px solid #1e4976;
      margin-bottom: 2rem;
      position: relative;
      overflow: hidden;
  }
  .hero-section::before {
      content: '';
      position: absolute; top: -50%; left: -50%;
      width: 200%; height: 200%;
      background: radial-gradient(circle, rgba(59,130,246,0.06) 0%, transparent 60%);
      pointer-events: none;
  }
  .hero-title {
      font-size: 3rem; font-weight: 800;
      background: linear-gradient(90deg, #60a5fa, #34d399);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
  }
  .hero-subtitle { font-size: 1.15rem; color: #94a3b8; margin-bottom: 1.5rem; }
  .hero-badge {
      display: inline-block;
      background: rgba(59,130,246,0.15);
      border: 1px solid rgba(59,130,246,0.4);
      color: #60a5fa; border-radius: 50px;
      padding: 0.3rem 1rem;
      font-size: 0.82rem; font-weight: 600;
      letter-spacing: 0.05em; margin-bottom: 2rem;
  }

  /* ────────────────────────────────────────────
     TARJETAS GENÉRICAS
  ──────────────────────────────────────────── */
  .card {
      background: #132030; border: 1px solid #1e3a52;
      border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem;
      transition: border-color .25s, transform .2s;
  }
  .card:hover { border-color: #3b82f6; transform: translateY(-2px); }
  .card-title {
      font-size: 1rem; font-weight: 700; color: #60a5fa;
      margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.5rem;
  }

  /* ────────────────────────────────────────────
     TARJETAS DE VARIABLES
  ──────────────────────────────────────────── */
  .var-card {
      background: #0d1f31; border: 1px solid #1e3a52;
      border-radius: 12px; padding: 1rem 1.2rem;
      text-align: center; transition: all .25s;
  }
  .var-card:hover { border-color: #34d399; background: #0f2840; }
  .var-icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
  .var-name { font-size: 0.82rem; font-weight: 600; color: #94a3b8; }

  /* ────────────────────────────────────────────
     TARJETAS DE ALERTA
  ──────────────────────────────────────────── */
  .alert-card-red {
      background: linear-gradient(135deg, #3b0a0a, #5c1a1a);
      border: 2px solid #ef4444; border-radius: 18px;
      padding: 2.5rem; text-align: center;
      animation: pulsered 2s infinite;
  }
  @keyframes pulsered {
      0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
      50%      { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
  }
  .alert-card-green {
      background: linear-gradient(135deg, #062016, #0a3322);
      border: 2px solid #22c55e; border-radius: 18px;
      padding: 2.5rem; text-align: center;
  }
  .alert-label { font-size: 2.8rem; font-weight: 800; letter-spacing: 0.08em; }
  .alert-label-red   { color: #ef4444; }
  .alert-label-green { color: #22c55e; }

  /* ────────────────────────────────────────────
     CAJAS DE MÉTRICAS
  ──────────────────────────────────────────── */
  .metric-box {
      background: #0d1f31; border: 1px solid #1e3a52;
      border-radius: 12px; padding: 1.2rem; text-align: center;
  }
  .metric-value { font-size: 2rem; font-weight: 800; color: #60a5fa; }
  .metric-label {
      font-size: 0.78rem; color: #64748b;
      text-transform: uppercase; letter-spacing: 0.06em;
  }

  /* ────────────────────────────────────────────
     TARJETAS DE RECOMENDACIONES
  ──────────────────────────────────────────── */
  .rec-card-red {
      background: #200a0a; border-left: 4px solid #ef4444;
      border-radius: 8px; padding: 0.9rem 1.2rem; margin-bottom: 0.6rem;
      display: flex; align-items: center; gap: 0.7rem; font-size: 0.92rem;
  }
  .rec-card-green {
      background: #061810; border-left: 4px solid #22c55e;
      border-radius: 8px; padding: 0.9rem 1.2rem; margin-bottom: 0.6rem;
      display: flex; align-items: center; gap: 0.7rem; font-size: 0.92rem;
  }

  /* ────────────────────────────────────────────
     PASOS DE PROCESAMIENTO
  ──────────────────────────────────────────── */
  .step-item {
      display: flex; align-items: center; gap: 0.8rem;
      padding: 0.6rem 0; font-size: 0.9rem; color: #94a3b8;
      border-bottom: 1px solid #1e3a52;
  }
  .step-item:last-child { border-bottom: none; }
  .step-check { color: #34d399; font-size: 1.1rem; }

  /* ────────────────────────────────────────────
     DIVISOR VISUAL ENTRE SECCIONES
  ──────────────────────────────────────────── */
  .section-divider {
      display: flex; align-items: center; gap: 1rem;
      margin: 3rem 0 2.5rem; position: relative;
  }
  .section-divider::before, .section-divider::after {
      content: ''; flex: 1; height: 1px;
      background: linear-gradient(90deg, transparent, #1e4976, #3b82f6, #1e4976, transparent);
  }
  .divider-label {
      display: flex; align-items: center; gap: 0.5rem;
      background: #0d1f31; border: 1px solid #1e4976;
      border-radius: 50px; padding: 0.35rem 1.1rem;
      font-size: 0.78rem; font-weight: 600; color: #60a5fa;
      letter-spacing: 0.08em; text-transform: uppercase;
      white-space: nowrap; box-shadow: 0 0 20px rgba(59,130,246,0.15);
  }
  .divider-dot {
      width: 6px; height: 6px; border-radius: 50%; background: #3b82f6;
      animation: blink 1.8s ease-in-out infinite;
  }
  @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

  /* ────────────────────────────────────────────
     CABECERA DE SECCIÓN
  ──────────────────────────────────────────── */
  .section-header { display: flex; align-items: center; gap: 1rem; margin: 2.5rem 0 1.5rem; }
  .section-number {
      width: 36px; height: 36px;
      background: linear-gradient(135deg, #1d4ed8, #2563eb);
      border-radius: 50%; display: flex; align-items: center; justify-content: center;
      font-weight: 800; font-size: 0.9rem; color: #fff; flex-shrink: 0;
  }
  .section-title { font-size: 1.4rem; font-weight: 700; color: #e2e8f0; }
  .section-line { flex: 1; height: 1px; background: linear-gradient(90deg, #1e3a52, transparent); }

  /* ────────────────────────────────────────────
     BADGE DE MODO (REAL / SIMULADO)
  ──────────────────────────────────────────── */
  .badge-real {
      background: rgba(52,211,153,0.15); border: 1px solid #34d399;
      border-radius: 50px; padding: 0.25rem 0.9rem;
      font-size: 0.78rem; font-weight: 600; color: #34d399;
  }
  .badge-sim {
      background: rgba(245,158,11,0.15); border: 1px solid #f59e0b;
      border-radius: 50px; padding: 0.25rem 0.9rem;
      font-size: 0.78rem; font-weight: 600; color: #f59e0b;
  }

  /* ────────────────────────────────────────────
     FOOTER
  ──────────────────────────────────────────── */
  .footer-section {
      background: #0a1520; border: 1px solid #1e3a52;
      border-radius: 14px; padding: 2rem; text-align: center;
      margin-top: 3rem; color: #475569; font-size: 0.85rem;
  }
  .footer-section strong { color: #60a5fa; }

  /* ────────────────────────────────────────────
     BOTONES
  ──────────────────────────────────────────── */
  .stButton > button {
      background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
      color: white !important; border: none !important;
      border-radius: 10px !important; font-weight: 600 !important;
      letter-spacing: 0.03em !important; transition: all .25s !important;
      box-shadow: 0 4px 15px rgba(37,99,235,0.3) !important;
  }
  .stButton > button:hover {
      background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
      box-shadow: 0 6px 25px rgba(37,99,235,0.5) !important;
      transform: translateY(-1px) !important;
  }

  /* ────────────────────────────────────────────
     TABS
  ──────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
      background: #0d1f31; border-radius: 10px;
      padding: 4px; border: 1px solid #1e3a52;
  }
  .stTabs [data-baseweb="tab"]  { color: #64748b !important; font-weight: 500; border-radius: 8px; }
  .stTabs [aria-selected="true"]{ background: #1d4ed8 !important; color: white !important; }

  /* ────────────────────────────────────────────
     INPUTS Y LABELS
  ──────────────────────────────────────────── */
  label { color: #94a3b8 !important; font-size: 0.88rem !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 4. FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════

# ── 4a. Divisor visual ────────────────────────────────────────────
def section_divider(label: str, icon: str = "◆"):
    st.markdown(f"""
    <div class="section-divider">
        <div class="divider-label">
            <div class="divider-dot"></div>
            {icon}&nbsp;&nbsp;{label}
            <div class="divider-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── 4b. Cabecera numerada ─────────────────────────────────────────
def section_header(num: str, title: str):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">{num}</div>
        <span class="section-title">{title}</span>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)


# ── 4c. Velocímetro Plotly ────────────────────────────────────────
def gauge_chart(value: float, title: str = "Probabilidad de Alerta"):
    color = "#ef4444" if value >= 50 else "#f59e0b" if value >= 30 else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 14, "color": "#94a3b8"}},
        number={"suffix": "%", "font": {"size": 32, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569",
                     "tickfont": {"color": "#475569", "size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#0d1f31", "bordercolor": "#1e3a52",
            "steps": [
                {"range": [0,  30],  "color": "rgba(34,197,94,0.12)"},
                {"range": [30, 60],  "color": "rgba(245,158,11,0.12)"},
                {"range": [60, 100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {"line": {"color": "#f8fafc", "width": 3},
                          "thickness": 0.75, "value": value},
        },
    ))
    fig.update_layout(
        height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=20, r=20), font={"family": "Inter"},
    )
    return fig


# ── 4d. Construcción del vector de features para el modelo ────────
def construir_fila(inputs: dict) -> pd.DataFrame:
    """
    Recrea la misma transformación que se hizo en el notebook:
      1. Armar DataFrame con las columnas originales.
      2. get_dummies con drop_first=True sobre las columnas categóricas.
      3. Rellenar columnas que falten (combinaciones no vistas) con 0.
      4. Escalar las columnas numéricas con el scaler guardado.
      5. Reordenar al orden exacto con que se entrenó el modelo.
    """
    row = {
        "hora_dia":                              inputs["hora_dia"],
        "dia_semana":                            inputs["dia_semana"],
        "mes":                                   inputs["mes"],
        "turno_ventana":                         inputs["turno_ventana"],
        "tipo_bebida_actual":                    inputs["tipo_bebida_actual"],
        "tamano_actual":                         float(inputs["tamano_actual"]),
        "paradas_operativas_ult_30min":          inputs["paradas_ult_30"],
        "paradas_operativas_ult_60min":          inputs["paradas_ult_60"],
        "paradas_operativas_ult_120min":         inputs["paradas_ult_120"],
        "min_operativo_ult_120min":              inputs["min_operativo_120"],
        "tiempo_desde_ultima_parada_operativa_min": inputs["tiempo_ultima_parada"],
        "paradas_operativas_ult_8h":             inputs["paradas_ult_8h"],
        "min_operativo_ult_8h":                  inputs["min_operativo_8h"],
        "ultimo_nivel2":                         inputs["ultimo_nivel2"],
        "ultimo_nivel3":                         inputs["ultimo_nivel3"],
        "ultimo_nivel4":                         inputs["ultimo_nivel4"],
        "fallas_mismo_nivel4_ult_8h":            inputs["fallas_nivel4_8h"],
        "min_mismo_nivel4_ult_8h":               inputs["min_nivel4_8h"],
        "paros_mecanicos_ult_24h":               inputs["paros_mecanicos_24h"],
        "min_mecanico_ult_24h":                  inputs["min_mecanico_24h"],
        "paro_mecanico_ult_24h":                 int(inputs["paros_mecanicos_24h"] > 0),
    }

    df = pd.DataFrame([row])

    # Mismo encoding que en el notebook: get_dummies con drop_first=True
    cat_cols = ["dia_semana", "turno_ventana", "tipo_bebida_actual",
                "tamano_actual", "ultimo_nivel2", "ultimo_nivel3", "ultimo_nivel4"]
    # tamano_actual debe ser object para que get_dummies lo tome
    df["tamano_actual"] = df["tamano_actual"].astype(str)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Si hay columnas del modelo que no aparecieron (combo no vista), agregar con 0
    if columnas_modelo:
        for col in columnas_modelo:
            if col not in df.columns:
                df[col] = 0
        df = df[columnas_modelo]   # orden exacto de entrenamiento

    # Escalar las columnas numéricas
    if scaler_cargado is not None:
        cols_esc = [c for c in COLUMNAS_A_ESCALAR if c in df.columns]
        if cols_esc:
            df[cols_esc] = scaler_cargado.transform(df[cols_esc])

    return df


# ── 4e. Predicción con modelo real ───────────────────────────────
def predict_real(inputs: dict) -> dict:
    df = construir_fila(inputs)
    prob      = round(modelo_cargado.predict_proba(df)[0][1] * 100, 1)
    alerta    = prob >= 50
    riesgo    = "ALTO" if prob >= 70 else "MEDIO" if prob >= 40 else "BAJO"
    confianza = round(modelo_cargado.predict_proba(df)[0].max() * 100, 1)
    return {"alerta": alerta, "probabilidad": prob,
            "riesgo": riesgo, "confianza": confianza}


# ── 4f. Predicción simulada (modo demo sin modelo) ────────────────
def predict_simulado(inputs: dict) -> dict:
    score = 0.0
    score += inputs["hora_dia"] / 24 * 0.08
    if inputs["turno_ventana"] == "T3":  score += 0.12
    elif inputs["turno_ventana"] == "T2": score += 0.06
    score += min(inputs["tiempo_ultima_parada"] / 300, 1) * 0.18
    score += min(inputs["paradas_ult_30"] / 8,  1) * 0.22
    score += min(inputs["paradas_ult_60"] / 16, 1) * 0.18
    score += min(inputs["paradas_ult_8h"] / 30, 1) * 0.12
    if inputs["ultimo_nivel4"] == "LLENADORA (ENVASADORA)": score += 0.15
    elif inputs["ultimo_nivel4"] not in ["MINUTOS SIN DEFINIR"]: score += 0.08
    score = min(max(score + random.uniform(-0.03, 0.03), 0), 1)
    prob      = round(score * 100, 1)
    alerta    = prob >= 50
    riesgo    = "ALTO" if prob >= 70 else "MEDIO" if prob >= 40 else "BAJO"
    confianza = round(random.uniform(82, 95), 1)
    return {"alerta": alerta, "probabilidad": prob,
            "riesgo": riesgo, "confianza": confianza}


# ── 4g. Función de predicción unificada ──────────────────────────
def hacer_prediccion(inputs: dict) -> dict:
    if MODO_REAL:
        return predict_real(inputs)
    return predict_simulado(inputs)


# ── 4h. Dashboard histórico ───────────────────────────────────────
def dashboard_charts():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "turno":   np.random.choice(["T1", "T2", "T3"], n, p=[0.34, 0.33, 0.33]),
        "bebida":  np.random.choice(["Inca Kola ReducidaV2", "Coca-Cola Reducida",
                                     "San Luis CG", "Inca Kola Sin azúcar",
                                     "Coca-Cola Sin azúcar KAIZEN"], n,
                                    p=[0.35, 0.30, 0.15, 0.12, 0.08]),
        "hora":    np.random.randint(0, 24, n),
        "alerta":  np.random.choice([0, 1], n, p=[0.62, 0.38]),
    })

    BASE = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#94a3b8"},
        margin=dict(t=40, b=30, l=30, r=20),
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Distribución", "🕐 Por Turno", "🥤 Por Bebida",
        "⏰ Por Hora", "🎯 Importancia de Variables",
    ])

    # ── Tab 1: Distribución general ──
    with tab1:
        counts = df["alerta"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["Sin Alerta", "Con Alerta"],
            values=[counts.get(0, 0), counts.get(1, 0)],
            hole=0.5, marker_colors=["#22c55e", "#ef4444"],
            textfont={"color": "#fff"},
        ))
        fig.update_layout(title="Distribución Histórica de Alertas",
                          legend=dict(font={"color": "#94a3b8"}), **BASE)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Por turno ──
    with tab2:
        tc = df.groupby(["turno", "alerta"]).size().reset_index(name="count")
        fig = px.bar(tc, x="turno", y="count", color="alerta",
                     color_discrete_map={0: "#22c55e", 1: "#ef4444"},
                     labels={"turno": "Turno", "count": "Eventos"},
                     title="Alertas por Turno (T1=Mañana, T2=Tarde, T3=Noche)",
                     barmode="group", template="plotly_dark")
        fig.update_layout(**BASE)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Por bebida ──
    with tab3:
        bc = df.groupby(["bebida", "alerta"]).size().reset_index(name="count")
        fig = px.bar(bc, x="bebida", y="count", color="alerta",
                     color_discrete_map={0: "#22c55e", 1: "#ef4444"},
                     labels={"bebida": "Producto", "count": "Eventos"},
                     title="Alertas por Tipo de Producto",
                     barmode="group", template="plotly_dark")
        fig.update_layout(**BASE)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Por hora ──
    with tab4:
        hc = (df.groupby("hora")["alerta"].mean().reset_index()
                .rename(columns={"alerta": "tasa"}))
        hc["tasa"] *= 100
        fig = px.area(hc, x="hora", y="tasa",
                      labels={"hora": "Hora del Día", "tasa": "Tasa de Alerta (%)"},
                      title="Tasa de Alerta por Hora del Día",
                      template="plotly_dark", color_discrete_sequence=["#3b82f6"])
        fig.update_traces(fill="tozeroy", fillcolor="rgba(59,130,246,0.12)")
        fig.update_layout(**BASE)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Importancia de variables ──
    with tab5:
        variables = [
            "tiempo_desde_ultima_parada", "paradas_ult_60min",
            "paradas_ult_30min", "paradas_ult_8h", "min_operativo_ult_8h",
            "fallas_nivel4_ult_8h", "turno", "tipo_bebida",
        ]
        importances = [0.24, 0.20, 0.17, 0.13, 0.10, 0.08, 0.05, 0.03]
        colors = ["#3b82f6" if i < 3 else "#60a5fa" if i < 5 else "#93c5fd"
                  for i in range(len(variables))]
        fig = go.Figure(go.Bar(
            x=importances, y=variables, orientation="h",
            marker_color=colors,
            text=[f"{v*100:.0f}%" for v in importances],
            textposition="outside", textfont={"color": "#94a3b8"},
        ))
        fig.update_layout(title="Importancia Relativa de Variables",
                          xaxis_title="Importancia",
                          yaxis={"autorange": "reversed"}, **BASE)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 5. ESTADO DE SESIÓN
# ══════════════════════════════════════════════════════════════════

if "prediction"  not in st.session_state: st.session_state.prediction  = None
if "inputs_used" not in st.session_state: st.session_state.inputs_used = None


# ══════════════════════════════════════════════════════════════════
# SECCIÓN 1 — HERO / PÁGINA DE INICIO
# ══════════════════════════════════════════════════════════════════

# ── Badge de modo ─────────────────────────────────────────────────
modo_html = (
    '<span class="badge-real">🟢 Modelo Real Cargado</span>'
    if MODO_REAL else
    '<span class="badge-sim">🟡 Modo Simulado — agrega modelo_linea5.pkl para activar</span>'
)

st.markdown(f"""
<div class="hero-section">
    <div class="hero-badge">🏭 ARCA CONTINENTAL LINDLEY &nbsp;|&nbsp; LÍNEA 5 PET</div>
    <div class="hero-title">PredictLine 5 PET</div>
    <div class="hero-subtitle">
        Sistema Inteligente de Predicción de Paradas Operativas<br>
        mediante Clasificación con Machine Learning
    </div>
    {modo_html}
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""<div class="metric-box"><div style="font-size:2rem">⚡</div>
    <div class="metric-value">30 min</div><div class="metric-label">Anticipación</div></div>""",
    unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="metric-box"><div style="font-size:2rem">🎯</div>
    <div class="metric-value">~56.80%</div><div class="metric-label">Precisión del Modelo</div></div>""",
    unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="metric-box"><div style="font-size:2rem">🔄</div>
    <div class="metric-value">21 vars</div><div class="metric-label">Variables de Entrada</div></div>""",
    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Problema + Objetivo ───────────────────────────────────────────
cA, cB = st.columns([1.6, 1])
with cA:
    st.markdown("""
    <div class="card">
        <div class="card-title">⚠️ Problema</div>
        <p style="color:#cbd5e1;font-size:0.95rem;line-height:1.7">
        Las <strong style="color:#f59e0b">paradas operativas no planificadas</strong> en la Línea 5 PET
        interrumpen el flujo de producción y generan pérdidas económicas. Una parada puede implicar
        retrasos, pérdida de producto, mayor costo de mantenimiento correctivo y penalizaciones
        por incumplimiento de cuotas.
        </p>
        <div class="card-title" style="margin-top:1rem">🎯 Objetivo</div>
        <p style="color:#cbd5e1;font-size:0.95rem;line-height:1.7">
        Predecir con <strong style="color:#34d399">30 minutos de anticipación</strong> si ocurrirá
        una parada operativa, usando datos históricos de producción de la Línea 5 PET.
        </p>
    </div>
    """, unsafe_allow_html=True)
with cB:
    st.markdown("""
    <div class="card" style="height:100%;display:flex;flex-direction:column;
                             justify-content:center;align-items:center;gap:1.2rem;">
        <div style="font-size:5rem;text-align:center">🏭</div>
        <div style="text-align:center;color:#64748b;font-size:0.85rem;line-height:1.6">
            Línea de producción PET<br>
            <strong style="color:#60a5fa">Arca Continental Lindley</strong>
        </div>
        <div style="display:flex;gap:0.6rem;flex-wrap:wrap;justify-content:center">
            <span style="background:rgba(59,130,246,0.15);border:1px solid #3b82f6;
                  border-radius:20px;padding:4px 12px;font-size:0.78rem;color:#60a5fa">
                🤖 StackingClassifier
            </span>
            <span style="background:rgba(52,211,153,0.15);border:1px solid #34d399;
                  border-radius:20px;padding:4px 12px;font-size:0.78rem;color:#34d399">
                ⏱️ Tiempo Real
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ── DIVISOR ── SECCIÓN 1 → SECCIÓN 2
# ══════════════════════════════════════════════════════════════════
section_divider("Sección 2 — Información del Sistema", "📘")


# ══════════════════════════════════════════════════════════════════
# SECCIÓN 2 — INFORMACIÓN Y VARIABLES DEL MODELO
# ══════════════════════════════════════════════════════════════════
section_header("2", "Variables del Modelo")

with st.expander("📘 ¿Cómo funciona el sistema?", expanded=False):
    st.markdown("""
    <div style="color:#cbd5e1;font-size:0.95rem;line-height:1.8">
    El sistema usa un <strong style="color:#60a5fa">StackingClassifier</strong>
    entrenado con datos históricos de la Línea 5 PET. Combina 3 modelos base
    (Regresión Logística, Random Forest y XGBoost) cuyas predicciones alimentan
    a un meta-modelo final de Regresión Logística. El pipeline de predicción es:<br><br>
    <span style="color:#f59e0b">① Ingreso de variables operativas</span> →
    <span style="color:#f59e0b">② One-Hot Encoding de categóricas</span> →
    <span style="color:#f59e0b">③ Predicción de los 3 modelos base (LogReg, Random Forest, XGBoost)</span> →
    <span style="color:#f59e0b">④ Meta-modelo (Regresión Logística) combina las 3 predicciones</span> →
    <span style="color:#f59e0b">⑤ Predicción binaria + probabilidad</span>
    </div>
    """, unsafe_allow_html=True)

# ── Tarjetas de variables ──────────────────────────────────────────
st.markdown("""<div style="margin-bottom:0.5rem;color:#64748b;font-size:0.85rem;
font-weight:600;letter-spacing:0.05em;text-transform:uppercase">
Variables de entrada al modelo</div>""", unsafe_allow_html=True)

vars_data = [
    ("🕐", "Hora del Día"),    ("📅", "Día Semana"),
    ("📆", "Mes"),             ("🔄", "Turno"),
    ("🥤", "Tipo Bebida"),     ("📦", "Tamaño Envase"),
    ("⏱️", "T. Últ. Parada"),  ("🔢", "Paradas 30min"),
    ("🔢", "Paradas 60min"),   ("🔢", "Paradas 120min"),
    ("⏳", "Min Op. 120min"),  ("🔢", "Paradas 8h"),
    ("⏳", "Min Op. 8h"),      ("⚙️", "Último Nivel 2"),   ("⚙️", "Último Nivel 3"),
    ("⚙️", "Último Nivel 4"),   ("🔁", "Fallas N4 8h"),
    ("⏳", "Min N4 8h"),        ("🔧", "Paros Mec. 24h"),
    ("⏳", "Min Mec. 24h"),     ("🚨", "Paro Mec. 24h"),
]

rows = [vars_data[i:i+8] for i in range(0, len(vars_data), 8)]
for row in rows:
    cols = st.columns(8)
    for j, (icon, name) in enumerate(row):
        if j < len(cols):
            with cols[j]:
                st.markdown(f"""<div class="var-card">
                    <div class="var-icon">{icon}</div>
                    <div class="var-name">{name}</div>
                </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ── DIVISOR ── SECCIÓN 2 → SECCIÓN 3
# ══════════════════════════════════════════════════════════════════
section_divider("Sección 3 — Ingreso de Variables Operativas", "📋")


# ══════════════════════════════════════════════════════════════════
# SECCIÓN 3 — FORMULARIO DE INGRESO DE DATOS
# ══════════════════════════════════════════════════════════════════
section_header("3", "Ingreso de Variables Operativas")

st.markdown("""<div class="card" style="margin-bottom:1.5rem">
    <div class="card-title">📋 Instrucciones</div>
    <p style="color:#94a3b8;font-size:0.9rem;margin:0">
    Complete los valores con las condiciones actuales de la Línea 5 PET y presione
    <strong style="color:#60a5fa">Generar Predicción</strong>.
    </p></div>""", unsafe_allow_html=True)

# ── Fila 1: Tiempo / Turno / Producto ─────────────────────────────
f1c1, f1c2, f1c3 = st.columns(3)

with f1c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**🕐 Tiempo**")
    hora_dia   = st.slider("Hora del día", 0, 23, 8, help="Hora actual (0–23)")
    dia_semana = st.selectbox("Día de la semana", DIAS_SEMANA, index=3)
    mes        = st.number_input("Mes", 1, 12, 1, help="Mes actual del año")
    st.markdown('</div>', unsafe_allow_html=True)

with f1c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**🔄 Turno y Producto**")
    turno       = st.selectbox("Turno", TURNOS,
                               help="T1=Mañana (6–14h), T2=Tarde (14–22h), T3=Noche (22–6h)")
    tipo_bebida = st.selectbox("Tipo de bebida", BEBIDAS)
    tamano_lbl  = st.selectbox("Tamaño del envase (L)",
                               list(TAMANOS_LABEL.keys()), index=1)
    tamano      = TAMANOS_LABEL[tamano_lbl]
    st.markdown('</div>', unsafe_allow_html=True)

with f1c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**⚙️ Clasificación del último evento**")
    ultimo_nivel2 = st.selectbox("Último Nivel 2",  NIVEL2_OPCIONES,
                                 index=3,
                                 help="Categoría de nivel 2 del último evento registrado")
    ultimo_nivel3 = st.selectbox("Último Nivel 3", NIVEL3_OPCIONES,
                                 index=4,
                                 help="Categoría de nivel 3 del último evento registrado")
    ultimo_nivel4 = st.selectbox("Último Nivel 4", NIVEL4_OPCIONES,
                                 help="Equipo específico (nivel 4) del último evento")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Fila 2: Historial de paradas ──────────────────────────────────
f2c1, f2c2, f2c3 = st.columns(3)

with f2c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**⏱️ Historial reciente de paradas**")
    tiempo_ultima_parada = st.number_input(
        "Tiempo desde última parada (min)", 0.0, 15000.0, 60.0, step=0.5,
        help="Minutos transcurridos desde la última parada operativa")
    paradas_ult_30  = st.number_input("Paradas operativas ult. 30 min", 0, 10, 0)
    paradas_ult_60  = st.number_input("Paradas operativas ult. 60 min", 0, 16, 0)
    st.markdown('</div>', unsafe_allow_html=True)

with f2c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**🔢 Historial 2h y 8h**")
    paradas_ult_120    = st.number_input("Paradas operativas ult. 120 min", 0, 20, 0)
    min_operativo_120  = st.number_input(
        "Min. operativo acumulado ult. 120 min", 0.0, 125.0, 60.0, step=0.5)
    paradas_ult_8h     = st.number_input("Paradas operativas ult. 8h", 0, 50, 0)
    min_operativo_8h   = st.number_input(
        "Min. operativo acumulado ult. 8h", 0.0, 500.0, 250.0, step=0.5)
    st.markdown('</div>', unsafe_allow_html=True)

with f2c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**🔧 Paros mecánicos y Nivel 4 (ult. 24h)**")
    paros_mecanicos_24h = st.number_input("Cantidad de paros mecánicos (ult. 24h)", 0, 20, 0)
    min_mecanico_24h    = st.number_input(
        "Minutos de paro mecánico (ult. 24h)", 0.0, 480.0, 0.0, step=0.5)
    fallas_nivel4_8h = st.number_input(
        "Fallas mismo Nivel 4 (ult. 8h)", 0, 50, 0)
    min_nivel4_8h = st.number_input(
        "Minutos mismo Nivel 4 parado (ult. 8h)", 0.0, 480.0, 0.0, step=0.5)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Botón de predicción ───────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
btn_cols = st.columns([1, 2, 1])
with btn_cols[1]:
    predict_btn = st.button("⚡  Generar Predicción", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# ── DIVISOR ── SECCIÓN 3 → SECCIÓN 4
# ══════════════════════════════════════════════════════════════════
if predict_btn:
    section_divider("Sección 4 — Procesamiento del Modelo", "⚙️")


# ══════════════════════════════════════════════════════════════════
# SECCIÓN 4 — PROCESAMIENTO Y BARRA DE PROGRESO
# ══════════════════════════════════════════════════════════════════
if predict_btn:
    section_header("4", "Procesamiento del Modelo")

    # ── Pasos del pipeline real del notebook ─────────────────────
    steps = [
        ("🧹", "Limpieza y validación de datos",         0.15),
        ("🏷️",  "One-Hot Encoding de variables categóricas", 0.35),
        ("🔗", "Alineación de columnas al orden del modelo", 0.55),
        ("🌲", "Inferencia de modelos base (LogReg, Random Forest, XGBoost)", 0.75),
        ("🤖", "Meta-modelo combina las 3 predicciones (Stacking)", 0.90),
        ("✅", "Generación de resultado y probabilidad",  1.00),
    ]

    with st.container():
        st.markdown("""<div class="card">
            <div class="card-title">⚙️ Procesando información...</div>
        </div>""", unsafe_allow_html=True)
        progress_bar     = st.progress(0)
        step_placeholder = st.empty()

        for icon, label, prog in steps:
            progress_bar.progress(prog)
            step_placeholder.markdown(f"""
            <div class="step-item">
                <span class="step-check">✔</span>
                <span>{icon} {label}</span>
                <span style="margin-left:auto;color:#34d399;font-size:0.8rem">Completado</span>
            </div>""", unsafe_allow_html=True)
            time.sleep(0.4)

    # ── Construir inputs y predecir ──────────────────────────────
    inputs = {
        "hora_dia":             hora_dia,
        "dia_semana":           dia_semana,
        "mes":                  mes,
        "turno_ventana":        turno,
        "tipo_bebida_actual":   tipo_bebida,
        "tamano_actual":        tamano,
        "tiempo_ultima_parada": tiempo_ultima_parada,
        "paradas_ult_30":       paradas_ult_30,
        "paradas_ult_60":       paradas_ult_60,
        "paradas_ult_120":      paradas_ult_120,
        "min_operativo_120":    min_operativo_120,
        "paradas_ult_8h":       paradas_ult_8h,
        "min_operativo_8h":     min_operativo_8h,
        "ultimo_nivel2":        ultimo_nivel2,
        "ultimo_nivel3":        ultimo_nivel3,
        "ultimo_nivel4":        ultimo_nivel4,
        "fallas_nivel4_8h":     fallas_nivel4_8h,
        "min_nivel4_8h":        min_nivel4_8h,
        "paros_mecanicos_24h":  paros_mecanicos_24h,
        "min_mecanico_24h":     min_mecanico_24h,
    }

    result = hacer_prediccion(inputs)
    st.session_state.prediction  = result
    st.session_state.inputs_used = inputs


# ══════════════════════════════════════════════════════════════════
# ── DIVISOR ── SECCIÓN 4 → SECCIÓN 5
# ══════════════════════════════════════════════════════════════════
if st.session_state.prediction:
    section_divider("Sección 5 — Resultado de la Predicción", "📊")


# ══════════════════════════════════════════════════════════════════
# SECCIÓN 5 — RESULTADOS
# ══════════════════════════════════════════════════════════════════
if st.session_state.prediction:
    result = st.session_state.prediction
    alerta = result["alerta"]
    prob   = result["probabilidad"]
    riesgo = result["riesgo"]
    conf   = result["confianza"]

    section_header("5", "Resultado de la Predicción")

    res1, res2 = st.columns([1, 1.2])

    # ── Tarjeta principal ─────────────────────────────────────────
    with res1:
        if alerta:
            st.markdown(f"""<div class="alert-card-red">
                <div style="font-size:3rem;margin-bottom:0.5rem">🔴</div>
                <div class="alert-label alert-label-red">⚠️ ALERTA</div>
                <div style="color:#fca5a5;margin-top:0.8rem;font-size:0.95rem">
                    Alta probabilidad de parada operativa<br>en los próximos 30 minutos
                </div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="alert-card-green">
                <div style="font-size:3rem;margin-bottom:0.5rem">🟢</div>
                <div class="alert-label alert-label-green">✅ SIN ALERTA</div>
                <div style="color:#86efac;margin-top:0.8rem;font-size:0.95rem">
                    Operación dentro de parámetros normales.<br>Riesgo bajo de parada.
                </div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        color_riesgo = {"ALTO": "#ef4444", "MEDIO": "#f59e0b", "BAJO": "#22c55e"}
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-value" style="color:{'#ef4444' if alerta else '#22c55e'}">{prob}%</div>
                <div class="metric-label">Prob. de Alerta</div></div>""",
                unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-value" style="color:{color_riesgo[riesgo]}">{riesgo}</div>
                <div class="metric-label">Nivel de Riesgo</div></div>""",
                unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-value">{conf}%</div>
                <div class="metric-label">Confianza</div></div>""",
                unsafe_allow_html=True)

    # ── Gauge + interpretación ────────────────────────────────────
    with res2:
        st.plotly_chart(gauge_chart(prob), use_container_width=True)

        if alerta:
            exp = (f"Las variables ingresadas indican una "
                   f"<strong style='color:#ef4444'>probabilidad de {prob}%</strong> de que "
                   f"ocurra una parada operativa durante los próximos 30 minutos. "
                   f"Nivel de riesgo: <strong style='color:#ef4444'>{riesgo}</strong>. "
                   f"Se recomienda acción preventiva inmediata.")
        else:
            exp = (f"Las condiciones actuales muestran una "
                   f"<strong style='color:#22c55e'>probabilidad de {prob}%</strong> de parada, "
                   f"dentro del rango seguro. "
                   f"Nivel de riesgo: <strong style='color:#22c55e'>{riesgo}</strong>. "
                   f"Continúe el monitoreo de rutina.")

        st.markdown(f"""<div class="card" style="margin-top:0">
            <div class="card-title">💡 Interpretación</div>
            <p style="color:#cbd5e1;font-size:0.92rem;line-height:1.7;margin:0">{exp}</p>
        </div>""", unsafe_allow_html=True)

        # Modo activo
        modo_label = (
            "🟢 Predicción con modelo real (StackingClassifier entrenado)"
            if MODO_REAL else
            "🟡 Modo simulado — agrega modelo_linea5.pkl para usar el modelo real"
        )
        st.markdown(f"""<div style="color:#475569;font-size:0.78rem;
            text-align:right;margin-top:0.3rem">{modo_label}</div>""",
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ── DIVISOR ── SECCIÓN 5 → SECCIÓN 6
# ══════════════════════════════════════════════════════════════════
if st.session_state.prediction:
    section_divider("Sección 6 — Recomendaciones Operativas", "🛡️")


# ══════════════════════════════════════════════════════════════════
# SECCIÓN 6 — TOMA DE DECISIONES Y RECOMENDACIONES
# ══════════════════════════════════════════════════════════════════
if st.session_state.prediction:
    alerta = st.session_state.prediction["alerta"]
    section_header("6", "Recomendaciones Operativas")

    if alerta:
        recs  = [
            ("🔍", "Revisar equipos críticos: Llenadora, Tapadora y Etiquetadora."),
            ("📢", "Notificar al supervisor de turno de forma inmediata."),
            ("🔧", "Verificar que el mantenimiento preventivo esté al día."),
            ("👁️", "Activar monitoreo continuo e intensivo de la línea."),
            ("👷", "Preparar personal técnico para intervención rápida."),
            ("📋", "Registrar la alerta en el sistema CMMS."),
        ]
        style = "red"
    else:
        recs  = [
            ("▶️", "Continuar operación normal sin interrupciones."),
            ("📊", "Mantener el monitoreo estándar de parámetros de línea."),
            ("📝", "Registrar la evaluación en el log de turno."),
            ("🔄", "Ejecutar próxima evaluación en 15 minutos."),
            ("✅", "Confirmar estado OK al supervisor de turno."),
        ]
        style = "green"

    r1, r2 = st.columns([1.5, 1])
    with r1:
        for icon, text in recs:
            st.markdown(f"""<div class="rec-card-{style}">
                <span style="font-size:1.2rem">{icon}</span>
                <span style="color:#e2e8f0">{text}</span>
            </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""<div class="card" style="text-align:center">
            <div style="font-size:3.5rem">{'🚨' if alerta else '🛡️'}</div>
            <div style="font-size:1rem;font-weight:700;
                color:{'#ef4444' if alerta else '#22c55e'};margin:0.8rem 0 0.4rem">
                {'ACCIÓN REQUERIDA' if alerta else 'ESTADO NORMAL'}
            </div>
            <div style="color:#64748b;font-size:0.85rem;line-height:1.6">
                {'Implemente las recomendaciones en orden para minimizar el impacto.'
                 if alerta else 'Mantenga el ritmo de producción y realice seguimiento periódico.'}
            </div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ── DIVISOR ── SECCIÓN 6 → SECCIÓN 7
# ══════════════════════════════════════════════════════════════════
section_divider("Sección 7 — Dashboard Histórico", "📈")


# ══════════════════════════════════════════════════════════════════
# SECCIÓN 7 — DASHBOARD COMPLEMENTARIO
# ══════════════════════════════════════════════════════════════════
section_header("7", "Dashboard Histórico de Producción")

st.markdown("""<div class="card" style="margin-bottom:1rem">
    <div class="card-title">📈 Análisis Histórico</div>
    <p style="color:#94a3b8;font-size:0.88rem;margin:0">
    Datos históricos simulados de la Línea 5 PET (referencia visual para contextualizar
    la predicción actual).
    </p></div>""", unsafe_allow_html=True)

dashboard_charts()


# ══════════════════════════════════════════════════════════════════
# ── DIVISOR ── SECCIÓN 7 → FOOTER
# ══════════════════════════════════════════════════════════════════
section_divider("Información del Proyecto", "🎓")


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-section">
    <div style="font-size:1.3rem;margin-bottom:0.5rem">🏭 <strong>PredictLine 5 PET</strong></div>
    <div style="color:#334155;margin-bottom:0.8rem">
        Sistema de Predicción de Paradas Operativas — Línea 5 PET · Arca Continental Lindley
    </div>
    <div style="display:flex;gap:2rem;justify-content:center;flex-wrap:wrap;
                font-size:0.82rem;color:#475569">
        <span>📚 Curso: <strong style="color:#60a5fa">Applied Machine Learning</strong></span>
        <span>🏫 Universidad: <strong style="color:#60a5fa">UTEC</strong></span>
        <span>👥 Integrantes: <strong style="color:#60a5fa">Jayde Oriundo, Gianella Lira, Chealsie Figueroa y Luzsmith Rojas</strong></span>
        <span>📅 Año: <strong style="color:#60a5fa">2026</strong></span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# FIN DEL ARCHIVO
# ══════════════════════════════════════════════════════════════════
