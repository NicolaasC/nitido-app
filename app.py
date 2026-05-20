"""
NÍTIDO — App de Pre-filtrado de Candidatos
Examen 3 — Machine Learning II — Universidad Externado de Colombia
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, os, warnings
warnings.filterwarnings('ignore')

# ─── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="NÍTIDO — Sistema de Pre-filtrado",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2C3E50, #3498DB);
        padding: 20px 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 25px;
    }
    .resultado-aprobado {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px 20px;
        border-radius: 5px;
        font-size: 1.2em;
    }
    .resultado-rechazado {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px 20px;
        border-radius: 5px;
        font-size: 1.2em;
    }
    .resultado-borderline {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px 20px;
        border-radius: 5px;
        font-size: 1.2em;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .cf-card {
        background: #e8f4f8;
        border-left: 4px solid #3498DB;
        padding: 12px 18px;
        border-radius: 5px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 NÍTIDO — Sistema de Pre-filtrado de Candidatos</h1>
    <p>Modelo de IA para la selección inicial de candidatos | Universidad Externado de Colombia</p>
</div>
""", unsafe_allow_html=True)

# ─── Cargar modelo y artefactos ───────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado y el preprocessor."""
    with open('modelo_nitido.pkl', 'rb') as f:
        artefactos = pickle.load(f)
    return artefactos

@st.cache_data
def cargar_datos_train():
    df = pd.read_csv('candidatos_nitido.csv')
    return df

# ─── Definición de variables ──────────────────────────────────────────────────
VARS_CONTINUAS  = ['x1', 'x2', 'x6', 'x10', 'x11', 'x14', 'x15', 'x16']
VARS_ORDINALES  = ['x3', 'x4', 'x7', 'x8', 'x9', 'x12', 'x13', 'x18']
VARS_BINARIAS   = ['x5', 'x17']
ALL_FEATURES    = VARS_CONTINUAS + VARS_ORDINALES + VARS_BINARIAS

# Metadatos de variables para los controles de la UI
VAR_META = {
    'x1':  {'label': 'x1 — Puntaje A',          'min': 0.0,  'max': 25.0,  'step': 0.1,  'default': 5.0,  'type': 'float'},
    'x2':  {'label': 'x2 — Puntaje B (%)',       'min': 7.0,  'max': 100.0, 'step': 1.0,  'default': 65.0, 'type': 'float'},
    'x3':  {'label': 'x3 — Nivel (1–5)',          'min': 1,    'max': 5,     'step': 1,    'default': 3,    'type': 'int'},
    'x4':  {'label': 'x4 — Conteo (0–8)',         'min': 0,    'max': 8,     'step': 1,    'default': 2,    'type': 'int'},
    'x5':  {'label': 'x5 — Binaria A',            'options': [0, 1],                       'default': 0,    'type': 'binary'},
    'x6':  {'label': 'x6 — Variable C',           'min': 22.0, 'max': 65.0,  'step': 1.0,  'default': 35.0, 'type': 'float'},
    'x7':  {'label': 'x7 — Nivel (1–6)',          'min': 1,    'max': 6,     'step': 1,    'default': 3,    'type': 'int'},
    'x8':  {'label': 'x8 — Rating (1–10)',        'min': 1,    'max': 10,    'step': 1,    'default': 5,    'type': 'int'},
    'x9':  {'label': 'x9 — Categoría (0–3)',      'min': 0,    'max': 3,     'step': 1,    'default': 1,    'type': 'int'},
    'x10': {'label': 'x10 — Nota (3.0–5.0)',      'min': 3.0,  'max': 5.0,   'step': 0.1,  'default': 4.0,  'type': 'float'},
    'x11': {'label': 'x11 — Puntaje D',           'min': 0.1,  'max': 100.0, 'step': 0.5,  'default': 50.0, 'type': 'float'},
    'x12': {'label': 'x12 — Conteo (0–11)',       'min': 0,    'max': 11,    'step': 1,    'default': 3,    'type': 'int'},
    'x13': {'label': 'x13 — Categoría (0–3)',     'min': 0,    'max': 3,     'step': 1,    'default': 1,    'type': 'int'},
    'x14': {'label': 'x14 — Años (0–24)',         'min': 0.0,  'max': 24.0,  'step': 0.5,  'default': 3.0,  'type': 'float'},
    'x15': {'label': 'x15 — Puntaje E',          'min': 1.0,  'max': 50.0,  'step': 0.5,  'default': 15.0, 'type': 'float'},
    'x16': {'label': 'x16 — Puntaje F (%)',      'min': 0.0,  'max': 100.0, 'step': 1.0,  'default': 60.0, 'type': 'float'},
    'x17': {'label': 'x17 — Binaria B',           'options': [0, 1],                       'default': 0,    'type': 'binary'},
    'x18': {'label': 'x18 — Score (0–10)',        'min': 0,    'max': 10,    'step': 1,    'default': 4,    'type': 'int'},
}

# ─── Sidebar — Ingreso de datos ───────────────────────────────────────────────
st.sidebar.markdown("## 📋 Datos del Candidato")
st.sidebar.markdown("Ingresa los valores de las 18 variables para evaluar al candidato.")

valores_candidato = {}

for var in ALL_FEATURES:
    meta = VAR_META[var]
    if meta['type'] == 'binary':
        valores_candidato[var] = st.sidebar.selectbox(
            meta['label'], options=meta['options'],
            index=meta['options'].index(meta['default']),
            key=var
        )
    elif meta['type'] == 'float':
        valores_candidato[var] = st.sidebar.slider(
            meta['label'],
            min_value=float(meta['min']),
            max_value=float(meta['max']),
            value=float(meta['default']),
            step=float(meta['step']),
            key=var
        )
    else:  # int / ordinal
        valores_candidato[var] = st.sidebar.slider(
            meta['label'],
            min_value=int(meta['min']),
            max_value=int(meta['max']),
            value=int(meta['default']),
            step=int(meta['step']),
            key=var
        )

evaluar = st.sidebar.button("🔎 Evaluar Candidato", type="primary", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("*Sistema NÍTIDO v1.0 — Uso interno*")

# ─── Cuerpo principal ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("### 📊 Perfil del Candidato")
    candidato_df = pd.DataFrame([valores_candidato])

    # Mostrar radar / tabla de valores
    fig_perfil, ax = plt.subplots(figsize=(8, 3))
    colors = ['#3498DB' if v != 0 else '#E74C3C'
              for v in candidato_df[ALL_FEATURES].values[0]]
    ax.bar(ALL_FEATURES, candidato_df[ALL_FEATURES].values[0], color='#3498DB', alpha=0.7)
    ax.set_xticklabels(ALL_FEATURES, rotation=45, ha='right', fontsize=8)
    ax.set_title('Valores del candidato (escala original)', fontweight='bold')
    ax.set_ylabel('Valor')
    plt.tight_layout()
    st.pyplot(fig_perfil, use_container_width=True)
    plt.close()

with col_right:
    st.markdown("### 🎯 Resultado de la Evaluación")

    if not evaluar:
        st.info("👈 Ingresa los datos del candidato en el panel izquierdo y presiona **Evaluar Candidato**.")
    else:
        try:
            artefactos = cargar_modelo()
            modelo        = artefactos['modelo']
            preprocessor  = artefactos['preprocessor']
            threshold     = artefactos['threshold']
            feature_names = artefactos['feature_names']
            
            # ¡NUEVO! Creamos el explainer en vivo en lugar de cargarlo del pkl
            import shap
            explainer = shap.TreeExplainer(modelo)

            # Preprocesar
            candidato_t = preprocessor.transform(candidato_df[ALL_FEATURES])
            # ... el resto de tu código sigue igual ...
            
        except FileNotFoundError:
            st.warning("⚠️ Modelo no encontrado. Asegúrate de ejecutar primero `train_and_save_model.py`.")
            st.info("**Demo mode:** Para ver la app funcional, ejecuta el script de entrenamiento.")
            st.stop() # <-- Recuerda dejar el stop que pusimos antes

            # Predicción
            prob = modelo.predict_proba(candidato_proc)[0, 1]
            pred = int(prob >= threshold)

            # Mostrar resultado
            if pred == 1 and prob >= 0.70:
                st.markdown(f"""
                <div class="resultado-aprobado">
                    ✅ <strong>AVANZA</strong><br>
                    Probabilidad de aprobación: <strong>{prob:.1%}</strong><br>
                    El candidato es recomendado para la siguiente etapa.
                </div>""", unsafe_allow_html=True)
            elif pred == 0 and prob < 0.30:
                st.markdown(f"""
                <div class="resultado-rechazado">
                    ❌ <strong>RECHAZADO</strong><br>
                    Probabilidad de aprobación: <strong>{prob:.1%}</strong><br>
                    El candidato no cumple el perfil requerido.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="resultado-borderline">
                    ⚠️ <strong>CASO BORDERLINE</strong><br>
                    Probabilidad de aprobación: <strong>{prob:.1%}</strong><br>
                    Se recomienda revisión humana de este candidato.
                </div>""", unsafe_allow_html=True)

            st.metric("P(avanza)", f"{prob:.3f}", delta=f"Umbral: {threshold:.2f}")

        except FileNotFoundError:
            st.warning("⚠️ Modelo no encontrado. Asegúrate de ejecutar primero `train_and_save_model.py`.")
            st.info("**Demo mode:** Para ver la app funcional, ejecuta el script de entrenamiento.")

# ─── Sección SHAP y Contrafactual ────────────────────────────────────────────
if evaluar:
    st.markdown("---")
    col_shap, col_cf = st.columns(2)

    with col_shap:
        st.markdown("### 🔍 Explicación SHAP Local")
        try:
            import shap
            shap_vals = explainer.shap_values(candidato_proc)
            base_val  = explainer.expected_value

            fig_shap, ax = plt.subplots(figsize=(8, 5))

            # Waterfall manual
            shap_dict = dict(zip(feature_names, shap_vals[0]))
            shap_sorted = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            feats_  = [s[0] for s in shap_sorted]
            vals_   = [s[1] for s in shap_sorted]

            colors_ = [('#2ECC71' if v > 0 else '#E74C3C') for v in vals_]
            ax.barh(feats_[::-1], vals_[::-1], color=colors_[::-1])
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_xlabel('Valor SHAP (contribución a la predicción)')
            ax.set_title(f'Top 10 variables — Explicación local\n(Valor base: {base_val:.3f}  →  P(avanza)={prob:.3f})',
                         fontweight='bold', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_shap, use_container_width=True)
            plt.close()

            st.caption("🟢 Verde = empuja hacia **aprobado** | 🔴 Rojo = empuja hacia **rechazado**")
        except Exception as e:
            st.error(f"No se pudo generar el gráfico SHAP: {e}")

    with col_cf:
        st.markdown("### 💡 Contrafactual Sugerido")
        if pred == 0:
            try:
                df_train = cargar_datos_train()
                # Simple counterfactual: sugerir los valores del percentil 75 de aprobados
                aprobados = df_train[df_train['avanza'] == 1][ALL_FEATURES]
                sugerencias = []

                shap_abs = np.abs(shap_vals[0])
                orden_shap = np.argsort(shap_abs)[::-1]

                candidato_mod = candidato_proc.copy()
                prob_mod = prob

                for fi in orden_shap:
                    feat = feature_names[fi]
                    if feat not in ALL_FEATURES:
                        continue
                    val_orig = candidato_df[feat].values[0]
                    val_p75 = aprobados[feat].quantile(0.75)

                    if val_p75 > val_orig + 0.01:
                        sugerencias.append({
                            'Variable': feat,
                            'Tu valor': round(val_orig, 2),
                            'Valor sugerido': round(val_p75, 2),
                        })
                        candidato_mod[feat] = preprocessor.named_transformers_.get(
                            'cont', None
                        ) and val_p75  # simplified

                        if len(sugerencias) >= 5:
                            break

                if sugerencias:
                    st.markdown("""
                    <div class="cf-card">
                    <strong>Para mejorar tus posibilidades de aprobación, considera:</strong>
                    </div>
                    """, unsafe_allow_html=True)

                    cf_df = pd.DataFrame(sugerencias)
                    st.dataframe(cf_df, use_container_width=True, hide_index=True)
                    st.caption("*Los valores sugeridos corresponden al percentil 75 de candidatos aprobados históricamente.*")
                else:
                    st.info("No se encontraron cambios simples que mejoren la predicción.")

            except Exception as e:
                st.error(f"Error generando contrafactual: {e}")
        else:
            st.success("✅ El candidato ya supera el umbral. No se requiere contrafactual.")
            st.markdown("**Factores positivos principales:**")
            try:
                top_pos = [(feature_names[i], shap_vals[0][i])
                           for i in np.argsort(shap_vals[0])[::-1][:3]]
                for feat, val in top_pos:
                    st.write(f"• **{feat}**: contribución positiva de {val:+.3f}")
            except:
                pass

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85em;'>"
    "NÍTIDO v1.0 — Sistema de pre-filtrado de hojas de vida con IA | "
    "Pregrado en Ciencia de Datos — Universidad Externado de Colombia"
    "</div>",
    unsafe_allow_html=True
)
