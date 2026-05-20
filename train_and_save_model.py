"""
train_and_save_model.py
Entrena el modelo final de NÍTIDO y guarda todos los artefactos necesarios
para la app de Streamlit.

Ejecutar ANTES de lanzar la app:
    python train_and_save_model.py
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import xgboost as xgb
import shap

print("=" * 60)
print("  NÍTIDO — Entrenamiento del modelo final")
print("=" * 60)

# ── Variables ──────────────────────────────────────────────────────────────────
VARS_CONTINUAS = ['x1', 'x2', 'x6', 'x10', 'x11', 'x14', 'x15', 'x16']
VARS_ORDINALES = ['x3', 'x4', 'x7', 'x8', 'x9', 'x12', 'x13', 'x18']
VARS_BINARIAS  = ['x5', 'x17']
ALL_FEATURES   = VARS_CONTINUAS + VARS_ORDINALES + VARS_BINARIAS
TARGET = 'avanza'

# ── Carga de datos ─────────────────────────────────────────────────────────────
print("\n[1/6] Cargando datos...")
df = pd.read_csv('candidatos_nitido.csv')
print(f"  Dataset: {df.shape[0]} candidatos, {df.shape[1]-1} variables")

X = df[ALL_FEATURES]
y = df[TARGET]

# ── Partición ─────────────────────────────────────────────────────────────────
print("[2/6] Particionando datos (60/20/20)...")
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25,
                                                    random_state=42, stratify=y_temp)

# ── Preprocesamiento ──────────────────────────────────────────────────────────
print("[3/6] Ajustando preprocesador...")
preprocessor = ColumnTransformer(transformers=[
    ('cont', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ]), VARS_CONTINUAS),
    ('ord', SimpleImputer(strategy='median'), VARS_ORDINALES),
    ('bin', SimpleImputer(strategy='most_frequent'), VARS_BINARIAS),
], remainder='drop')

preprocessor.fit(X_train)

X_train_t = preprocessor.transform(X_train)
X_val_t   = preprocessor.transform(X_val)
X_test_t  = preprocessor.transform(X_test)

X_train_df = pd.DataFrame(X_train_t, columns=ALL_FEATURES)
X_val_df   = pd.DataFrame(X_val_t,   columns=ALL_FEATURES)
X_test_df  = pd.DataFrame(X_test_t,  columns=ALL_FEATURES)

# ── Entrenamiento con mejores hiperparámetros ─────────────────────────────────
print("[4/6] Entrenando modelo XGBoost (búsqueda de hiperparámetros)...")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

param_grid = {
    'n_estimators':     [200, 400],
    'max_depth':        [3, 5],
    'learning_rate':    [0.05, 0.1],
    'subsample':        [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs = GridSearchCV(
    xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    ),
    param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1
)
gs.fit(X_train_df, y_train)

model = gs.best_estimator_
print(f"  Mejores parámetros: {gs.best_params_}")
print(f"  AUC CV: {gs.best_score_:.4f}")

# ── Umbral óptimo ─────────────────────────────────────────────────────────────
val_probs = model.predict_proba(X_val_df)[:, 1]
prec, rec, thresholds = precision_recall_curve(y_val, val_probs)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
threshold = thresholds[np.argmax(f1s[:-1])]
print(f"  Umbral óptimo (F1 val): {threshold:.3f}")

# ── SHAP explainer ────────────────────────────────────────────────────────────
print("[5/6] Generando SHAP explainer...")
explainer = shap.TreeExplainer(model)

# ── Evaluación final en test ──────────────────────────────────────────────────
test_probs = model.predict_proba(X_test_df)[:, 1]
test_preds = (test_probs >= threshold).astype(int)
auc_test   = roc_auc_score(y_test, test_probs)
f1_test    = f1_score(y_test, test_preds)
print(f"\n  === Evaluación final TEST ===")
print(f"  AUC:  {auc_test:.4f}")
print(f"  F1:   {f1_test:.4f}")

# ── Guardar artefactos ────────────────────────────────────────────────────────
print("[6/6] Guardando artefactos...")
artefactos = {
    'modelo':        model,
    'preprocessor':  preprocessor,
    'threshold':     threshold,
    'explainer':     explainer,
    'feature_names': ALL_FEATURES,
    'auc_test':      auc_test,
    'f1_test':       f1_test,
    'best_params':   gs.best_params_,
}

with open('modelo_nitido.pkl', 'wb') as f:
    pickle.dump(artefactos, f)

print("\n✅ Artefactos guardados en 'modelo_nitido.pkl'")
print("   Ahora puedes ejecutar la app con: streamlit run app.py")
print("=" * 60)
