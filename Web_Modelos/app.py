import os
import sys
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.base import BaseEstimator, TransformerMixin

# ── Monkey-patch compat ────────────────────────────────────────────────────────
if not hasattr(sys.modules.get('sklearn.compose._column_transformer', None), '_RemainderColsList'):
    from sklearn.compose._column_transformer import ColumnTransformer
    class _RemainderColsList(list):
        pass
    sys.modules['sklearn.compose._column_transformer']._RemainderColsList = _RemainderColsList

class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):   return X

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "alzheimer_database_final.csv")

st.set_page_config(page_title="Alzheimer ML Dashboard", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card{background:linear-gradient(135deg,#1e3a5f,#2e6da4);border-radius:12px;
  padding:20px;text-align:center;color:white;margin-bottom:10px;}
.metric-card h2{font-size:2rem;margin:0;}
.metric-card p{margin:4px 0 0;opacity:.8;}
.alzheimer-box{background:#c0392b;border-radius:10px;padding:18px;text-align:center;color:white;}
.normal-box   {background:#27ae60;border-radius:10px;padding:18px;text-align:center;color:white;}
.grupo-box    {border-radius:10px;padding:18px;text-align:center;color:white;font-size:1.1rem;}
.info-box     {background:#1a2940;border-radius:10px;padding:14px;color:#cce;margin:6px 0;}
</style>
""", unsafe_allow_html=True)

# ── Carga pipelines ────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    dt    = joblib.load(os.path.join(MODEL_DIR, "decision_tree_pipeline.pkl"))
    logit = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_pipeline.pkl"))
    km    = joblib.load(os.path.join(MODEL_DIR, "kmeans_pipeline.pkl"))
    return dt, logit, km

try:
    pipeline_dt, pipeline_logit, pipeline_kmeans = cargar_modelos()
except Exception as e:
    st.error("❌ Error al cargar los pipelines.")
    st.exception(e); st.stop()

def get_feature_names(pipeline):
    final = pipeline.steps[-1][1]
    if hasattr(final, "feature_names_in_"): return list(final.feature_names_in_)
    for _, step in pipeline.steps:
        if hasattr(step, "feature_names_in_"): return list(step.feature_names_in_)
    return []

variables_dt     = get_feature_names(pipeline_dt)
variables_logit  = get_feature_names(pipeline_logit)
variables_kmeans = get_feature_names(pipeline_kmeans)

# variables_all excluye Diagnosis y otras columnas resultado para el formulario manual
# pero las conserva en los pipelines que las necesitan (K-Means)
COLS_SOLO_KMEANS = [c for c in variables_kmeans
                    if c not in variables_dt and c not in variables_logit]
variables_all    = list(dict.fromkeys(variables_kmeans + variables_logit + variables_dt))
EXCLUIR_FORM     = {"Diagnosis", "Pred_DT", "Pred_Logistica", "Grupo_KMeans",
                    "Cluster", "PatientID", "ID"}

def completar_para_kmeans(dm):
    """Añade al df_manual las columnas extra que K-Means necesita
    (ej. Diagnosis) usando la mediana del dataset base como valor neutro."""
    dm_k = dm.copy()
    for col in variables_kmeans:
        if col not in dm_k.columns:
            if col in df_base.columns:
                dm_k[col] = float(df_base[col].median())
            else:
                dm_k[col] = 0.0
    return dm_k

try:
    df_base = pd.read_csv(DATASET_PATH)
except Exception as e:
    st.error("❌ No se pudo leer alzheimer_database_final.csv")
    st.exception(e); st.stop()

# ── Constantes de color ────────────────────────────────────────────────────────
COLORES_DIAG   = {0: "#27ae60", 1: "#c0392b"}
ETIQUETAS_DIAG = {0: "No Alzheimer", 1: "Alzheimer"}
_PALETA_FIJA   = {"Alto Riesgo":"#e74c3c","Riesgo Moderado-Alto":"#e67e22",
                  "Riesgo Moderado-Bajo":"#f39c12","Bajo Riesgo":"#27ae60"}
_COLS_FB       = ["#e74c3c","#e67e22","#f39c12","#27ae60","#7f8c8d"]

DESCRIPCIONES_GRUPO = {
    "Alto Riesgo":          ("⚠️ Alto Riesgo",
                             "Mayor probabilidad de Alzheimer. Múltiples factores de riesgo y síntomas cognitivos avanzados.",
                             "#e74c3c"),
    "Riesgo Moderado-Alto": ("🔶 Riesgo Moderado-Alto",
                             "Pacientes con varios factores activos: hipertensión, diabetes, depresión.",
                             "#e67e22"),
    "Riesgo Moderado-Bajo": ("🟡 Riesgo Moderado-Bajo",
                             "Algún factor de riesgo aislado. Síntomas leves o ausentes.",
                             "#f39c12"),
    "Bajo Riesgo":          ("✅ Bajo Riesgo",
                             "Sin factores de riesgo relevantes. Bajo índice de síntomas cognitivos.",
                             "#27ae60"),
}

# ── LABEL MAP K-MEANS ─────────────────────────────────────────────────────────
@st.cache_data
def construir_label_map(_pipeline_kmeans, _df_base, _vars_km):
    nombres = ["Alto Riesgo","Riesgo Moderado-Alto","Riesgo Moderado-Bajo","Bajo Riesgo"]
    try:
        cols_ok = [c for c in _vars_km if c in _df_base.columns]
        if not cols_ok: raise ValueError()
        clusters = _pipeline_kmeans.predict(_df_base[_vars_km])
        tmp = _df_base.copy(); tmp["_cluster"] = clusters
        if "Diagnosis" in tmp.columns:
            perfil = (tmp.groupby("_cluster")["Diagnosis"]
                      .mean().reset_index()
                      .sort_values("Diagnosis", ascending=False))
            return {int(r["_cluster"]): nombres[i]
                    for i, (_, r) in enumerate(perfil.iterrows())}
        return {int(c): nombres[i % 4]
                for i, c in enumerate(sorted(tmp["_cluster"].unique()))}
    except Exception:
        return {i: f"Grupo {i}" for i in range(4)}

_lmp = os.path.join(MODEL_DIR, "kmeans_label_map.pkl")
LABEL_MAP = joblib.load(_lmp) if os.path.exists(_lmp) else \
            construir_label_map(pipeline_kmeans, df_base, variables_kmeans)

_used        = list(dict.fromkeys(LABEL_MAP.values()))
ORDEN_GRUPOS = [e for e in _PALETA_FIJA if e in _used] or _used
for _l in _used:
    if _l not in DESCRIPCIONES_GRUPO:
        DESCRIPCIONES_GRUPO[_l] = (_l, "Grupo sin descripción.", "#7f8c8d")
COLORES_GRUPO_MAP = {l: _PALETA_FIJA.get(l, _COLS_FB[i % 5])
                     for i, l in enumerate(ORDEN_GRUPOS)}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS GRAFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════
def _fig(w=5, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="none")
    ax.set_facecolor("none"); fig.patch.set_alpha(0)
    return fig, ax

def _style(ax):
    ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
    ax.yaxis.label.set_color("white"); ax.xaxis.label.set_color("white")
    ax.title.set_color("white")

def grafico_dona(valores, etiquetas, colores, titulo):
    valores = [float(v) if (v is not None and v == v) else 0.0 for v in valores]
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
    ax.set_facecolor("none"); fig.patch.set_alpha(0)
    if sum(valores) == 0:
        ax.text(0, 0, "Sin datos", ha="center", va="center", color="white", fontsize=13)
        ax.set_title(titulo, color="white", fontsize=13); return fig
    wedges, _, autotexts = ax.pie(
        valores, labels=None, autopct="%1.1f%%", colors=colores, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        textprops=dict(color="white"))
    for at in autotexts: at.set_fontsize(11); at.set_color("white")
    ax.set_title(titulo, color="white", fontsize=13, pad=14)
    parches = [mpatches.Patch(color=colores[i], label=f"{etiquetas[i]} ({int(v)})")
               for i, v in enumerate(valores)]
    ax.legend(handles=parches, loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=9, framealpha=0, labelcolor="white")
    return fig

def grafico_radar(vals_norm, etiquetas, titulo):
    N = len(etiquetas)
    ang = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals = vals_norm + [vals_norm[0]]; ang += ang[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True), facecolor="none")
    ax.plot(ang, vals, "o-", lw=2, color="#3498db")
    ax.fill(ang, vals, alpha=0.3, color="#3498db")
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(etiquetas, size=8, color="white")
    ax.set_yticklabels([]); ax.spines["polar"].set_edgecolor("gray")
    ax.set_facecolor("none"); fig.patch.set_alpha(0)
    ax.set_title(titulo, color="white", fontsize=12, y=1.12)
    return fig

def grafico_barras_prob(prob_dt, prob_log):
    fig, ax = _fig(6, 3)
    x = np.arange(2); w = 0.3
    b1 = ax.bar(x - w/2, [prob_dt[0]*100,  prob_dt[1]*100],  w, label="Árbol Dec.",
                color=["#27ae60","#c0392b"], alpha=.85)
    b2 = ax.bar(x + w/2, [prob_log[0]*100, prob_log[1]*100], w, label="Log. Reg.",
                color=["#2ecc71","#e74c3c"], alpha=.85)
    ax.set_xticks(x); ax.set_xticklabels(["No Alzheimer","Alzheimer"], color="white")
    ax.set_ylabel("Probabilidad (%)"); ax.set_ylim(0, 115)
    ax.bar_label(b1, fmt="%.1f%%", color="white", fontsize=9)
    ax.bar_label(b2, fmt="%.1f%%", color="white", fontsize=9)
    ax.legend(fontsize=9, framealpha=0, labelcolor="white")
    _style(ax); return fig

def grafico_gauge(valor, titulo, color):
    """Mini gauge semicircular para confianza."""
    fig, ax = plt.subplots(figsize=(3, 1.8), subplot_kw=dict(polar=True), facecolor="none")
    theta = np.linspace(0, np.pi, 200)
    ax.plot(theta, [1]*200, lw=12, color="#2c3e50", solid_capstyle="round")
    ax.plot(np.linspace(0, np.pi * valor/100, 200), [1]*200,
            lw=12, color=color, solid_capstyle="round")
    ax.set_xlim(0, np.pi); ax.set_ylim(0, 1.4)
    ax.set_theta_zero_location("W"); ax.set_theta_direction(-1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_visible(False); ax.set_facecolor("none"); fig.patch.set_alpha(0)
    ax.text(np.pi/2, 0.35, f"{valor:.1f}%", ha="center", va="center",
            color="white", fontsize=14, fontweight="bold")
    ax.text(np.pi/2, -0.1, titulo, ha="center", va="center", color="white", fontsize=9)
    return fig

def grafico_conf_matrix(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = _fig(4, 3)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No Alz","Alz"], color="white")
    ax.set_yticklabels(["No Alz","Alz"], color="white")
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(titulo, fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
    _style(ax); return fig

def grafico_barras_metricas(df_metrics):
    fig, ax = _fig(7, 3.5)
    x = np.arange(len(df_metrics))
    metricas = ["Accuracy","Precision","Recall","F1","AUC"]
    colores  = ["#3498db","#9b59b6","#e67e22","#27ae60","#e74c3c"]
    w = 0.15
    for j, (met, col) in enumerate(zip(metricas, colores)):
        if met in df_metrics.columns:
            bars = ax.bar(x + j*w, df_metrics[met]*100, w, label=met, color=col, alpha=.85)
            ax.bar_label(bars, fmt="%.1f", color="white", fontsize=7, padding=2)
    ax.set_xticks(x + w*2); ax.set_xticklabels(df_metrics["Modelo"], color="white")
    ax.set_ylabel("Valor (%)"); ax.set_ylim(0, 118)
    ax.legend(fontsize=8, framealpha=0, labelcolor="white", ncol=5)
    _style(ax); return fig

def grafico_correlacion(df):
    cont = [c for c in ['BMI','AlcoholConsumption','PhysicalActivity','DietQuality',
                         'SleepQuality','SystolicBP','DiastolicBP','CholesterolTotal',
                         'CholesterolLDL','CholesterolHDL','CholesterolTriglycerides',
                         'MMSE','FunctionalAssessment','ADL']
            if c in df.columns]
    corr = df[cont + ['Diagnosis']].corr()['Diagnosis'].drop('Diagnosis').sort_values()
    colores = ["#e74c3c" if v > 0 else "#27ae60" for v in corr.values]
    fig, ax = _fig(6, 4.5)
    bars = ax.barh(corr.index, corr.values, color=colores, edgecolor="none")
    ax.axvline(0, color="white", lw=0.8, ls="--")
    ax.set_title("Correlación con Diagnosis", fontsize=12)
    ax.set_xlabel("Pearson r")
    ax.bar_label(bars, fmt="%.3f", padding=3, color="white", fontsize=8)
    _style(ax); return fig

def grafico_riesgo_binario(df):
    """Diferencia en tasa de Alzheimer según variables binarias."""
    cols = [c for c in ['MemoryComplaints','BehavioralProblems','Hypertension',
                         'CardiovascularDisease','Confusion','Disorientation',
                         'PersonalityChanges','DifficultyCompletingTasks',
                         'FamilyHistoryAlzheimers','Depression','Smoking',
                         'Diabetes','HeadInjury','Forgetfulness']
            if c in df.columns]
    diffs, labels = [], []
    for col in cols:
        t = df.groupby(col)['Diagnosis'].mean()
        diff = t.get(1, 0) - t.get(0, 0)
        diffs.append(diff); labels.append(col)
    order = np.argsort(diffs)[::-1]
    diffs = [diffs[i] for i in order]
    labels = [labels[i] for i in order]
    colores = ["#e74c3c" if d > 0.05 else ("#e67e22" if d > 0 else "#27ae60") for d in diffs]
    fig, ax = _fig(6, 5)
    bars = ax.barh(labels, diffs, color=colores, edgecolor="none")
    ax.axvline(0, color="white", lw=0.8, ls="--")
    ax.set_title("Impacto de factores binarios en Alzheimer", fontsize=11)
    ax.set_xlabel("Δ Tasa Alzheimer (con − sin factor)")
    ax.bar_label(bars, fmt="%.3f", padding=3, color="white", fontsize=8)
    _style(ax); return fig

def grafico_medias_grupo(df_res, variable):
    """Boxplot / media por grupo K-Means."""
    if variable not in df_res.columns: return None
    grupos = ORDEN_GRUPOS
    medias = [df_res[df_res["Grupo_KMeans"]==g][variable].mean() for g in grupos]
    colores= [COLORES_GRUPO_MAP.get(g,"#7f8c8d") for g in grupos]
    fig, ax = _fig(5, 3)
    bars = ax.bar(grupos, medias, color=colores, edgecolor="white")
    ax.set_title(f"Media de {variable} por grupo", fontsize=10)
    ax.set_ylabel(variable)
    ax.bar_label(bars, fmt="%.2f", color="white", padding=2, fontsize=9)
    ax.set_xticklabels(grupos, rotation=15, ha="right", fontsize=8, color="white")
    _style(ax); return fig

def grafico_variables_cluster(df_res):
    """Barras horizontales: media normalizada de cada variable continua por cluster."""
    cont = [c for c in ['MMSE','FunctionalAssessment','ADL','SleepQuality',
                         'PhysicalActivity','DietQuality','BMI','SystolicBP',
                         'DiastolicBP','CholesterolTotal']
            if c in df_res.columns]
    grupos = ORDEN_GRUPOS
    medias = {}
    for g in grupos:
        sub = df_res[df_res["Grupo_KMeans"]==g]
        if len(sub) == 0:
            medias[g] = [0]*len(cont)
            continue
        vals = sub[cont].mean().values
        v_min = df_res[cont].min().values
        v_max = df_res[cont].max().values
        denom = np.where((v_max - v_min)==0, 1, v_max - v_min)
        medias[g] = ((vals - v_min) / denom).tolist()

    x = np.arange(len(cont)); w = 0.18
    fig, ax = _fig(8, 4)
    for i, g in enumerate(grupos):
        c = COLORES_GRUPO_MAP.get(g, _COLS_FB[i])
        ax.bar(x + i*w, medias[g], w, label=g, color=c, alpha=0.85, edgecolor="none")
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels([c[:12] for c in cont], rotation=30, ha="right", fontsize=8, color="white")
    ax.set_ylabel("Media normalizada (0-1)")
    ax.set_title("Perfil de variables continuas por cluster", fontsize=11)
    ax.legend(fontsize=8, framealpha=0, labelcolor="white", ncol=2)
    ax.set_ylim(0, 1.15); _style(ax); return fig

def grafico_binarias_cluster(df_res):
    """Tasa de prevalencia de factores binarios por cluster."""
    bins = [c for c in ['MemoryComplaints','BehavioralProblems','Hypertension',
                         'Confusion','Depression','Diabetes','Smoking',
                         'CardiovascularDisease','FamilyHistoryAlzheimers']
            if c in df_res.columns]
    grupos = ORDEN_GRUPOS
    x = np.arange(len(bins)); w = 0.18
    fig, ax = _fig(8, 4)
    for i, g in enumerate(grupos):
        sub = df_res[df_res["Grupo_KMeans"]==g]
        tasas = [sub[b].mean() if len(sub)>0 else 0 for b in bins]
        c = COLORES_GRUPO_MAP.get(g, _COLS_FB[i])
        ax.bar(x + i*w, [t*100 for t in tasas], w, label=g, color=c, alpha=0.85)
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels([b[:14] for b in bins], rotation=30, ha="right", fontsize=8, color="white")
    ax.set_ylabel("Prevalencia (%)"); ax.set_ylim(0, 115)
    ax.set_title("Prevalencia de factores de riesgo por cluster", fontsize=11)
    ax.legend(fontsize=8, framealpha=0, labelcolor="white", ncol=2)
    _style(ax); return fig

def grafico_concordancia(df_res):
    conc = (df_res["Pred_DT"] == df_res["Pred_Logistica"]).mean() * 100
    fig, ax = _fig(5, 2.5)
    ax.barh(["Concuerdan","Difieren"], [conc, 100-conc],
            color=["#27ae60","#e74c3c"], edgecolor="white")
    ax.set_xlim(0, 115)
    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=4, color="white")
    ax.set_title("Concordancia entre modelos", fontsize=11)
    _style(ax); return fig

def tabla_cluster_summary(df_res):
    """Tabla resumen de clusters con métricas clave."""
    rows = []
    for g in ORDEN_GRUPOS:
        sub = df_res[df_res["Grupo_KMeans"]==g]
        n   = len(sub)
        if n == 0: continue
        row = {"Grupo": g, "N pacientes": n,
               "% del total": f"{n/len(df_res)*100:.1f}%"}
        if "Diagnosis" in sub.columns:
            row["% Alzheimer"] = f"{sub['Diagnosis'].mean()*100:.1f}%"
        for col in ["MMSE","FunctionalAssessment","ADL","SleepQuality"]:
            if col in sub.columns:
                row[f"Media {col}"] = round(sub[col].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows) if rows else None

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Alzheimer ML Dashboard")
st.caption("Árbol de Decisión · Regresión Logística · K-Means")

c1, c2, c3 = st.columns(3)
c1.metric("Modelos cargados", "3")
c2.metric("Variables de entrada", len(variables_all))
c3.metric("Pacientes en dataset", len(df_base))

tabs = st.tabs(["📂 Cargar CSV","✏️ Captura manual",
                "🔮 Predicciones","🗂 Grupos","📊 Hallazgos"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 – CARGAR CSV
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Cargar CSV")
    archivo = st.file_uploader("Sube tu CSV", type=["csv"])

    if archivo is not None:
        df = pd.read_csv(archivo)
        st.session_state["df"] = df
        st.success(f"✅ Archivo cargado — {len(df)} filas, {len(df.columns)} columnas")
        st.dataframe(df.head())

        err = []
        for pipe_name, vars_ in [("DT", variables_dt),
                                  ("Logística", variables_logit),
                                  ("K-Means", variables_kmeans)]:
            miss = [c for c in vars_ if c not in df.columns]
            if miss: err.append(f"{pipe_name} – faltan: {miss}")
        if err:
            for e in err: st.error(e)
        else:
            df_res = df.copy()
            df_res["Pred_DT"]       = pipeline_dt.predict(df[variables_dt])
            df_res["Pred_Logistica"]= pipeline_logit.predict(df[variables_logit])
            cluster_nums            = pipeline_kmeans.predict(df[variables_kmeans])
            df_res["Grupo_KMeans"]  = [LABEL_MAP.get(int(c), str(c)) for c in cluster_nums]
            st.session_state["df_resultados"] = df_res

            st.subheader("Vista previa de resultados")
            st.dataframe(df_res.head())



    elif "df" in st.session_state:
        st.info("Ya cargaste un CSV. Ve a las otras pestañas.")
    else:
        st.info("Sube un CSV para activar predicciones y gráficas.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – CAPTURA MANUAL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Captura manual de datos")

    # ── Grupos dummy: un selector por categoría ───────────────────────────────
    GRUPOS_DUMMY = {
        "AgeGroup": {
            "label": "Grupo de edad",
            "cols":    ["AgeGroup_<65","AgeGroup_65-75","AgeGroup_75-85","AgeGroup_>85"],
            "opciones":["<65","65-75","75-85",">85"],
        },
        "EducationLevel": {
            "label": "Nivel educativo",
            "cols":    ["EducationLevel_Primaria","EducationLevel_Secundaria",
                        "EducationLevel_Sin educación formal","EducationLevel_Universitaria o superior"],
            "opciones":["Primaria","Secundaria","Sin educación formal","Universitaria o superior"],
        },
        "Ethnicity": {
            "label": "Etnia",
            "cols":    ["Ethnicity_Afroamericano","Ethnicity_Asiático",
                        "Ethnicity_Caucásico","Ethnicity_Otro"],
            "opciones":["Afroamericano","Asiático","Caucásico","Otro"],
        },
    }
    DUMMY_COLS = {c for g in GRUPOS_DUMMY.values() for c in g["cols"]}

    binarias = ["Gender","Smoking","FamilyHistoryAlzheimers","CardiovascularDisease",
                "Diabetes","Depression","HeadInjury","Hypertension",
                "MemoryComplaints","BehavioralProblems","Confusion",
                "Disorientation","PersonalityChanges","DifficultyCompletingTasks","Forgetfulness"]

    datos = {}

    # ── Selectores agrupados (demografía) ─────────────────────────────────────
    st.markdown("**Datos demográficos**")
    dem_cols_ui = st.columns(3)
    for idx, (gkey, ginfo) in enumerate(GRUPOS_DUMMY.items()):
        presentes = [c for c in ginfo["cols"] if c in variables_all]
        if not presentes:
            continue
        sel = dem_cols_ui[idx % 3].selectbox(ginfo["label"], ginfo["opciones"], key=f"_g_{gkey}")
        for col, opc in zip(ginfo["cols"], ginfo["opciones"]):
            if col in variables_all:
                datos[col] = 1 if sel == opc else 0

    st.markdown("---")
    st.markdown("**Variables clínicas y de estilo de vida**")

    # Campos individuales (excluir columnas dummy ya manejadas arriba y columnas resultado)
    campos = [c for c in variables_all if c not in EXCLUIR_FORM and c not in DUMMY_COLS]

    col_a, col_b, col_c = st.columns(3)
    col_ciclo = [col_a, col_b, col_c]

    for i, col in enumerate(campos):
        dest = col_ciclo[i % 3]
        if col in binarias:
            datos[col] = dest.selectbox(col, [0, 1], key=col)
        else:
            v_def = float(df_base[col].median()) if col in df_base.columns else 0.0
            v_min = float(df_base[col].min())    if col in df_base.columns else 0.0
            v_max = float(df_base[col].max())    if col in df_base.columns else 1.0
            datos[col] = dest.number_input(col, v_min, v_max, v_def, key=col)

    df_manual = pd.DataFrame([datos])
    st.session_state["df_manual"] = df_manual
    st.markdown("---")
    st.write("**Datos capturados:**")
    st.dataframe(df_manual, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – PREDICCIONES SUPERVISADAS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🔮 Predicciones supervisadas")

    if "df_manual" not in st.session_state:
        st.warning("⚠️ Primero captura datos en **Captura manual**.")
    else:
        dm = st.session_state["df_manual"]

        pred_dt  = pipeline_dt.predict(dm[variables_dt])[0]
        prob_dt  = pipeline_dt.predict_proba(dm[variables_dt])[0]
        pred_log = pipeline_logit.predict(dm[variables_logit])[0]
        prob_log = pipeline_logit.predict_proba(dm[variables_logit])[0]

        # ── Tarjetas resultado ────────────────────────────────────────────────
        st.markdown("### 🏥 Resultado del paciente")
        col_k, col_l = st.columns(2)
        for col_ui, nombre, pred, prob in [
            (col_k, "Árbol de Decisión", pred_dt,  prob_dt),
            (col_l, "Reg. Logística",    pred_log, prob_log),
        ]:
            cls = ETIQUETAS_DIAG[pred]
            conf = prob[pred]*100
            css  = "alzheimer-box" if pred==1 else "normal-box"
            col_ui.markdown(f"""
                <div class="{css}">
                    <h3>{nombre}</h3>
                    <h2>{'⚠️' if pred==1 else '✅'} {cls}</h2>
                    <p>Confianza: <b>{conf:.1f}%</b></p>
                </div>""", unsafe_allow_html=True)

        # ── Probabilidades barras ────────────────────────────────────────────
        st.markdown("### 📊 Probabilidades por clase")
        st.pyplot(grafico_barras_prob(prob_dt, prob_log))

        # ── Tabla comparativa de modelos ──────────────────────────────────────
        st.markdown("### 🔍 Comparación de resultados para este paciente")
        comp_df = pd.DataFrame({
            "Modelo":          ["Árbol de Decisión", "Reg. Logística"],
            "Predicción":      [ETIQUETAS_DIAG[pred_dt], ETIQUETAS_DIAG[pred_log]],
            "Confianza (%)":   [f"{prob_dt[pred_dt]*100:.1f}%",
                                f"{prob_log[pred_log]*100:.1f}%"],
            "P(No Alzheimer)": [f"{prob_dt[0]*100:.1f}%",  f"{prob_log[0]*100:.1f}%"],
            "P(Alzheimer)":    [f"{prob_dt[1]*100:.1f}%",  f"{prob_log[1]*100:.1f}%"],
            "Acuerdo":         ["✅ Sí" if pred_dt==pred_log else "❌ No"]*2,
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # ── Distribución del CSV si cargado ───────────────────────────────────
        if "df_resultados" in st.session_state:
            st.markdown("### 📈 Distribución en el CSV cargado")
            df_res = st.session_state["df_resultados"]
            col_a, col_b = st.columns(2)
            for col_ui, nombre, col_pred in [
                (col_a, "Árbol de Decisión", "Pred_DT"),
                (col_b, "Reg. Logística",    "Pred_Logistica"),
            ]:
                vc    = df_res[col_pred].value_counts().sort_index()
                vals_ = [int(vc.get(k, 0)) for k in [0, 1]]
                col_ui.pyplot(grafico_dona(vals_,
                    [ETIQUETAS_DIAG[0],ETIQUETAS_DIAG[1]],
                    [COLORES_DIAG[0],COLORES_DIAG[1]], nombre))

            # Concordancia
            st.markdown("### 🔁 Concordancia entre modelos (CSV)")
            st.pyplot(grafico_concordancia(df_res))

            # Tablas de métricas si hay Diagnosis
            if "Diagnosis" in df_res.columns:
                y_true = df_res["Diagnosis"]
                st.markdown("### 🏅 Métricas sobre el CSV")
                rows = []
                for nombre, y_pred in [("Árbol de Decisión","Pred_DT"),
                                        ("Reg. Logística","Pred_Logistica")]:
                    yp = df_res[y_pred]
                    try: auc = roc_auc_score(y_true, yp)
                    except: auc = float("nan")
                    rows.append({
                        "Modelo":    nombre,
                        "Accuracy":  accuracy_score(y_true, yp),
                        "Precision": precision_score(y_true, yp, zero_division=0),
                        "Recall":    recall_score(y_true, yp, zero_division=0),
                        "F1":        f1_score(y_true, yp, zero_division=0),
                        "AUC":       auc,
                    })
                df_m = pd.DataFrame(rows)
                st.pyplot(grafico_barras_metricas(df_m))
                st.dataframe(
                    df_m.style.format({c: "{:.2%}" for c in
                                       ["Accuracy","Precision","Recall","F1","AUC"]})
                    .background_gradient(subset=["Accuracy","Precision","Recall","F1"],
                                         cmap="Greens"),
                    use_container_width=True)

                # Matrices de confusión
                st.markdown("### 🔢 Matrices de confusión")
                cm1, cm2 = st.columns(2)
                cm1.pyplot(grafico_conf_matrix(y_true, df_res["Pred_DT"],
                                               "Árbol de Decisión"))
                cm2.pyplot(grafico_conf_matrix(y_true, df_res["Pred_Logistica"],
                                               "Reg. Logística"))
        else:
            st.info("💡 Carga un CSV en la primera pestaña para ver más análisis.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – GRUPOS K-MEANS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🗂 Segmentación K-Means")

    if "df_manual" not in st.session_state:
        st.warning("⚠️ Primero captura datos en **Captura manual**.")
    else:
        dm = st.session_state["df_manual"]
        cluster_num = int(pipeline_kmeans.predict(completar_para_kmeans(dm)[variables_kmeans])[0])
        grupo = LABEL_MAP.get(cluster_num, str(cluster_num))
        titulo_g, desc_g, color_g = DESCRIPCIONES_GRUPO.get(
            grupo, (grupo, "Grupo sin descripción.", "#7f8c8d"))

        st.markdown(f"""
            <div class="grupo-box" style="background:{color_g};">
                <h3>Grupo asignado: {grupo}</h3>
                <h2>{titulo_g}</h2>
                <p>{desc_g}</p>
            </div>""", unsafe_allow_html=True)
        st.markdown("")

        # ── Radar del paciente ────────────────────────────────────────────────
        VARS_RADAR_COLS   = ["MMSE","FunctionalAssessment","ADL","SleepQuality",
                             "DietQuality","CholesterolTotal","BMI"]
        VARS_RADAR_LABELS = ["Desempeño\ncognitivo","Funcionalidad\ngeneral",
                             "Autonomía\ndiaria","Calidad\ndel sueño",
                             "Calidad\nalimentación","Factor\ncardiovascular",
                             "Estado\nnutricional"]
        VARS_RADAR_TABLA  = ["Desempeño cognitivo","Funcionalidad general",
                             "Autonomía en act. diarias","Calidad del sueño",
                             "Calidad de la alimentación","Factor metab./cardiovascular",
                             "Estado nutricional"]

        vars_radar   = [v for v in VARS_RADAR_COLS if v in df_base.columns]
        labels_radar = [VARS_RADAR_LABELS[VARS_RADAR_COLS.index(v)] for v in vars_radar]
        labels_tabla = [VARS_RADAR_TABLA[VARS_RADAR_COLS.index(v)]  for v in vars_radar]

        dm_k = completar_para_kmeans(dm)
        if vars_radar:
            vals_raw  = dm_k[vars_radar].values[0]
            v_min_r   = df_base[vars_radar].min().values
            v_max_r   = df_base[vars_radar].max().values
            denom     = np.where((v_max_r - v_min_r)==0, 1, v_max_r - v_min_r)
            vals_norm = ((vals_raw - v_min_r) / denom).tolist()

            col_r1, col_r2 = st.columns([1, 1])
            col_r1.pyplot(grafico_radar(vals_norm, labels_radar,
                                        f"Perfil clínico del paciente\n({grupo})"))
            df_ri = pd.DataFrame({
                "Dimensión":       labels_tabla,
                "Variable":        vars_radar,
                "Valor":           vals_raw.round(2),
                "Mín dataset":     v_min_r.round(2),
                "Máx dataset":     v_max_r.round(2),
                "% relativo":      [f"{v*100:.0f}%" for v in vals_norm],
            })
            col_r2.markdown("**Perfil clínico del paciente vs rango del dataset**")
            col_r2.dataframe(df_ri, use_container_width=True, hide_index=True)

        if "df_resultados" in st.session_state:
            df_res = st.session_state["df_resultados"]

            # ── Distribución dona ──────────────────────────────────────────────
            st.markdown("### 🍩 Distribución de grupos")
            col_d, col_e = st.columns([1, 1])
            vc_km   = df_res["Grupo_KMeans"].value_counts()
            g_vals  = [int(vc_km.get(lbl, 0)) for lbl in ORDEN_GRUPOS]
            g_cols  = [COLORES_GRUPO_MAP.get(lbl,"#7f8c8d") for lbl in ORDEN_GRUPOS]
            col_d.pyplot(grafico_dona(g_vals, ORDEN_GRUPOS, g_cols, "K-Means"))

            fig_bar, ax2 = plt.subplots(figsize=(5, 3.5), facecolor="none")
            ax2.barh(ORDEN_GRUPOS[::-1], g_vals[::-1],
                     color=g_cols[::-1], edgecolor="white")
            ax2.set_facecolor("none"); ax2.spines[:].set_visible(False)
            ax2.tick_params(colors="white")
            ax2.set_title("Pacientes por grupo", color="white")
            for i, v in enumerate(g_vals[::-1]):
                ax2.text(v+1, i, str(v), va="center", color="white", fontsize=10)
            ax2.xaxis.set_visible(False); fig_bar.patch.set_alpha(0)
            col_e.pyplot(fig_bar)

            # ── Tabla resumen de clusters ──────────────────────────────────────
            st.markdown("### 📋 Tabla resumen de clusters")
            tbl = tabla_cluster_summary(df_res)
            if tbl is not None:
                st.dataframe(tbl, use_container_width=True, hide_index=True)

            # ── Tarjetas descriptivas ──────────────────────────────────────────
            st.markdown("### 🏷️ Descripción de los grupos")
            total = sum(g_vals)
            cols_desc = st.columns(len(ORDEN_GRUPOS))
            for i, lbl in enumerate(ORDEN_GRUPOS):
                t, d, c = DESCRIPCIONES_GRUPO.get(lbl, (lbl, "", "#7f8c8d"))
                n   = vc_km.get(lbl, 0)
                pct = n/total*100 if total > 0 else 0
                cols_desc[i].markdown(f"""
                    <div class="grupo-box" style="background:{c};padding:14px;">
                        <b>{t}</b><br>
                        <small>{n} pacientes ({pct:.1f}%)</small><br>
                        <small style="opacity:.85">{d}</small>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("💡 Carga un CSV en la primera pestaña para ver análisis de clusters.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – HALLAZGOS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("📊 Hallazgos principales")

    st.markdown("""
### 📌 Hallazgos clave
- El **35.4%** de los pacientes del dataset presentan diagnóstico de Alzheimer.
- **MMSE, Functional Assessment y ADL** son las variables clínicas más asociadas al diagnóstico.
- **Memory Complaints y Behavioral Problems** son los factores binarios con mayor relación al Alzheimer.
- Los pacientes con Alzheimer muestran una **reducción importante** en indicadores cognitivos y funcionales.
- K-Means identificó **cuatro perfiles** de pacientes para apoyar el seguimiento clínico.
""")

    with st.expander("ℹ️ Descripción de los modelos", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.info("**Árbol de Decisión**\nPipeline con preprocesamiento integrado. Clasifica Alzheimer vs No Alzheimer.")
        c2.info("**Regresión Logística**\nPipeline con escalado interno. Variables clínicas, estilo de vida y demografía.")
        c3.info("**K-Means**\nPipeline de clustering. Agrupa en 4 perfiles de riesgo.")

    # ── Análisis del dataset base ─────────────────────────────────────────────
    st.markdown("### 🔬 Análisis del dataset base")

    col_h1, col_h2 = st.columns(2)
    # Distribución real
    vc_base = df_base["Diagnosis"].value_counts().sort_index()
    rv_base = [int(vc_base.get(k, 0)) for k in [0, 1]]
    col_h1.markdown("**Distribución de Diagnosis en el dataset**")
    col_h1.pyplot(grafico_dona(rv_base,
        [ETIQUETAS_DIAG[0],ETIQUETAS_DIAG[1]],
        [COLORES_DIAG[0],COLORES_DIAG[1]], "Dataset base"))

    # Correlación variables continuas
    col_h2.markdown("**Correlación de variables continuas con Diagnosis**")
    col_h2.pyplot(grafico_correlacion(df_base))

    # Impacto de factores binarios
    st.markdown("### ⚕️ Impacto de factores de riesgo (dataset base)")
    st.pyplot(grafico_riesgo_binario(df_base))

    # Medias continuas por Diagnosis
    st.markdown("### 📐 Comparación de medias continuas: Alzheimer vs No Alzheimer")
    cont_cols_base = [c for c in ["MMSE","FunctionalAssessment","ADL","SleepQuality",
                                   "BMI","SystolicBP","CholesterolTotal","PhysicalActivity"]
                      if c in df_base.columns]
    media_alz  = df_base[df_base["Diagnosis"]==1][cont_cols_base].mean()
    media_noalz= df_base[df_base["Diagnosis"]==0][cont_cols_base].mean()

    fig_comp, ax_c = plt.subplots(figsize=(8, 3.5), facecolor="none")
    ax_c.set_facecolor("none")
    fig_comp.patch.set_alpha(0)
    x_c = np.arange(len(cont_cols_base)); w_c = 0.35
    b1  = ax_c.bar(x_c - w_c/2, media_noalz.values, w_c,
                   label="No Alzheimer", color="#27ae60", alpha=0.85)
    b2  = ax_c.bar(x_c + w_c/2, media_alz.values,   w_c,
                   label="Alzheimer",    color="#e74c3c", alpha=0.85)
    ax_c.bar_label(b1, fmt="%.1f", color="white", fontsize=7, padding=2)
    ax_c.bar_label(b2, fmt="%.1f", color="white", fontsize=7, padding=2)
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels([c[:14] for c in cont_cols_base],
                         rotation=25, ha="right", fontsize=8, color="white")
    ax_c.legend(fontsize=9, framealpha=0, labelcolor="white")
    ax_c.set_title("Medias por Diagnosis", fontsize=11)
    _style(ax_c); st.pyplot(fig_comp)

    # Tabla de medias formateada
    df_medias = pd.DataFrame({
        "Variable":      cont_cols_base,
        "No Alzheimer":  media_noalz.values.round(2),
        "Alzheimer":     media_alz.values.round(2),
        "Diferencia":    (media_alz - media_noalz).values.round(3),
        "Δ%":           [f"{(a-n)/abs(n)*100:.1f}%" if n!=0 else "—"
                         for a, n in zip(media_alz, media_noalz)],
    })
    st.dataframe(df_medias, use_container_width=True, hide_index=True)

    # ── Análisis del CSV cargado ───────────────────────────────────────────────
    if "df_resultados" not in st.session_state:
        st.info("📂 Sube un CSV en la primera pestaña para análisis adicional.")