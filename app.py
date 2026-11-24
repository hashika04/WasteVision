"""WasteVision — dark glass Streamlit presentation app.
Run: python -m streamlit run WasteVision_Stellar.py
Keep this file beside your six .keras/.joblib checkpoints and comparison CSV.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from datetime import datetime
import hashlib
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
LABELS = ["dry", "wet", "recyclable", "ewaste"]
LABEL_NAMES = {"dry": "Dry Waste", "wet": "Wet Waste", "recyclable": "Recyclable", "ewaste": "E-Waste"}
LABEL_ICONS = {"dry": "◈", "wet": "❋", "recyclable": "♻", "ewaste": "ϟ"}
MODEL_FILES = {
    "MobileNetV2 Transfer": ("cnn", "mobilenetv2_transfer_clean.keras"),
    "CNN Balanced + Augmentation": ("cnn", "cnn_balanced_augmented_clean.keras"),
    "Class-Weighted CNN": ("cnn", "cnn_weighted_clean.keras"),
    "Baseline CNN": ("cnn", "cnn_baseline_clean.keras"),
    "Class-Weighted Random Forest": ("rf", "rf_weighted_clean.joblib"),
    "Baseline Random Forest": ("rf", "rf_baseline_clean.joblib"),
}

st.set_page_config(page_title="WasteVision • Intelligent Waste Classification", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {color-scheme:dark}
html,body,[data-testid="stApp"], [data-testid="stSidebar"] {font-family:'DM Sans',sans-serif; color:#e9e7f3;}
[data-testid="stAppViewContainer"] {background:radial-gradient(ellipse 38% 32% at 92% 1%,rgba(132,46,213,.18),transparent 80%),radial-gradient(ellipse 38% 33% at 10% 88%,rgba(202,69,74,.10),transparent 80%),#0b0a12;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stToolbar"] {right:1rem;}
[data-testid="stMain"] {background:transparent;}
.block-container {max-width:1380px;padding:1.8rem 3rem 4rem;}
[data-testid="stSidebar"] {background:linear-gradient(165deg,#15131f,#100f19);border-right:1px solid rgba(195,164,255,.12);}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:1rem;}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {gap:9px;}
[data-testid="stSidebar"] .stRadio label {background:rgba(255,255,255,.022);border:1px solid rgba(255,255,255,.03);border-radius:13px;padding:.65rem .7rem;transition:all .2s;}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {border:1px solid rgba(189,92,252,.75);background:linear-gradient(95deg,rgba(126,47,222,.35),rgba(113,36,198,.12));box-shadow:inset 3px 0 0 #bd5cfc,0 0 21px rgba(160,44,255,.18);}
[data-testid="stSidebar"] hr {border-color:rgba(255,255,255,.08)}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p {color:#c2bcd3;}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {background:#1b1828;border-color:#3c3353;border-radius:13px;color:#f4efff;}
[data-testid="stFileUploader"] {border:1px solid rgba(183,137,255,.18);background:linear-gradient(135deg,rgba(27,23,42,.85),rgba(17,16,27,.9));border-radius:24px;padding:1.2rem;box-shadow:0 16px 60px #0003;}
[data-testid="stFileUploaderDropzone"] {min-height:160px;border:1.5px dashed #674586;border-radius:18px;background:radial-gradient(ellipse at 50% 20%,rgba(139,57,221,.17),transparent 70%),#141120;}
[data-testid="stFileUploaderDropzone"] button {background:#8d42d5;border:0;color:white;border-radius:11px;}
[data-testid="stFileUploader"] small {color:#aaa2be;}
[data-testid="stImage"] img {border-radius:16px;border:1px solid rgba(224,191,255,.10);}
div.stButton>button[kind="primary"] {background:linear-gradient(100deg,#9138dd,#b650ed 62%,#dc6fba);border:1px solid #c475ff;border-radius:14px;color:white;font-weight:800;box-shadow:0 5px 27px rgba(173,47,238,.32);min-height:49px;}
div.stButton>button[kind="primary"]:hover {border-color:#dfb3ff;filter:brightness(1.1);box-shadow:0 5px 32px rgba(189,75,245,.43);}
div.stButton>button:not([kind="primary"]) {border:1px solid #493858;background:#201a2d;color:#eee4fc;border-radius:12px;}
[data-testid="stMetric"] {background:linear-gradient(135deg,rgba(36,29,53,.9),rgba(21,19,33,.94));border:1px solid rgba(187,135,255,.18);padding:17px;border-radius:18px;}
[data-testid="stMetricLabel"] {color:#a99bbf;}
[data-testid="stMetricValue"] {font-family:'Space Grotesk',sans-serif;color:#f3eafd;font-weight:700;}
[data-testid="stDataFrame"] {border:1px solid #342b49;border-radius:13px;overflow:hidden;}
[data-testid="stExpander"] {border-color:#3b304c;background:#171420;border-radius:15px;}
[data-testid="stAlert"] {background:#1c162b;color:#d5c5f3;border:1px solid #483565;}
[data-testid="stCaptionContainer"],.stCaption {color:#9288a7;}
[data-testid="stMarkdownContainer"] a {color:#d18aff;}
hr {border-color:#30263b;}
.brand {display:flex;gap:12px;align-items:center;margin:7px 0 16px;}
.brand-icon {display:grid;place-items:center;width:43px;height:43px;border-radius:14px;background:linear-gradient(135deg,#6925b4,#bb4af0);box-shadow:0 0 25px #9d40fa66;font-size:25px;color:white;}
.brand-name {font-family:'Space Grotesk';font-size:23px;font-weight:700;letter-spacing:-1.1px;line-height:1.05;}
.brand-sub {font-size:10px;color:#9890ad;letter-spacing:1.6px;margin-top:4px;text-transform:uppercase;}
.side-caption {text-transform:uppercase;letter-spacing:1.5px;font-size:10px;color:#9382ae;font-weight:800;margin:12px 0 4px;}
.side-foot {font-size:11px;color:#8d839d;line-height:1.7;}
.topbar {display:flex;justify-content:space-between;align-items:center;gap:20px;border-bottom:1px solid #30283c;padding:4px 0 19px;}
.kicker {font-size:11px;letter-spacing:2px;font-weight:800;color:#c879fd;text-transform:uppercase;}
.status {display:inline-block;font-size:11px;color:#dbcbef;padding:9px 12px;border:1px solid #4c365f;border-radius:99px;background:#1f192c;}
.hero {position:relative;overflow:hidden;background:radial-gradient(ellipse 55% 95% at 92% 16%,rgba(151,50,231,.29),transparent 82%),linear-gradient(115deg,#1b152a,#10101c 75%);border:1px solid #40304f;border-radius:26px;margin:27px 0 26px;padding:40px 43px;box-shadow:0 20px 80px #0004;}
.hero:after {content:'';position:absolute;width:170px;height:170px;border:1px solid rgba(177,91,255,.16);border-radius:37px;right:55px;top:15px;transform:rotate(-20deg);pointer-events:none;box-shadow:0 0 80px #9d39f844;}
.hero h1 {font-family:'Space Grotesk',sans-serif;letter-spacing:-2px;font-size:clamp(34px,4vw,57px);line-height:1.1;margin:12px 0;color:#f8f4ff;max-width:790px;}
.hero h1 span {background:linear-gradient(100deg,#f0caff,#cb7bff,#ef8dbe);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p {color:#aca4bc;font-size:15px;max-width:665px;line-height:1.7;margin:0;}
.section-head {font-family:'Space Grotesk';font-size:22px;font-weight:700;letter-spacing:-.7px;color:#f0eafa;margin:9px 0 15px;}
.pill {display:inline-block;padding:6px 10px;font-size:11px;color:#d5b1ef;background:#392149;border:1px solid #6d3b88;border-radius:99px;margin-bottom:8px;}
.card {background:linear-gradient(140deg,rgba(32,26,46,.95),rgba(19,17,29,.97));border:1px solid #393048;border-radius:23px;padding:25px;box-shadow:0 15px 45px #0003;}
.card.highlight {border-color:#7843a2;box-shadow:0 0 40px rgba(127,42,195,.16),0 15px 45px #0003;}
.label {font-size:10px;text-transform:uppercase;letter-spacing:1.7px;font-weight:800;color:#aa97c5;margin-bottom:8px;}
.answer {font-family:'Space Grotesk';font-size:clamp(28px,3.2vw,43px);font-weight:700;color:#f8e7ff;letter-spacing:-1.5px;line-height:1.1;margin:8px 0;}
.confidence {font-size:14px;color:#cf9bfb;margin:10px 0 21px;}
.divider {height:1px;background:#41324c;margin:18px 0 22px;}
.prob-row {margin:15px 0 19px;}
.prob-label {display:flex;justify-content:space-between;color:#d5cee0;font-size:13px;font-weight:600;margin-bottom:9px;}
.track {height:9px;border-radius:99px;background:#34283f;overflow:hidden;}
.fill {height:100%;border-radius:99px;background:#665278;}
.fill.winner {background:linear-gradient(90deg,#873adb,#ca5df0,#ed9ab7);box-shadow:0 0 13px #c75cf477;}
.placeholder {text-align:center;padding:62px 20px;color:#a89aba;}
.placeholder .glyph {font-size:40px;margin-bottom:15px;color:#bd83f5;}
.note {font-size:12px;line-height:1.65;color:#9188a7;}
@media(max-width:900px) {.block-container{padding:1rem 1.1rem 4rem}.hero{padding:30px 24px}.hero:after{right:-78px}.hero h1{font-size:36px}.status{font-size:10px}.card{padding:20px}.topbar{flex-wrap:wrap}}
@media(max-width:520px) {.hero{padding:24px 20px}.hero h1{font-size:31px}.hero p{font-size:13px}.answer{font-size:30px}.status{display:none}}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading trained classifier…")
def load_classifier(family: str, filename: str):
    path = ROOT / filename
    if family == "rf":
        import joblib
        return joblib.load(path)
    from tensorflow import keras
    return keras.models.load_model(path)

@st.cache_data(show_spinner=False)
def validation_metrics():
    path = ROOT / "six_model_validation_comparison.csv"
    if not path.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if {"Model", "Macro F1"}.issubset(df.columns):
            return df.sort_values("Macro F1", ascending=False)
    except (OSError, ValueError, pd.errors.ParserError):
        pass
    return pd.DataFrame()

def rf_features(image: Image.Image):
    """Match the training notebook's 96 RGB bins + 16 GLCM values."""
    from skimage.color import rgb2gray
    from skimage.feature import graycomatrix, graycoprops
    arr = np.asarray(image.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    colour = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=32, range=(0, 1))
        hist = hist.astype(np.float32)
        hist /= max(hist.sum(), 1)
        colour.extend(hist)
    grayscale = np.clip((rgb2gray(arr) * 32).astype(np.uint8), 0, 31)
    matrix = graycomatrix(grayscale, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    texture = []
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        texture.extend(graycoprops(matrix, prop).ravel())
    features = np.concatenate((colour, texture)).astype(np.float32)
    if features.shape != (112,):
        raise ValueError("Incorrect Random Forest feature shape")
    return features[None, :]

def classify(image: Image.Image, model_name: str):
    family, filename = MODEL_FILES[model_name]
    model = load_classifier(family, filename)
    if family == "rf":
        probabilities = model.predict_proba(rf_features(image))[0]
        scores = dict(zip([str(x) for x in model.classes_], map(float, probabilities)))
    else:
        # RGB 0–255 values; the Keras model handles rescaling internally.
        arr = np.asarray(image.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
        probabilities = model.predict(arr[None, ...], verbose=0)[0]
        scores = dict(zip(LABELS, map(float, probabilities)))
    return {label: max(0.0, float(scores.get(label, 0.0))) for label in LABELS}

def bars(scores):
    winner = max(scores, key=scores.get)
    output = []
    for label in sorted(LABELS, key=lambda x: scores[x], reverse=True):
        value = max(0, min(100, scores[label] * 100))
        cls = "winner" if label == winner else ""
        output.append(f'<div class="prob-row"><div class="prob-label"><span>{LABEL_ICONS[label]} &nbsp; {LABEL_NAMES[label]}</span><span>{value:.1f}%</span></div><div class="track"><div class="fill {cls}" style="width:{value:.1f}%"></div></div></div>')
    return "".join(output)


# Plotly charts share the same neon-dark theme as the Streamlit dashboard.
PLOT_COLORS = {
    "MobileNetV2 Transfer": "#c66bff",
    "CNN Balanced + Augmentation": "#fd8d64",
    "Class-Weighted CNN": "#e68ab5",
    "Baseline CNN": "#8b78db",
    "Class-Weighted Random Forest": "#45c9d5",
    "Baseline Random Forest": "#7384ab",
}


def configure_plot(fig, height=430, legend=True):
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#ded6ed", size=12),
        margin=dict(l=20, r=25, t=35, b=20),
        showlegend=legend,
        legend=dict(orientation="h", y=-0.21, x=0),
        hoverlabel=dict(bgcolor="#201832", font_color="#f5efff"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#40334f", tickfont=dict(color="#a99bbf"))
    fig.update_yaxes(gridcolor="rgba(178,140,226,.12)", zeroline=False, tickfont=dict(color="#a99bbf"))
    return fig


@st.cache_data(show_spinner=False)
def test_metrics():
    path = ROOT / "six_model_test_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def confusion_data():
    path = ROOT / "validation_confusion_matrices.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if {"Model", "True", "Predicted", "Count"}.issubset(df.columns):
            return df
    except (OSError, ValueError, pd.errors.ParserError):
        pass
    return pd.DataFrame()


def metric_comparison_chart(df, metric, choices):
    selected = df[df["Model"].isin(choices)].sort_values(metric)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=selected[metric] * 100, y=selected["Model"], orientation="h",
        marker=dict(color=[PLOT_COLORS.get(name, "#a97dfb") for name in selected["Model"]], line_width=0),
        text=[f"{v:.1f}%" for v in selected[metric] * 100], textposition="outside",
        customdata=selected["Model"], hovertemplate="%{customdata}<br>" + metric + ": %{x:.2f}%<extra></extra>"
    ))
    fig.update_xaxes(range=[0, 110], ticksuffix="%")
    return configure_plot(fig, height=max(300, 85 + 56 * len(selected)), legend=False)


def scatter_chart(df, choices):
    selected = df[df["Model"].isin(choices)]
    fig = go.Figure()
    for _, row in selected.iterrows():
        name = str(row["Model"])
        fig.add_trace(go.Scatter(
            x=[row["Accuracy"] * 100], y=[row["Macro F1"] * 100],
            mode="markers", name=name,
            marker=dict(size=20, color=PLOT_COLORS.get(name, "#bb8aff"),
                        line=dict(width=2, color="#f3eaff")),
            hovertemplate=f"{name}<br>Accuracy: %{{x:.2f}}%<br>Macro F1: %{{y:.2f}}%<extra></extra>"
        ))
    fig.update_xaxes(title="Accuracy (%)", range=[0, 100])
    fig.update_yaxes(title="Macro F1 (%)", range=[0, 100])
    return configure_plot(fig, height=415)


def recall_table(df):
    result = df[df["True"] == df["Predicted"]][["Model", "True", "Count"]].copy()
    total = df.groupby(["Model", "True"], as_index=False)["Count"].sum().rename(columns={"Count": "total"})
    result = result.merge(total, on=["Model", "True"], how="left")
    result["Recall"] = result["Count"] / result["total"].replace(0, np.nan)
    return result.pivot(index="Model", columns="True", values="Recall").reindex(columns=LABELS)


def recall_heatmap(confusions, choices):
    matrix = recall_table(confusions).reindex(choices).dropna(how="all")
    fig = go.Figure(data=go.Heatmap(
        z=matrix.to_numpy() * 100,
        x=[LABEL_NAMES[x] for x in matrix.columns], y=matrix.index.tolist(),
        colorscale=[[0, "#26152e"], [.35, "#5b2b91"], [.7, "#a347e1"], [1, "#f4a0e6"]],
        zmin=0, zmax=100,
        text=np.round(matrix.to_numpy() * 100, 1), texttemplate="%{text:.1f}%",
        textfont=dict(color="#f9f5ff", size=11),
        hovertemplate="%{y}<br>%{x} recall: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Recall %", ticksuffix="%"),
    ))
    fig.update_yaxes(autorange="reversed")
    return configure_plot(fig, height=max(320, 100 + 52 * len(matrix)), legend=False)


def confusion_chart(confusions, model, normalized):
    data = confusions[confusions["Model"] == model]
    matrix = data.pivot_table(index="True", columns="Predicted", values="Count", aggfunc="sum", fill_value=0).reindex(index=LABELS, columns=LABELS, fill_value=0)
    values = matrix.to_numpy(dtype=float)
    if normalized:
        row_sum = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum > 0) * 100
    fig = go.Figure(data=go.Heatmap(
        z=values, x=[LABEL_NAMES[x] for x in LABELS], y=[LABEL_NAMES[x] for x in LABELS],
        colorscale=[[0, "#1b1728"], [.25, "#573484"], [.7, "#b44de8"], [1, "#ef98ca"]],
        text=np.round(values, 1) if normalized else values.astype(int),
        texttemplate="%{text:.1f}%" if normalized else "%{text:.0f}",
        textfont=dict(color="white"),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>" + ("Recall share: %{z:.1f}%" if normalized else "Images: %{z:.0f}") + "<extra></extra>",
        showscale=False,
    ))
    fig.update_xaxes(title="Predicted category", side="bottom")
    fig.update_yaxes(title="Actual category", autorange="reversed")
    return configure_plot(fig, height=445, legend=False)


def confidence_chart(scores):
    order = sorted(LABELS, key=lambda x: scores[x])
    leader = max(scores, key=scores.get)
    fig = go.Figure(go.Bar(
        x=[scores[x] * 100 for x in order], y=[LABEL_NAMES[x] for x in order], orientation="h",
        marker=dict(color=["#d477f8" if x == leader else "#574471" for x in order], line_width=0),
        text=[f"{scores[x] * 100:.1f}%" for x in order], textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>"
    ))
    fig.update_xaxes(range=[0, 108], ticksuffix="%")
    return configure_plot(fig, height=290, legend=False)

available = [name for name, (_, filename) in MODEL_FILES.items() if (ROOT / filename).is_file()]
metrics = validation_metrics()
preferred = "MobileNetV2 Transfer"
if not metrics.empty:
    valid = metrics[metrics["Model"].isin(available)]
    if not valid.empty:
        preferred = str(valid.iloc[0]["Model"])

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">✦</div><div><div class="brand-name">WasteVision</div><div class="brand-sub">Intelligent classification</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-caption">Workspace</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["✧  Classify image", "▦  Model insights", "◷  Prediction history"], label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="side-caption">Classification engine</div>', unsafe_allow_html=True)
    if not available:
        st.error("No saved models found. Place the six .keras/.joblib checkpoints beside this Python file.")
        st.stop()
    model_name = st.selectbox("Select model", available, index=available.index(preferred) if preferred in available else 0)
    if model_name == preferred:
        st.caption("✦ Highest validation macro F1 among available models")
    st.divider()
    st.markdown('<div class="side-caption">Project</div><div class="side-foot">Four waste categories · Six classification experiments · Local inference</div>', unsafe_allow_html=True)

st.markdown('<div class="topbar"><div class="kicker">✦ &nbsp; Hashika&#39;s WasteVision</div><span class="status">● &nbsp; Local model inference</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><span class="pill">✦ &nbsp; Computer vision project</span><h1>Classify waste intelligently</h1><p>Explore image recognition across six trained models. Upload a photo, analyse predicted waste categories and compare model performance in one workspace.</p></div>', unsafe_allow_html=True)

if "history_records" not in st.session_state:
    st.session_state.history_records = []

if page.startswith("▦"):
    st.markdown('<div class="section-head">Performance intelligence</div>', unsafe_allow_html=True)
    if metrics.empty:
        st.info("Add six_model_validation_comparison.csv beside this app to enable model analytics.")
    else:
        # All charts use the CSV values produced by the notebook, not sample numbers.
        top = metrics.iloc[0]
        left, middle, right = st.columns(3)
        with left:
            st.metric("Top validation model", str(top["Model"]))
        with middle:
            st.metric("Validation accuracy", f'{float(top["Accuracy"]) * 100:.2f}%')
        with right:
            st.metric("Validation macro F1", f'{float(top["Macro F1"]) * 100:.2f}%')

        st.markdown('<div class="section-head" style="margin-top:28px">Explore classification performance</div>', unsafe_allow_html=True)
        metric_options = [x for x in ("Macro F1", "Accuracy", "Macro Precision", "Macro Recall") if x in metrics.columns]
        control1, control2 = st.columns([1, 2])
        with control1:
            chosen_metric = st.selectbox("Performance metric", metric_options)
        with control2:
            chosen_models = st.multiselect("Compare models", metrics["Model"].tolist(), default=metrics["Model"].tolist())
        if chosen_models:
            tab1, tab2, tab3 = st.tabs(["Ranking", "Accuracy vs F1", "Full results"])
            with tab1:
                st.plotly_chart(metric_comparison_chart(metrics, chosen_metric, chosen_models), use_container_width=True, config={"displaylogo": False})
            with tab2:
                if {"Accuracy", "Macro F1"}.issubset(metrics.columns):
                    st.plotly_chart(scatter_chart(metrics, chosen_models), use_container_width=True, config={"displaylogo": False})
            with tab3:
                st.dataframe(metrics[metrics["Model"].isin(chosen_models)].round(4), hide_index=True, use_container_width=True)

            conf = confusion_data()
            if not conf.empty:
                st.markdown('<div class="section-head" style="margin-top:32px">Class-specific recognition</div>', unsafe_allow_html=True)
                st.caption("Recall is the share of actual images of a class that the model correctly identifies. The charts are calculated from the saved validation confusion matrices.")
                st.plotly_chart(recall_heatmap(conf, chosen_models), use_container_width=True, config={"displaylogo": False})
                st.markdown('<div class="section-head" style="margin-top:24px">Explore misclassifications</div>', unsafe_allow_html=True)
                left_choice, right_choice = st.columns([2, 1])
                options = [m for m in chosen_models if m in conf["Model"].unique()]
                if options:
                    with left_choice:
                        inspect_model = st.selectbox("Inspect model", options)
                    with right_choice:
                        normalized = st.toggle("Show row percentages", value=True)
                    st.plotly_chart(confusion_chart(conf, inspect_model, normalized), use_container_width=True, config={"displaylogo": False})
            else:
                st.caption("Add validation_confusion_matrices.csv to enable class-wise recall and interactive confusion matrices.")

            test_df = test_metrics()
            if not test_df.empty and {"Accuracy", "Macro F1"}.issubset(test_df.columns):
                st.markdown('<div class="section-head" style="margin-top:30px">Generalisation: validation vs test</div>', unsafe_allow_html=True)
                st.caption("Test results are shown for reporting only. The best model was chosen using validation macro F1.")
                combined = pd.merge(metrics[["Model", "Macro F1"]], test_df[["Model", "Macro F1"]], on="Model", suffixes=(" validation", " test"))
                combined = combined[combined["Model"].isin(chosen_models)]
                fig = go.Figure()
                for name, colr in [("Validation", "#b05df0"), ("Test", "#fb9b75")]:
                    fig.add_trace(go.Bar(
                        x=combined["Model"], y=combined[f"Macro F1 {name.lower()}"] * 100,
                        name=name, marker_color=colr,
                        hovertemplate="%{x}<br>" + name + " macro F1: %{y:.2f}%<extra></extra>"
                    ))
                fig.update_layout(barmode="group")
                fig.update_yaxes(title="Macro F1 (%)", range=[0, 105])
                fig.update_xaxes(tickangle=-20)
                st.plotly_chart(configure_plot(fig, height=440), use_container_width=True, config={"displaylogo": False})
        else:
            st.info("Choose at least one model to display its analytics.")
    st.stop()

if page.startswith("◷"):
    st.markdown('<div class="section-head">Prediction history</div>', unsafe_allow_html=True)
    if st.session_state.history_records:
        st.dataframe(pd.DataFrame(st.session_state.history_records[::-1]), hide_index=True, use_container_width=True)
        if st.button("Clear history"):
            st.session_state.history_records = []
            st.rerun()
    else:
        st.markdown('<div class="card"><div class="placeholder"><div class="glyph">◷</div>No predictions recorded yet.<div class="note">Images you classify in this browser session will appear here.</div></div></div>', unsafe_allow_html=True)
    st.stop()

st.markdown('<div class="section-head">Image classification</div>', unsafe_allow_html=True)
left, right = st.columns([1.03, 1], gap="large")
with left:
    st.markdown('<div class="label">01 / Image input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a waste image", type=["png", "jpg", "jpeg", "webp"], help="JPG, PNG or WebP, up to 15MB")
    image = None
    if uploaded:
        try:
            if uploaded.size > 15 * 1024 * 1024:
                st.error("Please upload an image smaller than 15 MB.")
            else:
                with Image.open(BytesIO(uploaded.getvalue())) as raw:
                    image = ImageOps.exif_transpose(raw).convert("RGB").copy()
                image.thumbnail((4096, 4096))
                st.image(image, caption=f"{uploaded.name} · {image.width} × {image.height}", use_container_width=True)
        except (OSError, UnidentifiedImageError, ValueError):
            st.error("This image couldn't be opened. Please try another JPG, PNG or WebP file.")
    clicked = st.button("✦  Analyse image", type="primary", use_container_width=True, disabled=image is None)
    st.caption("Your image is processed by the selected classifier running in the local Python environment.")

with right:
    st.markdown('<div class="label">02 / Classification output</div>', unsafe_allow_html=True)
    if clicked and image is not None:
        try:
            with st.spinner("Running image recognition…"):
                result = classify(image, model_name)
            key = (hashlib.sha256(uploaded.getvalue()).hexdigest(), model_name)
            st.session_state.last_prediction = (key, result)
            predicted = max(result, key=result.get)
            st.session_state.history_records.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Image": uploaded.name,
                "Model": model_name,
                "Prediction": LABEL_NAMES[predicted],
                "Probability": f"{result[predicted] * 100:.1f}%",
            })
        except Exception as exc:
            st.error(f"Model prediction failed: {exc}")
    valid_key = (hashlib.sha256(uploaded.getvalue()).hexdigest(), model_name) if uploaded else None
    saved = st.session_state.get("last_prediction")
    if valid_key and saved and saved[0] == valid_key:
        result = saved[1]
        predicted = max(result, key=result.get)
        st.markdown('<div class="card highlight"><div class="label">Classification complete</div><div class="answer">'+html.escape(LABEL_NAMES[predicted])+'</div><div class="confidence">✦ &nbsp; '+f'{result[predicted] * 100:.1f}% estimated probability'+'</div><div class="divider"></div><div class="note">Model: '+html.escape(model_name)+'<br/>Classification estimates do not confirm whether an item is accepted by local recycling facilities.</div></div>', unsafe_allow_html=True)
        st.plotly_chart(confidence_chart(result), use_container_width=True, config={"displaylogo": False})
        st.caption("Hover over the bars to see the model's estimated probabilities. These are not calibrated measures of certainty.")
    else:
        st.markdown('<div class="card"><div class="placeholder"><div class="glyph">✧</div><strong>Your results will appear here.</strong><div class="note" style="margin-top:12px">Upload an image, select a model, and click Analyse image to view the prediction and category probabilities.</div></div></div>', unsafe_allow_html=True)

with st.expander("About WasteVision and the models"):
    st.write("WasteVision compares baseline and class-weighted Random Forest classifiers, baseline and class-weighted CNNs, MobileNetV2 transfer learning, and a CNN using balanced sampling with augmentation. All models predict four categories: dry, wet, recyclable and e-waste.")
    st.caption("A classroom demonstration; predictions are estimates, not authoritative disposal advice.")
