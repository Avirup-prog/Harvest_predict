import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Post-Harvest Loss Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS – Green Theme ───────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0d1f0f; }

  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a2210 0%, #122b14 60%, #0d1f0f 100%);
    border-right: 1px solid #1e4d22;
  }
  [data-testid="stSidebar"] * { color: #a8e6a3 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stNumberInput label { color: #6fcf70 !important; font-weight: 600; }

  body, .stMarkdown, p, li, span { color: #c8f0c5; }

  h1 { color: #4cdf6c !important; font-size: 2.2rem !important; letter-spacing: -0.5px; }
  h2 { color: #3ecf5a !important; border-bottom: 1px solid #1e4d22; padding-bottom: 6px; }
  h3 { color: #5de078 !important; }

  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #122b14 0%, #1a3d1c 100%);
    border: 1px solid #2a6b2e;
    border-radius: 12px;
    padding: 16px 20px;
  }
  [data-testid="metric-container"] label { color: #6fcf70 !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 1px; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #4cdf6c !important; font-size: 2rem !important; }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #88d68a !important; }

  .stButton > button {
    background: linear-gradient(135deg, #1a6b2a 0%, #2d8c3e 100%);
    color: #e8f8e5 !important;
    border: 1px solid #3aad50;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 700;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #2d8c3e 0%, #3aad50 100%);
    border-color: #4cdf6c;
    box-shadow: 0 0 16px rgba(76,223,108,0.3);
    transform: translateY(-1px);
  }

  .stTabs [data-baseweb="tab-list"] { background-color: #122b14; border-radius: 10px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { color: #6fcf70 !important; border-radius: 8px; font-weight: 600; }
  .stTabs [aria-selected="true"] { background-color: #1e5c25 !important; color: #4cdf6c !important; }

  .stSelectbox > div > div,
  .stMultiSelect > div > div,
  .stNumberInput > div > div > input {
    background-color: #122b14 !important;
    border: 1px solid #2a6b2e !important;
    border-radius: 8px !important;
    color: #c8f0c5 !important;
  }

  [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background: #4cdf6c !important; }

  .stDataFrame { border: 1px solid #2a6b2e; border-radius: 8px; overflow: hidden; }
  .stAlert { border-radius: 10px !important; }
  .stSuccess { background-color: #0f3312 !important; border-color: #2d8c3e !important; }
  .stInfo    { background-color: #0a2210 !important; border-color: #1e5c25 !important; }
  .stWarning { background-color: #2a1f00 !important; border-color: #7a5c00 !important; }

  .hero-banner {
    background: linear-gradient(135deg, #0f3312 0%, #1a4d1e 50%, #122b14 100%);
    border: 1px solid #2a6b2e;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .hero-tag {
    display: inline-block;
    background-color: #1a4d1e;
    color: #4cdf6c;
    border: 1px solid #3aad50;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-right: 8px;
    margin-bottom: 12px;
  }
  .green-card {
    background: linear-gradient(135deg, #0f2e12 0%, #172e18 100%);
    border: 1px solid #2a6b2e;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }
  .tune-card {
    background: linear-gradient(135deg, #0a1f0c 0%, #112614 100%);
    border: 1px solid #1e5c25;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
  }
  .best-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1a6b2a, #3aad50);
    color: #e8f8e5;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.5px;
  }
  .improvement-up {
    color: #4cdf6c;
    font-weight: 700;
    font-size: 1.1rem;
  }
  .improvement-down {
    color: #e74c3c;
    font-weight: 700;
    font-size: 1.1rem;
  }
  .predict-high {
    background: linear-gradient(135deg, #3d1010 0%, #4d1a1a 100%);
    border: 2px solid #c0392b;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
  }
  .predict-low {
    background: linear-gradient(135deg, #0d2e0f 0%, #163318 100%);
    border: 2px solid #27ae60;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
  }
  .result-label { font-size: 2.2rem; font-weight: 800; margin-bottom: 4px; }
  .result-sub   { font-size: 0.95rem; color: #a8e6a3; }

  hr { border-color: #1e4d22 !important; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d1f0f; }
  ::-webkit-scrollbar-thumb { background: #2a6b2e; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
GREEN_PALETTE = ["#4cdf6c","#2d8c3e","#88d68a","#1a6b2a","#a8e6a3",
                 "#5de078","#3ecf5a","#6fcf70","#0f6623","#c8f0c5"]

def green_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f2e12",
        font=dict(color="#c8f0c5", family="Inter, sans-serif"),
        title_font=dict(color="#4cdf6c", size=15),
        legend=dict(bgcolor="#122b14", bordercolor="#2a6b2e", borderwidth=1,
                    font=dict(color="#c8f0c5")),
        margin=dict(l=30, r=20, t=50, b=30),
        xaxis=dict(gridcolor="#1e4d22", linecolor="#2a6b2e", tickfont=dict(color="#88d68a")),
        yaxis=dict(gridcolor="#1e4d22", linecolor="#2a6b2e", tickfont=dict(color="#88d68a")),
    )
    return fig

# ── Base data loading (cached separately — never changes) ─────────────────────
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "Dataset.csv")
    df = pd.read_csv(csv_path)
    df_model = df[df["Element"] == "Agricultural Use"].copy()
    df_model.dropna(subset=["value","Food loss percentage","Item","Sub-region Name"], inplace=True)
    df_model.drop_duplicates(inplace=True)
    median_loss = df_model["Food loss percentage"].median()
    df_model["Loss_Class"] = (df_model["Food loss percentage"] >= median_loss).astype(int)
    le_item   = LabelEncoder()
    le_region = LabelEncoder()
    df_model["Item_enc"]   = le_item.fit_transform(df_model["Item"])
    df_model["Region_enc"] = le_region.fit_transform(df_model["Sub-region Name"])
    return df, df_model, le_item, le_region, median_loss

# ── Train with given hyperparams (cached per param combo) ─────────────────────
@st.cache_data
def train_model(C, penalty, solver, max_iter):
    _, df_model, _, _, _ = load_data()
    X = df_model[["value","Year","Item_enc","Region_enc"]]
    y = df_model["Loss_Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    model = LogisticRegression(C=C, penalty=penalty, solver=solver,
                               max_iter=max_iter, random_state=42)
    model.fit(X_train_sc, y_train)
    y_pred   = model.predict(X_test_sc)
    accuracy = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                     target_names=["Low Loss","High Loss"],
                                     output_dict=True)
    # Cross-validation score (5-fold stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring="accuracy")
    return model, scaler, accuracy, cm, report, cv_scores

# ── GridSearchCV (cached — runs once, heavy) ──────────────────────────────────
@st.cache_data
def run_grid_search():
    _, df_model, _, _, _ = load_data()
    X = df_model[["value","Year","Item_enc","Region_enc"]]
    y = df_model["Loss_Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    param_grid = [
        {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l2"],
         "solver": ["lbfgs","saga"], "max_iter": [500,1000]},
        {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1"],
         "solver": ["liblinear","saga"], "max_iter": [500,1000]},
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(LogisticRegression(random_state=42), param_grid,
                      cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train_sc, y_train)

    best = gs.best_estimator_
    y_pred   = best.predict(X_test_sc)
    accuracy = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                     target_names=["Low Loss","High Loss"],
                                     output_dict=True)
    results_df = pd.DataFrame(gs.cv_results_)
    return gs.best_params_, gs.best_score_, accuracy, cm, report, results_df, best, scaler

# ── Load base data ────────────────────────────────────────────────────────────
df_raw, df_model, le_item, le_region, median_loss = load_data()

# Default model (baseline)
default_model, default_scaler, default_accuracy, default_cm, default_report, default_cv = \
    train_model(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Navigation")
    page = st.radio("", ["🏠 Overview", "📊 Explore Data",
                          "🤖 Model Insights", "⚙️ Hyperparameter Tuning", "🔮 Predict"])
    st.markdown("---")
    st.markdown("### 📋 Dataset Info")
    st.metric("Total Records",   f"{len(df_raw):,}")
    st.metric("Sub-regions",     df_raw["Sub-region Name"].nunique())
    st.metric("Pesticide Types", df_raw["Item"].nunique())
    st.metric("Year Range",      f"{df_raw['Year'].min()}–{df_raw['Year'].max()}")
    st.markdown("---")
    st.markdown("""
    <div style='color:#6fcf70;font-size:0.78rem;line-height:1.7'>
    <b>SDG Goal 2 – Zero Hunger</b><br>
    Predicts post-harvest food loss based on pesticide usage patterns using Logistic Regression + GridSearchCV tuning.
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
page_key = page.split(" ", 1)[1]

st.markdown(f"""
<div class='hero-banner'>
  <span class='hero-tag'>SDG 2</span>
  <span class='hero-tag'>Zero Hunger</span>
  <span class='hero-tag'>Logistic Regression</span>
  <span class='hero-tag'>GridSearchCV</span>
  <h1 style='margin:0 0 6px 0;'>🌾 Post-Harvest Loss Prediction</h1>
  <p style='color:#88d68a;margin:0;font-size:1rem;'>
    Analyzing pesticide usage patterns to classify food loss risk by sub-region.
    Currently viewing: <b style='color:#4cdf6c'>{page_key}</b>
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page_key == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Accuracy", f"{default_accuracy*100:.2f}%", "Default params")
    c2.metric("Median Loss Threshold", f"{median_loss:.2f}%", "Binary split")
    c3.metric("High-Loss Samples", str(df_model["Loss_Class"].sum()), "class 1")
    c4.metric("Low-Loss Samples",  str((df_model["Loss_Class"]==0).sum()), "class 0")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("🗺️ Average Food Loss by Sub-region")
        region_loss = (df_model.groupby("Sub-region Name")["Food loss percentage"]
                       .mean().sort_values(ascending=True).reset_index())
        fig = px.bar(region_loss, x="Food loss percentage", y="Sub-region Name",
                     orientation="h", color="Food loss percentage",
                     color_continuous_scale=[[0,"#1a6b2a"],[0.5,"#3ecf5a"],[1,"#4cdf6c"]],
                     labels={"Sub-region Name":"","Food loss percentage":"Avg Loss %"})
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(green_fig(fig), use_container_width=True)

    with col_r:
        st.subheader("🥧 Efficiency Group Distribution")
        eff = df_raw["Efficiency Group"].value_counts().reset_index()
        eff.columns = ["Group","Count"]
        fig2 = px.pie(eff, names="Group", values="Count",
                      color_discrete_sequence=GREEN_PALETTE, hole=0.45)
        fig2.update_traces(textinfo="percent+label", textfont_color="#0d1f0f", textfont_size=11)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(green_fig(fig2), use_container_width=True)

    st.subheader("📈 Food Loss % Over Time")
    year_loss = df_model.groupby("Year")["Food loss percentage"].mean().reset_index()
    fig3 = px.area(year_loss, x="Year", y="Food loss percentage",
                   color_discrete_sequence=["#4cdf6c"])
    fig3.update_traces(fillcolor="rgba(76,223,108,0.15)", line_width=2.5)
    fig3.update_layout(yaxis_title="Avg Food Loss %")
    st.plotly_chart(green_fig(fig3), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – EXPLORE DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page_key == "Explore Data":
    st.subheader("🔍 Dataset Explorer")
    col1, col2 = st.columns(2)
    region_filter = col1.multiselect("Filter by Sub-region",
        options=sorted(df_raw["Sub-region Name"].dropna().unique()), default=[])
    item_filter = col2.multiselect("Filter by Pesticide Type",
        options=sorted(df_raw["Item"].dropna().unique()), default=[])

    df_view = df_raw.copy()
    if region_filter: df_view = df_view[df_view["Sub-region Name"].isin(region_filter)]
    if item_filter:   df_view = df_view[df_view["Item"].isin(item_filter)]

    st.info(f"📦 Showing **{len(df_view):,}** records")
    st.dataframe(df_view[["Area","Item","Year","value","Sub-region Name",
                           "Food loss percentage","Efficiency Group"]].head(200),
                 use_container_width=True, height=300)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🧪 Pesticide Value Distribution")
        fig = px.histogram(df_view, x="value", nbins=40,
                           color_discrete_sequence=["#4cdf6c"],
                           labels={"value":"Pesticide Quantity (tonnes)"})
        fig.update_traces(opacity=0.85)
        st.plotly_chart(green_fig(fig), use_container_width=True)

    with col_b:
        st.subheader("📦 Food Loss % Distribution")
        fig = px.histogram(df_view, x="Food loss percentage", nbins=40,
                           color_discrete_sequence=["#3ecf5a"],
                           labels={"Food loss percentage":"Loss %"})
        fig.update_traces(opacity=0.85)
        st.plotly_chart(green_fig(fig), use_container_width=True)

    st.subheader("🔗 Pesticide Value vs Food Loss %")
    scatter_df = df_view[df_view["Element"]=="Agricultural Use"].sample(min(1500, len(df_view)))
    fig = px.scatter(scatter_df, x="value", y="Food loss percentage",
                     color="Efficiency Group", color_discrete_sequence=GREEN_PALETTE,
                     hover_data=["Area","Item","Sub-region Name","Year"], opacity=0.7,
                     labels={"value":"Pesticide Qty (tonnes)","Food loss percentage":"Loss %"})
    st.plotly_chart(green_fig(fig), use_container_width=True)

    st.subheader("📊 Top 10 Pesticide Types by Usage")
    top_items = (df_view.groupby("Item")["value"].sum()
                 .sort_values(ascending=False).head(10).reset_index())
    fig = px.bar(top_items, x="value", y="Item", orientation="h", color="value",
                 color_continuous_scale=[[0,"#1a6b2a"],[1,"#4cdf6c"]],
                 labels={"value":"Total Usage (tonnes)","Item":""})
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(green_fig(fig), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page_key == "Model Insights":
    st.subheader("📐 Baseline Model Performance  (Default Params)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",         f"{default_accuracy*100:.2f}%")
    c2.metric("Precision (High)", f"{default_report['High Loss']['precision']*100:.1f}%")
    c3.metric("Recall (High)",    f"{default_report['High Loss']['recall']*100:.1f}%")
    c4.metric("F1-Score (High)",  f"{default_report['High Loss']['f1-score']*100:.1f}%")

    st.markdown("---")
    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.subheader("🎯 Confusion Matrix")
        fig = px.imshow(default_cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Low Loss","High Loss"], y=["Low Loss","High Loss"],
                        text_auto=True,
                        color_continuous_scale=[[0,"#0d1f0f"],[0.3,"#1a6b2a"],
                                                [0.7,"#3ecf5a"],[1,"#4cdf6c"]])
        fig.update_traces(textfont=dict(size=20, color="white"))
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(green_fig(fig), use_container_width=True)

    with col_fi:
        st.subheader("📊 Feature Importance")
        feature_names = ["Pesticide Value","Year","Pesticide Type","Sub-region"]
        coefs = np.abs(default_model.coef_[0])
        fi_df = pd.DataFrame({"Feature":feature_names,"Importance":coefs}).sort_values("Importance")
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance",
                     color_continuous_scale=[[0,"#1a6b2a"],[1,"#4cdf6c"]],
                     labels={"Importance":"|Coefficient|","Feature":""})
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(green_fig(fig), use_container_width=True)

    st.subheader("📋 Full Classification Report")
    report_df = pd.DataFrame(default_report).transpose()
    report_df = report_df[report_df.index.isin(["Low Loss","High Loss","macro avg","weighted avg"])]
    report_df = report_df[["precision","recall","f1-score","support"]].round(3)
    st.dataframe(report_df.style.background_gradient(
        cmap="Greens", subset=["precision","recall","f1-score"]), use_container_width=True)

    st.markdown("---")
    st.subheader("📉 Class Balance")
    balance = df_model["Loss_Class"].value_counts().reset_index()
    balance.columns = ["Class","Count"]
    balance["Class"] = balance["Class"].map({0:"Low Loss (0)",1:"High Loss (1)"})
    fig = px.bar(balance, x="Class", y="Count", color="Class",
                 color_discrete_sequence=["#4cdf6c","#2d8c3e"], text="Count")
    fig.update_traces(textfont=dict(color="white"), textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="Sample Count")
    st.plotly_chart(green_fig(fig), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════
elif page_key == "Hyperparameter Tuning":

    st.subheader("⚙️ Hyperparameter Tuning")
    st.markdown("<p style='color:#88d68a'>Two modes: <b>Manual</b> — pick params and see results instantly. <b>Auto GridSearchCV</b> — exhaustive search finds the globally best combination.</p>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎛️ Manual Tuning", "🔬 Auto GridSearchCV"])

    # ── TAB 1: MANUAL ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Adjust Hyperparameters")
        st.markdown("<p style='color:#88d68a;font-size:0.9rem'>Change any parameter and the model retrains instantly. Compare against the baseline (C=1, l2, lbfgs).</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class='tune-card'>
            <b style='color:#4cdf6c'>C — Regularization Strength</b><br>
            <span style='color:#88d68a;font-size:0.82rem'>Lower = stronger regularization (prevents overfitting).<br>Higher = weaker regularization (fits more aggressively).</span>
            </div>""", unsafe_allow_html=True)
            C_val = st.select_slider("C value", options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)

        with col2:
            st.markdown("""
            <div class='tune-card'>
            <b style='color:#4cdf6c'>Penalty — Regularization Type</b><br>
            <span style='color:#88d68a;font-size:0.82rem'>L2 (Ridge): shrinks all coefficients.<br>L1 (Lasso): drives some to zero (feature selection).</span>
            </div>""", unsafe_allow_html=True)
            penalty_val = st.selectbox("Penalty", ["l2", "l1"])

        with col3:
            st.markdown("""
            <div class='tune-card'>
            <b style='color:#4cdf6c'>Solver — Optimization Algorithm</b><br>
            <span style='color:#88d68a;font-size:0.82rem'>lbfgs: fast, l2 only.<br>liblinear: small datasets, l1/l2.<br>saga: large datasets, all penalties.</span>
            </div>""", unsafe_allow_html=True)
            solver_options = {"l2": ["lbfgs","saga","liblinear"], "l1": ["liblinear","saga"]}
            solver_val = st.selectbox("Solver", solver_options[penalty_val])

        max_iter_val = st.slider("Max Iterations", 100, 2000, 1000, step=100)

        st.markdown("---")

        # Train with chosen params
        try:
            tuned_model, tuned_scaler, tuned_acc, tuned_cm, tuned_report, tuned_cv = \
                train_model(C=C_val, penalty=penalty_val, solver=solver_val, max_iter=max_iter_val)

            # ── Comparison metrics ─────────────────────────────────────────
            delta_acc = tuned_acc - default_accuracy
            delta_f1  = tuned_report["High Loss"]["f1-score"] - default_report["High Loss"]["f1-score"]
            delta_pre = tuned_report["High Loss"]["precision"] - default_report["High Loss"]["precision"]
            delta_rec = tuned_report["High Loss"]["recall"]    - default_report["High Loss"]["recall"]

            def fmt_delta(d):
                sign = "▲" if d >= 0 else "▼"
                color = "#4cdf6c" if d >= 0 else "#e74c3c"
                return f"<span style='color:{color};font-weight:700'>{sign} {abs(d)*100:.2f}%</span>"

            st.markdown("### 📊 Results vs Baseline")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",         f"{tuned_acc*100:.2f}%",
                      f"{'↑' if delta_acc>=0 else '↓'} {abs(delta_acc)*100:.2f}% vs baseline")
            c2.metric("Precision (High)", f"{tuned_report['High Loss']['precision']*100:.1f}%",
                      f"{'↑' if delta_pre>=0 else '↓'} {abs(delta_pre)*100:.1f}%")
            c3.metric("Recall (High)",    f"{tuned_report['High Loss']['recall']*100:.1f}%",
                      f"{'↑' if delta_rec>=0 else '↓'} {abs(delta_rec)*100:.1f}%")
            c4.metric("F1-Score (High)",  f"{tuned_report['High Loss']['f1-score']*100:.1f}%",
                      f"{'↑' if delta_f1>=0 else '↓'} {abs(delta_f1)*100:.1f}%")

            # ── CV scores ─────────────────────────────────────────────────
            st.markdown("### 🔁 5-Fold Cross-Validation Scores")
            cv_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(5)],
                "Accuracy": tuned_cv
            })
            fig = px.bar(cv_df, x="Fold", y="Accuracy",
                         color="Accuracy",
                         color_continuous_scale=[[0,"#1a6b2a"],[0.5,"#3ecf5a"],[1,"#4cdf6c"]],
                         text=[f"{v*100:.1f}%" for v in tuned_cv],
                         range_y=[0, 1])
            fig.update_traces(textfont_color="white", textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(yaxis_tickformat=".0%", yaxis_title="CV Accuracy")
            # Add mean line
            fig.add_hline(y=tuned_cv.mean(), line_dash="dash", line_color="#4cdf6c",
                          annotation_text=f"Mean: {tuned_cv.mean()*100:.2f}%",
                          annotation_font_color="#4cdf6c")
            st.plotly_chart(green_fig(fig), use_container_width=True)

            st.markdown(f"""
            <div class='green-card'>
              <b style='color:#4cdf6c'>Cross-Validation Summary</b><br><br>
              📊 Mean CV Accuracy: <b style='color:#4cdf6c'>{tuned_cv.mean()*100:.2f}%</b> &nbsp;|&nbsp;
              Std Dev: <b style='color:#88d68a'>±{tuned_cv.std()*100:.2f}%</b><br>
              <span style='font-size:0.85rem;color:#88d68a'>Low std dev = stable model. High std dev = model is sensitive to data split.</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Side by side confusion matrices ───────────────────────────
            st.markdown("### 🎯 Confusion Matrix Comparison")
            col_base, col_tuned = st.columns(2)

            with col_base:
                st.markdown("<p style='color:#88d68a;text-align:center'>Baseline (C=1, l2, lbfgs)</p>", unsafe_allow_html=True)
                fig = px.imshow(default_cm,
                                labels=dict(x="Predicted",y="Actual"),
                                x=["Low Loss","High Loss"], y=["Low Loss","High Loss"],
                                text_auto=True,
                                color_continuous_scale=[[0,"#0d1f0f"],[0.3,"#1a3d1c"],[1,"#3ecf5a"]])
                fig.update_traces(textfont=dict(size=18, color="white"))
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(green_fig(fig), use_container_width=True)

            with col_tuned:
                st.markdown(f"<p style='color:#4cdf6c;text-align:center;font-weight:700'>Tuned (C={C_val}, {penalty_val}, {solver_val})</p>", unsafe_allow_html=True)
                fig = px.imshow(tuned_cm,
                                labels=dict(x="Predicted",y="Actual"),
                                x=["Low Loss","High Loss"], y=["Low Loss","High Loss"],
                                text_auto=True,
                                color_continuous_scale=[[0,"#0d1f0f"],[0.3,"#1a6b2a"],[1,"#4cdf6c"]])
                fig.update_traces(textfont=dict(size=18, color="white"))
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(green_fig(fig), use_container_width=True)

            # ── Accuracy bar comparison ────────────────────────────────────
            st.markdown("### 📈 Accuracy Comparison")
            comp_df = pd.DataFrame({
                "Model":    ["Baseline\n(C=1, l2, lbfgs)", f"Tuned\n(C={C_val}, {penalty_val}, {solver_val})"],
                "Accuracy": [default_accuracy, tuned_acc],
                "Type":     ["Baseline","Tuned"]
            })
            fig = px.bar(comp_df, x="Model", y="Accuracy",
                         color="Type",
                         color_discrete_map={"Baseline":"#2d8c3e","Tuned":"#4cdf6c"},
                         text=[f"{v*100:.2f}%" for v in comp_df["Accuracy"]],
                         range_y=[0, 1])
            fig.update_traces(textfont_color="white", textposition="outside")
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
            st.plotly_chart(green_fig(fig), use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ This solver/penalty combination is not compatible: {e}")
            st.info("Try: **l2 + lbfgs**, **l1 + liblinear**, or **l1/l2 + saga**")

    # ── TAB 2: AUTO GRIDSEARCHCV ──────────────────────────────────────────────
    with tab2:
        st.markdown("### 🔬 Automatic GridSearchCV")
        st.markdown("""
        <div class='tune-card'>
        <b style='color:#4cdf6c'>What is GridSearchCV?</b><br>
        <span style='color:#88d68a;font-size:0.88rem'>
        It systematically tries <b>every combination</b> of hyperparameters in a defined grid,
        evaluates each with <b>5-fold cross-validation</b>, and returns the combination
        that achieves the highest mean CV accuracy. This removes guesswork entirely.
        </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Parameter Grid being searched:**")
        grid_info = pd.DataFrame([
            {"Parameter": "C",       "Values Tried": "0.001, 0.01, 0.1, 1, 10, 100"},
            {"Parameter": "penalty", "Values Tried": "l1, l2"},
            {"Parameter": "solver",  "Values Tried": "lbfgs, liblinear, saga (matched to penalty)"},
            {"Parameter": "max_iter","Values Tried": "500, 1000"},
        ])
        st.dataframe(grid_info, use_container_width=True, hide_index=True)

        st.markdown(f"<p style='color:#88d68a;font-size:0.85rem'>Total combinations: ~40 × 5 folds = ~200 model fits</p>", unsafe_allow_html=True)

        if st.button("🚀 Run GridSearchCV  (may take ~30 seconds)", use_container_width=True):
            with st.spinner("🔍 Searching all hyperparameter combinations... please wait"):
                best_params, best_cv_score, gs_accuracy, gs_cm, gs_report, results_df, best_model, gs_scaler = run_grid_search()

            st.success("✅ GridSearchCV complete!")

            # ── Best params card ──────────────────────────────────────────
            st.markdown(f"""
            <div class='green-card'>
              <b style='color:#4cdf6c;font-size:1.1rem'>🏆 Best Parameters Found</b><br><br>
              <span class='best-badge'>C = {best_params['C']}</span>&nbsp;
              <span class='best-badge'>penalty = {best_params['penalty']}</span>&nbsp;
              <span class='best-badge'>solver = {best_params['solver']}</span>&nbsp;
              <span class='best-badge'>max_iter = {best_params['max_iter']}</span><br><br>
              📊 Best CV Accuracy: <b style='color:#4cdf6c'>{best_cv_score*100:.2f}%</b>
            </div>
            """, unsafe_allow_html=True)

            # ── Metrics comparison ─────────────────────────────────────────
            st.markdown("### 📊 Best Model vs Baseline")
            delta_acc = gs_accuracy - default_accuracy
            delta_f1  = gs_report["High Loss"]["f1-score"] - default_report["High Loss"]["f1-score"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best Accuracy",    f"{gs_accuracy*100:.2f}%",
                      f"{'↑' if delta_acc>=0 else '↓'} {abs(delta_acc)*100:.2f}% vs baseline")
            c2.metric("Precision (High)", f"{gs_report['High Loss']['precision']*100:.1f}%")
            c3.metric("Recall (High)",    f"{gs_report['High Loss']['recall']*100:.1f}%")
            c4.metric("F1-Score (High)",  f"{gs_report['High Loss']['f1-score']*100:.1f}%",
                      f"{'↑' if delta_f1>=0 else '↓'} {abs(delta_f1)*100:.1f}% vs baseline")

            # ── Top 15 results table ───────────────────────────────────────
            st.markdown("### 📋 Top 15 Parameter Combinations")
            top_results = results_df[["param_C","param_penalty","param_solver",
                                      "param_max_iter","mean_test_score","std_test_score","rank_test_score"]]
            top_results = top_results.rename(columns={
                "param_C":"C", "param_penalty":"Penalty",
                "param_solver":"Solver","param_max_iter":"Max Iter",
                "mean_test_score":"Mean CV Acc","std_test_score":"Std Dev",
                "rank_test_score":"Rank"
            }).sort_values("Rank").head(15).round(4)
            st.dataframe(top_results.style.background_gradient(
                cmap="Greens", subset=["Mean CV Acc"]), use_container_width=True)

            # ── C vs Accuracy plot ─────────────────────────────────────────
            st.markdown("### 📈 C Value vs CV Accuracy (by Penalty)")
            plot_df = results_df[["param_C","param_penalty","mean_test_score"]].copy()
            plot_df.columns = ["C","Penalty","CV Accuracy"]
            plot_df["C"] = pd.to_numeric(plot_df["C"])
            fig = px.line(plot_df.sort_values("C"), x="C", y="CV Accuracy",
                          color="Penalty", markers=True,
                          color_discrete_map={"l1":"#4cdf6c","l2":"#88d68a"},
                          log_x=True,
                          labels={"CV Accuracy":"Mean CV Accuracy","C":"C (log scale)"})
            fig.update_traces(line_width=2.5, marker_size=8)
            st.plotly_chart(green_fig(fig), use_container_width=True)

            # ── Best model confusion matrix ────────────────────────────────
            col_base, col_best = st.columns(2)
            with col_base:
                st.markdown("<p style='color:#88d68a;text-align:center'>Baseline</p>", unsafe_allow_html=True)
                fig = px.imshow(default_cm, text_auto=True,
                                x=["Low Loss","High Loss"], y=["Low Loss","High Loss"],
                                color_continuous_scale=[[0,"#0d1f0f"],[1,"#3ecf5a"]])
                fig.update_traces(textfont=dict(size=18, color="white"))
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(green_fig(fig), use_container_width=True)

            with col_best:
                st.markdown("<p style='color:#4cdf6c;text-align:center;font-weight:700'>Best (GridSearchCV)</p>", unsafe_allow_html=True)
                fig = px.imshow(gs_cm, text_auto=True,
                                x=["Low Loss","High Loss"], y=["Low Loss","High Loss"],
                                color_continuous_scale=[[0,"#0d1f0f"],[1,"#4cdf6c"]])
                fig.update_traces(textfont=dict(size=18, color="white"))
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(green_fig(fig), use_container_width=True)

            # Store best params in session for Predict page
            st.session_state["best_params"]  = best_params
            st.session_state["best_model"]   = best_model
            st.session_state["best_scaler"]  = gs_scaler

        else:
            st.info("👆 Click **Run GridSearchCV** to start the automated search.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page_key == "Predict":
    st.subheader("🔮 Predict Food Loss Risk")

    # Choose which model to use
    use_best = False
    if "best_model" in st.session_state:
        bp = st.session_state["best_params"]
        choice = st.radio(
            "Which model to use for prediction?",
            [f"✅ Best Tuned Model  (C={bp['C']}, {bp['penalty']}, {bp['solver']})",
             "⚙️ Baseline Model  (C=1, l2, lbfgs)"],
            horizontal=True
        )
        use_best = choice.startswith("✅")
    else:
        st.info("💡 Run **GridSearchCV** on the Hyperparameter Tuning page to unlock the best model for prediction.")

    active_model  = st.session_state["best_model"]  if use_best else default_model
    active_scaler = st.session_state["best_scaler"] if use_best else default_scaler

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌿 Pesticide Info")
        pest_value = st.slider("Pesticide Quantity (tonnes)", 0.0, 5000.0, 150.0, step=10.0)
        pest_type  = st.selectbox("Pesticide Type", sorted(le_item.classes_))

    with col2:
        st.markdown("#### 🗺️ Location & Time")
        year   = st.slider("Year", 2000, 2025, 2020)
        region = st.selectbox("Sub-region", sorted(le_region.classes_))

    st.markdown("---")

    if st.button("🚀 Run Prediction", use_container_width=True):
        item_enc   = le_item.transform([pest_type])[0]
        region_enc = le_region.transform([region])[0]

        sample = pd.DataFrame({
            "value":      [pest_value],
            "Year":       [year],
            "Item_enc":   [item_enc],
            "Region_enc": [region_enc]
        })
        sample_sc  = active_scaler.transform(sample)
        prediction = active_model.predict(sample_sc)[0]
        proba      = active_model.predict_proba(sample_sc)[0]

        if prediction == 1:
            st.markdown("""
            <div class='predict-high'>
              <div class='result-label' style='color:#e74c3c;'>⚠️ HIGH LOSS</div>
              <div class='result-sub'>This region is likely to experience <b>above-median</b> post-harvest food loss.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='predict-low'>
              <div class='result-label' style='color:#4cdf6c;'>✅ LOW LOSS</div>
              <div class='result-sub'>This region is likely to experience <b>below-median</b> post-harvest food loss.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Category":    ["Low Loss","High Loss"],
            "Probability": [proba[0], proba[1]]
        })
        fig = px.bar(prob_df, x="Category", y="Probability",
                     color="Category",
                     color_discrete_map={"Low Loss":"#4cdf6c","High Loss":"#e74c3c"},
                     text=[f"{p*100:.1f}%" for p in prob_df["Probability"]],
                     range_y=[0, 1])
        fig.update_traces(textfont_color="white", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(green_fig(fig), use_container_width=True)

        model_label = f"Best Tuned (C={st.session_state['best_params']['C']})" \
                      if use_best else "Baseline (C=1, l2, lbfgs)"
        st.markdown(f"""
        <div class='green-card'>
          <b style='color:#4cdf6c'>Input Summary</b><br><br>
          🌿 <b>Pesticide:</b> {pest_type} — {pest_value} tonnes<br>
          🗺️ <b>Sub-region:</b> {region}<br>
          📅 <b>Year:</b> {year}<br>
          🤖 <b>Model used:</b> {model_label}<br>
          📊 <b>Median threshold:</b> {median_loss:.2f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👆 Configure the inputs above and click **Run Prediction** to get a result.")
