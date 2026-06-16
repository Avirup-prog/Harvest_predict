import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
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
  /* ── Base background ── */
  .stApp { background-color: #0d1f0f; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a2210 0%, #122b14 60%, #0d1f0f 100%);
    border-right: 1px solid #1e4d22;
  }
  [data-testid="stSidebar"] * { color: #a8e6a3 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stNumberInput label { color: #6fcf70 !important; font-weight: 600; }

  /* ── Main text ── */
  body, .stMarkdown, p, li, span { color: #c8f0c5; }

  /* ── Headings ── */
  h1 { color: #4cdf6c !important; font-size: 2.2rem !important; letter-spacing: -0.5px; }
  h2 { color: #3ecf5a !important; border-bottom: 1px solid #1e4d22; padding-bottom: 6px; }
  h3 { color: #5de078 !important; }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #122b14 0%, #1a3d1c 100%);
    border: 1px solid #2a6b2e;
    border-radius: 12px;
    padding: 16px 20px;
  }
  [data-testid="metric-container"] label { color: #6fcf70 !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 1px; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #4cdf6c !important; font-size: 2rem !important; }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #88d68a !important; }

  /* ── Buttons ── */
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

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] { background-color: #122b14; border-radius: 10px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { color: #6fcf70 !important; border-radius: 8px; font-weight: 600; }
  .stTabs [aria-selected="true"] { background-color: #1e5c25 !important; color: #4cdf6c !important; }

  /* ── Inputs ── */
  .stSelectbox > div > div,
  .stMultiSelect > div > div,
  .stNumberInput > div > div > input {
    background-color: #122b14 !important;
    border: 1px solid #2a6b2e !important;
    border-radius: 8px !important;
    color: #c8f0c5 !important;
  }

  /* ── Sliders ── */
  [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background: #4cdf6c !important; }
  [data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] { color: #4cdf6c !important; }

  /* ── Dataframes ── */
  .stDataFrame { border: 1px solid #2a6b2e; border-radius: 8px; overflow: hidden; }

  /* ── Info / success boxes ── */
  .stAlert { border-radius: 10px !important; }
  .stSuccess { background-color: #0f3312 !important; border-color: #2d8c3e !important; }
  .stInfo    { background-color: #0a2210 !important; border-color: #1e5c25 !important; }

  /* ── Hero banner ── */
  .hero-banner {
    background: linear-gradient(135deg, #0f3312 0%, #1a4d1e 50%, #122b14 100%);
    border: 1px solid #2a6b2e;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute; top: 0; right: 0; bottom: 0;
    width: 40%; opacity: 0.07;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%234cdf6c' fill-opacity='1'%3E%3Ccircle cx='30' cy='30' r='4'/%3E%3C/g%3E%3C/svg%3E");
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

  /* ── Card container ── */
  .green-card {
    background: linear-gradient(135deg, #0f2e12 0%, #172e18 100%);
    border: 1px solid #2a6b2e;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }

  /* ── Predict result ── */
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

  /* ── Divider ── */
  hr { border-color: #1e4d22 !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d1f0f; }
  ::-webkit-scrollbar-thumb { background: #2a6b2e; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme helper ────────────────────────────────────────────────────────
GREEN_PALETTE = ["#4cdf6c", "#2d8c3e", "#88d68a", "#1a6b2a", "#a8e6a3",
                 "#5de078", "#3ecf5a", "#6fcf70", "#0f6623", "#c8f0c5"]

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

# ── Data loading & model training (cached) ────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("/mnt/user-data/uploads/Dataset.csv")

    df_model = df[df["Element"] == "Agricultural Use"].copy()
    df_model.dropna(subset=["value", "Food loss percentage", "Item", "Sub-region Name"],
                    inplace=True)
    df_model.drop_duplicates(inplace=True)

    median_loss = df_model["Food loss percentage"].median()
    df_model["Loss_Class"] = (df_model["Food loss percentage"] >= median_loss).astype(int)

    le_item   = LabelEncoder()
    le_region = LabelEncoder()
    df_model["Item_enc"]   = le_item.fit_transform(df_model["Item"])
    df_model["Region_enc"] = le_region.fit_transform(df_model["Sub-region Name"])

    X = df_model[["value", "Year", "Item_enc", "Region_enc"]]
    y = df_model["Loss_Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)

    y_pred   = model.predict(X_test_sc)
    accuracy = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                     target_names=["Low Loss", "High Loss"],
                                     output_dict=True)

    return df, df_model, model, scaler, le_item, le_region, accuracy, cm, report, median_loss

df_raw, df_model, model, scaler, le_item, le_region, accuracy, cm, report, median_loss = load_and_train()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Navigation")
    page = st.radio("", ["🏠 Overview", "📊 Explore Data", "🤖 Model Insights", "🔮 Predict"])
    st.markdown("---")
    st.markdown("### 📋 Dataset Info")
    st.metric("Total Records", f"{len(df_raw):,}")
    st.metric("Sub-regions", df_raw["Sub-region Name"].nunique())
    st.metric("Pesticide Types", df_raw["Item"].nunique())
    st.metric("Year Range", f"{df_raw['Year'].min()}–{df_raw['Year'].max()}")
    st.markdown("---")
    st.markdown("""
    <div style='color:#6fcf70;font-size:0.78rem;line-height:1.7'>
    <b>SDG Goal 2 – Zero Hunger</b><br>
    Predicts post-harvest food loss based on pesticide usage patterns using Logistic Regression.
    </div>
    """, unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
page_key = page.split(" ", 1)[1]  # strip emoji

st.markdown(f"""
<div class='hero-banner'>
  <span class='hero-tag'>SDG 2</span>
  <span class='hero-tag'>Zero Hunger</span>
  <span class='hero-tag'>Logistic Regression</span>
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
    c1.metric("Model Accuracy", f"{accuracy*100:.2f}%", "Logistic Regression")
    c2.metric("Median Loss Threshold", f"{median_loss:.2f}%", "Binary split")
    c3.metric("High-Loss Regions", str(df_model["Loss_Class"].sum()), "class 1")
    c4.metric("Low-Loss Regions",  str((df_model["Loss_Class"]==0).sum()), "class 0")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("🗺️ Average Food Loss by Sub-region")
        region_loss = (df_model.groupby("Sub-region Name")["Food loss percentage"]
                       .mean().sort_values(ascending=True).reset_index())
        fig = px.bar(region_loss, x="Food loss percentage", y="Sub-region Name",
                     orientation="h",
                     color="Food loss percentage",
                     color_continuous_scale=[[0,"#1a6b2a"],[0.5,"#3ecf5a"],[1,"#4cdf6c"]],
                     labels={"Sub-region Name": "", "Food loss percentage": "Avg Loss %"})
        fig.update_coloraxes(showscale=False)
        fig = green_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("🥧 Efficiency Group Distribution")
        eff = df_raw["Efficiency Group"].value_counts().reset_index()
        eff.columns = ["Group", "Count"]
        fig2 = px.pie(eff, names="Group", values="Count",
                      color_discrete_sequence=GREEN_PALETTE,
                      hole=0.45)
        fig2.update_traces(textinfo="percent+label", textfont_color="#0d1f0f",
                           textfont_size=11)
        fig2 = green_fig(fig2)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📈 Food Loss % Over Time")
    year_loss = (df_model.groupby("Year")["Food loss percentage"]
                 .mean().reset_index())
    fig3 = px.area(year_loss, x="Year", y="Food loss percentage",
                   color_discrete_sequence=["#4cdf6c"])
    fig3.update_traces(fillcolor="rgba(76,223,108,0.15)", line_width=2.5)
    fig3.update_layout(yaxis_title="Avg Food Loss %")
    fig3 = green_fig(fig3)
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – EXPLORE DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page_key == "Explore Data":

    st.subheader("🔍 Dataset Explorer")
    col1, col2 = st.columns(2)
    region_filter = col1.multiselect(
        "Filter by Sub-region",
        options=sorted(df_raw["Sub-region Name"].dropna().unique()),
        default=[]
    )
    item_filter = col2.multiselect(
        "Filter by Pesticide Type",
        options=sorted(df_raw["Item"].dropna().unique()),
        default=[]
    )

    df_view = df_raw.copy()
    if region_filter:
        df_view = df_view[df_view["Sub-region Name"].isin(region_filter)]
    if item_filter:
        df_view = df_view[df_view["Item"].isin(item_filter)]

    st.info(f"📦 Showing **{len(df_view):,}** records")
    st.dataframe(
        df_view[["Area","Item","Year","value","Sub-region Name",
                 "Food loss percentage","Efficiency Group"]].head(200),
        use_container_width=True, height=300
    )

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🧪 Pesticide Value Distribution")
        fig = px.histogram(df_view, x="value", nbins=40,
                           color_discrete_sequence=["#4cdf6c"],
                           labels={"value": "Pesticide Quantity (tonnes)"})
        fig.update_traces(opacity=0.85)
        fig = green_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("📦 Food Loss % Distribution")
        fig = px.histogram(df_view, x="Food loss percentage", nbins=40,
                           color_discrete_sequence=["#3ecf5a"],
                           labels={"Food loss percentage": "Loss %"})
        fig.update_traces(opacity=0.85)
        fig = green_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔗 Pesticide Value vs Food Loss %")
    scatter_df = df_view[df_view["Element"] == "Agricultural Use"].sample(min(1500, len(df_view)))
    fig = px.scatter(scatter_df, x="value", y="Food loss percentage",
                     color="Efficiency Group",
                     color_discrete_sequence=GREEN_PALETTE,
                     hover_data=["Area","Item","Sub-region Name","Year"],
                     opacity=0.7,
                     labels={"value": "Pesticide Qty (tonnes)",
                             "Food loss percentage": "Loss %"})
    fig = green_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Top 10 Pesticide Types by Usage")
    top_items = (df_view.groupby("Item")["value"].sum()
                 .sort_values(ascending=False).head(10).reset_index())
    fig = px.bar(top_items, x="value", y="Item", orientation="h",
                 color="value",
                 color_continuous_scale=[[0,"#1a6b2a"],[1,"#4cdf6c"]],
                 labels={"value": "Total Usage (tonnes)", "Item": ""})
    fig.update_coloraxes(showscale=False)
    fig = green_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page_key == "Model Insights":

    st.subheader("📐 Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{accuracy*100:.2f}%")
    c2.metric("Precision (High)", f"{report['High Loss']['precision']*100:.1f}%")
    c3.metric("Recall (High)",    f"{report['High Loss']['recall']*100:.1f}%")
    c4.metric("F1-Score (High)",  f"{report['High Loss']['f1-score']*100:.1f}%")

    st.markdown("---")
    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.subheader("🎯 Confusion Matrix")
        fig = px.imshow(cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Low Loss","High Loss"],
                        y=["Low Loss","High Loss"],
                        text_auto=True,
                        color_continuous_scale=[[0,"#0d1f0f"],[0.3,"#1a6b2a"],
                                                 [0.7,"#3ecf5a"],[1,"#4cdf6c"]])
        fig.update_traces(textfont=dict(size=20, color="white"))
        fig = green_fig(fig)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_fi:
        st.subheader("📊 Feature Importance")
        feature_names = ["Pesticide Value", "Year", "Pesticide Type", "Sub-region"]
        coefs = np.abs(model.coef_[0])
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": coefs})
        fi_df = fi_df.sort_values("Importance", ascending=True)

        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance",
                     color_continuous_scale=[[0,"#1a6b2a"],[1,"#4cdf6c"]],
                     labels={"Importance": "|Coefficient|", "Feature": ""})
        fig.update_coloraxes(showscale=False)
        fig = green_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Full Classification Report")
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df[report_df.index.isin(["Low Loss", "High Loss",
                                                  "macro avg", "weighted avg"])]
    report_df = report_df[["precision", "recall", "f1-score", "support"]].round(3)
    st.dataframe(report_df.style.background_gradient(
        cmap="Greens", subset=["precision","recall","f1-score"]),
        use_container_width=True)

    st.markdown("---")
    st.subheader("📉 Class Balance in Training Data")
    balance = df_model["Loss_Class"].value_counts().reset_index()
    balance.columns = ["Class", "Count"]
    balance["Class"] = balance["Class"].map({0: "Low Loss (0)", 1: "High Loss (1)"})
    fig = px.bar(balance, x="Class", y="Count",
                 color="Class",
                 color_discrete_sequence=["#4cdf6c","#2d8c3e"],
                 text="Count")
    fig.update_traces(textfont=dict(color="white"), textposition="outside")
    fig = green_fig(fig)
    fig.update_layout(showlegend=False, yaxis_title="Sample Count")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page_key == "Predict":

    st.subheader("🔮 Predict Food Loss Risk")
    st.markdown("<p style='color:#88d68a'>Fill in the fields below to predict whether a region will experience <b>High</b> or <b>Low</b> post-harvest food loss.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌿 Pesticide Info")
        pest_value = st.slider("Pesticide Quantity (tonnes)", 0.0, 5000.0, 150.0, step=10.0)
        pest_type  = st.selectbox("Pesticide Type", sorted(le_item.classes_))

    with col2:
        st.markdown("#### 🗺️ Location & Time")
        year       = st.slider("Year", 2000, 2025, 2020)
        region     = st.selectbox("Sub-region", sorted(le_region.classes_))

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
        sample_sc  = scaler.transform(sample)
        prediction = model.predict(sample_sc)[0]
        proba      = model.predict_proba(sample_sc)[0]

        if prediction == 1:
            st.markdown(f"""
            <div class='predict-high'>
              <div class='result-label' style='color:#e74c3c;'>⚠️ HIGH LOSS</div>
              <div class='result-sub'>This region is likely to experience <b>above-median</b> post-harvest food loss.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='predict-low'>
              <div class='result-label' style='color:#4cdf6c;'>✅ LOW LOSS</div>
              <div class='result-sub'>This region is likely to experience <b>below-median</b> post-harvest food loss.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Category":    ["Low Loss", "High Loss"],
            "Probability": [proba[0], proba[1]]
        })
        fig = px.bar(prob_df, x="Category", y="Probability",
                     color="Category",
                     color_discrete_map={"Low Loss": "#4cdf6c", "High Loss": "#e74c3c"},
                     text=[f"{p*100:.1f}%" for p in prob_df["Probability"]],
                     range_y=[0, 1])
        fig.update_traces(textfont_color="white", textposition="outside")
        fig = green_fig(fig)
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class='green-card'>
          <b style='color:#4cdf6c'>Input Summary</b><br><br>
          🌿 <b>Pesticide:</b> {pest_type} — {pest_value} tonnes<br>
          🗺️ <b>Sub-region:</b> {region}<br>
          📅 <b>Year:</b> {year}<br>
          📊 <b>Median threshold:</b> {median_loss:.2f}%
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("👆 Configure the inputs above and click **Run Prediction** to get a result.")
