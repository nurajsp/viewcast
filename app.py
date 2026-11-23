# ============================================================
# STEP 7 - ViewCast LK: Streamlit dashboard
# Run in Terminal (inside .venv):  streamlit run app.py
# Needs (made by the notebook): videos_clean.csv, models/, results/
# ============================================================
import datetime as dt
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from features import make_features
import hybrid_models  # noqa: F401  (needed so joblib can load the hybrid models)

st.set_page_config(page_title="ViewCast LK", page_icon="🎬", layout="wide")


@st.cache_resource
def load_models():
    return joblib.load("models/all_models.joblib"), joblib.load("models/meta.joblib")


@st.cache_data
def load_data():
    df = pd.read_csv("videos_clean.csv", parse_dates=["publish_date"])
    table = pd.read_csv("results/model_comparison.csv")
    try:
        imp = pd.read_csv("results/feature_importance.csv", index_col="feature")
    except FileNotFoundError:
        imp = None
    return df, table, imp


models, meta = load_models()
df, table, imp = load_data()
FEATURES, X_test, y_test = meta["features"], meta["X_test"], meta["y_test"]
model_names = table["Model"].tolist()            # already sorted best -> worst

# ---------------- Sidebar ----------------
st.sidebar.title("🎬 ViewCast LK")
st.sidebar.caption("Predicting YouTube views for Sri Lankan creators before upload")
chosen = st.sidebar.selectbox("Choose a model", model_names, index=0)
st.sidebar.markdown(f"**Best model (CV):** {meta['best_model']}")
row = table.set_index("Model").loc[chosen]
st.sidebar.metric("Test R² (log views)", f"{row['Test R2 (log)']:.3f}")
st.sidebar.metric("CV RMSE (log views)", f"{row['CV RMSE (log) mean']:.3f} ± {row['CV RMSE (log) std']:.3f}")
st.sidebar.metric("Fit time", f"{row['Fit time (s)']:.2f} s")

st.title("ViewCast LK: Ensemble Learning Dashboard")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Dataset", "🏆 Model comparison", "🎯 Actual vs predicted", "🔍 Feature importance", "🔮 Make a prediction"])

# ---------------- Tab 1: dataset overview ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Videos", len(df))
    c2.metric("Total views", f"{df['views'].sum():,.0f}")
    c3.metric("Median views / video", f"{df['views'].median():,.0f}")
    c4.metric("Shorts", f"{df['is_short'].sum()} / {len(df)}")

    left, right = st.columns(2)
    fig = px.scatter(df, x="publish_date", y="views", color=np.where(df["is_short"] == 1, "Short", "Long"),
                     hover_data=["title"], log_y=True, title="Views per video over time",
                     color_discrete_map={"Long": "#4C9BE8", "Short": "#F2A65A"},
                     labels={"publish_date": "Publish date", "views": "Views (log scale)"})
    fig.update_layout(legend_title_text="Format")
    left.plotly_chart(fig, width="stretch")
    fig = px.box(df, x="title_lang", y="views", log_y=True, points="all",
                 category_orders={"title_lang": ["English", "Mixed", "Sinhala"]},
                 title="Views by title language", hover_data=["title"],
                 labels={"title_lang": "Title language", "views": "Views (log scale)"})
    right.plotly_chart(fig, width="stretch")

    st.subheader("Data sample")
    st.dataframe(df[["title", "publish_date", "duration_sec", "views", "likes", "comments",
                     "title_lang", "is_short"]].sort_values("views", ascending=False),
                 width="stretch", height=300)

# ---------------- Tab 2: model comparison ----------------
with tab2:
    st.subheader("All models: same train/test split, same metrics")
    show_cols = ["Model", "Family", "CV RMSE (log) mean", "CV RMSE (log) std", "Test R2 (log)",
                 "Test MAE (log)", "Test RMSE (log)", "Test MAE (views)", "Test MAPE (views) %",
                 "Fit time (s)", "Tuning time (s)"]
    st.dataframe(table[show_cols].style.format(precision=3)
                 .highlight_min(subset=["CV RMSE (log) mean"], color="#c6efce")
                 .highlight_max(subset=["Test R2 (log)"], color="#c6efce"),
                 width="stretch", hide_index=True)
    metric = st.radio("Chart metric", ["CV RMSE (log) mean", "Test R2 (log)", "Test MAE (views)", "Fit time (s)"],
                      horizontal=True)
    ascending = metric != "Test R2 (log)"
    t = table.sort_values(metric, ascending=not ascending)
    fig = px.bar(t, x=metric, y="Model", color="Family", orientation="h",
                 error_x="CV RMSE (log) std" if metric == "CV RMSE (log) mean" else None,
                 title=f"{metric} by model")
    fig.update_layout(height=500, yaxis={"categoryorder": "array", "categoryarray": t["Model"].tolist()})
    st.plotly_chart(fig, width="stretch")

# ---------------- Tab 3: actual vs predicted ----------------
with tab3:
    st.subheader(f"Actual vs predicted views on the test set: {chosen}")
    actual = np.expm1(y_test.values)
    pred = np.expm1(meta["test_preds"][chosen])
    titles = df.loc[X_test.index, "title"].values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=pred, mode="markers", marker=dict(size=10, color="#C44E52"),
                             text=titles, hovertemplate="%{text}<br>Actual %{x:,.0f}<br>Predicted %{y:,.0f}"))
    lo, hi = max(10, min(actual.min(), pred.min()) * 0.8), max(actual.max(), pred.max()) * 1.2
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="black"),
                             name="Perfect prediction"))
    fig.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="Actual views",
                      yaxis_title="Predicted views", showlegend=False, height=550)
    st.plotly_chart(fig, width="stretch")
    st.caption("Points on the dashed line are perfect predictions. Viral videos (far right) are usually under-predicted.")

# ---------------- Tab 4: feature importance ----------------
with tab4:
    m = models[chosen]
    if hasattr(m, "feature_importances_"):
        fi = pd.Series(m.feature_importances_, index=FEATURES)
        fi = fi / fi.sum()
        source = f"Built-in importance of {chosen}"
    elif imp is not None:
        fi = imp["mean_abs_shap"]
        source = f"{chosen} has no built-in importance, so mean |SHAP| of {meta['best_model']} is shown"
    else:
        fi = None
    if fi is not None:
        fi = fi.sort_values().tail(15)
        fig = px.bar(x=fi.values, y=fi.index, orientation="h", title=source,
                     labels={"x": "Importance", "y": "Feature"})
        fig.update_layout(height=550)
        st.plotly_chart(fig, width="stretch")
    st.markdown("""
**How to read this (Sri Lankan context)**
- `kw_uom`: titles about the University of Moratuwa attract a big local student audience.
- `is_short`: Shorts are pushed by the YouTube feed and get many more views on this channel.
- `publish_year` / `video_age_days`: the channel grew strongly from 2025, so newer videos get more views.
- `kw_vlog`: personal vlogs reach a smaller audience than university-guidance content.
""")
    try:
        st.image("figures/12_shap_summary.png", caption="SHAP summary (from the notebook)")
    except Exception:
        pass

# ---------------- Tab 5: make a prediction ----------------
with tab5:
    st.subheader("Plan a new video and estimate its views")
    with st.form("predict"):
        title = st.text_input("Video title", "A/L ලියන අය අනිවාර්යයෙන් බලන්න! 💡 | Uni Life | University of Moratuwa 🎓")
        c1, c2, c3 = st.columns(3)
        minutes = c1.number_input("Duration (minutes)", min_value=0.1, max_value=60.0, value=3.0, step=0.5)
        publish = c2.date_input("Planned publish date", dt.date.today())
        horizon = c3.selectbox("Estimate views after", [30, 90, 180, 365], index=1,
                               format_func=lambda d: f"{d} days")
        submitted = st.form_submit_button("Predict views")
    if submitted:
        feats = make_features(title, minutes * 60, publish, horizon)
        X_new = pd.DataFrame([feats])[FEATURES]
        p_log = float(models[chosen].predict(X_new)[0])
        rmse = float(row["CV RMSE (log) mean"])
        est, low, high = np.expm1(p_log), np.expm1(p_log - rmse), np.expm1(p_log + rmse)
        c1, c2, c3 = st.columns(3)
        c1.metric("Likely low", f"{low:,.0f}")
        c2.metric("Best estimate", f"{est:,.0f}")
        c3.metric("Likely high", f"{high:,.0f}")
        st.caption(f"Model: {chosen}. The range is the estimate ± 1 CV RMSE on the log scale (about 68% of past videos fall inside).")
        with st.expander("Features the model used"):
            st.json({k: (round(v, 3) if isinstance(v, float) else v) for k, v in feats.items()})
        st.warning("Viral hits cannot be predicted from title and timing alone. Treat this as a planning guide, not a promise.")
