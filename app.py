"""
Ames Housing Price Predictor — All-in-One
==========================================
Run:
    pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
    streamlit run app.py

Place AmesHousing_1_.csv in the same folder as this file.
The script trains all models on first run and caches them automatically.
"""

import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
import joblib
from pathlib import Path

from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
DATA_PATH = BASE / "AmesHousing_1_.csv"
MODEL_PKL = BASE / "best_model.pkl"
META_JSON = BASE / "model_meta.json"

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
html, body, [data-testid="stAppViewContainer"] { background: #0f0f1a; }
[data-testid="stSidebar"]  { background: #090912; border-right: 1px solid #1e1e2e; }

/* ── header banner ── */
.hdr {
    background: linear-gradient(120deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.6rem;
}
.hdr h1 { color:#fff; font-size:2rem; margin:0 0 .3rem; }
.hdr p  { color:#8899bb; font-size:.95rem; margin:0; }

/* ── metric card ── */
.kpi {
    background:#141428;
    border:1px solid #1e2a4a;
    border-radius:12px;
    padding:1rem 1.25rem;
    text-align:center;
}
.kpi .lbl { font-size:.7rem; text-transform:uppercase; letter-spacing:1.2px;
             color:#556; margin-bottom:.25rem; }
.kpi .val { font-size:1.7rem; font-weight:700; color:#e94560; }
.kpi .sub { font-size:.72rem; color:#445; margin-top:.15rem; }

/* ── result box ── */
.result-box {
    background:linear-gradient(135deg,#0f3460,#1a1a2e);
    border:1px solid #e9456050;
    border-radius:14px;
    padding:1.5rem 2rem;
    text-align:center;
    margin-top:1rem;
}
.result-box .price { font-size:2.8rem; font-weight:800; color:#e94560; letter-spacing:-1px; }
.result-box .sub   { color:#8899bb; font-size:.85rem; margin-top:.4rem; }

/* ── section label ── */
.slbl {
    font-size:.7rem; font-weight:700; text-transform:uppercase;
    letter-spacing:1.5px; color:#e94560;
    border-bottom:1px solid #e9456025;
    margin:1.4rem 0 .6rem; padding-bottom:.3rem;
}

/* ── buttons ── */
div.stButton > button {
    background: linear-gradient(135deg,#e94560,#c23152) !important;
    color:#fff !important;
    border:none !important;
    border-radius:10px !important;
    font-size:1rem !important;
    font-weight:600 !important;
    padding:.6rem 1.6rem !important;
    width:100% !important;
    box-shadow:0 4px 14px #e9456040 !important;
    transition:transform .15s !important;
}
div.stButton > button:hover { transform:translateY(-2px) !important; }

/* ── sidebar progress label ── */
.best-badge {
    display:inline-block;
    background:#e9456020;
    color:#e94560;
    border:1px solid #e9456050;
    border-radius:6px;
    font-size:.72rem;
    padding:2px 8px;
    margin-left:6px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ① DATA LOADING
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}. Place AmesHousing_1_.csv next to app.py.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Order", "PID"], errors="ignore")
    return df

df_raw = load_data()


# ═══════════════════════════════════════════════════════════════
# ② FEATURE ENGINEERING & PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def build_pipeline_and_train():
    df   = df_raw.copy()
    y    = np.log1p(df["SalePrice"])
    X    = df.drop(columns=["SalePrice"])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    # ── all candidate models ──────────────────────────────────
    candidates = {
        "Ridge":             Ridge(alpha=10),
        "Lasso":             Lasso(alpha=0.001, max_iter=5000),
        "Decision Tree":     DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=15,
                                                   random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                        learning_rate=0.05, random_state=42),
        "Extra Trees":       ExtraTreesRegressor(n_estimators=200, max_depth=15,
                                                 random_state=42, n_jobs=-1),
    }

    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in candidates.items():
        pipe   = Pipeline([("prep", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X, y, cv=kf,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
        results[name] = {"RMSE": float(-scores.mean()), "STD": float(scores.std())}

    # ── best model → retrain on full data ────────────────────
    best_name = min(results, key=lambda k: results[k]["RMSE"])
    best_pipe = Pipeline([("prep", preprocessor), ("model", candidates[best_name])])
    best_pipe.fit(X, y)

    # ── feature importances (for GBR / tree models) ──────────
    feat_imp = None
    try:
        ohe       = best_pipe["prep"].named_transformers_["cat"]["encoder"]
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
        all_names = num_cols + cat_names
        raw_imp   = best_pipe["model"].feature_importances_
        top_idx   = np.argsort(raw_imp)[::-1][:20]
        feat_imp  = [(all_names[i], float(raw_imp[i])) for i in top_idx]
    except Exception:
        pass

    # ── num / cat stats for UI defaults ──────────────────────
    num_stats  = {c: {"min":   float(df[c].min()),
                       "max":   float(df[c].max()),
                       "median":float(df[c].median())} for c in num_cols}
    cat_unique = {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in cat_cols}
    cat_default= {c: str(df[c].mode()[0]) for c in cat_cols}

    meta = {
        "best_model":  best_name,
        "results":     results,
        "num_cols":    num_cols,
        "cat_cols":    cat_cols,
        "num_stats":   num_stats,
        "cat_unique":  cat_unique,
        "cat_default": cat_default,
        "feat_imp":    feat_imp,
    }
    return best_pipe, meta, X, y


with st.spinner("⚙️ Training models (first run only — takes ~30 s) …"):
    best_pipe, meta, X_full, y_full = build_pipeline_and_train()

best_name  = meta["best_model"]
results    = meta["results"]
num_cols   = meta["num_cols"]
cat_cols   = meta["cat_cols"]
num_stats  = meta["num_stats"]
cat_unique = meta["cat_unique"]
cat_default= meta["cat_default"]
feat_imp   = meta["feat_imp"]


# ═══════════════════════════════════════════════════════════════
# ③ SIDEBAR — Model Leaderboard
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Model Leaderboard")
    st.caption("5-fold CV · log RMSE (lower = better)")

    sorted_res = sorted(results.items(), key=lambda x: x[1]["RMSE"])
    best_rmse  = sorted_res[0][1]["RMSE"]
    worst_rmse = sorted_res[-1][1]["RMSE"]

    for rank, (name, r) in enumerate(sorted_res):
        badge = '<span class="best-badge">✓ deployed</span>' if name == best_name else ""
        st.markdown(f"**{rank+1}. {name}** {badge}", unsafe_allow_html=True)

        bar_pct = int((1 - (r["RMSE"] - best_rmse) / (worst_rmse - best_rmse + 1e-9)) * 100)
        bar_col = "#e94560" if name == best_name else "#2a3a5a"
        st.markdown(f"""
        <div style="background:#0d0d1a;border-radius:6px;height:8px;margin:2px 0 6px">
          <div style="width:{bar_pct}%;background:{bar_col};height:8px;border-radius:6px"></div>
        </div>
        <span style="font-size:.75rem;color:#667">RMSE {r['RMSE']:.4f} &nbsp;±{r['STD']:.4f}</span>
        """, unsafe_allow_html=True)
        st.markdown("---")

    st.caption(f"Dataset: Ames, Iowa · 2,930 rows · 80 features")


# ═══════════════════════════════════════════════════════════════
# ④ HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hdr">
  <h1>🏠 Ames House Price Predictor</h1>
  <p>Enter property details → get an AI-powered price estimate.
     Best model: <strong>{best_name}</strong>
     (CV RMSE {results[best_name]['RMSE']:.4f}) trained on 2,930 Iowa homes.</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ⑤ TABS
# ═══════════════════════════════════════════════════════════════
tab_pred, tab_perf, tab_data = st.tabs(
    ["🔮 Predict Price", "📈 Model Performance", "📊 Data Insights"]
)


# ────────────────────────────────────────────────────────────────
# TAB 1 · PREDICT
# ────────────────────────────────────────────────────────────────
with tab_pred:

    user_input = {}

    # ── KEY NUMERIC ──────────────────────────────────────────
    st.markdown('<div class="slbl">Core Property Details</div>', unsafe_allow_html=True)

    priority_num = [
        "Overall Qual","Gr Liv Area","Total Bsmt SF","Garage Area",
        "1st Flr SF","Year Built","Year Remod/Add","Lot Area",
        "TotRms AbvGrd","Full Bath","Bedroom AbvGr",
    ]
    rest_num = [c for c in num_cols if c not in priority_num]

    cols = st.columns(3)
    for i, col in enumerate(priority_num):
        s = num_stats[col]
        with cols[i % 3]:
            if col == "Overall Qual":
                user_input[col] = st.slider(
                    col, min_value=1, max_value=10,
                    value=int(s["median"]), help="1=Very Poor · 10=Very Excellent"
                )
            else:
                user_input[col] = st.number_input(
                    col, min_value=float(max(0, s["min"])),
                    max_value=float(s["max"] * 2),
                    value=float(s["median"]), step=1.0,
                )

    # ── KEY CATEGORICAL ──────────────────────────────────────
    st.markdown('<div class="slbl">Location & Style</div>', unsafe_allow_html=True)

    priority_cat = [
        "Neighborhood","MS Zoning","Exter Qual","Kitchen Qual",
        "Bsmt Qual","Heating QC","Central Air","Garage Type",
        "Foundation","House Style","Sale Condition","Bldg Type",
    ]
    rest_cat = [c for c in cat_cols if c not in priority_cat]

    cols = st.columns(3)
    for i, col in enumerate(priority_cat):
        opts = cat_unique.get(col, ["Missing"])
        dflt = cat_default.get(col, opts[0])
        idx  = opts.index(dflt) if dflt in opts else 0
        with cols[i % 3]:
            user_input[col] = st.selectbox(col, options=opts, index=idx)

    # ── ADVANCED (collapsed) ─────────────────────────────────
    with st.expander("⚙️  Additional numeric features"):
        cols = st.columns(3)
        for i, col in enumerate(rest_num):
            s = num_stats[col]
            with cols[i % 3]:
                user_input[col] = st.number_input(
                    col, min_value=float(max(0, s["min"])),
                    max_value=float(s["max"] * 2),
                    value=float(s["median"]), step=1.0, key=f"rn_{col}"
                )

    with st.expander("🔖  Additional categorical features"):
        cols = st.columns(3)
        for i, col in enumerate(rest_cat):
            opts = cat_unique.get(col, ["Missing"])
            dflt = cat_default.get(col, opts[0])
            idx  = opts.index(dflt) if dflt in opts else 0
            with cols[i % 3]:
                user_input[col] = st.selectbox(col, options=opts, index=idx, key=f"rc_{col}")

    # ── PREDICT BUTTON ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col, res_col = st.columns([1, 2])

    with btn_col:
        predict_clicked = st.button("🔮 Predict Sale Price")

    if predict_clicked:
        # Build input df
        row = {}
        for c in num_cols:
            row[c] = float(user_input.get(c, num_stats[c]["median"]))
        for c in cat_cols:
            row[c] = str(user_input.get(c, cat_default.get(c, "Missing")))

        inp_df    = pd.DataFrame([row])[num_cols + cat_cols]
        log_pred  = best_pipe.predict(inp_df)[0]
        price     = float(np.expm1(log_pred))

        # ── Result box ────────────────────────────────────────
        with res_col:
            st.markdown(f"""
            <div class="result-box">
              <div style="color:#8899bb;font-size:.8rem;margin-bottom:.3rem">Estimated Sale Price</div>
              <div class="price">${price:,.0f}</div>
              <div class="sub">Range: ${price*0.9:,.0f} – ${price*1.1:,.0f} &nbsp;(±10 %)</div>
              <div class="sub" style="margin-top:.6rem;opacity:.5">
                Model: {best_name} &nbsp;|&nbsp; Log-RMSE: {results[best_name]['RMSE']:.4f}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── KPI row ───────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        for col_obj, label, val, sub in [
            (k1, "Conservative",    price * 0.90, "−10 %"),
            (k2, "Best Estimate",   price,         "Model prediction"),
            (k3, "Optimistic",      price * 1.10,  "+10 %"),
        ]:
            with col_obj:
                st.markdown(f"""
                <div class="kpi">
                  <div class="lbl">{label}</div>
                  <div class="val">${val:,.0f}</div>
                  <div class="sub">{sub}</div>
                </div>
                """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# TAB 2 · MODEL PERFORMANCE
# ────────────────────────────────────────────────────────────────
with tab_perf:
    st.markdown("### 5-fold Cross-Validation Results")
    st.caption("Target: log1p(SalePrice) · Lower RMSE = better")

    # ── Bar chart ─────────────────────────────────────────────
    sorted_names = [n for n, _ in sorted(results.items(), key=lambda x: x[1]["RMSE"])]
    rmse_vals    = [results[n]["RMSE"] for n in sorted_names]
    std_vals     = [results[n]["STD"]  for n in sorted_names]
    colors       = ["#e94560" if n == best_name else "#1e3a5f" for n in sorted_names]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    bars = ax.barh(sorted_names[::-1], rmse_vals[::-1],
                   xerr=std_vals[::-1], color=colors[::-1],
                   error_kw=dict(ecolor="#445", capsize=4, lw=1.2),
                   height=0.55, zorder=3)

    for bar, val in zip(bars, rmse_vals[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=10, color="#ccc")

    ax.set_xlabel("CV RMSE (log scale)", color="#778", fontsize=10)
    ax.tick_params(colors="#aaa", labelsize=10)
    ax.spines[:].set_color("#1e2a4a")
    ax.xaxis.label.set_color("#778")
    ax.set_xlim(0, max(rmse_vals) * 1.18)
    ax.grid(axis="x", color="#1e2a4a", linewidth=0.6, zorder=0)

    deployed = mpatches.Patch(color="#e94560", label=f"Deployed: {best_name}")
    others   = mpatches.Patch(color="#1e3a5f", label="Other models")
    ax.legend(handles=[deployed, others], facecolor="#141428",
              edgecolor="#1e2a4a", labelcolor="#ccc", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Metrics table ─────────────────────────────────────────
    st.markdown("### Detailed Metrics")
    table_rows = []
    for name, r in sorted(results.items(), key=lambda x: x[1]["RMSE"]):
        table_rows.append({
            "Model":       name,
            "CV RMSE":     f"{r['RMSE']:.4f}",
            "Std Dev":     f"±{r['STD']:.4f}",
            "vs Best":     "—" if name == best_name else f"+{r['RMSE']-results[best_name]['RMSE']:.4f}",
            "Status":      "✅ Deployed" if name == best_name else "",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ── Feature importances ───────────────────────────────────
    if feat_imp:
        st.markdown("### Feature Importances (Top 20)")
        names_imp = [f for f, _ in feat_imp]
        vals_imp  = [v for _, v in feat_imp]

        fig2, ax2 = plt.subplots(figsize=(9, 6))
        fig2.patch.set_facecolor("#0f0f1a")
        ax2.set_facecolor("#0f0f1a")

        bar_colors = ["#e94560" if i == 0 else "#1e3a5f" for i in range(len(names_imp))]
        ax2.barh(names_imp[::-1], vals_imp[::-1], color=bar_colors[::-1], height=0.6, zorder=3)

        for i, (nm, vl) in enumerate(zip(names_imp[::-1], vals_imp[::-1])):
            ax2.text(vl + 0.002, i, f"{vl*100:.1f}%",
                     va="center", ha="left", fontsize=9, color="#aaa")

        ax2.set_xlabel("Feature importance", color="#778", fontsize=10)
        ax2.tick_params(colors="#aaa", labelsize=9)
        ax2.spines[:].set_color("#1e2a4a")
        ax2.grid(axis="x", color="#1e2a4a", linewidth=0.6, zorder=0)
        ax2.set_xlim(0, max(vals_imp) * 1.2)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ────────────────────────────────────────────────────────────────
# TAB 3 · DATA INSIGHTS
# ────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("### Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    for col_obj, lbl, val, sub in [
        (c1, "Total Homes",      f"{len(df_raw):,}",                    "Ames, Iowa"),
        (c2, "Features",         str(len(df_raw.columns) - 1),           "after dropping IDs"),
        (c3, "Median Price",     f"${int(df_raw['SalePrice'].median()):,}", ""),
        (c4, "Price Range",      f"${int(df_raw['SalePrice'].min()/1e3)}K–${int(df_raw['SalePrice'].max()/1e3)}K", ""),
    ]:
        with col_obj:
            st.markdown(f"""
            <div class="kpi">
              <div class="lbl">{lbl}</div>
              <div class="val" style="font-size:1.4rem">{val}</div>
              <div class="sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Price distribution ────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Sale price distribution**")
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        fig3.patch.set_facecolor("#0f0f1a")
        ax3.set_facecolor("#0f0f1a")
        ax3.hist(df_raw["SalePrice"]/1e3, bins=50, color="#e94560", alpha=0.8, edgecolor="#0f0f1a")
        ax3.set_xlabel("Sale price ($K)", color="#aaa", fontsize=9)
        ax3.set_ylabel("Count", color="#aaa", fontsize=9)
        ax3.tick_params(colors="#aaa", labelsize=8)
        ax3.spines[:].set_color("#1e2a4a")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_b:
        st.markdown("**Price vs. Overall Quality**")
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        fig4.patch.set_facecolor("#0f0f1a")
        ax4.set_facecolor("#0f0f1a")
        grp = df_raw.groupby("Overall Qual")["SalePrice"].median() / 1e3
        ax4.bar(grp.index, grp.values, color="#e94560", alpha=0.85, width=0.65)
        ax4.set_xlabel("Overall Quality (1–10)", color="#aaa", fontsize=9)
        ax4.set_ylabel("Median price ($K)", color="#aaa", fontsize=9)
        ax4.tick_params(colors="#aaa", labelsize=8)
        ax4.spines[:].set_color("#1e2a4a")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    # ── Neighbourhood median prices ───────────────────────────
    st.markdown("**Median sale price by neighbourhood**")
    nbhd = df_raw.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False) / 1e3

    fig5, ax5 = plt.subplots(figsize=(11, 3.8))
    fig5.patch.set_facecolor("#0f0f1a")
    ax5.set_facecolor("#0f0f1a")
    bar_c = ["#e94560" if v > nbhd.median() else "#1e3a5f" for v in nbhd.values]
    ax5.bar(nbhd.index, nbhd.values, color=bar_c, width=0.7, zorder=3)
    ax5.axhline(nbhd.median(), color="#e94560", linewidth=1, linestyle="--", alpha=0.6)
    ax5.set_ylabel("Median price ($K)", color="#aaa", fontsize=9)
    ax5.tick_params(colors="#aaa", labelsize=8, axis="x", rotation=45)
    ax5.tick_params(colors="#aaa", labelsize=8, axis="y")
    ax5.spines[:].set_color("#1e2a4a")
    ax5.grid(axis="y", color="#1e2a4a", linewidth=0.5, zorder=0)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    # ── Correlation heatmap (top numeric) ─────────────────────
    st.markdown("**Correlation with sale price (top numeric features)**")
    top_corr_cols = (df_raw[num_cols + ["SalePrice"]]
                     .corr()["SalePrice"]
                     .abs()
                     .sort_values(ascending=False)
                     .head(12)
                     .index.tolist())
    corr_mat = df_raw[top_corr_cols].corr()

    fig6, ax6 = plt.subplots(figsize=(9, 7))
    fig6.patch.set_facecolor("#0f0f1a")
    ax6.set_facecolor("#0f0f1a")
    sns.heatmap(corr_mat, ax=ax6, cmap="RdYlGn", annot=True, fmt=".2f",
                annot_kws={"size": 8}, linewidths=0.4, linecolor="#1e2a4a",
                cbar_kws={"shrink": 0.8})
    ax6.tick_params(colors="#ccc", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

    # ── Raw data preview ──────────────────────────────────────
    with st.expander("📋 Raw data preview (first 50 rows)"):
        st.dataframe(df_raw.head(50), use_container_width=True)
