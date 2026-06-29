"""
Ames Housing Price Predictor — All-in-One (v3 — Cloud Compatible)
==================================================================
Compatible with:
  Python      3.9 – 3.14
  scikit-learn >= 1.2  (handles sparse_output / sparse kwarg automatically)
  pandas      >= 2.0
  streamlit   >= 1.32
  numpy       >= 1.24

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Cloud:
    Push app.py + requirements.txt + AmesHousing_1_.csv to a GitHub repo
    and connect on share.streamlit.io.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # must be set before any other matplotlib import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from pathlib import Path

import sklearn
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.impute           import SimpleImputer
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.ensemble         import (GradientBoostingRegressor,
                                      RandomForestRegressor,
                                      ExtraTreesRegressor)
from sklearn.linear_model     import Ridge, Lasso
from sklearn.tree             import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ── OneHotEncoder helper — works on sklearn 1.0 through 1.9+ ─────────────────
def _ohe():
    """Return a dense OneHotEncoder regardless of sklearn version."""
    sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if sk_ver >= (1, 2):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
DATA_PATH = BASE / "AmesHousing_1_.csv"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"]{ background:#090912; border-right:1px solid #1e1e2e; }
.hdr{
    background:linear-gradient(120deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
    border:1px solid #1e3a5f; border-radius:14px;
    padding:1.6rem 2rem; margin-bottom:1.6rem;
}
.hdr h1{ color:#fff; font-size:2rem; margin:0 0 .3rem; }
.hdr p { color:#8899bb; font-size:.95rem; margin:0; }
.kpi{
    background:#141428; border:1px solid #1e2a4a;
    border-radius:12px; padding:1rem 1.25rem; text-align:center;
}
.kpi .lbl{ font-size:.7rem; text-transform:uppercase;
           letter-spacing:1.2px; color:#556; margin-bottom:.25rem; }
.kpi .val{ font-size:1.7rem; font-weight:700; color:#e94560; }
.kpi .sub{ font-size:.72rem; color:#445; margin-top:.15rem; }
.result-box{
    background:linear-gradient(135deg,#0f3460,#1a1a2e);
    border:1px solid #e9456050; border-radius:14px;
    padding:1.5rem 2rem; text-align:center; margin-top:1rem;
}
.result-box .price{ font-size:2.8rem; font-weight:800;
                    color:#e94560; letter-spacing:-1px; }
.result-box .sub  { color:#8899bb; font-size:.85rem; margin-top:.4rem; }
.slbl{
    font-size:.7rem; font-weight:700; text-transform:uppercase;
    letter-spacing:1.5px; color:#e94560;
    border-bottom:1px solid #e9456025;
    margin:1.4rem 0 .6rem; padding-bottom:.3rem;
}
div.stButton > button{
    background:linear-gradient(135deg,#e94560,#c23152) !important;
    color:#fff !important; border:none !important;
    border-radius:10px !important; font-size:1rem !important;
    font-weight:600 !important; padding:.6rem 1.6rem !important;
    width:100% !important; box-shadow:0 4px 14px #e9456040 !important;
}
.best-badge{
    display:inline-block; background:#e9456020; color:#e94560;
    border:1px solid #e9456050; border-radius:6px;
    font-size:.72rem; padding:2px 8px; margin-left:6px;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(
            f"❌ **Dataset not found**: `{DATA_PATH}`\n\n"
            "Make sure `AmesHousing_1_.csv` is in the same folder as `app.py`."
        )
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Order", "PID"], errors="ignore")
    return df


df_raw = load_data()


# ═════════════════════════════════════════════════════════════════════════════
# 2. TRAINING PIPELINE  (runs once, then cached)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def build_and_train(_df):
    df = _df.copy()
    y  = np.log1p(df["SalePrice"].values.astype(float))
    X  = df.drop(columns=["SalePrice"])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", _ohe()),
    ])

    # verbose_feature_names_out available from sklearn 1.1
    try:
        preprocessor = ColumnTransformer(
            [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
            verbose_feature_names_out=False,
        )
    except TypeError:
        preprocessor = ColumnTransformer(
            [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
        )

    candidates = {
        "Ridge":             Ridge(alpha=10),
        "Lasso":             Lasso(alpha=0.001, max_iter=5000),
        "Decision Tree":     DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=12,
                                                   random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                                        learning_rate=0.07,
                                                        random_state=42),
        "Extra Trees":       ExtraTreesRegressor(n_estimators=100, max_depth=12,
                                                 random_state=42, n_jobs=-1),
    }

    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in candidates.items():
        pipe   = Pipeline([("prep", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X, y, cv=kf,
                                 scoring="neg_root_mean_squared_error")
        results[name] = {
            "RMSE": float(-scores.mean()),
            "STD":  float(scores.std()),
        }

    best_name = min(results, key=lambda k: results[k]["RMSE"])
    best_pipe = Pipeline([("prep", preprocessor), ("model", candidates[best_name])])
    best_pipe.fit(X, y)

    # Feature importances (tree-based models only)
    feat_imp = None
    try:
        raw_imp    = best_pipe.named_steps["model"].feature_importances_
        feat_names = best_pipe.named_steps["prep"].get_feature_names_out()
        top_idx    = np.argsort(raw_imp)[::-1][:20]
        feat_imp   = [(str(feat_names[i]), float(raw_imp[i])) for i in top_idx]
    except Exception:
        pass

    # UI metadata
    num_stats  = {
        c: {"min": float(df[c].min()), "max": float(df[c].max()),
            "median": float(df[c].median())}
        for c in num_cols
    }
    cat_unique  = {c: sorted(df[c].dropna().astype(str).unique().tolist())
                   for c in cat_cols}
    cat_default = {c: str(df[c].mode()[0]) for c in cat_cols}

    return best_pipe, {
        "best_model":  best_name,
        "results":     results,
        "num_cols":    num_cols,
        "cat_cols":    cat_cols,
        "num_stats":   num_stats,
        "cat_unique":  cat_unique,
        "cat_default": cat_default,
        "feat_imp":    feat_imp,
    }, X, y


with st.spinner("⚙️  Training 6 models on first run — please wait (~30–60 s)…"):
    best_pipe, meta, X_full, y_full = build_and_train(df_raw)

best_name   = meta["best_model"]
results     = meta["results"]
num_cols    = meta["num_cols"]
cat_cols    = meta["cat_cols"]
num_stats   = meta["num_stats"]
cat_unique  = meta["cat_unique"]
cat_default = meta["cat_default"]
feat_imp    = meta["feat_imp"]


# ═════════════════════════════════════════════════════════════════════════════
# 3. SIDEBAR — Model Leaderboard
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Model Leaderboard")
    st.caption("5-fold CV · log RMSE — lower is better")

    sorted_res = sorted(results.items(), key=lambda x: x[1]["RMSE"])
    best_rmse  = sorted_res[0][1]["RMSE"]
    worst_rmse = sorted_res[-1][1]["RMSE"]
    spread     = max(worst_rmse - best_rmse, 1e-9)

    for rank, (name, r) in enumerate(sorted_res):
        badge = '<span class="best-badge">✓ deployed</span>' if name == best_name else ""
        st.markdown(f"**{rank+1}. {name}** {badge}", unsafe_allow_html=True)
        bar_pct = int((1 - (r["RMSE"] - best_rmse) / spread) * 100)
        bar_col = "#e94560" if name == best_name else "#2a3a5a"
        st.markdown(f"""
        <div style="background:#0d0d1a;border-radius:6px;height:8px;margin:2px 0 4px">
          <div style="width:{bar_pct}%;background:{bar_col};
               height:8px;border-radius:6px"></div>
        </div>
        <span style="font-size:.75rem;color:#667">
          RMSE {r['RMSE']:.4f} &nbsp;±{r['STD']:.4f}
        </span>
        """, unsafe_allow_html=True)
        st.markdown("---")

    st.caption(f"Dataset · Ames Iowa · {len(df_raw):,} rows · 79 features")


# ═════════════════════════════════════════════════════════════════════════════
# 4. HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hdr">
  <h1>🏠 Ames House Price Predictor</h1>
  <p>Enter property details and get an AI-powered price estimate.
     Best model: <strong>{best_name}</strong>
     (CV RMSE {results[best_name]['RMSE']:.4f}) · trained on {len(df_raw):,} Iowa homes.</p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 5. TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_pred, tab_perf, tab_data = st.tabs(
    ["🔮 Predict Price", "📈 Model Performance", "📊 Data Insights"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tab_pred:
    user_input = {}

    priority_num = [
        "Overall Qual", "Gr Liv Area", "Total Bsmt SF", "Garage Area",
        "1st Flr SF", "Year Built", "Year Remod/Add", "Lot Area",
        "TotRms AbvGrd", "Full Bath", "Bedroom AbvGr",
    ]
    priority_cat = [
        "Neighborhood", "MS Zoning", "Exter Qual", "Kitchen Qual",
        "Bsmt Qual", "Heating QC", "Central Air", "Garage Type",
        "Foundation", "House Style", "Sale Condition", "Bldg Type",
    ]
    rest_num = [c for c in num_cols if c not in priority_num]
    rest_cat = [c for c in cat_cols if c not in priority_cat]

    # ── Key numeric ──────────────────────────────────────────
    st.markdown('<div class="slbl">Core Property Details</div>',
                unsafe_allow_html=True)
    cols = st.columns(3)
    for i, col in enumerate(priority_num):
        s = num_stats.get(col, {"min": 0, "max": 9999, "median": 0})
        with cols[i % 3]:
            if col == "Overall Qual":
                user_input[col] = st.slider(
                    col, min_value=1, max_value=10,
                    value=int(s["median"]),
                    help="1 = Very Poor · 10 = Very Excellent",
                )
            else:
                user_input[col] = st.number_input(
                    col,
                    min_value=float(max(0, s["min"])),
                    max_value=float(s["max"] * 2),
                    value=float(s["median"]),
                    step=1.0,
                )

    # ── Key categorical ───────────────────────────────────────
    st.markdown('<div class="slbl">Location & Style</div>',
                unsafe_allow_html=True)
    cols = st.columns(3)
    for i, col in enumerate(priority_cat):
        opts = cat_unique.get(col, ["Missing"])
        dflt = cat_default.get(col, opts[0])
        idx  = opts.index(dflt) if dflt in opts else 0
        with cols[i % 3]:
            user_input[col] = st.selectbox(col, opts, index=idx)

    # ── Extra numeric (collapsed) ────────────────────────────
    with st.expander("⚙️  Additional numeric features"):
        cols = st.columns(3)
        for i, col in enumerate(rest_num):
            s = num_stats.get(col, {"min": 0, "max": 9999, "median": 0})
            with cols[i % 3]:
                user_input[col] = st.number_input(
                    col,
                    min_value=float(max(0, s["min"])),
                    max_value=float(s["max"] * 2),
                    value=float(s["median"]),
                    step=1.0, key=f"rn_{col}",
                )

    # ── Extra categorical (collapsed) ────────────────────────
    with st.expander("🔖  Additional categorical features"):
        cols = st.columns(3)
        for i, col in enumerate(rest_cat):
            opts = cat_unique.get(col, ["Missing"])
            dflt = cat_default.get(col, opts[0])
            idx  = opts.index(dflt) if dflt in opts else 0
            with cols[i % 3]:
                user_input[col] = st.selectbox(
                    col, opts, index=idx, key=f"rc_{col}"
                )

    st.markdown("<br>", unsafe_allow_html=True)
    btn_col, res_col = st.columns([1, 2])

    with btn_col:
        predict_clicked = st.button("🔮 Predict Sale Price")

    if predict_clicked:
        row = {}
        for c in num_cols:
            row[c] = float(user_input.get(c, num_stats[c]["median"]))
        for c in cat_cols:
            row[c] = str(user_input.get(c, cat_default.get(c, "Missing")))

        inp_df   = pd.DataFrame([row])[num_cols + cat_cols]
        log_pred = float(best_pipe.predict(inp_df)[0])
        price    = float(np.expm1(log_pred))

        with res_col:
            st.markdown(f"""
            <div class="result-box">
              <div style="color:#8899bb;font-size:.8rem;margin-bottom:.3rem">
                Estimated Sale Price</div>
              <div class="price">${price:,.0f}</div>
              <div class="sub">
                Range: ${price*0.9:,.0f} – ${price*1.1:,.0f} &nbsp;(±10 %)
              </div>
              <div class="sub" style="margin-top:.6rem;opacity:.5">
                Model: {best_name} &nbsp;|&nbsp;
                Log-RMSE: {results[best_name]['RMSE']:.4f}
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        for col_obj, label, val, sub in [
            (k1, "Conservative",  price * 0.90, "−10 %"),
            (k2, "Best Estimate", price,         "Model prediction"),
            (k3, "Optimistic",    price * 1.10,  "+10 %"),
        ]:
            with col_obj:
                st.markdown(f"""
                <div class="kpi">
                  <div class="lbl">{label}</div>
                  <div class="val">${val:,.0f}</div>
                  <div class="sub">{sub}</div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab_perf:
    st.markdown("### 5-fold Cross-Validation Results")
    st.caption("Target: log1p(SalePrice) · Lower RMSE = better")

    sorted_names = [n for n, _ in sorted(results.items(), key=lambda x: x[1]["RMSE"])]
    rmse_vals    = [results[n]["RMSE"] for n in sorted_names]
    std_vals     = [results[n]["STD"]  for n in sorted_names]
    bar_colors   = ["#e94560" if n == best_name else "#1e3a5f" for n in sorted_names]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    bars = ax.barh(
        sorted_names[::-1], rmse_vals[::-1],
        xerr=std_vals[::-1], color=bar_colors[::-1],
        error_kw=dict(ecolor="#445", capsize=4, lw=1.2),
        height=0.55, zorder=3,
    )
    for bar, val in zip(bars, rmse_vals[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=10, color="#ccc")
    ax.set_xlabel("CV RMSE", color="#778", fontsize=10)
    ax.tick_params(colors="#aaa", labelsize=10)
    for sp in ax.spines.values():
        sp.set_color("#1e2a4a")
    ax.set_xlim(0, max(rmse_vals) * 1.18)
    ax.grid(axis="x", color="#1e2a4a", linewidth=0.6, zorder=0)
    ax.legend(
        handles=[
            mpatches.Patch(color="#e94560", label=f"Deployed: {best_name}"),
            mpatches.Patch(color="#1e3a5f", label="Other models"),
        ],
        facecolor="#141428", edgecolor="#1e2a4a", labelcolor="#ccc", fontsize=9,
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Detailed Metrics Table")
    table_rows = []
    for name, r in sorted(results.items(), key=lambda x: x[1]["RMSE"]):
        table_rows.append({
            "Model":   name,
            "CV RMSE": f"{r['RMSE']:.4f}",
            "Std Dev": f"±{r['STD']:.4f}",
            "vs Best": "—" if name == best_name
                       else f"+{r['RMSE'] - results[best_name]['RMSE']:.4f}",
            "Status":  "✅ Deployed" if name == best_name else "",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True,
                 hide_index=True)

    if feat_imp:
        st.markdown("### Feature Importances (Top 20)")
        names_imp = [f for f, _ in feat_imp]
        vals_imp  = [v for _, v in feat_imp]
        fi_colors = ["#e94560" if i == 0 else "#1e3a5f"
                     for i in range(len(names_imp))]

        fig2, ax2 = plt.subplots(figsize=(9, 6))
        fig2.patch.set_facecolor("#0f0f1a")
        ax2.set_facecolor("#0f0f1a")
        ax2.barh(names_imp[::-1], vals_imp[::-1],
                 color=fi_colors[::-1], height=0.6, zorder=3)
        for i, vl in enumerate(vals_imp[::-1]):
            ax2.text(vl + 0.002, i, f"{vl*100:.1f}%",
                     va="center", ha="left", fontsize=9, color="#aaa")
        ax2.set_xlabel("Feature importance", color="#778", fontsize=10)
        ax2.tick_params(colors="#aaa", labelsize=9)
        for sp in ax2.spines.values():
            sp.set_color("#1e2a4a")
        ax2.set_xlim(0, max(vals_imp) * 1.2)
        ax2.grid(axis="x", color="#1e2a4a", linewidth=0.6, zorder=0)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DATA INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("### Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    for col_obj, lbl, val, sub in [
        (c1, "Total Homes",  f"{len(df_raw):,}",
             "Ames, Iowa"),
        (c2, "Features",     str(len(df_raw.columns) - 1),
             "after dropping IDs"),
        (c3, "Median Price", f"${int(df_raw['SalePrice'].median()):,}",
             ""),
        (c4, "Price Range",
             f"${int(df_raw['SalePrice'].min()/1e3)}K–"
             f"${int(df_raw['SalePrice'].max()/1e3)}K", ""),
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

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Sale price distribution**")
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        fig3.patch.set_facecolor("#0f0f1a")
        ax3.set_facecolor("#0f0f1a")
        ax3.hist(df_raw["SalePrice"] / 1e3, bins=50,
                 color="#e94560", alpha=0.8, edgecolor="#0f0f1a")
        ax3.set_xlabel("Sale price ($K)", color="#aaa", fontsize=9)
        ax3.set_ylabel("Count",           color="#aaa", fontsize=9)
        ax3.tick_params(colors="#aaa", labelsize=8)
        for sp in ax3.spines.values():
            sp.set_color("#1e2a4a")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col_b:
        st.markdown("**Price vs. Overall Quality**")
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        fig4.patch.set_facecolor("#0f0f1a")
        ax4.set_facecolor("#0f0f1a")
        grp = df_raw.groupby("Overall Qual")["SalePrice"].median() / 1e3
        ax4.bar(grp.index, grp.values, color="#e94560", alpha=0.85, width=0.65)
        ax4.set_xlabel("Overall Quality (1–10)", color="#aaa", fontsize=9)
        ax4.set_ylabel("Median price ($K)",       color="#aaa", fontsize=9)
        ax4.tick_params(colors="#aaa", labelsize=8)
        for sp in ax4.spines.values():
            sp.set_color("#1e2a4a")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown("**Median sale price by neighbourhood**")
    nbhd = (df_raw.groupby("Neighborhood")["SalePrice"]
            .median()
            .sort_values(ascending=False) / 1e3)
    nbhd_median = float(nbhd.median())

    fig5, ax5 = plt.subplots(figsize=(11, 3.8))
    fig5.patch.set_facecolor("#0f0f1a")
    ax5.set_facecolor("#0f0f1a")
    bar_c5 = ["#e94560" if v > nbhd_median else "#1e3a5f"
              for v in nbhd.values]
    ax5.bar(nbhd.index, nbhd.values, color=bar_c5, width=0.7, zorder=3)
    ax5.axhline(nbhd_median, color="#e94560", linewidth=1,
                linestyle="--", alpha=0.6)
    ax5.set_ylabel("Median price ($K)", color="#aaa", fontsize=9)
    ax5.tick_params(colors="#aaa", labelsize=8, axis="x", rotation=45)
    ax5.tick_params(colors="#aaa", labelsize=8, axis="y")
    for sp in ax5.spines.values():
        sp.set_color("#1e2a4a")
    ax5.grid(axis="y", color="#1e2a4a", linewidth=0.5, zorder=0)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    st.markdown("**Correlation with sale price (top numeric features)**")
    top_corr_cols = (
        df_raw[num_cols + ["SalePrice"]]
        .corr()["SalePrice"]
        .abs()
        .sort_values(ascending=False)
        .head(12)
        .index.tolist()
    )
    corr_mat = df_raw[top_corr_cols].corr()
    fig6, ax6 = plt.subplots(figsize=(9, 7))
    fig6.patch.set_facecolor("#0f0f1a")
    ax6.set_facecolor("#0f0f1a")
    sns.heatmap(
        corr_mat, ax=ax6, cmap="RdYlGn", annot=True, fmt=".2f",
        annot_kws={"size": 8}, linewidths=0.4, linecolor="#1e2a4a",
        cbar_kws={"shrink": 0.8},
    )
    ax6.tick_params(colors="#ccc", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)

    with st.expander("📋 Raw data preview (first 50 rows)"):
        st.dataframe(df_raw.head(50), use_container_width=True)
