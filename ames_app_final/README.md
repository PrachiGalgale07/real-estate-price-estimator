# 🏠 Ames House Price Predictor — All-in-One

Everything is in **one file**: `app.py`  
No pre-training step, no separate script — just run it and go.

---

## Folder structure

```
ames_predictor/
├── app.py                 ← the entire app (UI + training + charts)
├── requirements.txt
└── AmesHousing_1_.csv     ← put your dataset here
```

---

## Setup in VS Code

### 1. Install dependencies
Open a terminal inside VS Code (`Ctrl+` `` ` ``) and run:

```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

### 3. First-run note
On first launch, all 6 models are trained with 5-fold CV (~30 seconds).  
Results are cached — subsequent runs are instant.

---

## What's inside `app.py`

| Section | Description |
|---------|-------------|
| Data loading | Reads CSV, drops identifier columns |
| Preprocessing | Median imputation → StandardScaler (numeric), OHE (categorical) |
| Model training | Ridge, Lasso, Decision Tree, Random Forest, Extra Trees, **Gradient Boosting** |
| Model selection | 5-fold CV on log1p(SalePrice) · best model auto-selected |
| UI Tab 1 | Interactive price predictor with sliders + dropdowns |
| UI Tab 2 | Model leaderboard chart + feature importance bar chart |
| UI Tab 3 | Data insights: price distribution, quality bands, neighbourhood chart, correlation heatmap |

---

## Model Results

| Model | CV RMSE |
|---|---|
| **Gradient Boosting ✅** | **0.1229** |
| Ridge | 0.1341 |
| Extra Trees | 0.1361 |
| Lasso | 0.1369 |
| Random Forest | 0.1393 |
| Decision Tree | 0.1946 |

RMSE on log1p(SalePrice) · 5-fold CV · 2,930 samples · 79 features
