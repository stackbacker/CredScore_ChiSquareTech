# Credit Risk Analysis — German Credit Dataset

End-to-end ML pipeline to predict **Good vs Bad credit risk** from structured applicant data (German Credit). Includes clean preprocessing, statistical screening, model comparison, hyperparameter tuning, and final evaluation with visualizations.

> **Why this matters:** in credit underwriting, the most expensive mistake is approving a **Bad Risk**. This repo shows how to build, evaluate, and *tune decision thresholds* with that business constraint in mind.

---

##  Project Highlights
- Clean **data mapping** from coded categories (e.g., `A11`, `A34`) to human-readable labels.
- **EDA visuals:** class-split category distributions and boxplots for numeric features.
- **Statistical screening:** Chi-square (categoricals) & one-way ANOVA (numerics) to pick signal-bearing features.
- **Feature engineering:** one-hot encoding; optional **PCA** for dimensionality reduction.
- **Model zoo:** Logistic Regression, Random Forest, SVM, Gradient Boosting, XGBoost, LightGBM, Extra Trees, AdaBoost.
- **Tuning:** `GridSearchCV` on the top model (LightGBM in the notebook).
- **Evaluation:** F1, ROC AUC, confusion matrix, classification report, ROC curve; feature importances (if supported).
- **Risk-aware tips:** class weights & **threshold tuning** to reduce *Bad→Good* errors.

---

##  Repository Structure (suggested)
```
.
├── Credit_Risk_Analysis_German_Credit.ipynb   # Main notebook (end-to-end)
├── data/
│   └── german.data                            # Raw data (see Dataset note below)
├── reports/                                   # (optional) saved figures/metrics
├── src/                                       # (optional) if you factor out utilities
└── README.md
```

> The notebook path may differ depending on where you saved it. Adjust as needed.

---

##  Dataset
- **Name:** German Credit
- **Size:** 1,000 rows; **Features:** 20 (7 numeric, 13 categorical); **Target:** Good(1) / Bad(2).
- **Note:** The raw file is typically named `german.data`. If licensing prevents bundling, download it yourself (commonly available via the UCI Machine Learning Repository) and place it under `./data/german.data`.

The notebook reads:
```python
df = pd.read_csv("german.data", sep=" ", header=None)
```

---

## Environment
- **Python:** 3.9+
- **Libraries:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `xgboost`, `lightgbm`, `jupyter`

Quick setup:
```bash
# (Recommended) create a virtual env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install deps
pip install -U pip
pip install numpy pandas matplotlib seaborn scikit-learn scipy xgboost lightgbm jupyter
```

---

##  How to Run
1. **Place the data**
   - Put `german.data` under `./data/` or next to the notebook. If you change the path, update the `pd.read_csv(...)` line.
2. **Launch Jupyter**
   ```bash
   jupyter notebook
   # or: jupyter lab
   ```
3. **Open** `Credit_Risk_Analysis_German_Credit.ipynb` and run all cells, top to bottom.

---

## Methodology (what the notebook does)
1. **Preprocessing**
   - Map coded categoricals (e.g., checking account, credit history, purpose, etc.) to readable labels.
   - Create **one-hot** dummies for selected categorical features.
   - Map target: `Good Risk → 1`, `Bad Risk → 0`.
2. **EDA**
   - **Category distributions** split by class (pyramid-style bar charts).
   - **Boxplots** for numeric features by class (median, spread, outliers).
3. **Univariate screening**
   - **Chi-square** for categorials → association with class.
   - **ANOVA** for numerics → mean differences across classes.
4. **Dimensionality reduction (optional)**
   - **PCA** to 16 components (or set a variance target).
   - *Note:* tree/boosting models usually don’t need PCA.
5. **Modeling & tuning**
   - Compare 8+ models on **F1** (stratified train/test split).
   - `GridSearchCV` on the best model (LightGBM in this project).
6. **Final evaluation**
   - **F1** (positive class defaults to `Good=1`), **ROC AUC**.
   - **Confusion matrix** & **classification report**.
   - **ROC curve**; feature importance if available.

---

## Results & Interpretation
- **F1** summarizes precision/recall for the positive class.
- **ROC AUC** measures separability (threshold-independent).
- **Confusion matrix** highlights *which* mistakes happen:
  - The costliest in credit is **predicting Good when actually Bad**.
- **Tip:** If your goal is catching Bad applicants, either:
  - Flip the positive class (treat **Bad = 1**), or
  - Keep `Good = 1` but **raise the decision threshold** to reduce false-positives for Good.

Threshold tuning example:
```python
thr = 0.6  # try 0.5 → 0.8 grid
y_pred_custom = (best_model.predict_proba(X_test)[:, 1] >= thr).astype(int)
```

---

##  Reproducibility
- Stratified train/test split with fixed `random_state`.
- `GridSearchCV` with a specified CV and scoring.
- For full reproducibility, pin exact library versions via `requirements.txt` or `pip-tools` / `poetry`.

---

## Extensions
- **Explainability:** SHAP on the best non-PCA model (human-readable features).
- **Cost-sensitive learning:** class weights, custom loss aligned to cost matrix.
- **Calibration:** Platt or isotonic for better probability estimates.
- **Production:** export a `Pipeline` (preprocess + model) and serve via FastAPI; add monitoring & drift checks.

---

## License & Data
- Code: choose a license (e.g., MIT) and add `LICENSE`.
- Data: respect the dataset’s original terms. If not redistributing, add a small fetch script or instructions to download.

---

##  Contact
If you have questions or want to collaborate, feel free to open an issue or reach out at bishankaseal@gmail.com

