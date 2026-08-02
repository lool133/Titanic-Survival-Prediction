# 🚢 Titanic Survival Prediction

A complete Machine Learning workflow on the [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic/data?select=train.csv), including cross-validation, model comparison, hyperparameter tuning, model persistence with `joblib`, and a **Streamlit** web app for live predictions.

## 📁 Project Structure

```
├── titanic_churn_task1_task2.ipynb   # Notebook: CV + model comparison + save/load model
├── train.csv                          # Raw Titanic dataset
├── titanic_best_model.joblib          # Saved best pipeline (preprocessing + model)
├── app.py                             # Streamlit app for live predictions
├── requirements.txt                   # Python dependencies
└── README.md
```

## 📌 Tasks Covered

### Task 1 — Cross Validation
Applied 5-fold Stratified Cross Validation on a Logistic Regression baseline pipeline (with full preprocessing: imputation, encoding, scaling) to get a reliable estimate of model performance.

### Task 2 — Best Model Selection + Save/Load
- Compared 7 different classifiers (Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, Naive Bayes) using cross-validation.
- Tuned the top 3 candidates with `GridSearchCV`.
- **Best model: SVM (RBF kernel)** — ~82.6% CV accuracy, ~80.4% test accuracy.
- Saved the full pipeline (preprocessing + model) using `joblib`.
- Loaded the saved pipeline and tested it directly on raw, unprocessed passenger data.

### Task 3 — Streamlit App
Built an interactive web app (`app.py`) that loads the saved pipeline and predicts survival probability for any passenger, given raw inputs (no manual preprocessing needed — it all happens inside the pipeline).

## 🚀 Running the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 Model Pipeline

The saved model is a single `sklearn.Pipeline` combining:
- **Preprocessing**: median imputation + scaling for numeric features (`Age`, `SibSp`, `Parch`, `Fare`), most-frequent imputation + one-hot encoding for categorical features (`Pclass`, `Sex`, `Embarked`).
- **Classifier**: tuned `SVC` (RBF kernel, `C=10`, `gamma='auto'`, `probability=True`).

This means raw passenger data can be fed directly into `model.predict()` — no manual preprocessing required.

## 📊 Results

| Model | CV Accuracy |
|---|---|
| Logistic Regression | 80.2% |
| KNN | 79.6% |
| SVM (tuned) | **82.6%** |
| Decision Tree | 77.0% |
| Random Forest (tuned) | 82.6% |
| Gradient Boosting (tuned) | 82.6% |
| Naive Bayes | 78.9% |

Final SVM pipeline test set accuracy: **80.4%**
