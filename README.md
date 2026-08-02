# 🚢 Titanic Survival Prediction

A complete Machine Learning workflow on the Kaggle Titanic dataset, including cross-validation, model comparison, hyperparameter tuning, model persistence with `joblib`, and a **Streamlit** web app for live predictions.

---

## 📁 Project Structure

```text
├── walaa.ipynb                       # Notebook: CV + model comparison + save/load model
├── train.csv                         # Raw Titanic dataset
├── titanic_best_model.joblib         # Saved best pipeline (preprocessing + model)
├── app.py                            # Streamlit app for live predictions
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 📌 Tasks Covered

### Task 1 — Cross Validation

Applied **5-fold Stratified Cross Validation** on a Logistic Regression baseline pipeline (with full preprocessing: imputation, encoding, and scaling) to obtain a reliable estimate of model performance.

### Task 2 — Best Model Selection + Save/Load

- Compared **7 different classifiers**:
  - Logistic Regression
  - KNN
  - SVM
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Naive Bayes
- Tuned the top 3 candidates using **GridSearchCV**.
- **Best model:** SVM (RBF kernel)
  - Cross-validation Accuracy: **82.6%**
  - Test Accuracy: **80.4%**
- Saved the complete pipeline (preprocessing + model) using `joblib`.
- Loaded the saved pipeline and tested it directly on raw, unprocessed passenger data.

### Task 3 — Streamlit App

Built an interactive **Streamlit** web application (`app.py`) that loads the saved pipeline and predicts passenger survival probability from raw user inputs. No manual preprocessing is required because preprocessing is integrated into the pipeline.

---

## 🚀 Running the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Model Pipeline

The saved model is a single `sklearn.Pipeline` containing:

- **Preprocessing**
  - Median imputation and scaling for numerical features:
    - Age
    - SibSp
    - Parch
    - Fare
  - Most-frequent imputation and One-Hot Encoding for categorical features:
    - Pclass
    - Sex
    - Embarked

- **Classifier**
  - Tuned SVM (`SVC`)
  - RBF kernel
  - `C = 10`
  - `gamma = "auto"`
  - `probability = True`

This allows raw passenger data to be passed directly into:

```python
model.predict()
```

without any manual preprocessing.

---

## 📊 Results

| Model | CV Accuracy |
|----------------------|------------|
| Logistic Regression | 80.2% |
| KNN | 79.6% |
| SVM (Tuned) | **82.6%** |
| Decision Tree | 77.0% |
| Random Forest (Tuned) | 82.6% |
| Gradient Boosting (Tuned) | 82.6% |
| Naive Bayes | 78.9% |

### Final Performance 
Final SVM pipeline test set accuracy: 80.4%
ا
- **Best Model:** SVM (RBF Kernel)
- **Cross-Validation Accuracy:** **82.6%**
- **Test Accuracy:** **80.4%**
