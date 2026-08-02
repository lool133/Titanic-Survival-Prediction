import streamlit as st
import pandas as pd
import joblib

# ---------- Page Config ----------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return joblib.load("titanic_best_model.joblib")

model = load_model()

# ---------- Header ----------
st.title("🚢 Titanic Survival Predictor")
st.write(
    "This app uses a Machine Learning pipeline (trained with cross-validation "
    "and hyperparameter tuning) to predict whether a Titanic passenger would "
    "have survived, based on their personal details."
)

st.divider()

# ---------- Input Form ----------
st.subheader("Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", options=[1, 2, 3], index=2,
                           help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
    sex = st.selectbox("Sex", options=["male", "female"])
    age = st.slider("Age", min_value=0, max_value=100, value=30)
    embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"],
                             help="S = Southampton, C = Cherbourg, Q = Queenstown")

with col2:
    sibsp = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents / Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=32.0, step=0.5)

st.divider()

# ---------- Prediction ----------
if st.button("Predict Survival", type="primary", use_container_width=True):
    raw_input = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])

    prediction = model.predict(raw_input)[0]
    probability = model.predict_proba(raw_input)[0][1]

    if prediction == 1:
        st.success(f"✅ This passenger would likely **SURVIVE**")
    else:
        st.error(f"❌ This passenger would likely **NOT SURVIVE**")

    st.metric("Survival Probability", f"{probability * 100:.1f}%")
    st.progress(float(probability))

    with st.expander("See input data sent to the model"):
        st.dataframe(raw_input)

st.divider()
st.caption(
    "Model: Best pipeline selected via cross-validation & GridSearchCV "
    "(preprocessing + classifier) — saved and loaded with joblib."
)
