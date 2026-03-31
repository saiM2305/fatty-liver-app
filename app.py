import streamlit as st
import pickle
import pandas as pd

# Load files
model = pickle.load(open("fatty_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Fatty Liver App", layout="centered")

# Title
st.title("🩺 Fatty Liver Disease Predictor")
st.markdown("---")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("ML model to detect fatty liver disease")

# Section
st.subheader("🧾 Patient Details")

# Inputs
age = st.number_input("Age", 1, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])

total_bilirubin = st.number_input("Total Bilirubin", 0.0, 10.0, 0.5)
direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 5.0, 0.1)
alk_phos = st.number_input("Alkaline Phosphotase", 0, 1000, 200)
alt = st.number_input("ALT", 0, 1000, 30)
ast = st.number_input("AST", 0, 1000, 30)
total_protein = st.number_input("Total Proteins", 0.0, 10.0, 6.5)
albumin = st.number_input("Albumin", 0.0, 6.0, 3.5)
ag_ratio = st.number_input("A/G Ratio", 0.0, 3.0, 1.0)

# Encoding
gender = 1 if gender == "Male" else 0

if age <= 35:
    age_group = 0
elif age <= 60:
    age_group = 1
else:
    age_group = 2

# Input dataframe
input_data = pd.DataFrame([[
    age, gender, total_bilirubin, direct_bilirubin,
    alk_phos, alt, ast,
    total_protein, albumin, ag_ratio, age_group
]], columns=columns)

input_scaled = scaler.transform(input_data)

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("### 🧪 Prediction Result")

    if prediction == 1:
        st.error("⚠️ HIGH RISK")
    else:
        st.success("✅ LOW RISK")

    st.write(f"### Probability: {round(prob*100,2)}%")
    st.progress(int(prob * 100))

    st.info("This prediction is based on clinical parameters.")