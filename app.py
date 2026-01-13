import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

st.write("VERSION 2 LOADED")
st.title("Bank Credit Risk & Profit App")

# Categorical columns
label_cols = ["term", "grade", "sub_grade", "emp_length", "home_ownership"]


use_cols = [
    "loan_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership",
    "annual_inc", "loan_status"
]

DATA_URL = "https://drive.google.com/uc?id=1Y_0LR1NgX9koOO1GmPb7nlkORZBFXlF7"
data = pd.read_csv(DATA_URL, usecols=use_cols)


# Create target
data["risk"] = data["loan_status"].apply(
    lambda x: 1 if x in ["Charged Off", "Default", "Late (31-120 days)"] else 0
)


ml_data = data.copy()

le = {}   # one encoder per column

for col in label_cols:
    ml_data[col] = ml_data[col].astype(str).str.strip()   # remove spaces
    le[col] = LabelEncoder()
    ml_data[col] = le[col].fit_transform(ml_data[col])



X = ml_data.drop(["loan_status", "risk"], axis=1)
y = ml_data["risk"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@st.cache_resource
def train_model():
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

model = train_model()
preds = model.predict(X_test)



accuracy = accuracy_score(y_test, preds)
st.subheader("Credit Risk Model Performance")
st.write("Model Accuracy:", round(accuracy * 100, 2), "%")


st.subheader("New Loan Application")

col1, col2 = st.columns(2)

with col1:
    loan_amnt_in = st.number_input("Loan Amount ($)", 1000, 50000, 10000)
    term_in = st.selectbox("Loan Term", ["36 months", "60 months"])
    int_rate_in = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
    grade_in = st.selectbox("Grade", ["A","B","C","D","E","F","G"])
    emp_len_in = st.selectbox("Employment Length", ["<1 year","1 year","2 years","3 years","5 years","10+ years"])

with col2:
    installment_in = st.number_input("Monthly Installment ($)", 50, 2000, 300)
    subgrade_in = st.selectbox("Sub Grade", ["A1","A2","A3","B1","B2","C1","C2","D1","D2","E1","E2","F1","F2"])
    home_in = st.selectbox("Home Ownership", ["RENT","MORTGAGE","OWN"])
    annual_inc_in = st.number_input("Annual Income ($)", 10000, 200000, 50000)


input_dict = {
    "loan_amnt": loan_amnt_in,
    "term": term_in,
    "int_rate": int_rate_in,
    "installment": installment_in,
    "grade": grade_in,
    "sub_grade": subgrade_in,
    "emp_length": emp_len_in,
    "home_ownership": home_in,
    "annual_inc": annual_inc_in
}

input_df = pd.DataFrame([input_dict])

# Normalize user inputs to match training data
emp_map = {
    "<1 year": "1 year",
    "1 year": "1 year",
    "2 years": "2 years",
    "3 years": "3 years",
    "5 years": "5 years",
    "10+ years": "10+ years"
}

term_map = {
    "36 months": "36 months",
    "60 months": "60 months"
}

input_df["emp_length"] = input_df["emp_length"].map(emp_map)
input_df["term"] = input_df["term"].map(term_map)

# Encode categorical fields
for col in label_cols:
    input_df[col] = input_df[col].astype(str).str.strip()
    input_df[col] = le[col].transform(input_df[col])



if st.button("Check Loan Eligibility"):
    risk_pred = model.predict(input_df)[0]
    risk_prob = model.predict_proba(input_df)[0][1]

    if risk_pred == 0:
        st.success("✅ Loan Approved (Low Risk)")
    else:
        st.error("❌ Loan Rejected (High Risk)")

    st.write("Default Risk Probability:", round(risk_prob * 100, 2), "%")
    
# ------------------ PROFIT SIMULATION ------------------

loan_amount = X_test["loan_amnt"]
interest_rate = X_test["int_rate"] / 100

X_test_clean = X_test.copy()
X_test_clean["risk"] = y_test.values
X_test_clean["pred"] = preds

X_test_clean = X_test_clean.dropna(subset=["loan_amnt", "int_rate"])

loan_amount = X_test_clean["loan_amnt"]
interest_rate = X_test_clean["int_rate"] / 100
pred_clean = X_test_clean["pred"]

profit = []

for i in range(len(X_test_clean)):
    if pred_clean.iloc[i] == 0:
        profit.append(loan_amount.iloc[i] * interest_rate.iloc[i])
    else:
        profit.append(-loan_amount.iloc[i])

total_profit = sum(profit)

st.subheader("Bank Profit Simulation Using AI")
st.write("Total Profit from test customers ($):", round(total_profit, 2))




