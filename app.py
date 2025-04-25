import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Cancer Prediction App", layout="wide")

# Title
st.title("🧬 Cancer Prediction using Gene Expression")
st.markdown("Built with **Random Forest** on the Breast Cancer Wisconsin dataset.")

# Load the dataset
@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y, data.target_names

X, y, target_names = load_data()

# Sidebar
st.sidebar.header("📊 Dataset Preview Options")
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Dataset")
    st.dataframe(X)

if st.sidebar.checkbox("Show class distribution"):
    st.subheader("Class Distribution")
    st.bar_chart(y.value_counts())

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
st.subheader("✅ Model Evaluation")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {accuracy:.4f}")

report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
st.write("**Classification Report:**")
st.dataframe(pd.DataFrame(report).transpose())

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig)

# Optional Prediction
st.sidebar.header("🔍 Predict on New Input")
user_input = {}
for feature in X.columns:
    val = st.sidebar.slider(f"{feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
    user_input[feature] = val

if st.sidebar.button("Predict Cancer Status"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    result = target_names[prediction]
    st.success(f"🧾 Prediction: **{result}**")
