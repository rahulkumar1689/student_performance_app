import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Page Setup
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")
st.markdown("Analyze student data, visualize patterns, and build machine learning models to predict performance categories.")

# ---------------------------------------------------------
# Sidebar: Dataset Loading
# ---------------------------------------------------------
st.sidebar.header("📁 Data Source")

@st.cache_data
def generate_sample_data():
    """Generates a synthetic dataset if no file is uploaded."""
    np.random.seed(42)
    n = 300
    study_hours = np.random.uniform(1, 10, n).round(1)
    attendance = np.random.uniform(50, 100, n).round(1)
    previous_score = np.random.uniform(40, 100, n).round(1)
    sleep_hours = np.random.uniform(4, 9, n).round(1)
    tutoring = np.random.choice(["Yes", "No"], n)
    
    # Generate Target based on logic + noise
    composite = (study_hours * 3) + (attendance * 0.4) + (previous_score * 0.5) + np.random.normal(0, 5, n)
    
    categories = []
    for score in composite:
        if score > 85:
            categories.append("High")
        elif score > 65:
            categories.append("Medium")
        else:
            categories.append("Low")
            
    df = pd.DataFrame({
        "StudyHours": study_hours,
        "Attendance": attendance,
        "PreviousScore": previous_score,
        "SleepHours": sleep_hours,
        "Tutoring": tutoring,
        "PerformanceCategory": categories
    })
    
    # Introduce some missing values for realistic cleaning demo
    df.loc[np.random.choice(df.index, 10), "StudyHours"] = np.nan
    df.loc[np.random.choice(df.index, 10), "Attendance"] = np.nan
    return df

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom dataset loaded successfully!")
else:
    df_raw = generate_sample_data()
    st.sidebar.info("Using built-in sample dataset.")

# ---------------------------------------------------------
# Module 1: Data Cleaning & Preprocessing
# ---------------------------------------------------------
st.header("1. Data Preprocessing")

df = df_raw.copy()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(), use_container_width=True)

with col2:
    st.subheader("Missing Values Check")
    missing_info = df.isnull().sum()
    st.dataframe(missing_info[missing_info > 0], use_container_width=True)

# Impute missing numerical values with median
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Impute missing categorical values with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

st.success("Missing and inconsistent values imputed successfully!")

# ---------------------------------------------------------
# Module 2: Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
st.header("2. Exploratory Data Analysis")

eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    st.subheader("Target Distribution")
    target_col = st.selectbox("Select Target Variable", options=df.columns, index=len(df.columns)-1)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    df[target_col].value_counts().plot(kind='bar', ax=ax, color=['#4C72B0', '#55A868', '#C44E52'])
    plt.title(f"Distribution of {target_col}")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    st.pyplot(fig)

with eda_col2:
    st.subheader("Feature vs Performance")
    numeric_features = [c for c in num_cols if c != target_col]
    selected_feature = st.selectbox("Select Numerical Feature", options=numeric_features)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    for category in df[target_col].unique():
        subset = df[df[target_col] == category]
        ax.hist(subset[selected_feature], alpha=0.5, label=str(category), bins=15)
    plt.title(f"{selected_feature} by {target_col}")
    plt.xlabel(selected_feature)
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(fig)

# ---------------------------------------------------------
# Module 3: Model Training & Evaluation
# ---------------------------------------------------------
st.header("3. Machine Learning Model Training")

# Feature Matrix & Target Vector Setup
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode Categorical Features
encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Fitting
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Evaluation Metrics")
metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.metric(label="Model Accuracy", value=f"{acc * 100:.2f}%")
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred))

with metric_col2:
    st.subheader("Feature Importance")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    importances.plot(kind='barh', ax=ax, color='#4C72B0')
    plt.title("Random Forest Feature Importances")
    st.pyplot(fig)

# ---------------------------------------------------------
# Module 4: Real-time Student Performance Prediction
# ---------------------------------------------------------
st.header("4. Predict Performance for a New Student")

st.markdown("Adjust the feature values below to get an instant performance prediction:")

user_inputs = {}
input_cols = st.columns(len(X.columns))

for i, col in enumerate(X.columns):
    with input_cols[i % len(input_cols)]:
        if col in encoders:
            # Handle categorical inputs
            options = list(encoders[col].classes_)
            selected = st.selectbox(f"{col}", options=options)
            user_inputs[col] = encoders[col].transform([selected])[0]
        else:
            # Handle numerical inputs
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            mean_val = float(X[col].mean())
            user_inputs[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

input_df = pd.DataFrame([user_inputs])

if st.button("Predict Performance Category", type="primary"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    st.markdown(f"### Predicted Performance Category: **{prediction}**")
    
    # Display Prediction Probabilities
    prob_df = pd.DataFrame({
        'Category': model.classes_,
        'Probability': probabilities
    }).sort_values(by='Probability', ascending=False)
    
    st.dataframe(prob_df, hide_index=True)