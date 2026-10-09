# 🎓 Student Performance Prediction System

An end-to-end Machine Learning web application built with Python and Streamlit that uses student academic, study, and lifestyle data to analyze patterns and predict performance categories (`High`, `Medium`, `Low`).

---

## 📌 Features

* **📁 Dynamic Data Source:** Upload your own CSV dataset or automatically generate synthetic student data for demonstration.
* **🧹 Automated Preprocessing:** Handles missing values dynamically using statistical imputation (median for numerical variables, mode for categorical variables).
* **📊 Exploratory Data Analysis (EDA):** Interactive data visualisations built with Matplotlib, including target variable distributions and feature distribution comparisons.
* **🤖 Machine Learning Model:** Trains a Random Forest Classifier with automatic feature encoding and displays classification metrics and feature importances.
* **⚡ Live Interactive Predictions:** Adjustable UI sliders and dropdowns to predict performance categories and view probabilities for new student data in real time.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Web Framework:** [Streamlit](https://streamlit.io/)
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Data Visualization:** Matplotlib

---

## 📁 Project Structure

```text
student_performance_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation