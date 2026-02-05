# 🩺 Pima Indians Diabetes Prediction

An end-to-end **Machine Learning classification project** that predicts whether a patient has diabetes based on medical diagnostic features.  
This project is built using **real-world data from Kaggle** and follows industry-standard ML practices.

---

## 📌 Project Overview

Diabetes is a chronic disease that affects millions of people worldwide.  
This project uses the **Pima Indians Diabetes Dataset** to build a machine learning model that predicts the likelihood of diabetes in a patient based on health metrics such as glucose level, BMI, age, insulin, and blood pressure.

The project demonstrates a **complete ML workflow** from data preprocessing to model saving and prediction, making it suitable for learning, interviews, and deployment.

---

## 🎯 Objectives

- Understand and preprocess real medical data  
- Handle invalid values and missing data  
- Detect and treat outliers using statistical methods  
- Build a clean and reusable ML pipeline  
- Train and evaluate multiple ML models  
- Save and load the trained model for future use  

---

## 📊 Dataset Information

- **Dataset Name:** Pima Indians Diabetes Database  
- **Source:** Kaggle  
- **Type:** Binary Classification  

### Target Variable
| Value | Meaning |
|-----|--------|
| 0 | No Diabetes |
| 1 | Diabetes |

### Features
- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  

---

## 🧹 Data Preprocessing

- Replaced invalid `0` values in medical features with median values  
- Handled outliers using the **IQR (Interquartile Range)** method  
- Applied feature scaling using **StandardScaler**  
- Prevented data leakage using **scikit-learn Pipeline**

---

## 🧠 Machine Learning Models Used

- Logistic Regression (Baseline)  
- Support Vector Machine (SVM)  
- Random Forest Classifier  
- Gradient Boosting Classifier  

---

## 🛠️ Technologies & Libraries

- **Python**
- **Pandas** – Data manipulation  
- **NumPy** – Numerical computing  
- **Matplotlib** – Data visualization  
- **Scikit-learn** – ML models & pipelines  
- **Joblib / Pickle** – Model persistence  

---

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy Score  
- Confusion Matrix  
- Classification Report  
- ROC Curve  
- Precision-Recall Curve  

The Logistic Regression pipeline provided a strong and interpretable baseline, while ensemble models improved performance.

---

## 🗂️ Project Structure

