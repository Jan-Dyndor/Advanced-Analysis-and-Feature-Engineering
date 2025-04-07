# 📊 Advanced Analysis and Feature Engineering

## 🧾 Project Description

This project focuses on advanced exploratory data analysis (EDA), feature selection, and engineering techniques using a tabular dataset. The objective was to identify and understand relationships between variables, engineer meaningful new features, and evaluate how these transformations affect model performance. 

In this project, I compared the impact of various feature engineering techniques on model performance across different algorithm types. These included shallow learning algorithms such as XGBoost, a simple deep learning model, and an ensemble method using Stacking Regressor (blender). The goal was to evaluate how preprocessing and feature transformations influence predictive power in both traditional and modern machine learning pipelines.

The notebook walks through a complete data science workflow — from raw data cleaning and inspection to evaluation of machine learning models trained on different feature engineering steps.
---

## 🎯 Objective

The main goals of this project are:

- 🔍 Perform deep exploratory data analysis to understand the data distribution and interdependencies.
- 🛠️ Apply feature engineering techniques such as:
  - Encoding categorical variables
  - Creating interaction features
  - Transforming numerical variables
- 🧪 Evaluate model performance on two approaches to data engineering.
- 📈 Compare the impact of engineered features using RMSE metric.
- Compare shallow learning algorithms to deep learning
  
Finally compare this notebook with deeper data underestaning and data engineering to my previous **[notebook](https://www.kaggle.com/code/jandyndor/house-prices-advanced-regression-techniques#Predicting-the-prices-of-a-House-based-on-the-given-features)** done on the same dataset, to compare the results.

---

## 📁 Dataset

- **Type:** Tabular data with a mix of numerical and categorical features
- **Source:** Public dataset on [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Preprocessing:**
  - Handling missing values
  - Creating missing indicators
  - Ordinal encoding and one-hot encoding
  - Variable scaling and transformation
  - Visualization of distributions and feature interactions
  - Gathering preprocessing steps into Pipeline

---

## 🤖 Models and Techniques

- Logistic Regression
- Random Forest
- Elastic Net
- Simple Deep Learning model
- XGBoost
- GridSearchCV and RandomizedSearchCV
- **StackingRegressor with Random Forest and XGBoost**

---

## 🧠 Key Insights

- Feature engineering has significant impact on  model performance.
- More features do not always lead to better performance.
- StackingRegressor is powerful tool
---

## 🛠️ Tools and Libraries

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- PyTorch

---

## 📊 Results

| Model                | RMSE | 
|---------------------|-------------------|
| XBGoost (previous notebook)       | 0.13717             | 
| XGBoost (orginal  preprocessing)            | 0.366243        | 
| Deep Learning (orginal preprocessing)      | 0.31196              | 
| XGBoost ("clever"  preprocessing)      |  0.14302              | 
| StackingRegressor  (orginal  preprocessing)      |   0.13332              | 



---

## 🙋‍♂️ Author

**Jan Dyndor**  
ML Engineer & Pharmacist  
📧 dyndorjan@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/jan-dyndor-156101322/)  
📊 [Kaggle](https://www.kaggle.com/jandyndor)

---

## 🧠 Keywords

EDA, feature engineering, machine learning, SHAP, logistic regression, random forest, model evaluation, interpretability, tabular data
