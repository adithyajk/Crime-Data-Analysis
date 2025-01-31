
# 🚔 Crime Data Analysis  

## 📌 Overview  
Crime Data Analysis is a project aimed at predicting crime patterns using historical data. By leveraging structured data analysis techniques, we provide insights into crime trends, enabling law enforcement agencies to make informed decisions.  

## 📊 Project Workflow  
1. **Dataset Collection & Inspection**  
   - Used a structured crime dataset from Kaggle (22 columns, 20,000 rows).  
   - Identified missing values and data inconsistencies.  

2. **Exploratory Data Analysis (EDA)**  
   - **Univariate Analysis**: Analyzed individual columns using histograms and box plots.  
   - **Bivariate Analysis**: Explored relationships between two columns via heatmaps and chi-square tests.  
   - **Multivariate Analysis**: Correlation study using pair plots and heatmaps.  

3. **Data Cleaning & Preprocessing**  
   - Handled missing values and removed non-essential columns.  
   - Standardized categorical values (e.g., label encoding for `Victim_Sex`).  
   - Scaled numerical data using Min-Max, Standardization, and Robust Scaling.  

4. **Dataset Splitting & Normalization**  
   - Split dataset into **80% training** and **20% testing** for model development.  
   - Normalized data for improved model accuracy.  

5. **Model Building**  
   - **Logistic Regression**: For understanding relationships between features and crime categories.  
   - **Decision Tree**: For explainability and feature importance analysis.  
   - Future scope includes advanced models for better accuracy.  

## 🎯 Goals  
- Predict crime categories based on historical data.  
- Identify recurring crime patterns.  
- Improve investigation efficiency through data-driven insights.  

## 🔧 Technologies Used  
- **Python**: Data processing and model training.  
- **Pandas, NumPy**: Data manipulation.  
- **Matplotlib, Seaborn**: Data visualization.  
- **Scikit-learn**: Machine learning model implementation.  

