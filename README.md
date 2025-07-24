# **Customer Churn Prediction in Telecom - README**  

## **📌 Introduction**  
Customer churn (the loss of customers) is a critical challenge in the telecom industry, where acquiring new customers is often more expensive than retaining existing ones. This project leverages machine learning to predict which customers are likely to churn, allowing telecom companies to take proactive retention measures.  

Using Python and machine learning techniques, I built a predictive model that identifies at-risk customers based on their service usage, contract details, and payment behavior. The goal is to help businesses reduce churn rates and improve customer retention strategies.  

---  

## **📌 Background**  
Customer churn is a major concern for telecom companies due to high competition and customer switching costs. By analyzing historical customer data, we can:  
- **Identify patterns** leading to churn  
- **Predict high-risk customers** before they leave  
- **Recommend targeted retention strategies**  

This project uses the **Telco Customer Churn dataset** from Kaggle, containing 7,043 customer records with features like:  
- **Contract type** (month-to-month, yearly)  
- **Payment method** (electronic check, credit card)  
- **Internet service type** (DSL, fiber optic)  
- **Tenure** (how long the customer has been with the company)  

---  

## **🛠 Tools & Technologies Used**  
- **Python** (Primary programming language)  
- **Pandas & NumPy** (Data manipulation)  
- **Scikit-learn** (Machine learning models)  
- **Matplotlib & Seaborn** (Data visualization)  
- **Imbalanced-learn** (Handling class imbalance)  
- **Jupyter Notebook** (Interactive analysis)  

---  

## **📊 Analysis & Key Findings**  

### **1. Data Exploration & Cleaning**  
- **Missing Values:** Fixed `TotalCharges` by filling missing values with the median.  
- **Class Imbalance:** Original dataset had **73% non-churners vs. 27% churners**, requiring balancing techniques.  

### **2. Feature Engineering**  
Created new features to improve model performance:  
- **AvgMonthlyCharge** = `TotalCharges / tenure`  
- **NumServices** = Count of subscribed services  
- **TenureGroup** = Binned tenure into categories (e.g., 0-1yr, 1-2yr)  

### **3. Model Training & Optimization**  
- **Baseline Model (Random Forest):**  
  - **Accuracy:** 79.9%  
  - **Recall (Churn Detection):** 45% (too low)  

- **After Optimization (SMOTE + Hyperparameter Tuning):**  
  - **Best Model:** Random Forest (optimized for **recall**)  
  - **Recall Improved to 80.2%** (captures more churners)  
  - **Precision:** 47.9% (trade-off for higher recall)  
  - **ROC-AUC:** 0.837 (good discriminatory power)  

### **4. Key Business Insights**  
- **Highest Churn Factors:**  
  - **Contract Type:** Month-to-month customers churn **3× more** than yearly contracts.  
  - **Payment Method:** Electronic check users churn at **45%**, vs. 15% for credit card users.  
  - **Internet Service:** Fiber optic users churn at **42%**, vs. 19% for DSL.  

- **Top Predictive Features:**  
  1. **Tenure** (how long a customer stays)  
  2. **Fiber Optic Internet** (high churn correlation)  
  3. **Electronic Check Payments** (risk factor)  

---  

## **📖 What I Learned**  
✅ **Handling Class Imbalance:** SMOTE and undersampling improved recall but didn’t always boost accuracy.  
✅ **Feature Engineering Matters:** Derived features (`AvgMonthlyCharge`, `NumServices`) improved predictions.  
✅ **Recall vs. Precision Trade-off:** Optimizing for recall captures more churners, even if some false alarms occur.  
✅ **Business Impact:** The model helps prioritize retention efforts on high-risk customers.  

---  

## **🎯 Conclusion & Next Steps**  
This project successfully predicts telecom customer churn with **80% recall**, allowing businesses to intervene before customers leave.  

