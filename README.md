# 🧠 Customer Churn Prediction with Explainable AI & Carbon Impact Monitoring

This project predicts **customer churn** using machine learning models and explains predictions through **Explainable AI (XAI)** techniques.  
It also integrates **carbon footprint tracking** to align with **Responsible AI** principles — ensuring that AI systems are **transparent, fair, and environmentally responsible**.

---

## 🎯 Project Objective

- **Primary Goal:**  
  Predict whether a customer will **churn** (leave the service) and **explain** the reasoning behind each prediction in human-understandable terms.

- **Business Value:**  
  Understanding *why* customers churn helps businesses improve retention strategies, optimize pricing, and design targeted offers.

- **Responsible AI Focus:**  
  - Ensure transparency and interpretability in model decisions.  
  - Evaluate and track the **environmental impact** of model training.  
  - Support compliance with frameworks like the **EU AI Act** and **GDPR**.  

---

## 🛠 Environment, Models & Techniques  

- **Environment:**  
  - Python **3.12.3**  
  - [CodeCarbon 3.0.1](https://mlco2.github.io/codecarbon/) for carbon footprint monitoring  

- **Models Used:**  
  - Multi-Layer Perceptron (**MLP**)  
  - **Random Forest** Classifier (with class-weighting for imbalance)  

- **Explainability Tools:**  
  - **Global Explanations:**  
    - Permutation Feature Importance  
    - SHAP Summary Plots  
    - Partial Dependence Plots (PDPs)  
  - **Local Explanations:**  
    - SHAP Waterfall Plots  
    - LIME Local Explanations  
    - Gauge Indicators (customer-level churn risk visualization)  
  - **Interactive Dashboards:**  
    - [Shapash](https://github.com/MAIF/shapash) for interactive XAI reports  

- **Carbon Footprint Monitoring:**  
  - [CodeCarbon](https://mlco2.github.io/codecarbon/) to measure CO₂ emissions during model training.  
  - Power usage breakdown (CPU, GPU, RAM) for each training run.  

---

## 📊 Dataset

**Source:** [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  

- **Numerical Features:** `tenure`, `MonthlyCharges`, `TotalCharges`  
- **Categorical Features:** `Contract`, `InternetService`, `PaymentMethod`, `TechSupport`, etc.  
- **Target Variable:** `Churn` (Yes/No)  
- **Challenges Addressed:**  
  - Handling missing values in `TotalCharges`  
  - Encoding multiple categorical variables  
  - Managing class imbalance with weighted models  

---

## 🌍 Responsible AI & Sustainability

This project aligns with ENGIE’s **Responsible AI** strategy:  
- **Explainability** — making model predictions interpretable for all stakeholders.  
- **Fairness** — reducing bias by examining feature impact.  
- **Transparency** — providing full visibility into model decision logic.  
- **Sustainability** — measuring and reducing the carbon footprint of AI.  

By combining **XAI** with **carbon tracking**, we ensure that our models are not only *accurate and fair*, but also *energy-conscious*.  

---

## 📚 References

- [SHAP Documentation](https://shap.readthedocs.io)  
- [Lundberg & Lee (2017) – SHAP Paper](https://arxiv.org/abs/1705.07874)  
- [ELI5 Documentation](https://eli5.readthedocs.io)  
- [scikit-learn Documentation](https://scikit-learn.org)  
- [PDPbox GitHub](https://github.com/SauceCat/PDPbox)  
- [CodeCarbon Documentation](https://mlco2.github.io/codecarbon/)  

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone <repo_url>
   cd <repo_folder>
2. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```