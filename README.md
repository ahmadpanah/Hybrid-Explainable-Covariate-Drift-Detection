# Hybrid Explainable Covariate Drift Detection Framework  

This repository implements a **Hybrid Explainable Covariate Drift Detection Framework** for serverless environments. It combines statistical methods, machine learning models, and explainable AI (XAI) tools like SHAP and LIME to detect and explain covariate drift in real-time. Designed for scalability and transparency, it supports datasets like PaySim, Credit Card Fraud, and IoT-23.  

---

## Features  

- **Hybrid Drift Detection**: Combines statistical tests (Kolmogorov-Smirnov, Chi-Square) and ML models (Isolation Forest).  
- **Explainability**: Uses SHAP for global feature importance and LIME for local interpretability.  
- **Modular Design**: Separates data ingestion, drift detection, explanation, and notification layers.  
- **Serverless Ready**: Deployable in AWS Lambda for real-time, cost-efficient drift detection.  
- **Multi-Dataset Support**: Works with PaySim, Credit Card Fraud, and IoT-23 datasets.  

---

## Installation  

1. Clone the repository:  
   `git clone https://github.com/ahmadpanah/hybrid-drift-detection.git`  
   `cd hybrid-drift-detection`  

2. Install dependencies:  
   `pip install -r requirements.txt`  

3. Download datasets:  
   - [PaySim](https://www.kaggle.com/ealaxi/paysim1)  
   - [Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
   - [IoT-23](https://www.stratosphereips.org/datasets-iot23)  
   Place them in the `data/` directory.  

---

## Usage  

### 1. Local Execution  
Run the framework on a dataset:  
`python src/main.py --dataset CreditCard`  (Options: CreditCard, PaySim, IoT-23)  

### 2. Serverless Deployment  
1. Package the code:  
   `zip -r lambda_package.zip .`  
2. Upload to AWS Lambda and configure the trigger.  

---

## Framework Components  

1. **Data Ingestion Layer**: Loads and preprocesses datasets.  
2. **Drift Detection Service**: Combines statistical and ML-based drift detection.  
3. **Explanation Service**: Provides explainability using SHAP and LIME.  
4. **Notification and Action Layer**: Generates alerts and triggers actions based on drift severity.  

---

## Serverless Integration  

The `lambda_handler.py` script enables deployment in AWS Lambda. It processes incoming events, performs drift detection, and returns results.  

### Example Lambda Event:  
```json  
{  
  "dataset": "CreditCard"  
}
```

### Example Output:

```json 
{  
  "statusCode": 200,  
  "body": {  
    "message": "Drift detection completed successfully.",  
    "hybrid_score": 0.85,  
    "statistical_results": {  
      "feature1": {"KS Statistic": 0.12, "KS p-value": 0.05},  
      "feature2": {"Chi-Square Statistic": 15.3, "Chi-Square p-value": 0.01}  
    },  
    "ml_drift_score": 0.78,  
    "alert_level": "Critical"  
  }  
}  
```


