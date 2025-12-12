# Comparative Analysis of Classification Models — ML\_PROJECT\_2

**Authors:** Vivek Joshi (MT2025133)  
**Course:** Machine Learning (AIT 511)  
**Institution:** IIIT Bangalore

* * *

## 🚀 Project Overview

This project presents a comparative evaluation of machine learning models for two real-world classification tasks:

1.  **Smoker Status Prediction** (Binary classification)
    
2.  **Forest Cover Type Classification** (Multiclass classification)
    

Both classical ML models and deep neural networks were trained and compared on the basis of accuracy, computational efficiency, and behavior under data imbalance.

* * *

## 📊 Key Results

### **Smoker Dataset (38,984 samples)**

-   Best Model: **Neural Network (MLP)**
    
-   **Accuracy:** **75.41%**
    

### **Forest Cover Dataset (581,012 samples, 7 classes)**

-   Best Model: **Deep Neural Network (MLP)**
    
-   **Accuracy:** **91.03%**
    

Linear baselines (Logistic Regression, Linear SVM) achieved ~71–73%.

* * *

## 🗂️ Datasets

**1\. Smoker Status (Bio-Signals)**  
[https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals](https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals)

**2\. Forest Cover Type (UCI/Kaggle)**  
[https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset)

* * *

## 🧹 Preprocessing Summary

-   Removed duplicates and outliers (IQR-based capping).
    
-   Imputation: median (continuous), mode (categorical).
    
-   Feature selection to remove multicollinearity:
    
    -   Removed **Cholesterol** (high correlation with LDL)
        
    -   Removed **waist(cm)** (high correlation with weight)
        
-   Scaling: StandardScaler for continuous features.
    
-   Forest dataset: applied **class weights** to handle imbalance.
    

* * *

## 🤖 Models & Hyperparameters

### **Smoker Dataset**

-   Logistic Regression (`C=1.0`, `lbfgs`)
    
-   Linear SVM (`C=1.0`)
    
-   RBF SVM (`C=1.0`, `gamma=scale`)
    
-   **Neural Network (MLP)**:
    
    -   Layers: `[128, 32]`
        
    -   Activation: ReLU
        
    -   Dropout: 0.30
        
    -   Optimizer: Adam (lr=1e-3)
        
    -   Epochs: 50
        

### **Forest Cover Dataset**

-   Logistic Regression (`multinomial`, `max_iter=2000`)
    
-   Linear SVM (`C=10.0`, `dual=False`)
    
-   **Deep MLP**:
    
    -   Layers: `[1024, 512, 256]`
        
    -   Activation: ReLU
        
    -   Dropout: 0.15
        
    -   L2 Regularization: `1e-5`
        
    -   BatchNorm: Yes
        
    -   Optimizer: Adam (lr=3e-4, cosine decay)
        
    -   Batch size: 1024
        

* * *

## 📈 Evaluation Metrics

-   Accuracy
    
-   Precision / Recall / F1-score
    
-   Confusion Matrix
    
-   ROC-AUC (Smoker dataset)
    

All metrics computed using scikit-learn.

* * *

## ▶️ How to Run

**1\. Clone the repo**

`git clone https://github.com/vivekvj18/ML-PROJECT-2.git cd ML-PROJECT-2`

**2\. Install dependencies**

`pip install -r requirements.txt`

**3\. Open the notebooks**

-   `ML_2_Smoking.ipynb`
    
-   `ML_2_Logistic_linearSVM_Forest.ipynb`
    
-   `ML_2_Neural_Forest.ipynb`
    

**4\. Download datasets** into `data/` as instructed above.

* * *

## 🔮 Future Work

-   Use Bayesian Optimization (Optuna) for hyperparameter tuning.
    
-   Try ensemble methods combining tree models + neural nets.
    
-   Use SHAP/LIME to interpret feature contributions.
    
-   Engineer more domain features (e.g., BMI, interaction terms).
    

* * *

## 📜 License

MIT License — see `LICENSE` file.

* * *

## 👨‍💻 Contact

**Vivek Joshi**  
GitHub: [https://github.com/vivekvj18](https://github.com/vivekvj18)

* * *

⭐ _If you found this project useful, consider giving the repository a star!_ ⭐
