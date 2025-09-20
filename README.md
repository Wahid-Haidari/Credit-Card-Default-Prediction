# Credit Card Default Prediction

## Overview
This project applies supervised learning algorithms to predict the likelihood of credit card clients defaulting on payments in the following month. Using the UCI “Default of Credit Card Clients” dataset, we implement, compare, and analyze four models:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Neural Networks

The goal is to identify the best-performing model for credit risk prediction, a problem of high importance to banks and financial institutions.

---

## Repository Structure
```
Credit-Card-Default-Prediction/
│
├── Decision-Tree.py                 # Decision Tree implementation
├── Logistic-Regression.py            # Logistic Regression implementation
├── Neural_network1.py                # Simple Neural Network implementation
├── randomForest.py                   # Random Forest implementation
│
├── Supervised Learning/              # Supporting resources
│   ├── *.sty                         # LaTeX style files
│   ├── *.png                         # Model visualization figures
│   ├── ProjectProposal.tex           # Project writeup in LaTeX
│   ├── refs.bib                      # References for report
│
├── SL Project Checkpoint.pdf         # Mid-project checkpoint report
├── Supervised_Learning.pdf           # Final project report
└── README.md                         # Project documentation
```

---

## Dataset
- **Source:** [UCI Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
- **Instances:** 30,000  
- **Attributes:** 24 (demographics, credit limit, repayment status, bill amount, prior payments, etc.)  
- **Target:** Whether the client defaulted on payment the following month (binary classification)

---

## Models and Results
### Logistic Regression
- Accuracy: 0.78  
- Pros: Fast, interpretable  
- Cons: Lower accuracy compared to ensemble methods

### Decision Tree
- Validation Accuracy: 0.842  
- Test Accuracy: 0.832  
- Pros: High interpretability  
- Cons: Risk of overfitting without proper tuning

### Random Forest
- Accuracy: 0.82  
- Configuration: 7 trees, max depth = 3  
- Strengths: Robust against overfitting, models complex feature interactions effectively

### Neural Network
- One hidden layer, sigmoid activation  
- Cross-validation MSE: 0.356 ± 0.161  
- Test Set MSE: 0.25  
- Strengths: Handles complex non-linear patterns  
- Limitations: Sensitive to overfitting, requires regularization

---

## Hypotheses and Findings
1. Younger clients are more likely to default  
   - Confirmed but effect size small (default rate: 22.84% under 30 vs 21.78% over 30, p = 0.0383).  

2. Random Forest will outperform Logistic Regression  
   - Confirmed (Random Forest: 0.82 vs Logistic Regression: 0.78).  

3. Sex, education, and marital status are weak predictors  
   - Confirmed; accuracy remained strong after removing these features.  

4. Neural Networks outperform traditional models  
   - Partially supported; lower error achieved, but risk of overfitting observed.  

---

## Learning Curves
- Logistic Regression: Stable, capped near 0.78  
- Random Forest: Gradual improvement, peaked at 0.82  
- Decision Tree: Strong early performance near 0.83  
- Neural Network: Variable performance, prone to overfitting  

---

## Getting Started

### Clone Repository
```bash
git clone https://github.com/Wahid-Haidari/Credit-Card-Default-Prediction.git
cd Credit-Card-Default-Prediction
```

### Install Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Run Models
Example:
```bash
python Decision-Tree.py
```

---

## References
- Yeh, I. C. (2016). *Default of Credit Card Clients Dataset*. UCI ML Repository.  
- Hassan, M. M., & Mirza, T. (2020). *Credit Card Default Prediction Using Artificial Neural Networks*.  
- Sayjadah, Y., Hashem, I. A. T., Alotaibi, F., & Kasmiran, K. A. (2018). *Credit Card Default Prediction Using Machine Learning Techniques*.  

---

## Future Work
- Feature engineering to extract more informative attributes  
- Hyperparameter tuning, especially for Random Forest and Neural Networks  
- Real-time prediction system for financial institutions  
