```markdown
# Credit Card Default Prediction

## Overview
This project builds machine learning models to predict whether a credit card client will default on their payment the next month. It uses the “Default of Credit Card Clients” dataset from Taiwan, and includes steps from data preprocessing through model training and evaluation.

## Repository Structure
```

Credit-Card-Default-Prediction/
│
├── data/
│   └── UCI\_Credit\_Card.csv          # Dataset used for modeling
│
├── notebooks/
│   └── EDA\_and\_Modeling.ipynb        # Exploratory data analysis and initial modeling
│
├── models/
│   └── LogR\_Model.pkl                # Saved logistic regression model
│
├── app/
│   └── main.py                       # Script for running predictions / deployment
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation

````

## Dataset
- Source: UCI “Default of Credit Card Clients” dataset  
- Features include demographic factors (e.g. age, sex, education, marriage), credit limit, past repayment status for six months, bill amounts, and payment amounts.  
- Target: whether the client will default next month (binary: 1 = default, 0 = non-default).

## Preprocessing
- Clean missing or inconsistent data  
- Encode categorical variables  
- Split into training and testing sets  
- Feature scaling as needed

## Models & Evaluation
- Logistic Regression  
- Other baseline/classification models  
- Evaluation metrics include accuracy, precision, recall, F1-score, ROC-AUC  
- The logistic regression model is saved and used in deployment

## Usage

1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
````

2. Explore data and train models via the notebook in `notebooks/`.

3. Use the saved model in `models/` to make predictions via `app/main.py`.

## Results

* The logistic regression model achieved balanced performance across precision, recall, and F1-score.
* ROC-AUC indicates ability to distinguish between defaulters and non-defaulters.

## Future Work

* Handling class imbalance more robustly
* Deploying a user interface for interactive predictions

## Requirements

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Other libraries as listed in `requirements.txt`

```
::contentReference[oaicite:0]{index=0}
```
