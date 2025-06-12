# Credit Card Fraud Detection

This project uses machine learning to detect fraudulent credit card transactions. The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Project Overview
- Preprocessed and scaled features for balanced model input.
- Trained a Random Forest Classifier to detect fraud.
- Evaluated model performance using:
  - Confusion Matrix
  - Feature Importance
  - ROC Curve & AUC Score
- Performed cross-validation for robust performance estimation.

## Modules Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib

## Dataset
Due to size limitations, the dataset is not included in this repository but can be downloaded from [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Key Visualizations
### Confusion Matrix
![Confusion_matrix](https://github.com/user-attachments/assets/869bad60-f319-4e71-afab-e9eb173ae6a6)

### Feature Correleation
![feature_correlation](https://github.com/user-attachments/assets/0f47f1df-19ec-4518-8a7d-20a792a29e47)

### Feature Importance
![importance_score](https://github.com/user-attachments/assets/5878bd0e-50f7-45e9-ba30-fe59d5463531)

### ROC Curve
![roc_curve](https://github.com/user-attachments/assets/321351b3-d85a-4a7f-bfca-9e60c6d50209)

## How to Run
1. Clone the repo.
2. Install dependencies:
```bash
pip install -r requirements.txt
