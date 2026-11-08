# bankruptcy_prediction: Multi-Model ML Classification Dashboard

## a. Problem statement
Corporate bankruptcy prediction is an essential operational task in credit risk rating, underwriting, and capital allocation. The objective of this project is to implement, evaluate, and benchmark five classical machine learning classification algorithms to predict corporate failure using continuous financial ratios. The project delivers an interactive Streamlit application deployed on the cloud to evaluate model generalizability on unseen test data without retraining.

## b. Dataset description
- **Source:** Kaggle (Taiwan Economic Journal Company Bankruptcy Dataset)
- **Target Variable:** `Bankrupt?` (Binary: `1` for Bankrupt, `0` for Non-Bankrupt)
- **Instances:** 6,819 rows
- **Features:** 95 numeric continuous financial ratios (Return on Assets, Net Worth Growth, Total Debt/Net Worth, Quick Ratio, Operating Margins, etc.)
- **Class Distribution:** Imbalanced (~96.8% Non-Bankrupt [0], ~3.2% Bankrupt [1])

## c. Github Repository Link
https://github.com/prakharraj29/bankruptcy_prediction

## d. Models used:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8783 | 0.9171 | 0.9671 | 0.8783 | 0.9130 | 0.3533 |
| **Decision Tree** | 0.9164 | 0.8273 | 0.9660 | 0.9164 | 0.9361 | 0.3859 |
| **KNN** | 0.9707 | 0.7401 | 0.9634 | 0.9707 | 0.9633 | 0.3519 |
| **Naive Bayes** | 0.2903 | 0.7487 | 0.9530 | 0.2903 | 0.4138 | 0.0539 |
| **Support Vector Machine** | 0.8996 | 0.9174 | 0.9611 | 0.8996 | 0.9248 | 0.3085 |

*(Note: Calculated on the 20% stratified test split stored in `test_data.csv`.)*

### Observations on model performance:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | By applying balanced class weights, Logistic Regression functions as a strong linear baseline, achieving a near-top AUC of 0.9171 while balancing minority detection against an acceptable volume of false alarms. |
| **Decision Tree** | Segmenting orthogonal non-linear thresholds across solvency and liquidity ratios proved effective. With a constrained maximum depth, it achieved the highest balanced correlation score (MCC = 0.3859) and a solid F1 of 0.9361 without runaway overfitting. |
| **KNN** | Shows inflated raw accuracy (0.9707) by defaulting heavily toward the majority non-bankrupt class. In a 95-dimensional ratio space, Euclidean distance suffers from the curse of dimensionality, pulling down its ranking capability (AUC = 0.7401). |
| **Naive Bayes** | Struggles severely on overall accuracy (0.2903) and MCC (0.0539). Gaussian Naive Bayes assumes independent features, which collapses under the heavy collinearity present among corporate accounting ratios. |
| **Support Vector Machine** | The RBF kernel captures non-linear decision boundaries effectively in the 95-dimensional feature space, securing the highest overall discriminatory ranking score (AUC = 0.9174) with a strong F1 of 0.9248. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** is the overall winner for discriminatory capability (highest AUC = 0.9174), with **Decision Tree** serving as the best balanced-threshold classifier (highest MCC = 0.3859). |
