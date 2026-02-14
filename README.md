# A Comparison study on Classification Models (Machine Learning Assignment 2)

## 1. Problem Statement

The objective of this assignment is to design, implement, and evaluate multiple machine learning classification models on a real-world dataset. The assignment covers the complete machine learning workflow, including dataset selection, model training, evaluation using standard metrics, development of an interactive Streamlit web application, and deployment on Streamlit Community Cloud.

Six different classification models are implemented and evaluated on the same dataset. Their performance is also compared using multiple evaluation metrics to analyze strengths and limitations of each implementation.

---

## 2. Dataset Description

The **Human Activity Recognition (HAR)** [dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) from the UCI Machine Learning Repository is used for this assignment. 

The dataset was collected using accelerometer and gyroscope sensors embedded in smartphones, with the goal of classifying human activities based on sensor signal patterns.

### Dataset Characteristics
- **Problem Type**: Multi-class classification
- **Number of Classes**: 6
  - Walking
  - Walking Upstairs
  - Walking Downstairs
  - Sitting
  - Standing
  - Laying
- **Number of Features**: 561 numerical features
- **Number of Instances**: 10,299 samples

The original dataset is provided in separate training and test sections. For this assignment, the data was merged and converted into a single CSV file to enable unified model training & evaluation.

---

## 3. Models Used and Evaluation Metrics

All models were trained and evaluated on the same dataset using a **stratified 80/20 train–test split**, ensuring balanced representation of all activity classes in both training and test sets.

### Classification Models Implemented
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN) Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest Classifier (Ensemble)
6. XGBoost Classifier (Ensemble)

### Evaluation Metrics
The following evaluation metrics were computed for each model:
- Accuracy
- Area Under the ROC Curve (AUC)
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1 Score (macro-averaged)
- Matthews Correlation Coefficient (MCC)

All metrics were computed **offline during model training** and stored in [`metrics.csv`](./results/metrics.csv). These values serve as the authoritative results for model comparison and reporting.

---

## 4. Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|---------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.984951 | 0.999576 | 0.985757 | 0.985922 | 0.985823 | 0.981905 |
| Decision Tree | 0.938350 | 0.962036 | 0.937282 | 0.936378 | 0.936707 | 0.925866 |
| KNN | 0.963592 | 0.996169 | 0.965831 | 0.965051 | 0.965373 | 0.956226 |
| Naive Bayes | 0.754854 | 0.962115 | 0.781261 | 0.758661 | 0.748912 | 0.715104 |
| Random Forest (Ensemble) | 0.981553 | 0.999426 | 0.981724 | 0.981411 | 0.981527 | 0.977822 |
| XGBoost (Ensemble) | **0.992233** | **0.999878** | **0.992413** | **0.992598** | **0.992499** | **0.990660** |

---

## 5. Model-wise Observations

| Model | Observation |
|------|-------------|
| Logistic Regression | Achieved strong performance due to the high-dimensional and well-engineered feature space, which exhibits near-linear separability for several activity classes. |
| Decision Tree | Demonstrated reasonable performance but showed lower generalization compared to ensemble methods, likely due to overfitting on complex feature interactions. |
| KNN | Performed strongly due to dense numeric feature representation, though computational cost increases with dataset size. |
| Naive Bayes | Showed comparatively lower performance due to the strong feature independence assumption, which is violated in correlated sensor data. |
| Random Forest (Ensemble) | Achieved high accuracy and robustness by aggregating multiple decision trees, reducing variance and improving generalization. |
| XGBoost (Ensemble) | Delivered the best overall performance across all metrics, benefiting from gradient boosting, regularization, and effective modeling of complex non-linear patterns. |

---

## 6. Streamlit Application & Deployment

An interactive Streamlit web application was developed and deployed using **Streamlit Community Cloud**.

### Features of the Streamlit App
- Upload of a CSV test dataset (Quick download of a small test dataset is enabled)
- Model selection dropdown to choose among the six trained models
- Display of offline evaluation metrics for all models
- Class-wise performance table showing Precision, Recall, and F1-score
- Confusion matrix for the selected model
- Class-wise metric heatmap for multi-class performance visualization

### Important Note on Metrics
- Overall Model comparison metrics are computed during training and loaded from [`metrics.csv`](./results/metrics.csv) for visualization. No model training is performed inside the Streamlit application.
- Uploaded dataset evaluation is performed on the uploaded test data only, for demonstration purposes.


The Streamlit application was deployed on **Streamlit Community Cloud** using the project GitHub repository. The deployed app provides a fully interactive frontend for evaluating the trained classification models.

---
