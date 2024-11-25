# Diabetes Prediction Project

This project leverages machine learning models to predict the likelihood of diabetes based on health-related features. The dataset contains synthetic records with various medical indicators, and the goal is to identify the most effective model for accurate predictions.

---

## Project Overview

Diabetes is a chronic condition that affects millions worldwide, making early prediction crucial for better health outcomes. This project uses a dataset with 400 samples and 9 features to train and evaluate several machine learning models, including:

- Naive Bayes (NB)
- K-Nearest Neighbors (KNN)
- Random Forest (RF)
- Gated Recurrent Unit (GRU)

The project evaluates these models based on their ability to predict diabetes using metrics such as accuracy, precision, recall, F1-score, and AUC.

---

## Dataset

### Features
The dataset includes the following features:
- **Pregnancies**: Number of pregnancies.
- **Glucose**: Blood glucose level (mg/dL).
- **Blood Pressure**: Measurement (mmHg).
- **Skin Thickness**: Skinfold thickness (mm).
- **Insulin**: Serum insulin level (µU/mL).
- **BMI**: Body mass index.
- **Diabetes Pedigree Function**: Genetic predisposition measure.
- **Age**: Age of the individual (years).
- **Outcome**: Target variable (1 = Diabetic, 0 = Non-diabetic).

### Preprocessing Steps
1. Missing values replaced with median values for consistency.
2. Standardized features to have a mean of 0 and a standard deviation of 1.
3. Split dataset into training (90%) and testing (10%) using stratified sampling.

---

## Models Implemented

### 1. Naive Bayes
- Simple probabilistic model based on Bayes' theorem.
- Assumes feature independence.

### 2. K-Nearest Neighbors (KNN)
- Non-parametric method using the majority vote of nearest neighbors.
- Configured with `k=5`.

### 3. Random Forest
- Ensemble method using multiple decision trees for robust predictions.
- Reduces overfitting and captures complex relationships.

### 4. Gated Recurrent Unit (GRU)
- Recurrent Neural Network model for capturing dependencies.
- Designed to evaluate the effectiveness of deep learning for this task.

---

## Evaluation Metrics
- **Accuracy**: Overall prediction correctness.
- **Precision**: Proportion of correctly identified diabetic cases.
- **Recall**: Proportion of actual diabetic cases identified correctly.
- **F1-Score**: Harmonic mean of precision and recall.
- **AUC**: Ability to distinguish between classes.
- **Confusion Matrix**: Breakdown of true/false positives and negatives.

---

## Results

| Model           | Accuracy | AUC  | F1-Score |
|------------------|----------|------|----------|
| Naive Bayes      | 71%      | 0.73 | 0.72     |
| K-Nearest Neighbors | 75%   | 0.77 | 0.74     |
| Random Forest    | 74%      | 0.75 | 0.73     |
| GRU              | 73%      | 0.76 | 0.72     |

### Best Model
The **K-Nearest Neighbors (KNN)** model outperformed others with:
- **Accuracy**: 75%
- **AUC**: 0.77
- **F1-Score**: 0.74

---

## Conclusion

This project demonstrates the potential of machine learning in healthcare for early disease detection. The KNN model proved to be the most effective, making it a suitable choice for predicting diabetes in similar datasets.

---

## Recommendations
- Deploy the KNN model in healthcare systems for initial diabetes screenings.
- Explore hyperparameter tuning for all models to enhance performance.
- Integrate additional features, such as genetic data or medical history, for more accurate predictions.

---

## Files Included
- `FinalProject_sm3736.ipynb`: Jupyter Notebook with code implementation.
- `diabetes_dataSet_sm3736.csv`: Dataset used in this project.
- `FinalProject.docx`: Detailed report of the project.

---

## How to Run

1. Clone the repository or download the project files.
2. Open `FinalProject_sm3736.ipynb` in Jupyter Notebook.
3. Ensure required libraries are installed:
   ```bash
   pip install numpy pandas scikit-learn tensorflow
