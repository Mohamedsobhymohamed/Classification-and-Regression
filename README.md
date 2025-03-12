# Machine Learning Models: Classification and Regression

## Overview
This repository contains two Jupyter Notebook files implementing machine learning models:
1. **Classification using K-Nearest Neighbors (KNN)**
2. **Regression using Linear, Lasso, and Ridge Regression**

## Files in the Repository
- `classification_knn.ipynb`: Implements K-Nearest Neighbors (KNN) for a classification task using the MAGIC gamma telescope dataset.
- `California_Housing_Regression.ipynb`: Implements Linear Regression, Lasso Regression, and Ridge Regression on the California Housing dataset.

## Classification Task
### Dataset
- The dataset contains two classes: gamma (signal) and hadrons (background).
- The dataset is imbalanced, requiring preprocessing to balance class distributions.

### Steps
1. **Data Preprocessing**
   - Load dataset and balance class distribution.
   - Split into training, validation, and test sets (70%-15%-15%).
2. **Model Training**
   - Implement K-Nearest Neighbors (KNN) classifier.
   - Tune hyperparameter `k` to find the best value based on F1-score.
3. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

## Regression Task
### Dataset
- The dataset consists of housing features and their corresponding median house value.

### Steps
1. **Data Preprocessing**
   - Handle missing values and scale features.
   - Split into training, validation, and test sets (70%-15%-15%).
2. **Model Training**
   - Train and compare Linear Regression, Lasso Regression, and Ridge Regression models.
3. **Evaluation Metrics**
   - Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
   - Identify the best model based on test set performance.

## How to Run the Notebooks
1. Install required dependencies using:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```
2. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Load and run `classification_knn.ipynb` or `California_Housing_Regression.ipynb`.

## Results
- **KNN Classification:** The best `k` value is determined based on F1-score, and model performance is reported using a confusion matrix.
- **Regression Models:** Linear Regression provides comparable performance to Lasso and Ridge, indicating minimal overfitting.

## Conclusion
- **KNN Classification:** Hyperparameter tuning significantly impacts classification accuracy.
- **Regression Models:** Feature scaling is essential, and regularization (Lasso/Ridge) may not always improve results.

## License
This project is open-source and available for educational purposes.

