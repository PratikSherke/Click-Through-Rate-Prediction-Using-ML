# Precision CTR Prediction with Multi-Model ML

Welcome to the **Click-Through Rate (CTR) Prediction** project, a robust machine learning application designed to predict the likelihood of users clicking on online advertisements. This project leverages advanced techniques including XGBoost, Random Forest, and Logistic Regression models, enhanced with cross-validation, hyperparameter tuning, and comprehensive exploratory data analysis (EDA). It provides actionable insights for optimizing ad campaigns and maximizing return on investment (ROI).

## Features

- **Multiple Machine Learning Models**:
  - Implements XGBoost, Random Forest, and Logistic Regression to predict CTR.
  - Uses GridSearchCV for hyperparameter tuning to optimize model performance.
  - Includes 5-fold cross-validation for robust performance estimation.

- **Exploratory Data Analysis (EDA)**:
  - Visualizes data distributions with histograms for numerical features.
  - Displays correlation heatmaps to identify relationships between features.
  - Generates box plots to compare feature distributions by click status.
  - Shows target variable distribution to understand class balance.

- **Model Evaluation and Comparison**:
  - Evaluates models using accuracy, precision, recall, F1-score, and AUC metrics.
  - Compares model performance in a tabular format and identifies the best model.
  - Visualizes ROC curves for all models in a single plot for easy comparison.

- **Feature Importance Analysis**:
  - Plots feature importance for each model, highlighting key predictors of ad clicks.

- **Synthetic Data Generation**:
  - Generates synthetic data using the Faker library for testing and evaluation.
  - Uses the best-performing model to assign realistic labels to synthetic data.

- **Robust Preprocessing**:
  - Handles datetime features, categorical encoding, and feature scaling.
  - Ensures consistent feature sets across all models for fair comparison.

- **Visualization Output**:
  - EDA, ROC curves, feature importance.
  - Provides clear, labeled visualizations using Matplotlib and Seaborn.

## Prerequisites

To run this project, ensure you have the following installed:

- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ctr-prediction.git
   cd ctr-prediction
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset**:
   - Ensure the `ad_records.csv` file is in the project directory. The dataset should contain columns: `Daily Time Spent on Site`, `Age`, `Area Income`, `Daily Internet Usage`, `City`, `Gender`, `Country`, `Timestamp`, and `Clicked on Ad`.

5. **Run the Application**:
   ```bash
   CTR_Project_main.ipynb 
   ```

   This will:
   - Perform EDA .
   - Train and evaluate models, displaying performance metrics and visualizations.
   - Generate and evaluate synthetic data using the best model.

## Usage

1. **Run the Script**:
   - Execute `CTR_Project_main.ipynb ` to process the dataset, train models, and generate outputs.
   - The script will:
     - Load and preprocess `ad_records.csv`.
     - Generate EDA visualizations (histograms, correlation heatmap, box plots, target distribution).
     - Train XGBoost, Random Forest, and Logistic Regression models with hyperparameter tuning and cross-validation.
     - Compare model performance and identify the best model.
     - Plot ROC curves and feature importance for each model.
     - Generate and evaluate synthetic data.

2. **View Outputs**:
     -  Distributions of numerical features.
     -  Feature correlations.
     -  Feature distributions by click status.
     -  Target variable distribution.
     -  ROC curves for all models.
     -  Feature importance for each model.
   - Review console output for model performance metrics, cross-validation scores, and synthetic data results.
 

## Dataset

The project expects a CSV file (`ad_records.csv`) with the following columns:
- `Daily Time Spent on Site`: Time spent on the website (numeric).
- `Age`: User age (numeric).
- `Area Income`: User income (numeric).
- `Daily Internet Usage`: Daily internet usage time (numeric).
- `City`: User city (categorical).
- `Gender`: User gender (categorical).
- `Country`: User country (categorical).
- `Timestamp`: Ad impression timestamp (datetime).
- `Clicked on Ad`: Target variable (0 or 1).

## Project Structure

```
ctr-prediction/
│
├── CTR_Project_main.ipynb             # Main script
├── requirements.txt                  # Python dependencies
├── ad_records.xls                    # dataset file
└── README.md                        # Project documentation
```

## Dependencies

The project relies on the following Python packages (listed in `requirements.txt`):
- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib
- seaborn
- faker
- joblib

## Notes

- **Performance**: Model training with GridSearchCV may be computationally intensive. Adjust the parameter grids or use fewer folds (`cv`) for faster execution.
- **Synthetic Data**: Generated using realistic ranges and the best model's predictions for labels, ensuring meaningful evaluation.
- **Scalability**: The script uses a consistent feature set and scaling for fair model comparison. Additional features (e.g., text from `Ad Topic Line`) can be added for enhanced predictions.
- **Error Handling**: Includes basic error handling for missing dataset files.


Happy predicting! 📊🚀
