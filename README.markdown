# Click-Through Rate (CTR) Prediction

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
  - Saves all plots (EDA, ROC curves, feature importance) to an `eda_plots` directory.
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
   python click_through_rate_prediction.py
   ```

   This will:
   - Perform EDA and save plots to the `eda_plots` directory.
   - Train and evaluate models, displaying performance metrics and visualizations.
   - Generate and evaluate synthetic data using the best model.

## Usage

1. **Run the Script**:
   - Execute `python click_through_rate_prediction.py` to process the dataset, train models, and generate outputs.
   - The script will:
     - Load and preprocess `ad_records.csv`.
     - Generate EDA visualizations (histograms, correlation heatmap, box plots, target distribution).
     - Train XGBoost, Random Forest, and Logistic Regression models with hyperparameter tuning and cross-validation.
     - Compare model performance and identify the best model.
     - Plot ROC curves and feature importance for each model.
     - Generate and evaluate synthetic data.

2. **View Outputs**:
   - Check the `eda_plots` directory for saved visualizations:
     - `numerical_histograms.png`: Distributions of numerical features.
     - `correlation_heatmap.png`: Feature correlations.
     - `box_plots.png`: Feature distributions by click status.
     - `target_distribution.png`: Target variable distribution.
     - `roc_curves.png`: ROC curves for all models.
     - `feature_importance_*.png`: Feature importance for each model.
   - Review console output for model performance metrics, cross-validation scores, and synthetic data results.
   - Trained models are saved as `xgboost_model.pkl`, `random_forest_model.pkl`, and `logistic_regression_model.pkl`.

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
├── click_through_rate_prediction.py  # Main script
├── requirements.txt                  # Python dependencies
├── .gitignore                       # Git ignore file
├── eda_plots/                       # Directory for saved plots
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

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please follow PEP 8 guidelines and include relevant documentation for your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please open an issue on GitHub or contact [your-email@example.com].

---

Happy predicting! 📊🚀