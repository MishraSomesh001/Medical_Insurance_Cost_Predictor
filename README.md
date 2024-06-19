# Insurance Cost Prediction Project

This project explores building a machine learning model to predict insurance costs based on various factors.

## Functionality

The code performs the following tasks:

### Data Collection & Analysis:
- Loads insurance data from a CSV file.
- Provides basic information about the data (dimensions, missing values, etc.).
- Analyzes the distribution of features through visualizations (age, gender, BMI, etc.).

### Data Pre-Processing:
- Encodes categorical features (sex, smoker, region) into numerical values.
- Splits the data into features (X) and target variable (charges) (Y).
- Splits the data further into training and testing sets.

### Model Training:
- Implements a Linear Regression model to predict insurance charges.
- Trains the model on the training data.

### Model Evaluation:
- Evaluates the model's performance on both training and testing data using R-squared score.

### Predictive System:
- Demonstrates how to use the trained model to predict insurance costs for a new data point.

## Dependencies

- `numpy`
- `pandas`
- `matplotlib.pyplot`
- `seaborn`
- `scikit-learn`

## Usage

1. Clone the repository.
2. Ensure the required libraries are installed:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. The code is written in a single Python file (*.py). Run the script to execute the data analysis, model training, and prediction functionalities.

## Future Enhancements
- Explore other machine learning models beyond Linear Regression (e.g., Random Forest, Gradient Boosting).
- Implement hyperparameter tuning to improve model performance.
- Develop a user-friendly interface for interactive cost prediction.