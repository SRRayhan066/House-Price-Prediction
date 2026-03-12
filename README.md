# House Price Prediction

A machine learning project that predicts house prices using Linear Regression.

## Overview

This project analyzes housing data and builds a predictive model to estimate median house values based on various features like income, population, and geographic attributes.

## Project Structure

```
house-price-prediction/
├── data/
│   └── dataset.csv          # Raw housing data
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Data loading and visualization
│   ├── preprocessing.py     # Feature preparation and cleaning
│   ├── train.py             # Model training
│   └── evaluate.py          # Model evaluation and metrics
├── main.py                  # Main pipeline script
└── README.md
```

## Features

- **Data Loading**: Loads housing data from CSV
- **Exploratory Data Analysis**: 
  - Summary statistics
  - Correlation matrix heatmap
  - Income vs price scatter plot
- **Preprocessing**:
  - Missing value handling (median imputation)
  - Categorical encoding (one-hot)
  - Train/test splitting
- **Model**: Linear Regression
- **Evaluation Metrics**:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Actual vs predicted visualization

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

Run the full pipeline:
```bash
python main.py
```

This will:
1. Load the dataset
2. Display data observations
3. Generate visualizations
4. Train the model
5. Evaluate and show metrics
6. Save plots to the `data/` folder

## Output

Generated plots (saved in `data/`):
- `correlation_matrix.png` - Heatmap showing feature correlations
- `income_vs_price.png` - Scatter plot of income vs house prices
- `actual_vs_predicted.png` - Bar chart comparing predictions to actual values

## Model

Currently uses **Linear Regression** from scikit-learn. The model can be swapped for other algorithms by modifying `src/train.py`.
