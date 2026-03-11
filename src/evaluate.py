import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(y_test, y_pred):
    """Calculate and print MSE, MAE, and RMSE."""
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error:      {mse:.2f}")
    print(f"Mean Absolute Error:     {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")

    return mse, mae, rmse


def plot_results(y_test, y_pred, n=50, save_path='data/actual_vs_predicted.png'):
    """
    Plot a grouped bar chart comparing actual vs predicted house prices
    for the first n test samples.
    """
    actual_sample = y_test.values[:n]
    predicted_sample = y_pred[:n]
    x_indices = np.arange(n)
    bar_width = 0.4

    plt.figure(figsize=(18, 6))
    plt.bar(x_indices - bar_width / 2, actual_sample, width=bar_width,
            label='Actual', color='steelblue', alpha=0.8)
    plt.bar(x_indices + bar_width / 2, predicted_sample, width=bar_width,
            label='Predicted', color='tomato', alpha=0.8)

    plt.xlabel('Sample Index')
    plt.ylabel('House Price')
    plt.title('Actual vs Predicted House Prices (First 50 Samples)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Graph saved to {save_path}")
