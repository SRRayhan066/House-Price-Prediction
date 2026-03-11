import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_features(df, target_col='median_house_value'):
    """Separate features (X) and target (y) from the DataFrame."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def handle_missing_values(X_train, X_test, column='total_bedrooms'):
    """
    Fill missing values using the median of the TRAINING set only.
    This avoids data leakage from the test set.
    """
    train_median = X_train[column].median()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[column] = X_train[column].fillna(train_median)
    X_test[column] = X_test[column].fillna(train_median)

    return X_train, X_test


def encode_categorical(X_train, X_test, columns=None):
    """
    Apply one-hot encoding to categorical columns.
    Aligns train and test to ensure they have the same columns.
    """
    if columns is None:
        columns = ['ocean_proximity']

    X_train = pd.get_dummies(X_train, columns=columns)
    X_test = pd.get_dummies(X_test, columns=columns)

    # Align to handle any categories missing in one of the sets
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    return X_train, X_test
