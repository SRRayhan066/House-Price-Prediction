from src.data_loader import load_data, observe_data, plot_income_vs_price
from src.preprocessing import prepare_features, split_data, handle_missing_values, encode_categorical
from src.train import train_model
from src.evaluate import calculate_metrics, plot_results


def main():
    # 1. Load Data
    df = load_data('data/dataset.csv')

    # 2. Observe Data
    observe_data(df)
    plot_income_vs_price(df)

    # 3. Prepare Features & Target
    X, y = prepare_features(df)

    # 4. Split Data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5. Preprocess
    X_train, X_test = handle_missing_values(X_train, X_test)
    X_train, X_test = encode_categorical(X_train, X_test)

    # 6. Train Model
    model = train_model(X_train, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test)
    calculate_metrics(y_test, y_pred)
    plot_results(y_test, y_pred)


if __name__ == '__main__':
    main()
