import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data_preparation import load_data, preprocess_data, split_data

def train_and_evaluate():
    df = load_data()
    X, y, preprocessor = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R²:", r2_score(y_test, y_pred))

    joblib.dump(model, "models/model.pkl")

if __name__ == "__main__":
    train_and_evaluate()