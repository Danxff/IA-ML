import joblib
import pandas as pd

def predict_new(data: dict):
    model = joblib.load("models/model.pkl")
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":
    sample = {
        "age": 40,
        "sex": "male",
        "bmi": 27.5,
        "children": 2,
        "smoker": "no",
        "region": "southwest"
    }

    result = predict_new(sample)
    print(f"Previsão de custo médico: R$ {result:.2f}")
