from src.train_model import train_and_evaluate
from src.predict import predict_new

if __name__ == "__main__":
    train_and_evaluate()

    # Previsão após treino
    data = {
        "age": 35,
        "sex": "female",
        "bmi": 26.0,
        "children": 1,
        "smoker": "no",
        "region": "southeast"
    }

    result = predict_new(data)
    print(f"Custo previsto: R$ {result:.2f}")