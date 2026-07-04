import joblib
from sklearn.linear_model import LogisticRegression

from src.common import config, load_data, preprocess, evaluate


def build_model():
    return LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)


def train_and_evaluate():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data.load_split()

    x_train_flat = preprocess.to_flat_input(x_train)
    x_test_flat = preprocess.to_flat_input(x_test)

    model = build_model()
    model.fit(x_train_flat, y_train)

    y_pred = model.predict(x_test_flat)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.LOGREG_MODEL_PATH)

    metrics = evaluate.save_metrics_json(
        config.METRICS_V1_PATH, "logreg_baseline", y_test, y_pred
    )
    print(f"V1 baseline accuracy: {metrics['accuracy']:.4f}")
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
