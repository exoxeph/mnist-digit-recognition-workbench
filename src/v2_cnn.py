import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.common import config, load_data, preprocess, evaluate


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, config.NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def make_loader(x, y, batch_size, shuffle):
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_and_evaluate():
    torch.manual_seed(config.RANDOM_SEED)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data.load_split()

    x_train_cnn = preprocess.to_cnn_input(x_train)
    x_val_cnn = preprocess.to_cnn_input(x_val)
    x_test_cnn = preprocess.to_cnn_input(x_test)

    train_loader = make_loader(x_train_cnn, y_train, config.CNN_BATCH_SIZE, True)
    val_loader = make_loader(x_val_cnn, y_val, config.CNN_BATCH_SIZE, False)

    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=config.CNN_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.CNN_EPOCHS):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_accuracy = correct / total
        print(f"Epoch {epoch + 1}/{config.CNN_EPOCHS} - val_accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), config.CNN_MODEL_PATH)

    model.load_state_dict(torch.load(config.CNN_MODEL_PATH, weights_only=True))
    model.eval()
    x_test_tensor = torch.from_numpy(x_test_cnn)
    with torch.no_grad():
        y_pred = model(x_test_tensor).argmax(dim=1).numpy()

    metrics = evaluate.save_metrics_json(config.METRICS_V2_PATH, "cnn", y_test, y_pred)
    evaluate.save_confusion_matrix_plot(y_test, y_pred, config.CONFUSION_MATRIX_PATH)
    evaluate.save_per_class_accuracy_csv(y_test, y_pred, config.PER_CLASS_ACCURACY_PATH)
    print(f"V2 CNN test accuracy: {metrics['accuracy']:.4f}")
    return metrics


def load_trained_model():
    if not config.CNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{config.CNN_MODEL_PATH} not found. Run `python -m src.v2_cnn` first."
        )
    model = CNN()
    model.load_state_dict(
        torch.load(config.CNN_MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


if __name__ == "__main__":
    train_and_evaluate()
