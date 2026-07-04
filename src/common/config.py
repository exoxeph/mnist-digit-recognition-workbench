from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_CSV_PATH = DATA_RAW_DIR / "train.csv"

IMAGE_SIZE = 28
NUM_CLASSES = 10

RANDOM_SEED = 42
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

CNN_EPOCHS = 10
CNN_BATCH_SIZE = 64
CNN_LEARNING_RATE = 1e-3

CONFIDENCE_HIGH_THRESHOLD = 0.9
CONFIDENCE_MEDIUM_THRESHOLD = 0.6

LOGREG_MODEL_PATH = MODELS_DIR / "logreg_baseline.joblib"
CNN_MODEL_PATH = MODELS_DIR / "best_cnn_model.pt"

METRICS_V1_PATH = REPORTS_DIR / "metrics_v1.json"
METRICS_V2_PATH = REPORTS_DIR / "metrics_v2.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
PER_CLASS_ACCURACY_PATH = REPORTS_DIR / "per_class_accuracy.csv"
MISCLASSIFIED_EXAMPLES_PATH = REPORTS_DIR / "misclassified_examples.png"
ROBUSTNESS_RESULTS_PATH = REPORTS_DIR / "robustness_results.csv"
ERROR_ANALYSIS_PATH = REPORTS_DIR / "error_analysis.md"
