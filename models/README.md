# models/

Trained model artifacts are not committed to this repository.

After running the training pipeline (`poetry run python main.py`, or
`src/v1_baseline.py` / `src/v2_cnn.py` individually), this directory will
contain:

- `logreg_baseline.joblib` - the V1 logistic regression baseline
- `best_cnn_model.pt` - the V2 CNN (PyTorch state_dict)
