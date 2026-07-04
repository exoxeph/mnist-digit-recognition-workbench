# models/

Trained model artifacts are not committed to this repository.

After running the training pipeline (`python main.py`, or `src/v1_baseline.py`
/ `src/v2_cnn.py` individually), this directory will contain:

- `logreg_baseline.joblib` â€” the V1 logistic regression baseline
- `best_cnn_model.pt` â€” the V2 CNN (PyTorch state_dict)
