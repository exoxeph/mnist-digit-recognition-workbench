from src import v1_baseline, v2_cnn, v3_error_analysis, v3_robustness


def main():
    print("=== V1: Baseline ===")
    v1_baseline.train_and_evaluate()

    print("=== V2: CNN ===")
    v2_cnn.train_and_evaluate()

    print("=== V3: Error Analysis ===")
    v3_error_analysis.run()

    print("=== V3: Robustness ===")
    v3_robustness.run()

    print("Pipeline complete. See reports/ for results.")


if __name__ == "__main__":
    main()
