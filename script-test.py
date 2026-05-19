from ml_exp import MLExp

ml_exp = MLExp(
    scores_target=["accuracy"],
    report_path="tests/local/reports/classification_ml_experiment",
    report_name="library_with_different_test_data"
)

ml_exp.add_test_data(
    test_data_name="test_data",
    X_test="tests/local/data/x_test.csv",
    y_test="tests/local/data/y_test.csv"
)

ml_exp.add_test_data(
    test_data_name="test_data_2",
    X_test="tests/local/data/x_test_2.csv",
    y_test="tests/local/data/y_test_2.csv"
)

ml_exp.add_context(
    context_name="model_3_sklearn",
    model_trained="tests/local/models/model_3.pkl",
    ref_test_data="test_data"
)

ml_exp.add_context(
    context_name="model_0_v2_sklearn",
    model_trained="tests/local/models/model_0_v2.pkl",
    ref_test_data="test_data_2"
)

ml_exp.run()