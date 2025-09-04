from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from ml_exp import BetterExperimentation
import numpy as np


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

pipeline_1 = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier())
])

pipeline_2 = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

pipeline_1.fit(X_train, y_train)
onnx_model = convert_sklearn(pipeline_1, initial_types=initial_type)
with open("tests/local/example_onnx/sklearn_pipeline_1.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

pipeline_2.fit(X_train, y_train)
onnx_model = convert_sklearn(pipeline_2, initial_types=initial_type)
with open("tests/local/example_onnx/sklearn_pipeline_2.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

X_test = np.tile(X_test, (10, 1))
y_test = np.tile(y_test, 10)

# using library with files paths (similar with console)
better_exp = BetterExperimentation(
    # models_trained="tests/local/example_onnx",
    # X_test=X_test,
    # y_test=y_test,
    scores_target="accuracy",
    report_name="library_with_onnx"
)
better_exp.add_test_data(
    test_data_name="test_data",
    X_test=X_test,
    y_test=y_test
)
better_exp.add_model(
    model_name="model_1_onnx",
    model_trained="tests/local/example_onnx/sklearn_pipeline_1.onnx",
    ref_test_data="test_data"
)
better_exp.add_model(
    model_name="model_2_onnx",
    model_trained="tests/local/example_onnx/sklearn_pipeline_2.onnx",
    ref_test_data="test_data"
)
better_exp.run()