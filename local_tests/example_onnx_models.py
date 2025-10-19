from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from ml_exp import MLExp
import numpy as np


X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

pipeline_1 = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LinearRegression())
])

pipeline_2 = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", Ridge())
])

initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

pipeline_1.fit(X_train, y_train)

onnx_model = convert_sklearn(pipeline_1, initial_types=initial_type)
with open("tests/local/models/sklearn_pipeline_1.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

pipeline_2.fit(X_train, y_train)
onnx_model = convert_sklearn(pipeline_2, initial_types=initial_type)
with open("tests/local/models/sklearn_pipeline_2.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

X_test = np.tile(X_test, (10, 1))
y_test = np.tile(y_test, 10)
print(X_test.shape)
print(y_test.shape)

# using library with files paths (similar with console)
ml_exp = MLExp(
    scores_target=["accuracy"],
    report_path="tests/local/reports/example_onnx_models",
    report_name="library_with_onnx"
)
ml_exp.add_test_data(
    test_data_name="test_data",
    X_test=X_test,
    y_test=y_test
)
ml_exp.add_context(
    context_name="model_1_onnx",
    model_trained="tests/local/models/sklearn_pipeline_1.onnx",
    ref_test_data="test_data"
)
ml_exp.add_context(
    context_name="model_2_onnx",
    model_trained="tests/local/models/sklearn_pipeline_2.onnx",
    ref_test_data="test_data"
)
ml_exp.run()