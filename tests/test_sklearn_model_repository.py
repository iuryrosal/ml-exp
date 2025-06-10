import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression

from tests.config.general_fixtures import sklearn_model_repository
from better_experimentation.model.ml_model import ModelTechnology, ModelType



def test_load_model_by_obj_classifier(sklearn_model_repository):
    model = LogisticRegression()
    ml_model = sklearn_model_repository.load_model_by_obj(model_idx=0, model_obj=model)

    assert ml_model.model_index == 0
    assert ml_model.model_name.startswith("LogisticRegression_")
    assert ml_model.model_object == model
    assert ml_model.model_technology == ModelTechnology.sklearn.value
    assert ml_model.model_type == ModelType.classifier.value


def test_load_model_by_obj_regressor(sklearn_model_repository):
    model = LinearRegression()
    ml_model = sklearn_model_repository.load_model_by_obj(model_idx=1, model_obj=model)

    assert ml_model.model_index == 1
    assert ml_model.model_name.startswith("LinearRegression_")
    assert ml_model.model_object == model
    assert ml_model.model_technology == ModelTechnology.sklearn.value
    assert ml_model.model_type == ModelType.regressor.value


def test_load_model_by_obj_invalid_type(sklearn_model_repository):
    class Dummy:
        pass

    dummy_model = Dummy()
    with pytest.raises(ValueError, match="Model have invalid type"):
        sklearn_model_repository.load_model_by_obj(model_idx=0, model_obj=dummy_model)


def test_load_model_by_path_with_mixed_models(sklearn_model_repository):
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        classifier = LogisticRegression()
        regressor = LinearRegression()

        with open(path / "classifier_model.pkl", "wb") as f:
            pickle.dump(classifier, f)
        with open(path / "regressor_model.obj", "wb") as f:
            pickle.dump(regressor, f)

        models = sklearn_model_repository.load_model_by_path(path)

        assert len(models) == 2
        assert any(m.model_type == ModelType.classifier.value for m in models)
        assert any(m.model_type == ModelType.regressor.value for m in models)
        assert all(m.model_technology == ModelTechnology.sklearn.value for m in models)
        assert set(m.model_name for m in models) == {"classifier_model.pkl", "regressor_model.obj"}


def test_load_model_by_path_with_invalid_model(sklearn_model_repository):
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        invalid_obj = {"a": 1}

        with open(path / "invalid_model.obj", "wb") as f:
            pickle.dump(invalid_obj, f)

        with pytest.raises(ValueError, match="Model have invalid type"):
            sklearn_model_repository.load_model_by_path(path)