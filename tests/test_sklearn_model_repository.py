import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression

from tests.config.general_fixtures import sklearn_model_repository
from ml_exp.model.ml_model import ModelTechnology, ModelType



def test_load_model_by_obj_classifier(sklearn_model_repository):
    model = LogisticRegression()
    ml_model = sklearn_model_repository.load_model_by_obj(context_name="test0", model_obj=model)

    assert ml_model.context_name == "test0"
    assert ml_model.model_object == model
    assert ml_model.model_technology == ModelTechnology.sklearn.value
    assert ml_model.model_type == ModelType.classifier.value


def test_load_model_by_obj_regressor(sklearn_model_repository):
    model = LinearRegression()
    ml_model = sklearn_model_repository.load_model_by_obj(context_name="test1", model_obj=model)

    assert ml_model.context_name == "test1"
    assert ml_model.model_object == model
    assert ml_model.model_technology == ModelTechnology.sklearn.value
    assert ml_model.model_type == ModelType.regressor.value


def test_load_model_by_obj_invalid_type(sklearn_model_repository):
    class Dummy:
        pass

    dummy_model = Dummy()
    with pytest.raises(ValueError, match="Model have invalid type"):
        sklearn_model_repository.load_model_by_obj(context_name="test2", model_obj=dummy_model)