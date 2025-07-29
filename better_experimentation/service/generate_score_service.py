from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from better_experimentation.utils.log_config import LogService, handle_exceptions
from better_experimentation.model.ml_model import ModelTechnology


class GenerateScoreService:
    __log_service = LogService()

    def __init__(self,
                 experiments: dict,
                 test_data: dict,
                 scores_target: str,
                 n_splits: int) -> None:
        self.__logger = self.__log_service.get_logger(__name__)
        self.__kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.scores = {}
        self.experiments = experiments
        self.test_data = test_data

        for score_target in scores_target:
            self.scores[score_target] = {}

            for experiment_name in self.experiments.keys():
                self.scores[score_target][experiment_name] = []

        for experiment_name, experiment_data in self.experiments.items():
            ml_model = experiment_data['ml_model']
            test_data_for_experiment = self.test_data[experiment_data['test_data_name']]
            X_test = test_data_for_experiment['x_test']
            y_test = test_data_for_experiment['y_test']

            data_test_split = self.__kf.split(X=X_test, y=y_test)
        
            for i, (train_index, test_index) in enumerate(data_test_split):
                X_fold, Y_fold = X_test.iloc[test_index], y_test.iloc[test_index]
                Y_fold = Y_fold.values.ravel()
                Y_pred = self.__collect_prediction_from_model(ml_model, X_fold)
                self.__collect_metric_result(ml_model, Y_fold, Y_pred)

    def __collect_prediction_from_model(self, model, X_fold):
        if model.model_technology == ModelTechnology.general_from_onnx.value:
            input_name = model.model_object.get_inputs()[0].name
            output_name = model.model_object.get_outputs()[0].name
            Y_pred_list = []
            for i in range(len(X_fold)):
                input_data = X_fold[i:i+1].astype(np.float32).to_numpy()
                output_data = model.model_object.run([output_name], {input_name: input_data})
                Y_pred_list.append(output_data[0][0])
            Y_pred = np.array(Y_pred_list)
        else:
            Y_pred = model.model_object.predict(X_fold)
        return Y_pred
                
    def __collect_metric_result(self, model, Y_fold, Y_pred):
        """Collects metrics for the given model and test data."""
        for score_target in self.scores.keys():
            if score_target == "accuracy":
                self.scores[score_target][model.model_name].append(accuracy_score(Y_fold, Y_pred))
            elif score_target == "roc_auc":
                self.scores[score_target][model.model_name].append(roc_auc_score(Y_fold, Y_pred))
            elif score_target == "mae":
                self.scores[score_target][model.model_name].append(mean_absolute_error(Y_fold, Y_pred))
            elif score_target == "mse":
                self.scores[score_target][model.model_name].append(mean_squared_error(Y_fold, Y_pred))
            elif score_target == "r2":
                self.scores[score_target][model.model_name].append(r2_score(Y_fold, Y_pred))

    def get_scores_data(self):
        return self.scores