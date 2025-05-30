from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from better_experimentation.utils.log_config import LogService, handle_exceptions


class PrepareDataService:
    __log_service = LogService()

    def __init__(self, models, X_test, y_test, scores_target, n_splits) -> None:
        self.__logger = self.__log_service.get_logger(__name__)
        self.scores = {}
        for score_target in scores_target:
            self.scores[score_target] = {}
            for i, model in enumerate(models):
                self.scores[score_target][f"{model.model_index}"] = []

        self.__kf = KFold(n_splits=n_splits, shuffle=True)
        data_test_split = self.__kf.split(X=X_test)
        for i, (train_index, test_index) in enumerate(data_test_split):
            X_fold, Y_fold = X_test.iloc[test_index], y_test.iloc[test_index]
            Y_fold = Y_fold.values.ravel()
            for i, model in enumerate(models):
                Y_pred = model.model_object.predict(X_fold)
                self.__collect_metric_result(model, Y_fold, Y_pred)

    @handle_exceptions(__log_service.get_logger(__name__))
    def __collect_metric_result(self, model, Y_fold, Y_pred):
        """Collects metrics for the given model and test data."""
        for score_target in self.scores.keys():
            if score_target == "accuracy":
                self.scores[score_target][str(model.model_index)].append(accuracy_score(Y_fold, Y_pred))
            elif score_target == "f1":
                self.scores[score_target][str(model.model_index)].append(f1_score(Y_fold, Y_pred))
            elif score_target == "precision":
                self.scores[score_target][str(model.model_index)].append(precision_score(Y_fold, Y_pred))
            elif score_target == "recall":
                self.scores[score_target][str(model.model_index)].append(recall_score(Y_fold, Y_pred))

    @handle_exceptions(__log_service.get_logger(__name__))
    def get_scores_data(self):
        return self.scores