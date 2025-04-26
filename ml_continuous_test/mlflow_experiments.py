import logging
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def process_data(dataframe):
	df = dataframe.sample(frac=1)

	# amount of fraud classes 492 rows.
	fraud_df = df.loc[df['Class'] == 1]
	non_fraud_df = df.loc[df['Class'] == 0][:492]

	normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

	# Shuffle dataframe rows
	new_df = normal_distributed_df.sample(frac=1, random_state=42)

	return new_df


if __name__ == "__main__":
	np.random.seed(40)

	csv_path = "data/creditcard/creditcard.csv"

	try:
		data = pd.read_csv(csv_path, sep=",")
	except Exception as e:
		logger.exception(
			"Unable to download training & test CSV, check your internet connecion. Error: %s", e
		)
	
	data = process_data(data)
	
	X = data.drop(["Class"], axis=1)
	Y = data[["Class"]]

	models = []
	models.append(("LR", LogisticRegression(solver="newton-cg")))
	models.append(("KNN", KNeighborsClassifier()))
	models.append(("CART", DecisionTreeClassifier()))
	models.append(("NB", GaussianNB()))
	models.append(("SVM", SVC()))

	scoring = "accuracy"
	num_folds = 10

	now_code = int(datetime.utcnow().timestamp())
	exp = mlflow.set_experiment(experiment_name=f"betterexp_{now_code}")
	for name, model in models:
		with mlflow.start_run(experiment_id=exp.experiment_id):
			mlflow.set_tag("mlflow.runName", f"{name}")
			kfold = KFold(n_splits=num_folds)
			cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

			print(f"{name} model:")
			print("    Accuracy Mean: %s" % cv_results.mean())
			print("    Accuracy STD: %s" % cv_results.std())
			
			metrics_data = {}
			for i, score in enumerate(cv_results):
				metrics_data[f"{name}_{scoring}_{i+1}"] = score
			mlflow.log_metrics(metrics_data)

			mlflow.log_params(model.get_params())
			
			mlflow.log_metric("accuracy mean", cv_results.mean())
			mlflow.log_metric("accuracy mean", cv_results.std())
			mlflow.log_metric("kfold", num_folds)

			mlflow.sklearn.log_model(model, f"model_{name}")