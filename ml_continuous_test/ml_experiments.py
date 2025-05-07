import logging
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from ml_continuous_test.service.experimental_pipeline_service import ExperimentalPipelineService
from ml_continuous_test.service.prepare_data_service import PrepareDataService

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
	non_fraud_df_to_test = data.loc[data['Class'] == 0][492:]
	non_fraud_df_to_test = non_fraud_df_to_test.drop(["Class"], axis=1)
	data = process_data(data)
	
	X = data.drop(["Class"], axis=1)
	Y = data[["Class"]]
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7)

	X_test = pd.concat([non_fraud_df_to_test, X_test])
	df_non_fraud_values_to_test = {'Class': [0 for _ in range(len(non_fraud_df_to_test.values))]}
	y_test = pd.concat([pd.DataFrame(data=df_non_fraud_values_to_test), y_test])

	models = []
	models.append(LogisticRegression(solver="newton-cg"))
	models.append(KNeighborsClassifier())
	models.append(DecisionTreeClassifier())
	models.append(GaussianNB())
	models.append(SVC())
	models_trained = []
	for model in models:
		model.fit(X_train, y_train)
		models_trained.append(model)
	
	scores = PrepareDataService(
		models_trained=models_trained,
		X_test=X_test,
		y_test=y_test,
		scores_target=["accuracy", "precision", "recall"],
		n_splits=1000).get_scores_data()
	# Persist result
	exp_pipe = ExperimentalPipelineService(scores_data=scores)
	exp_pipe.run_pipeline()