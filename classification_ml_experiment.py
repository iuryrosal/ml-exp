import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from ml_exp import BetterExperimentation

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

	# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
	csv_path = "data/creditcard/creditcard.csv"

	data = pd.read_csv(csv_path, sep=",")
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

	# using console
	for i, model in enumerate(models):
		with open(f'tests/local/classification/model_{i}.pkl','wb') as f:
			pickle.dump(model, f)
	
	X_test.to_csv("tests/local/classification/x_test.csv", index=False)
	y_test.to_csv("tests/local/classification/y_test.csv", index=False)

	# Generate models with different test data
	data_2 = pd.read_csv(csv_path, sep=",")
	non_fraud_df_to_test_2 = data_2.loc[data_2['Class'] == 0][492:]
	non_fraud_df_to_test_2 = non_fraud_df_to_test_2.drop(["Class", "Amount"], axis=1)
	data_2 = process_data(data_2)
	X_2 = data_2.drop(["Class", "Amount"], axis=1)
	Y_2 = data_2[["Class"]]
	X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, Y_2, test_size=0.3, train_size=0.7)

	X_test_2 = pd.concat([non_fraud_df_to_test_2, X_test_2])
	df_non_fraud_values_to_test_2 = {'Class': [0 for _ in range(len(non_fraud_df_to_test_2.values))]}
	y_test_2 = pd.concat([pd.DataFrame(data=df_non_fraud_values_to_test_2), y_test_2])

	models_2 = []
	models_2.append(DecisionTreeClassifier())
	models_trained_2 = []
	for model in models_2:
		model.fit(X_train_2, y_train_2)
		models_trained_2.append(model)

	# using console
	for i, model in enumerate(models_2):
		with open(f'tests/local/classification/model_{i}_v2.pkl','wb') as f:
			pickle.dump(model, f)

	X_test_2.to_csv("tests/local/classification/x_test_2.csv", index=False)
	y_test_2.to_csv("tests/local/classification/y_test_2.csv", index=False)

	# CLI: ml_exp tests/local/classification tests/local/classification/x_test.csv tests/local/classification/y_test.csv accuracy

	# using library with files paths (similar with console)
	better_exp = BetterExperimentation(
		scores_target=["accuracy", "roc_auc", "precision_recall"],
		report_name="library_with_path"
	)
	better_exp.add_test_data(
		test_data_name="test_data",
		X_test="tests/local/classification/x_test.csv",
		y_test="tests/local/classification/y_test.csv"
	)
	better_exp.add_context(
		context_name="model_0_sklearn",
		model_trained="tests/local/classification/model_0.pkl",
		ref_test_data="test_data"
	)
	better_exp.add_context(
		context_name="model_1_sklearn",
		model_trained="tests/local/classification/model_1.pkl",
		ref_test_data="test_data"
	)
	better_exp.add_context(
		context_name="model_2_sklearn",
		model_trained="tests/local/classification/model_2.pkl",
		ref_test_data="test_data"
	)
	better_exp.add_context(
		context_name="model_3_sklearn",
		model_trained="tests/local/classification/model_3.pkl",
		ref_test_data="test_data"
	)
	better_exp.add_context(
		context_name="model_4_sklearn",
		model_trained="tests/local/classification/model_4.pkl",
		ref_test_data="test_data"
	)
	better_exp.run()

	# using library with current objects
	better_exp = BetterExperimentation(
		scores_target=["accuracy"],
		report_name="library_with_objects"
	)
	better_exp.add_test_data(
		test_data_name="test_data",
		X_test=X_test,
		y_test=y_test
	)
	for i, model in enumerate(models_trained):
		better_exp.add_context(
			context_name=f"model_{i}_sklearn",
			model_trained=model,
			ref_test_data="test_data"
		)
	better_exp.run()

	# using both
	better_exp = BetterExperimentation(
		scores_target=["accuracy"],
		report_name="library_with_both"
	)
	better_exp.add_test_data(
		test_data_name="test_data",
		X_test=X_test,
		y_test=y_test
	)
	better_exp.add_context(
		context_name=f"model_1_sklearn",
		model_trained=models_trained[0],
		ref_test_data="test_data"
	)
	better_exp.add_context(
		context_name=f"model_2_sklearn",
		model_trained=models_trained[1],
		ref_test_data="test_data"
	)
	better_exp.add_context(
		context_name=f"model_3_sklearn",
		model_trained="tests/local/classification/model_3.pkl",
		ref_test_data="test_data"
	)
	better_exp.run()

	# models with different test data
	better_exp = BetterExperimentation(
		scores_target=["accuracy"],
		report_name="library_with_different_test_data"
	)
	better_exp.add_test_data(
		test_data_name="test_data",
		X_test=X_test,
		y_test=y_test
	)
	better_exp.add_test_data(
		test_data_name="test_data_2",
		X_test=X_test_2,
		y_test=y_test_2
	)
	better_exp.add_context(
		context_name="model_3_sklearn",
		model_trained="tests/local/classification/model_3.pkl",
		ref_test_data="test_data"
	)
	better_exp.add_context(
		context_name="model_0_v2_sklearn",
		model_trained="tests/local/classification/model_0_v2.pkl",
		ref_test_data="test_data_2"
	)
	better_exp.run()