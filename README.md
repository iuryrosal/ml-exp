# better-experimentation
The main objective of `better-experimentation` is to provide a better evaluation and comparison between supervised machine learning models, being a great way to apply continuous experimentation during model evaluation or even as a step in an MLOps pipeline.

## 🛠️ Installation

### From source (development environment)
Download the source code by cloning the repository or click on  "Download ZIP" to download the latest stable version.

Install it by navigating to the proper directory and running:

```sh
pip install -e .
```

The result of better_experimentation is written in HTML and CSS, which means a modern browser is required to see it correctly.

You need [Python 3](https://python3statement.github.io/) to run the package. Other dependencies can be found in the requirements files available in `pyproject.toml`. You can activate a virtual environment with all the project dependencies, as well as the version, using [Poetry](https://python-poetry.org).

With Poetry installed, simply run the following command in the project root folder (where `pyproject.toml` is present):   

```sh
poetry shell
```

## ▶️ Quickstart
During model training, you may be organizing the model objects into a list, as well as loading the feature set into a Pandas Dataframe (X_test) and the respective targets into another Pandas Dataframe (y_test). As shown in the example below:

```python
    all_data_csv_path = " "
	data = pd.read_csv(all_data_csv_path, sep=",")

    # logic to deal with unbalanced classes
	non_fraud_df_to_test = data.loc[data['Class'] == 0][492:]
	non_fraud_df_to_test = non_fraud_df_to_test.drop(["Class"], axis=1)

    # dataframe transformations and cleanings before applying data split
	data = process_data(data) 
	
    # applying split between features and target, in addition to the training and testing part.
	X = data.drop(["Class"], axis=1)
	Y = data[["Class"]]
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7)

    # aggregating the cases discarded during training to encompass the test scenarios
	X_test = pd.concat([non_fraud_df_to_test, X_test])
	df_non_fraud_values_to_test = {'Class': [0 for _ in range(len(non_fraud_df_to_test.values))]}
	y_test = pd.concat([pd.DataFrame(data=df_non_fraud_values_to_test), y_test])

    # generating models and training
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
```

Once we have the trained models, the test features and the respective test targets, we can apply continuous experimentation within our Python code. To do this, we use the `BetterExperimentation` object instantiation and pass the respective objects. You can use a single metric to apply continuous experimentation or a list of metrics. Like the example below:

```python
from better_experimentation import BetterExperimentation

better_exp = BetterExperimentation(
		models_trained=models_trained,
		X_test=X_test,
		y_test=y_test,
		scores_target="accuracy" # or ["accuracy"] or ["accuracy", "precision"]
	)

better_exp.run()
```

When executing the `run()` command you will apply the continuous experimentation pipeline and generate the report (which, if not specified, will always be generated in the root folder of your project within `reports/general_report`).

If you export your models to Pickle format, as well as test data to files (csv, parquet, txt, json...) you can pass the respective paths as arguments.

```python
from better_experimentation import BetterExperimentation

better_exp = BetterExperimentation(
		models_trained="tests/local/classification",
		X_test="tests/local/classification/x_test.csv",
		y_test="tests/local/classification/y_test.csv",
		scores_target="accuracy" # or ["accuracy"] or ["accuracy", "precision"]
	)
better_exp.run()
```

### Command Line Interface
You can use the command line to run continuous experimentation around a specific metric, generate a report, and capture the best model (if any) around a metric. You can check the available commands by running the following command:

```sh
better_experimentation --h
```

An example of using the command line by passing a folder with several Sklearn models saved in Pickle format (.pkl), an X_test and y_test saved in CSV format and indicating, in the optional parameter, the name of the report that will be generated.

```sh
better_experimentation tests/local/classification tests/local/classification/x_test.csv tests/local/classification/y_test.csv accuracy --report_name iury_teste
```

## 💎 Key features
- Generates different test clusters by applying KFold
- Generates certain metrics around these clusters to collect data for use in statistical tests, using the trained models
- Applies a descriptive summary of the distribution of the collected metrics: maximum value, minimum value, mean, median and standard deviation
- Applies a set of statistical tests to verify the existence of significant differences between the models around a metric
- Based on the significant differences, it will search for the best model around the metric in question by comparing the median of the distribution collected for each model
- Organizes all results in JSON and HTML to facilitate decision making

### Statistical Tests Flowchart


## 👀 Examples


## 🙋 Support


## ✨ Contributing

