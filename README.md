# MLExp
The main objective of `MLExp` is to provide a better evaluation and comparison between supervised machine learning models, being a great way to apply continuous experimentation during model evaluation or even as a step in an MLOps pipeline.

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

During model training, you may be organizing the model trained objects, as well as loading the feature set into a Pandas Dataframe (X_test) and the respective targets into another Pandas Dataframe (y_test). It's possible to save model trained object in Pickle, ONNX or MLFlow, while X_text and y_test can be store in data files that supported by Pandas library.

You can apply continuous experimentation within our Python code, using `MLExp` object instantiation with a reference in local variable. During instantiation you need provide a parameters that impacts how the experimentation works and location to store reports.

Using the local variable to reference the library instanciated, you need to add test data with `add_test_data()` instance method. In this function, you need to inform X_test (features), y_test (target) and name to refer own set of data (must be unique). The X_test and y_test can be a Pandas DataFrame objects or path to files supported by Pandas library (csv, parquet, txt, json...).

```python
ml_exp.add_test_data(
		test_data_name="test_data",
		X_test="tests/local/classification/x_test.csv",
		y_test="tests/local/classification/y_test.csv"
	)
```

To add context, combining a model trained with test data to use in experiment, you use `add_context()` instance method. During the call, you need to provide the model trained (can be a object or path), what test data will be applied in this model and a name to refer own context (must be unique). 

```python
ml_exp.add_context(
		context_name="model_0_sklearn",
		model_trained="tests/local/classification/model_0.pkl",
		ref_test_data="test_data"
	)
```

When executing the `run()` instance method, you will apply the continuous experimentation pipeline and generate the report (which, if not specified, will always be generated in the root folder of your project within `reports/general_report`).

### Command Line Interface
You can use the command line to run continuous experimentation around a specific metric, generate a report, and capture the best model (if any) around a metric. 

NOTE: From the command line it is only possible to generate the report and the best model result for a single metric at a time.

You can check the available commands by running the following command:

```sh
ml_exp --h
```

An example of using the command line by passing a folder with several Sklearn models saved in Pickle format (.pkl), an X_test and y_test saved in CSV format and indicating, in the optional parameter, the name of the report that will be generated.

```sh
ml_exp accuracy --test_data_paths tests/local/classification/x_test.csv tests/local/classification/y_test.csv test_data --contexts tests/local/classification/model_0.pkl test_data model_test_1 tests/local/classification/model_4.pkl test_data model_test_4 --report_name cli
```

## 💎 Key features and Details
- Generates different test data groups by applying KFold
- Generates certain metrics around these clusters to collect data for use in statistical tests, using the trained models
- Applies a descriptive summary of the distribution of the collected metrics: maximum value, minimum value, mean, median and standard deviation
- Applies a set of statistical tests to verify the existence of significant differences between the models around a metric
- Based on the significant differences, it will search for the best model around the metric in question by comparing the median of the distribution collected for each model
- Organizes all results in JSON and HTML to facilitate decision making

### Statistical Tests Flowchart
The experimentation flow will perform a set of statistical tests that will help in decision making around the existence of a better performance metric. The flow is summarized in the image below.

This flow will be applied for each defined performance metric involving all past trained models and test data.

![alt text](https://github.com/iuryrosal/better-exp/blob/main/images/docs/experimental_pipeline.png)

### Diagram Class
Class diagram of this project.

![alt text](https://github.com/iuryrosal/better-exp/blob/main/images/docs/class_diagram.png)




