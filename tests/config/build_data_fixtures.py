import pytest
import numpy as np


@pytest.fixture
def build_normal_context_2_models_with_not_significancy():
    media_0 = 50
    desvio_0 = 15

    media_1 = 70
    desvio_1 = 10

    valores_0 = np.random.normal(loc=media_0, scale=desvio_0, size=100)
    valores_1 = np.random.normal(loc=media_1, scale=desvio_1, size=100)

    valores_0 = np.clip(valores_0, 10, 95).tolist()
    valores_1 = np.clip(valores_1, 10, 95).tolist()

    dados = {
        "accuracy": {
            "0": valores_0,
            "1": valores_1
        }
    }

    return dados

@pytest.fixture
def build_normal_context_2_models_with_significancy():
    media_0 = 50
    desvio_0 = 5

    media_1 = 75
    desvio_1 = 5

    valores_0 = np.random.normal(loc=media_0, scale=desvio_0, size=100)
    valores_1 = np.random.normal(loc=media_1, scale=desvio_1, size=100)

    valores_0 = np.clip(valores_0, 10, 95).tolist()
    valores_1 = np.clip(valores_1, 10, 95).tolist()

    dados = {
        "accuracy": {
            "0": valores_0,
            "1": valores_1
        }
    }
    return dados

@pytest.fixture
def build_normal_and_homoscedastic_3_models_with_significancy():
    """To fill the dictionary with values ​​that follow a normal distribution and are homoscedastic, but with different means to create statistically significant difference
        Set different means for each group (0, 1, 2) to ensure statistical difference.
        Keep the same standard deviation (homoscedasticity).
        Make sure the values ​​are between 10 and 95 (we can truncate values ​​outside the range).
    """
    np.random.seed(42)  
    n = 1000
    std_dev = 2

    means = {
        "0": 20,
        "1": 50,
        "2": 80
    }

    accuracy_dict = {"accuracy": {"0": [], "1": [], "2": []}}

    for key in accuracy_dict["accuracy"]:
        data = np.random.normal(loc=means[key], scale=std_dev, size=n)
        data = np.clip(data, 10, 95)
        accuracy_dict["accuracy"][key] = data.tolist()

    return accuracy_dict

@pytest.fixture
def build_normal_and_homoscedastic_3_models_with_not_significancy():
    """To fill the dictionary with values ​​that follow a normal distribution and are homoscedastic, but without a significant difference between groups:
        Use the same mean for all groups (or very close means).
        Maintain homoscedasticity (same standard deviation).
        Ensure that variability does not mask nonexistent differences.
    """
    np.random.seed(42)  
    n = 1000
    std_dev = 5

    means = {
        "0": 50,
        "1": 50.1,
        "2": 50.2
    }

    accuracy_dict = {"accuracy": {"0": [], "1": [], "2": []}}

    for key in accuracy_dict["accuracy"]:
        data = np.random.normal(loc=means[key], scale=std_dev, size=n)
        data = np.clip(data, 10, 95)
        accuracy_dict["accuracy"][key] = data.tolist()

    return accuracy_dict

@pytest.fixture
def build_not_normal_and_homoscedastic_3_models_with_not_significancy():
    """To generate non-normal data without significant differences between groups, we can use distributions such as uniform, modified exponential or logistic with the same parameters for all groups.
        Uniform: All groups will have values ​​equally distributed between a range (e.g.: 20 to 80).
        Modified Exponential Distribution: For asymmetric data, but with the same mean and variance.
        Logistic Distribution: Similar to normal, but with heavier tails.
    """
    np.random.seed(42)
    n = 1000

    accuracy_dict = {"accuracy": {"0": [], "1": [], "2": []}}

    for key in accuracy_dict["accuracy"]:
        data = np.random.uniform(low=30, high=70, size=n)
        accuracy_dict["accuracy"][key] = data.tolist()

    return accuracy_dict

@pytest.fixture
def build_not_normal_and_homoscedastic_3_models_with_significancy():
    """
    To generate non-normal data with significant differences between groups, we can use distributions such as uniform, exponential or Poisson, but with different parameters for each group, ensuring that their means/medians are statistically distinct.
    Uniform Distribution (with different ranges per group).
    Exponential Distribution (with different means).
    Poisson Distribution (with different rates).
    """
    np.random.seed(42)
    n = 1000

    # Group 0: Uniform (20-40) → Mean ~30
    # Group 1: Uniform (40-60) → Mean ~50
    # Group 2: Uniform (60-80) → Mean ~70
    accuracy_dict = {
        "accuracy": {
            "0": np.random.uniform(20, 40, n).tolist(),
            "1": np.random.uniform(40, 60, n).tolist(),
            "2": np.random.uniform(60, 80, n).tolist()
        }
    }

    return accuracy_dict

@pytest.fixture
def build_normal_not_homoscedastic_3_models_with_not_significancy():
    """To generate non-homoscedastic normal data with no significant difference:
        Same population mean across all groups (e.g. μ = 50)
        Different variances (e.g. σ² = 1, 25, 100)
        Normality maintained within each group individually
    """
    np.random.seed(42)
    n = 1000

    mean = 50  

    std_devs = {"0": 5, "1": 15, "2": 30}

    accuracy_dict = {"accuracy": {"0": [], "1": [], "2": []}}

    for key in accuracy_dict["accuracy"]:
        data = np.random.normal(loc=mean, scale=std_devs[key], size=n)
        accuracy_dict["accuracy"][key] = data.tolist()
    return accuracy_dict

@pytest.fixture
def build_normal_not_homoscedastic_3_models_with_significancy():
    """To generate normal, non-homoscedastic, and significantly different data:
        Different means to ensure significant difference.
        Different variances to violate homoscedasticity.
        Normal distribution in each group.
    """
    np.random.seed(42)
    n = 500

    groups = {
        "0": {"mean": 50, "std": 1},    # Low variance
        "1": {"mean": 50, "std": 5},    # Average variance
        "2": {"mean": 50, "std": 10}    # High variance
    }
    accuracy_dict = {}
    accuracy_dict["accuracy"] = {name: np.random.normal(loc=params["mean"], 
                    scale=params["std"], 
                    size=n) 
            for name, params in groups.items()}
    return accuracy_dict