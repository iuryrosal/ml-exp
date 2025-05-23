import numpy as np

from service.experimental_pipeline_service import ExperimentalPipelineService


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

def build_normal_and_homoscedastic_3_models_with_significancy():
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


def build_normal_and_homoscedastic_3_models_with_not_significancy():
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

def build_not_normal_and_homoscedastic_3_models_with_not_significancy():
    np.random.seed(42)
    n = 1000

    accuracy_dict = {"accuracy": {"0": [], "1": [], "2": []}}

    for key in accuracy_dict["accuracy"]:
        data = np.random.uniform(low=30, high=70, size=n)
        accuracy_dict["accuracy"][key] = data.tolist()

    return accuracy_dict

def build_not_normal_and_homoscedastic_3_models_with_significancy():
    np.random.seed(42)
    n = 1000

    # Grupo 0: Uniforme (20-40) → Média ~30
    # Grupo 1: Uniforme (40-60) → Média ~50
    # Grupo 2: Uniforme (60-80) → Média ~70
    accuracy_dict = {
        "accuracy": {
            "0": np.random.uniform(20, 40, n).tolist(),
            "1": np.random.uniform(40, 60, n).tolist(),
            "2": np.random.uniform(60, 80, n).tolist()
        }
    }

    return accuracy_dict


def build_normal_not_homoscedastic_3_models_with_not_significancy():
    np.random.seed(42)
    n = 1000

    mean = 50  

    std_devs = {"0": 5, "1": 15, "2": 30}

    accuracy_dict = {"accuracy": {"0": [], "1": [], "2": []}}

    for key in accuracy_dict["accuracy"]:
        data = np.random.normal(loc=mean, scale=std_devs[key], size=n)
        accuracy_dict["accuracy"][key] = data.tolist()
    return accuracy_dict

def build_normal_not_homoscedastic_3_models_with_significancy():
    np.random.seed(42)
    n = 500

    groups = {
        "0": {"mean": 50, "std": 1},    # Baixa variância
        "1": {"mean": 50, "std": 5},    # Variância média
        "2": {"mean": 50, "std": 10}    # Alta variância
    }
    accuracy_dict = {}
    accuracy_dict["accuracy"] = {name: np.random.normal(loc=params["mean"], 
                    scale=params["std"], 
                    size=n) 
            for name, params in groups.items()}
    return accuracy_dict

if __name__ == "__main__":

    exp_1 = ExperimentalPipelineService(scores_data=build_normal_context_2_models_with_not_significancy(),
                                        report_path="reports/test/2_models_normal_not_significancy")
    
    exp_2 = ExperimentalPipelineService(scores_data=build_normal_context_2_models_with_significancy(),
                                        report_path="reports/test/2_models_normal_significancy")

    exp_3 = ExperimentalPipelineService(scores_data=build_normal_and_homoscedastic_3_models_with_significancy(),
                                        report_path="reports/test/3_models_normal_homoscedastic_significancy")
    
    exp_4 = ExperimentalPipelineService(scores_data=build_normal_and_homoscedastic_3_models_with_not_significancy(),
                                        report_path="reports/test/3_models_normal_homoscedastic_not_significancy")
    
    exp_5 = ExperimentalPipelineService(scores_data=build_not_normal_and_homoscedastic_3_models_with_not_significancy(),
                                        report_path="reports/test/3_models_not_normal_and_homoscedastic_not_significancy")
    
    exp_6 = ExperimentalPipelineService(scores_data=build_not_normal_and_homoscedastic_3_models_with_significancy(),
                                        report_path="reports/test/3_models_not_normal_and_homoscedastic_significancy")
    
    exp_7 = ExperimentalPipelineService(scores_data=build_normal_not_homoscedastic_3_models_with_not_significancy(),
                                        report_path="reports/test/3_models_normal_not_homoscedastic_not_significancy")
    
    exp_8 = ExperimentalPipelineService(scores_data=build_normal_not_homoscedastic_3_models_with_significancy(),
                                        report_path="reports/test/3_models_normal_not_homoscedastic_significancy")