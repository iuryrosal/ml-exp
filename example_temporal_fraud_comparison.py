import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from ml_exp import MLExp

def generate_noisy_fraud_data(n_samples=2000, seed=42):
    np.random.seed(seed)
    
    data = np.random.randn(n_samples, 10)
    columns = [f'V{i}' for i in range(1, 11)]
    df = pd.DataFrame(data, columns=columns)
    

    df['Class'] = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    
    df.loc[df['Class'] == 1, 'V1'] += 0.45 
    df.loc[df['Class'] == 1, 'V2'] -= 0.45
    
    return df

def ensure_dirs():
    paths = ["tests/local/data", "tests/local/models", "tests/local/reports"]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

if __name__ == "__main__":
    ensure_dirs()
        
    data_p1 = generate_noisy_fraud_data(n_samples=1000, seed=10)
    X1 = data_p1.drop(["Class"], axis=1)
    y1 = data_p1[["Class"]]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=1)

    data_p2 = generate_noisy_fraud_data(n_samples=1000, seed=20)
    data_p2['V3'] += 0.2
    X2 = data_p2.drop(["Class"], axis=1)
    y2 = data_p2[["Class"]]
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=1)

    model_v1 = LogisticRegression(C=0.1) 
    model_v1.fit(X_train1, y_train1.values.ravel())


    model_v2 = DecisionTreeClassifier(max_depth=3)
    model_v2.fit(X_train2, y_train2.values.ravel())

    with open('tests/local/models/model_v1.pkl', 'wb') as f:
        pickle.dump(model_v1, f)
    with open('tests/local/models/model_v2.pkl', 'wb') as f:
        pickle.dump(model_v2, f)

    X_test1.to_csv("tests/local/data/x_test_v1.csv", index=False)
    y_test1.to_csv("tests/local/data/y_test_v1.csv", index=False)
    X_test2.to_csv("tests/local/data/x_test_v2.csv", index=False)
    y_test2.to_csv("tests/local/data/y_test_v2.csv", index=False)

    # --- MLExp ---
    ml_exp = MLExp(
        scores_target=["accuracy", "roc_auc"],
        report_path="tests/local/reports/temporal_fraud_comparison",
        report_name="experiment_70_percent"
    )

    ml_exp.add_test_data(
        test_data_name="january_test",
        X_test=X_test1,
        y_test=y_test1
    )

    ml_exp.add_test_data(
        test_data_name="february_test",
        X_test=X_test2,
        y_test=y_test2
    )

    ml_exp.add_context(
        context_name="log_reg_v1_jan",
        model_trained="tests/local/models/model_v1.pkl",
        ref_test_data="january_test"
    )

    ml_exp.add_context(
        context_name="tree_v2_feb",
        model_trained="tests/local/models/model_v2.pkl",
        ref_test_data="february_test"
    )

    ml_exp.run()
    

