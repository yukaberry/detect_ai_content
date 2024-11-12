
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

def evaluate_model(model, X_test_processed, y_test):
    print(f'evaluate_model:{model}')
    y_pred = model.predict(X_test_processed)
    df = pd.DataFrame(data=y_pred)
    # print(df)
    if len(df.value_counts()) > 2:   # classes issue
        print("evaluate_model FIX predictions (continuous to Classes)")
        print(f"y_test: {y_test}")
        print(f"y_pred: {y_pred}")

        classes = []
        for (index, row) in df.iterrows():
            pred = row[0]
            if pred > 0.5:
                classes.append(1)
            else:
                classes.append(0)
        df[0] = classes

    y_pred = df[0]
    return {
        "accuracy_score": accuracy_score(y_true=y_test, y_pred=y_pred),
        "f1_score" : f1_score(y_true=y_test, y_pred=y_pred),
        "precision_score": precision_score(y_true=y_test, y_pred=y_pred),
        "recall_score" : recall_score(y_true=y_test, y_pred=y_pred)
    }
