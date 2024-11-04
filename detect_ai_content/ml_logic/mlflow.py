
import mlflow
from detect_ai_content.params import *

def load_model(model_name: str, is_tensorflow: bool, stage:str):
    model_uri = f"models:/{model_name}/{stage}"
    if is_tensorflow:
        model = mlflow.tensorflow.load_model(model_uri=model_uri)
    else:
        model = mlflow.sklearn.load_model(model_uri=model_uri)
    return model

def mlflow_save_params(
        training_set_size: int,
        row_count: int,
        dataset_huggingface_human_ai_generated_text: bool,
        dataset_kaggle_ai_generated_vs_human_text: bool,
        dataset_kaggle_daigt_v2_train_dataset: bool,
        additional_parameters = {}
    ) -> None:

    params = {
        "training_set_size": training_set_size,
        "row_count": row_count,
        "dataset_huggingface_human_ai_generated_text": dataset_huggingface_human_ai_generated_text,
        "dataset_kaggle_ai_generated_vs_human_text": dataset_kaggle_ai_generated_vs_human_text,
        "dataset_kaggle_daigt_v2_train_dataset": dataset_kaggle_daigt_v2_train_dataset,
    }

    for k in additional_parameters:
        params[k] = additional_parameters[k]

    mlflow.log_params(params=params)

def mlflow_save_metrics(
        f1_score: float,
        recall_score: float,
        precision_score: float,
        accuracy_score: float
    ) -> None:

    metrics = {
        "f1_score": f1_score,
        "recall_score": recall_score,
        "precision_score": precision_score,
        "accuracy_score": accuracy_score,
    }

    mlflow.log_metrics(metrics=metrics)

def mlflow_save_model(
                        model,
                        model_name: str,
                        is_tensorflow: bool,
                        input_example) -> None:

    # Save model
    if is_tensorflow:
        mlflow.tensorflow.log_model(model=model,
                                    artifact_path="model",
                                    registered_model_name=model_name,
                                    input_example=input_example)
        print("✅ tensorflow Model saved to mlflow")
    else:
        mlflow.sklearn.log_model(sk_model=model,
                                 artifact_path="model",
                                 registered_model_name=model_name,
                                 input_example=input_example)
        print("✅ sklearn Model saved to mlflow")
