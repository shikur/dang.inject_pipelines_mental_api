import pandas as pd
import numpy as np
from zenml.core.pipelines.training_pipeline import TrainingPipeline
from zenml.steps import BaseStep, Output, step
from zenml.steps.step_interfaces.base_step_config import BaseStepConfig
from zenml.repository import Repository
from zenml.artifacts import DataArtifact
from zenml.integrations.sklearn.helpers.digits import get_digits
from zenml import step


# Define the configuration classes for each step
class DataConfig(BaseStepConfig):
    training_data_path: str
    validation_data_path: str
    test_data_path: str

class PreprocessingConfig(BaseStepConfig):
    normalize: bool
    handle_missing_values: str
    feature_scaling: str

class TrainingConfig(BaseStepConfig):
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str

class EvaluationConfig(BaseStepConfig):
    metrics: list
    evaluation_data_path: str

class OutputConfig(BaseStepConfig):
    model_save_path: str
    metrics_save_path: str

# Step 1: Data Loading
@step
def data_loading_step(config: DataConfig) -> Output(training_data=pd.DataFrame, validation_data=pd.DataFrame, test_data=pd.DataFrame):
    training_data = pd.read_csv(config.training_data_path)
    validation_data = pd.read_csv(config.validation_data_path)
    test_data = pd.read_csv(config.test_data_path)
    return training_data, validation_data, test_data

# Step 2: Data Preprocessing
@step
def data_preprocessing_step(config: PreprocessingConfig, training_data: pd.DataFrame, validation_data: pd.DataFrame, test_data: pd.DataFrame) -> Output(processed_training_data=pd.DataFrame, processed_validation_data=pd.DataFrame, processed_test_data=pd.DataFrame):
    if config.normalize:
        training_data = (training_data - training_data.mean()) / training_data.std()
        validation_data = (validation_data - validation_data.mean()) / validation_data.std()
        test_data = (test_data - test_data.mean()) / test_data.std()
    if config.handle_missing_values == "mean":
        training_data.fillna(training_data.mean(), inplace=True)
        validation_data.fillna(validation_data.mean(), inplace=True)
        test_data.fillna(test_data.mean(), inplace=True)
    return training_data, validation_data, test_data

# Step 3: Model Training
@step
def model_training_step(config: TrainingConfig, training_data: pd.DataFrame, validation_data: pd.DataFrame) -> Output(model=object):
    from sklearn.linear_model import SGDClassifier
    
    X_train = training_data.drop('target', axis=1)
    y_train = training_data['target']
    X_val = validation_data.drop('target', axis=1)
    y_val = validation_data['target']
    
    model = SGDClassifier(learning_rate='constant', eta0=config.learning_rate, max_iter=config.epochs)
    model.fit(X_train, y_train)
    
    # Optionally validate the model
    val_score = model.score(X_val, y_val)
    print(f"Validation Score: {val_score}")
    
    return model

# Step 4: Model Evaluation
@step
def model_evaluation_step(config: EvaluationConfig, model: object, test_data: pd.DataFrame):
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro')
    }
    
    print(f"Evaluation Metrics: {metrics}")
    
    return metrics

# Step 5: Save Model and Metrics
@step
def save_outputs_step(config: OutputConfig, model: object, metrics: dict):
    import joblib
    import json
    import os

    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.metrics_save_path, exist_ok=True)
    
    model_path = os.path.join(config.model_save_path, "model.joblib")
    metrics_path = os.path.join(config.metrics_save_path, "metrics.json")
    
    joblib.dump(model, model_path)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")

# Define the pipeline
class MyMLPipeline(TrainingPipeline):
    def configure_pipeline(self):
        data_config = DataConfig(
            training_data_path="s3://my-bucket/training_data.csv",
            validation_data_path="s3://my-bucket/validation_data.csv",
            test_data_path="s3://my-bucket/test_data.csv"
        )
        preprocessing_config = PreprocessingConfig(
            normalize=True,
            handle_missing_values="mean",
            feature_scaling="standard"
        )
        training_config = TrainingConfig(
            batch_size=32,
            epochs=10,
            learning_rate=0.001,
            optimizer="adam"
        )
        evaluation_config = EvaluationConfig(
            metrics=["accuracy", "precision", "recall"],
            evaluation_data_path="s3://my-bucket/evaluation_data.csv"
        )
        output_config = OutputConfig(
            model_save_path="s3://my-bucket/models/model_v1/",
            metrics_save_path="s3://my-bucket/metrics/experiment_1/"
        )
        
        self.add_step(data_loading_step(data_config))
        self.add_step(data_preprocessing_step(preprocessing_config))
        self.add_step(model_training_step(training_config))
        self.add_step(model_evaluation_step(evaluation_config))
        self.add_step(save_outputs_step(output_config))

# Run the pipeline
if __name__ == "__main__":
    pipeline = MyMLPipeline()
    pipeline.run()
