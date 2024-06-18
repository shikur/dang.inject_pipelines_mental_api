import mlflow
from zenml import pipeline, step
import pandas as pd
import logging

@step
def load_data(path: str) -> pd.DataFrame:
    """Simulates loading of training data and labels."""
    mlflow.start_run()
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.85)
    mlflow.end_run()

    training_data = [[1, 2], [3, 4], [5, 6]]
    labels = [0, 1, 0]
    # data = pd.read_csv('/home/shiku/repo/dang.milvus_zenml_docker/data/input/qustions_answers/mental_health_faq.csv')
    data = pd.read_csv(path)
    logging.info(data.head(3))
    
    return data #{'features': training_data, 'labels': labels}

@step
def train_model(data:pd.DataFrame) -> None:
    """
    A mock 'training' process that also demonstrates using the input data.
    In a real-world scenario, this would be replaced with actual model fitting logic.
    """
    logging.info(data.head(4))
    # total_features = sum(map(sum, data['features']))
    # total_labels = sum(data['labels'])
    
    # print(f"Trained model using {len(data['features'])} data points. "
    #       f"Feature sum is {total_features}, label sum is {total_labels}")

@pipeline
def pipeline1(path: str):
    """Define a pipeline that connects the steps."""
    dataset = load_data(path)
    train_model(dataset)

if __name__ == "__main__":
    pipe = pipeline1()
    pipe.run()
    # You can now use the `run` object to see steps, outputs, etc.