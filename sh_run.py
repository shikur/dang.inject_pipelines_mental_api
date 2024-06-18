from zenml import pipeline, step
import pandas as pd
import logging

@step
def load_data() -> pd.DataFrame:
    """Simulates loading of training data and labels."""

    training_data = [[1, 2], [3, 4], [5, 6]]
    labels = [0, 1, 0]
    data = pd.read_csv('./data/input/qustions_answers/mental_health_faq.csv')
 
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
def simple_ml_pipeline():
    """Define a pipeline that connects the steps."""
    dataset = load_data()
    train_model(dataset)

if __name__ == "__main__":
    run = simple_ml_pipeline()
    # You can now use the `run` object to see steps, outputs, etc.