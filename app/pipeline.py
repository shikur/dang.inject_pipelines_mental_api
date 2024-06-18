import pandas as pd
from zenml import pipeline
from zenml import step
import mlflow

# from sh_run import simple_ml_pipeline
import psycopg2.extras as extras
from sqlalchemy import create_engine
import logging

from pydantic import BaseModel
from typing import Any

class DataFrameValidator(BaseModel):
    data: Any

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def validate(cls, value):
        if isinstance(value, pd.DataFrame):
            return value
        raise ValueError("Value must be a Pandas DataFrame")

def dataframe_validator(value):
    return DataFrameValidator.validate(value)

@step
def load_data() -> pd.DataFrame:
    # import pandas as pd
    data = pd.read_csv("/home/shiku/repo/dang.milvus_zenml_docker/data/input/qustions_answers/mental_health_faq.csv")
    mlflow.log_param("data_rows", data.shape[0])
    mlflow.log_param("data_columns", data.shape[1])
    # Question_ID,Questions,Answers
    data["target"] = 1
    return data

@step
def train_model(data: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    X = data.drop(columns=["target"])
    y = data["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    
    # predictions = model.predict(X_test)
    # mse = mean_squared_error(y_test, predictions)
    
    mlflow.log_param("mse", "mse")
    mlflow.log_param("model", "model")
    
    return data

@pipeline
def simple_ml_pipeline25():
    """Define a pipeline that connects the steps."""
    dataset = load_data()
    if dataset:
        model = train_model(data=dataset)
    
if __name__ == "__main__":
    from zenml.client import Client
    client = Client()

    # Run the pipeline
    # pipeline_instance = simple_ml_pipeline(data_loader=load_data, trainer=train_model)
    # simple_ml_pipeline25(video_path="path/to/video.mp4").run()
    simple_ml_pipeline25()

