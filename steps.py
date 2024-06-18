# steps.py

from zenml.steps import step, BaseParameters
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

class LoadDataParameters(BaseParameters):
    path: str

@step
def load_data(params: LoadDataParameters) -> pd.DataFrame:
    """Load data from a CSV file."""
    df = pd.read_csv(params.path)
    return df

@step
def train_model(data: pd.DataFrame):
    """Train a simple linear regression model."""
    model = LinearRegression()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    model.fit(X, y)
    joblib.dump(model, 'model.pkl')

class StepOneParameters(BaseParameters):
    param1: str

@step
def step_one(params: StepOneParameters):
    """Example step one."""
    print(f"Step one executed with param1: {params.param1}")

@step
def step_two():
    """Example step two."""
    print("Step two executed.")
