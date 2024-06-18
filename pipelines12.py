# pipelines.py

from zenml.pipelines import pipeline
from steps import load_data, train_model, step_one, step_two

@pipeline
def simple_ml_pipeline2(load_data, train_model):
    """Simple ML pipeline with data loading and model 
    training."""
    data = load_data()
    train_model(data)

@pipeline
def second_pipeline(step_one, step_two):
    """Second pipeline with two steps."""
    step_one()
    step_two()


# Register the pipeline
simple_ml_pipeline2_instance = simple_ml_pipeline2(
    load_data=load_data(),
    train_model=train_model()
)
simple_ml_pipeline2_instance.run()