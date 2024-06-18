# import pandas as pd
# from zenml import step

import pandas as pd
from zenml import step


@step(enable_cache=False)
def transform_load_data(data=pd.DataFrame) -> pd.DataFrame: 

    """Load a dataset."""
    df = data

    return df

# import pandas as pd
# from zenml.steps import BaseStep, step

# class transform(BaseStep):
#     def entrypoint(self, inputs) -> pd.DataFrame:
#          """Load a dataset."""

#          data = pd.read_csv('/home/shiku/repo/dang.milvus_zenml_docker/data/input/qustions_answers/mental_health_faq.csv')
         
#          return data
