
# import pandas as pd


# zenml importing
from zenml import step, log_artifact_metadata
from zenml.metadata.metadata_types import StorageSize
# from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

import pandas as pd


# from milvus_utils import connect_to_milvus, create_collection_if_not_exists,  insert_data_into_milvus



# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

@step
def load_data() -> pd.DataFrame:
   

    """Load a dataset."""

    data = pd.read_csv('/home/shiku/repo/dang.milvus_zenml_docker/data/input/qustions_answers/mental_health_faq.csv')

    return data


# import pandas as pd
# from zenml.steps import BaseStep, step

# class load_data(BaseStep):
#     def entrypoint(self, inputs) -> pd.DataFrame:
#          """Load a dataset."""

#          data = pd.read_csv('/app/data/input/qustions_answers/mental_health_faq.csv')
         
#          return data
