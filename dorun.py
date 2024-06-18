import pandas as pd
import sys
sys.path.append('/home/shiku/repo/dang.milvus_zenml_docker')
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections
# from newpipelines.pipeline_qa import injection_qa_pipeline
# from pipelines.qustions_answers.run import main
# from pipelines.qustions_answers.data_loading import load_data
# from pipelines.qustions_answers.insert_data_to_milvus import insert_into_milvus_qa
# from pipelines.qustions_answers.transform import transform_load_data # transform import transform_load_data
from typing import Annotated, Optional, Tuple

from zenml import pipeline, step
from zenml.client import Client

from pipelines_total.qustions_answers.milvus_utils import create_collection_if_not_exists
# from zenml import pipeline
from typing import Annotated, Optional, Tuple

# from zenml import get_step_context, pipeline, step
from zenml.client import Client
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

import pandas as pd

# from pipelines.qustions_answers.milvus_utils import connect_to_milvus, create_collection_if_not_exists,  insert_data_into_milvus

from transformers import AutoTokenizer, AutoModel
import torch
from zenml.steps import BaseStep, step

@step
def load_data() -> pd.DataFrame:
   

    """Load a dataset."""

    data = pd.read_csv('/home/shiku/repo/dang.milvus_zenml_docker/data/input/qustions_answers/mental_health_faq.csv')

    return data
# from pipelines.qustions_answers import pipeline_qa
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
def vectorize_text(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(1).numpy()  # Average pooling
    return embeddings.flatten()

@step
def transform_load_data(data:pd.DataFrame) -> pd.DataFrame: 

    """Load a dataset."""
    df = data

    return df
@step
def insert_into_milvus_qa(df: pd.DataFrame) -> pd.DataFrame:
    df['Answer_Vector'] = df['Answers'].apply(lambda x: vectorize_text(x, tokenizer, model))

# Connect to Milvus
# connecttmilvus =connect_to_milvus(context.resources.conversations_file_dirs_resource['host'], context.resources.conversations_file_dirs_resource['port'])
    milvus_host = "milvus-standalone"
    milvus_port = "19530"

    connections.connect(host=milvus_host, port=milvus_port, timeout=120)
    collection_name = "test_data"
    index_params = {
    "index_type": "IVF_FLAT",  # Example index type
    "params": {"nlist": 1024},  # Example parameter, adjust based on your data and requirements
    "metric_type": "L2"  # Example metric, choose "L2" or "IP" based on your use case
    }

    if not utility.has_collection(collection_name):
        id_field = FieldSchema(name="Question_ID", dtype=DataType.INT64, is_primary=True, auto_id=False)
        Answer_field = FieldSchema(name="Answers", dtype=DataType.VARCHAR, max_length=2000, description="foreign id of vector in a different database")
        vector_field = FieldSchema(name="Answer_Vector", dtype=DataType.FLOAT_VECTOR, dim=384)
        schema = CollectionSchema(fields=[id_field, Answer_field, vector_field], description="Question and Answers")
        create_collection_if_not_exists(collection_name, schema)

    collection = Collection(name=collection_name)
    collection.create_index(field_name="Answer_Vector", index_params=index_params)

    # Prepare data for insertion
    question_ids = df['Question_ID'].tolist()
    df['Answers']= df['Answers'].str.slice(stop=550)
    Answer= df['Answers'].tolist()
    vectors = df['Answer_Vector'].tolist()

    # Insert data into Milvus
    datauploaded = collection.insert([question_ids, Answer, vectors])
    # context.log.info(f"Inserted data into Milvus with IDs: {datauploaded.primary_keys}")

    # utility.flush([collection_name])
    collection.load()

    return df

@pipeline
def injection_qa_pipeline():
    raw_data_qa = load_data()  
    # print(raw_data_qa.head(10))  
    transform_data_qa = transform_load_data(raw_data_qa)
    data_saved = insert_into_milvus_qa(transform_data_qa) 

def main():
    training = injection_qa_pipeline()
    # # get_data = load_data(),
    # # transform_data_qa = transform_load_data(),
    # # post_data = insert_into_milvus_qa() )
    # # training=  pipeline_qa.injection_qa_pipeline()
    # training.run()


if __name__ == '__main__':
    main()
    

