# '''
# 1. load data from qa
# 2. remove any null row
# 3. save intermidate data into csv file
# 4. save data into milvus victore database

# '''
import logging
from fastapi import HTTPException
import torch
# from zenml import pipeline, step
import pandas as pd
import yaml
from zenml.pipelines import BasePipeline
from zenml.steps import BaseStep
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from transformers import AutoTokenizer, AutoModel
from pipelines_total.qustions_answers.milvus_utils import connect_to_milvus, create_collection_if_not_exists, insert_data_into_milvus
from zenml.steps import step
from zenml.pipelines import pipeline
from pipelines.util_pipelines import PostgressDBconnect, DataConfig, MilvusConfig

       


def vectorize_text(text, tokenizer, model):
    try:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(1).numpy()  # Average pooling
        return embeddings.flatten()
    except Exception as e:
        logging.error(f'Error in insert_milvus_qa: {e}')


# Define steps
@step
def Load_qa_step(path: str):
    # path = "/home/shiku/repo/dang.milvus_zenml_docker/data/input/qustions_answers/mental_health_faq.csv"
    df = pd.read_csv(path)
    df.head(10)
    logging.info(df.head(10))
    return df

@step
def preprocessing_qa(data: pd.DataFrame):
    return data.drop_duplicates().dropna() if not data.empty else data

@step
def intermediate_qa(data: pd.DataFrame, path: str):
    df = data.drop_duplicates().dropna() if not data.empty else data
    df.to_csv(path, index=False)
    return df

@step
def insert_milvus_qa(config: MilvusConfig, data: pd.DataFrame):
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_str)
        model = AutoModel.from_pretrained(config.modelstr)    
        data['Answer_Vector'] = data['Answers'].apply(lambda x: vectorize_text(x, tokenizer, model))

        connections.connect(host=config.milvus_host, port=config.milvus_port, timeout=120)
        collection_name = config.collection_name
        index_params = {"index_type": config.index_type, "params": config.params, "metric_type": config.metric_type}

        if not utility.has_collection(collection_name):
            id_field = FieldSchema(name="Question_ID", dtype=DataType.INT64, is_primary=True, auto_id=False)
            answer_field = FieldSchema(name="Answers", dtype=DataType.VARCHAR, max_length=2000, description="foreign id of vector in a different database")
            vector_field = FieldSchema(name="Answer_Vector", dtype=DataType.FLOAT_VECTOR, dim=384)
            schema = CollectionSchema(fields=[id_field, answer_field, vector_field], description="Question and Answers")
            create_collection_if_not_exists(collection_name, schema)

        collection = Collection(name=collection_name)
        collection.create_index(field_name="Answer_Vector", index_params=index_params)

        question_ids = data['Question_ID'].tolist()
        data['Answers'] = data['Answers'].str.slice(stop=550)
        answers = data['Answers'].tolist()
        vectors = data['Answer_Vector'].tolist()

        collection.insert([question_ids, answers, vectors])
        collection.load()
    except Exception as e:
        logging.error(f'Error in insert_milvus_qa: {e}')

def read_config(path :str):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

@pipeline
def pipeline_qa_with_milvus(Load_qa_step, preprocessing_qa, intermediate_qa, insert_milvus_qa): #, preprocessing_qa, intermediate_qa, insert_milvus_qa):
    data = Load_qa_step()
    preprocessing_qa_data = preprocessing_qa(data)
    intermediate_qa = intermediate_qa(preprocessing_qa_data)
    
    # intermediate_data = intermediate_qa(preprocessing_qa_data)
    insert_milvus_qa = insert_milvus_qa(data=preprocessing_qa_data)
