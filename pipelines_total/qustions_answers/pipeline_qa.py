
from typing import Annotated, Optional, Tuple

from zenml import get_step_context, pipeline, step
from zenml.client import Client
from pipelines_total.qustions_answers.transform           import  transform_load_data
from pipelines_total.qustions_answers.data_loading import load_data
from pipelines_total.qustions_answers.insert_data_to_milvus   import  insert_into_milvus_qa

# , transform_load_data, post_data_to_milvus

@pipeline
def injection_qa_pipeline():
    raw_data_qa = load_data()  
    # print(raw_data_qa.head(10))  
    transform_data_qa = transform_load_data(raw_data_qa)
    data_saved = insert_into_milvus_qa(transform_data_qa)   
    
    
    

# from zenml.pipelines import BasePipeline, pipeline

# class injection_qa_pipeline(BasePipeline):
#     def connect(self, raw_data_qa, transform_data_qa, insert_into_milvus_qa ):
#         output1 = raw_data_qa()
#         output2 = transform_data_qa(input=output1)
#         output3 = insert_into_milvus_qa(output2)

# To instantiate your pipeline
# my_pipeline = pipeline(MyPipeline)   