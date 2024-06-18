from .pipeline_qa import injection_qa_pipeline
# from pipelines.qustions_answers.run import main
# from pipelines.qustions_answers.data_loading import load_data
# from pipelines.qustions_answers.insert_data_to_milvus import insert_into_milvus_qa
# from pipelines.qustions_answers.transform import transform_load_data # transform import transform_load_data
from typing import Annotated, Optional, Tuple

from zenml import get_step_context, pipeline, step
from zenml.client import Client
from pipelines_total.qustions_answers import pipeline_qa
def main():
    training = injection_qa_pipeline()
    # get_data = load_data(),
    # transform_data_qa = transform_load_data(),
    # post_data = insert_into_milvus_qa() )
    # training=  pipeline_qa.injection_qa_pipeline()
    training.run()


if __name__ == '__main__':
    main()
    
    
#     #  raw_data_qa = get_data    
#     # transform_data_qa = transform_data(raw_data_qa)
#     # post_data_to_milvus(transform_data_qa)

# from zenml.core.repo import Repository


# def main():
#     repo = Repository()
#     pipeline = repo.get_pipeline(name='injection_qa_pipeline')
    
#     # Configure your pipeline run
#     config = {
#         'insert_into_milvus_qa': {
#             # Configuration options
#         }
#     }
    
#     pipeline.run(config)

# if __name__ == '__main__':
#     main()
