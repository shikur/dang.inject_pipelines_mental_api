



# from pymilvus import connections
# from pipelines.qustions_answers.milvus_utils import connect_to_milvus, create_collection_if_not_exists,  insert_data_into_milvus

# from transformers import AutoTokenizer, AutoModel
# import torch
# # from zenml import BaseStep, step
# import logging

# milvus_host = "milvus-standalone"
# milvus_port = "19530"
# try:
#     connections.connect(host=milvus_host, port=milvus_port, timeout=300)
#     logging.info("Successfully connected to Milvus server.")
# except Exception as e:
#     logging.info(f"Failed to connect to Milvus server: {e}")
#     # connections.connect(host=milvus_host, port=milvus_port, timeout