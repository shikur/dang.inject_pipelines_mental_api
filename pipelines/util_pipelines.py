from pydantic import BaseModel


class DataConfig(BaseModel):
    path_source: str = None
    path_target: str = None
        
class MilvusConfig(BaseModel):
    milvus_host: str
    milvus_port: int
    collection_name: str
    modelstr: str
    tokenizer_str: str
    index_type: str = "IVF_FLAT"
    params: dict
    metric_type: str
    def __init__(self, **data):
       super().__init__(**data)
       
       
class PostgressDBconnect(BaseModel):
    username: str
    password: str
    host: str
    port: str
    database: str
    def __init__(self, **data):
       super().__init__(**data)
       