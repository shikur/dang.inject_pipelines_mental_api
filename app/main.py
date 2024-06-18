'''
----------------------------------------------------------------------------------------------------------------------

           Objective: Use this service to expose MLOPS workflow service endpoints related to this process.
           
----------------------------------------------------------------------------------------------------------------------


Generic import packages required by fastapi endpoints, os and config
'''
import os
from fastapi import FastAPI, File, HTTPException, UploadFile, requests
from pydantic import BaseModel
import yaml
from fastapi.middleware.cors import CORSMiddleware
import subprocess
from dotenv import load_dotenv

from newpipelines.session_mh_video import read_video, session_video_op_graph
# from pipelines.pipeline_session_video import pipeline_session_video_op_graph,  step_read_video , step_save_data_to_psql
from pipelines.util_pipelines import PostgressDBconnect

'''
loging stracture per application and per API service endpoints
'''
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from logs.logging_config import logger

'''
packagers required for MLOPS realted activities from zenml
'''
from zenml import step, pipeline
import run as test
from zenml.client import Client
from zenml.client import Client

'''
import required module and  package per individual workflow\pipeline 
Note: More text based activities

'''
from pipelines.pipeline1 import pipeline1
# from pipelines.pipeline_qa import DataConfig, MilvusConfig, pipeline_qa_with_milvus, Load_qa_step, preprocessing_qa, intermediate_qa, insert_milvus_qa

from pipelines.pipeline_qa import  pipeline_qa_with_milvus, Load_qa_step, preprocessing_qa, intermediate_qa, insert_milvus_qa
from pipelines.pipeline_session_video import pipeline_session_video_op_graph,  step_read_video , step_save_data_to_psql
from pipelines_total.qustions_answers import run
from pipelines.pipeline_qa import  Load_qa_step,  pipeline_qa_with_milvus # pipeline related to document for qustion and answer
# from newpipelines.session_mh_video import session_video_op_graph, read_video
from app.api_util import  get_config_context, read_config, run_pipeline_with_config, get_config
from pipelines.util_pipelines import DataConfig, MilvusConfig,  PostgressDBconnect

'''
Application foused setup
'''
app = FastAPI()

logger.info("Coach Assistance API Service endpoint started\re-stared")

# Load the .env file
load_dotenv()

# Access the environment variables
global_config = os.getenv('CONFIG_APP')
secret_key = os.getenv('SECRET_KEY')
debug = os.getenv('DEBUG')
class PipelineInput(BaseModel):
    input_path: str
    
origins = ["*"]
app.add_middleware( CORSMiddleware,  allow_origins=["*"],  allow_credentials=True,  allow_methods=["*"], allow_headers=["*"], )
client = Client()
   


@app.post("/run-pipeline-config/")
def run_pipeline_config(input: PipelineInput):
    config_file = os.path.join("/app/", "pipeline_config.yml")
    try:
        subprocess.run(
            ["zenml", "pipeline", "run", "simple_ml_pipeline2", "--config", config_file],
            check=True
        )
        return {"status": "Pipeline executed successfully"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")


# looks this is working as expected  
@app.post("/session_emssion/")
def run_pipeline(input: PipelineInput):
    logger.info(f'Starting with vedio session with client (used to capture image anlysis for emation')  
    try:
        # Create a step instance with the provided input path
        step_instance = read_video(video_path=input.input_path)

        # Run the pipeline with the step instance
        session_video_op_graph(step_instance=step_instance)
        logger.info(f"status: Pipeline executed successfully")
        return {"status": "Pipeline executed successfully"}
    except subprocess.CalledProcessError as e:
      raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")

@app.post("/qustion_answer_data/{pipeline_name}")
def pipeline_name(pipeline_name: str):
    logger.info(f'Starting with data injestion related with mental health qustion and answer data') 
    try:
        #get service level config file, this used to get config per pipeline
        global_config_path = get_config(global_config)
        
        # Used to discover pipeline live config file location
        config_file_path = get_config_context(global_config_path, pipeline_name)
        
        # read config for current pipeline workflow
        config = read_config(config_file_path)
        
        # Extract step-specific configurations
        get_path_source = config["pipelines"][pipeline_name]["steps"][0]['parameters']["path"]
        data_config = DataConfig(path_source=get_path_source, path_target=config["pipelines"][pipeline_name]["steps"][0]['parameters']["path_output"])
        milvus_config = MilvusConfig(**global_config_path["milvus"])

        # set up pipeline workflow to run
        pipeline_instance = pipeline_qa_with_milvus(
            Load_qa_step=Load_qa_step().configure(parameters={"path": data_config.path_source}),
            preprocessing_qa=preprocessing_qa(),
            intermediate_qa=intermediate_qa().configure(parameters={"path": data_config.path_target}),
            insert_milvus_qa=insert_milvus_qa().configure(parameters={"config": milvus_config})
        )
        
        # Run the pipeline
        pipeline_instance.run()
        logger.info(f"status: Pipeline executed successfully")
     
    except Exception as e:
        logger.error(f'Error in running pipeline: {e}')
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/session/{session_emission}")
def pipesession_emission(pipeline_name: str):
    logger.info(f'Starting with data injestion related with mental health qustion and answer data') 
    try:
        #get service level config file, this used to get config per pipeline
        global_config_path = get_config(global_config)
        
        # Used to discover pipeline live config file location
        config_file_path = get_config_context(global_config_path, pipeline_name)
        
        # read config for current pipeline workflow
        config = read_config(config_file_path)
        
        # Extract step-specific configurations
        get_path_source = config["pipelines"][pipeline_name]["steps"][1]['parameters']["path"]
        data_config = DataConfig(path_source=get_path_source, path_target=config["pipelines"][pipeline_name]["steps"][1]['parameters']["path_output"])
        postgres_config = PostgressDBconnect(**global_config_path["postgres_connect"])

        # set up pipeline workflow to run
        pipeline_instancesession = pipeline_session_video_op_graph(
            step_read_video=step_read_video().configure(parameters={"path": data_config.path_source}),
            step_save_data_to_psql=step_save_data_to_psql().configure(parameters={"postgressdbconn": postgres_config})
        )
        
        # Run the pipeline
        pipeline_instancesession.run()
        logger.info(f"status: Pipeline executed successfully")
     
    except Exception as e:
        logger.error(f'Error in running pipeline: {e}')
        raise HTTPException(status_code=500, detail=str(e))
    
       