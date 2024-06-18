import subprocess
from fastapi import HTTPException
from logs.logging_config import logger
import yaml


#*********************************** Begin API Service Level Functions *****************************

def read_config(global_config :str):
    logger.info(f'global config for pipelines {global_config}')
    try:
        with open(global_config, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f'{global_config} file read with no  issue')   
        return config
    except Exception as e:
        logger.error(f'Error: Not able to read file {global_config} ')   
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")
    
def run_pipeline_with_config(pipeline_name: str, config_file: str):
    logger.info(f'global config for pipelines {pipeline_name} with config {config_file}')
    try:
        subprocess.run(
            ["zenml", "pipeline", "run", pipeline_name, "--config", config_file], 
            check=True
        )
        logger.info(f"status: Pipeline:{pipeline_name} run initiated successfully")
        return {"status": "Pipeline run initiated successfully"}
    except subprocess.CalledProcessError as e:
        logger.error(f'Error: Not able to read file {e} ')  
        return {"status": "Pipeline run failed", "error": str(e)}
    
def get_config_context(config_file, pipeline_name):
    logger.info(f'context config for pipelines {config_file}')
           
    for item in config_file.get('config_base', []):
        if item.get('name') == pipeline_name:
            return "".join([item.get('config_base_path'),pipeline_name,".yml"]) 
    logger.error(f'Error: Not able to read file {config_file} and {pipeline_name} ')
    return None 

def get_config(path):
    config = read_config(path)
    return config  

#*********************************** END API Service Level Functions *****************************