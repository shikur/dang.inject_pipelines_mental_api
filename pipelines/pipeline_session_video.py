

import os
import subprocess
import time
from zenml import pipeline, step
import psycopg2
from moviepy.editor import VideoFileClip
import pandas as pd
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
from PIL import Image
import numpy as np
from newpipelines.video_util  import detect_emotions 
import pandas as pd
import psycopg2.extras as extras
from sqlalchemy import create_engine
from pipelines.util_pipelines import PostgressDBconnect, DataConfig
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from logs.logging_config import logger
from zenml.steps import step
from zenml.pipelines import pipeline


@step  
def step_read_video(path: str):
    video_path = path
    logger.info(video_path)
                     
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The file {video_path} does not exist.")
    else:
        logger.info(f"file exists with path {video_path}")
        
    video = VideoFileClip(video_path, verbose=False)
    logger.info("this is test to fwork with")
    
    video = video.without_audio()
    video_data = np.array(list(video.iter_frames()))
    
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    df = pd.DataFrame(columns=emotions)

    for i, frame in enumerate(video_data):
        image = Image.fromarray(frame)
        face, class_probabilities = detect_emotions(image)
        
        new_data = class_probabilities
        # new_data['face'] = face
        
        new_row_df = pd.DataFrame(new_data, index=[i])
        df = pd.concat([df, new_row_df], ignore_index=True)
        
        if i >= 10:  # Adjusted the condition to include 10 frames
            break
    logger.info(f"Video duration: {video.duration}")
    logger.info(f"Dataframe: {df[emotions].head(10)}")
    
    return df


@step
def step_save_data_to_psql(df: pd.DataFrame, postgressdbconn: PostgressDBconnect):
    max_retries = 5
    retry_count = 0
    connection = None
    cursor = None 
    
    # while retry_count < max_retries:
    #     try:
    #         connection = psycopg2.connect(
    #             host= postgressdbconn.host, #"postgresql_interviewvedio",  # Note: In Docker, this should be the service name, not "localhost"
    #             port= postgressdbconn.port ,#"5432",
    #             user= postgressdbconn.username, #"postgres_user",
    #             password=postgressdbconn.password, # "postgres_password",  
    #             database=postgressdbconn.database #"session_db"
    #         )
    #         logger.info(f"Failed to connect to the database: good")
    #         break
    #     except Exception as e:
    #         logger.info(f"Failed to connect to the database: {str(e)}")
    #         time.sleep(3)  # Wait for 1 second before retrying
    #         retry_count += 1
    #         cursor = None
        
    df_f = df #[["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise", "data"]]
    df_f.head(10)
    
    logger.info(df.head())
    try:
        db_uri = f"postgresql+psycopg2://{postgressdbconn.username}:{postgressdbconn.password}@{postgressdbconn.host}:{postgressdbconn.port}/{postgressdbconn.database}"
        engine = create_engine(db_uri)
        df_f.to_sql('session_video', con=engine, if_exists='append', index=False)
       
    except (Exception, psycopg2.DatabaseError) as error:
        logger.info("Error: %s" % error)
        connection.rollback()
        cursor.close()
        return df_f
    
    logger.info("execute_batch() done")  
    logger.info("job done")   
    
    
    return df
    

@pipeline
def pipeline_session_video_op_graph(step_read_video, step_save_data_to_psql):
    video_df = step_read_video()
    step_save_data_to_psql = step_save_data_to_psql(video_df)
    
