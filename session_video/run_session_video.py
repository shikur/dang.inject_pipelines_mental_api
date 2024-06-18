
import logging
import os
import subprocess
import time

from zenml import step

import psycopg2
from moviepy.editor import VideoFileClip
import pandas as pd
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
from PIL import Image
import numpy as np



import pandas as pd
import os

import psycopg2.extras as extras
from sqlalchemy import create_engine
import logging



@step 
def read_video(video_path: str) -> pd.DataFrame:
    logging.log.info(video_path)
                     
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The file {video_path} does not exist.")
    else:
        logging.log.info(f"file exists with path {video_path}")
        
    video = VideoFileClip(video_path, verbose=False)
    logging.log.info("this is test to fwork with")
    
    video = video.without_audio()
    video_data = np.array(list(video.iter_frames()))
    
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    df = pd.DataFrame(columns=emotions)

    for i, frame in enumerate(video_data):
        image = Image.fromarray(frame)
        face, class_probabilities = detect_emotions(image)
        
        new_data = class_probabilities
        new_data['face'] = face
        
        new_row_df = pd.DataFrame(new_data, index=[i])
        df = pd.concat([df, new_row_df], ignore_index=True)
        
        if i >= 10:  # Adjusted the condition to include 10 frames
            break
        
    df = df.head(200)    
    df['data'] = ''
    df['anger'] = ''
    cols = ','.join(list(df.columns)) 
    max_retries = 5
    retry_count = 0
    connection = None
    cursor = None 
    tuples = [tuple(x) for x in df.to_numpy()]
   

# Replace these with your actual credentials
    username = 'postgres_user'
    password = 'postgres_password'
    host = 'postgresql_inputvedio'
    port = '5432'
    database = 'session_db'

    while retry_count < max_retries:
        try:
            connection = psycopg2.connect(
                host="postgresql_inputvedio",  # Note: In Docker, this should be the service name, not "localhost"
                port="5432",
                user="postgres_user",
                password="postgres_password",  
                database="session_db"
            )
            logging.log.info(f"Failed to connect to the database: good")
            break
        except Exception as e:
            logging.log.info(f"Failed to connect to the database: {str(e)}")
            time.sleep(3)  # Wait for 1 second before retrying
            retry_count += 1
            cursor = None
    tuples = [tuple(x) for x in df.to_numpy()]
    df['data'] = 'data'
    df[['angry']] = 0.1
    df['anger'] = 0.1
    df_f = df[["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise", "data"]]
    df_f.head(10)
    # Comma-separated dataframe columns
    cols = ','.join(list(df_f.columns))
    # SQL quert to execute
    table = 'session_video'
    page_size = 10

    logging.log.info(df.head())
    try:
        db_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
        engine = create_engine(db_uri)
        df_f.to_sql('session_video', con=engine, if_exists='append', index=False)
       
    except (Exception, psycopg2.DatabaseError) as error:
        logging.log.info("Error: %s" % error)
        connection.rollback()
        cursor.close()
        return df_f
    logging.log.info("execute_batch() done")
  
    logging.log.info("job done")   
    logging.log.info(f"Video duration: {video.duration}")
    logging.log.info(f"Dataframe: {df[emotions].head(10)}")
    
    return df
    
   
@step
def getTransofrmed(context, dfimagedata: pd.DataFrame) -> pd.DataFrame:
    try:
        context = logging
        dirsource = f"{context.resources.video_file_dirs['write_file_dir']}";
        filenamesource = f"{context.resources.video_file_dirs['writefilename']}"
        fullpath = dirsource + filenamesource
        context.info("this is test")
        context.info("Transforming data")
        context.info(dfimagedata.head(10))
        context.info(fullpath)  
        context.info(f"Dataframe saved to file ")
        context.info(f"Dataframe saved to file ")
        # Yielding or returning an Output object if your logic continues or connects to other operations.
        return dfimagedata
    except Exception as e:
        context.error(f"Failed to save dataframe to CSV: {e}")
        raise
   
@step
def extract_audio(context, dfimagedata2: pd.DataFrame) -> str:
    # audio_path = "./session_video/data/output/" + "temp_audio.wav"
    # video.audio.write_audiofile(audio_path)
    context.log.info("Extracted audio to temp_audio.wav")
    return 'audio_path'

@step
def classify_expression(context, video_path: str):
   
    context.log.info("Classified expression this is test")
    
    return "predicted_class_idx"

@graph
def session_video_op_graph():
    video = read_video()
    
    
