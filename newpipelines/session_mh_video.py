
import logging
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
import logging


@step  
def read_video(video_path: str) -> pd.DataFrame:
    logging.info(video_path)
                     
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The file {video_path} does not exist.")
    else:
        logging.info(f"file exists with path {video_path}")
        
    video = VideoFileClip(video_path, verbose=False)
    logging.info("this is test to fwork with")
    
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
    host = 'postgresql_interviewvedio'
    port = '5432'
    database = 'session_db'

# Create an engine
    # engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database}')
    # df.to_sql('session_video', con=engine, if_exists='append', index=False, chunksize=100)

    while retry_count < max_retries:
        try:
            connection = psycopg2.connect(
                host="postgresql_interviewvedio",  # Note: In Docker, this should be the service name, not "localhost"
                port="5432",
                user="postgres_user",
                password="postgres_password",  
                database="session_db"
            )
            logging.info(f"Failed to connect to the database: good")
            break
        except Exception as e:
            logging.info(f"Failed to connect to the database: {str(e)}")
            time.sleep(3)  # Wait for 1 second before retrying
            retry_count += 1
            cursor = None
    tuples = [tuple(x) for x in df.to_numpy()]
    df['data'] = 'data'
    df[['angry']] = 0.1
    df['anger'] = 0.1
    df_f = df[["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise", "data"]]
  
    logging.info(df.head())
    try:
        db_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
        engine = create_engine(db_uri)
        df_f.to_sql('session_video', con=engine, if_exists='append', index=False)
       
    except (Exception, psycopg2.DatabaseError) as error:
        logging.info("Error: %s" % error)
        connection.rollback()
        cursor.close()
        return df_f
    logging.info("execute_batch() done")
  
    logging.info("job done")   
    logging.info(f"Video duration: {video.duration}")
    logging.info(f"Dataframe: {df[emotions].head(10)}")
    
    return df
    
    
# @step
# def getTransofrmed(dfimagedata: pd.DataFrame) -> pd.DataFrame:
#     try:
#         # dirsource = f"{context.resources.video_file_dirs['write_file_dir']}";
#         # filenamesource = f"{context.resources.video_file_dirs['writefilename']}"
#         # fullpath = dirsource + filenamesource

#         logging.info("Transforming data")
#         logging.info(dfimagedata.head(10))
#         logging.info(fullpath)  
#         # '/opt/dagster/app/session_video/data/output/video_data.csv'  
#         # dfimagedata2 = dfimagedata[['angry','disgust','fear','happy','neutral','sad','surprise']] # angry,disgust,fear,happy,neutral,sad,surprise    
#         # with open(fullpath, 'w') as file:
#         #     context.log.error(f"Start writing to  CSV:")
#         #     dfimagedata2.to_csv(file, index=False)
#         #     file.flush()
#         #     file.close()
#         #     context.log.error(f"End writing to  CSV:")
#         # dfimagedata.to_csv('/opt/dagster/app/session_video/data/output/video_data.csv', index=False) 
#         logging.info(f"Dataframe saved to file ")
#         logging.info(f"Dataframe saved to file ")
#         # Yielding or returning an Output object if your logic continues or connects to other operations.
#         return dfimagedata
#     except Exception as e:
#         logging.log.error(f"Failed to save dataframe to CSV: {e}")
#         raise
    # context.info(df.head(10))   
        
    # context.info(f"Video count: {firamelist.count}s")
    # context.info("test works")
    # return "video"
    
    
# @op(required_resource_keys={"video_file_dirs" }) 
# @step
# def extract_audio(context, dfimagedata2: pd.DataFrame) -> str:
#     # audio_path = "./session_video/data/output/" + "temp_audio.wav"
#     # video.audio.write_audiofile(audio_path)
#     context.info("Extracted audio to temp_audio.wav")
#     return 'audio_path'

# @step
# def classify_expression(context, video_path: str):
    
#     context.info("Classified expression this is test")
    
#     return "predicted_class_idx"

@pipeline
def session_video_op_graph(step_instance):
    video_df = step_instance()
    
