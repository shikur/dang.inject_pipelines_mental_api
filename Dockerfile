



FROM python:3.12-slim

RUN pip install --upgrade pip && \
    pip install pyarrow pymilvus
# Install necessary Python packages

# RUN pip install --upgrade pip setuptools wheel cython
# # RUN pip install PyYAML


# RUN RUN apt-get update && apt-get install -y build-essential libffi-dev libssl-dev python3-dev
    

# RUN pip install PyYAML --only-binary :all:

RUN pip install --no-cache-dir \
    pandas \
    chardet \
    moviepy \
    transformers \
    torch \
    facenet-pytorch 
    
 


WORKDIR /opt/webapi/app

# COPY . /opt/dagster/app/session_video