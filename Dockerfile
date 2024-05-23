FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
      
WORKDIR /workfile

COPY requirements.txt .

RUN pip install -r requirements.txt
