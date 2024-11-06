# base image (our Virtual Environment is python 3.10.6)
FROM python:3.10-slim
# Working directory
WORKDIR /production
# install packages
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# RUN python -m spacy download en_core_web_sm
# test comment
# copy folders
COPY detect_ai_content detect_ai_content

# EXPOSE 8080
# deployment : uncomment below and use this for deployment
#CMD uvicorn detect_ai_content.api.fast:app --host 0.0.0.0 --port 8080
CMD uvicorn detect_ai_content.api.fast:app --host 0.0.0.0 --port $PORT
# local : use this before deployment
#CMD uvicorn detect_ai_content.api.fast:app --host 0.0.0.0
