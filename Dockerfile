# base image (our Virtual Environment is python 3.10.6)
FROM python:3.10-slim
# Working directory
WORKDIR /production
# install packages
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt
# larger version
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt

# RUN python -m spacy download en_core_web_sm
# test comment
# copy folders
COPY detect_ai_content detect_ai_content

# EXPOSE 8080
# deployment : uncomment below and use this for deployment
#CMD uvicorn detect_ai_content.api.fast:app --host 0.0.0.0 --port 8080

# comment out the below prior to creating a pull request

CMD uvicorn detect_ai_content.api.fast:app --host 0.0.0.0 --port $PORT

# local : use this before deployment
#CMD uvicorn detect_ai_content.api.fast:app --host 0.0.0.0

# api url 18 nov 2024
#URL=https://detect-ai-content-improved18nov-667980218208.europe-west1.run.app



# testing image api
#CMD uvicorn detect_ai_content.api.aban371818_api.image_classifier_api:app --host 0.0.0.0 --port $PORT
