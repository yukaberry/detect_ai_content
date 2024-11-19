# Current version as of 19-11-2024

from fastapi import FastAPI, File, UploadFile
from detect_ai_content.ml_logic.image_classifier_cnn import image_classifier_cnn
from io import BytesIO
import os

app = FastAPI()
classifier = image_classifier_cnn()

@app.get("/")
def root():
    return {"greeting": "hello"}

####### DO NOT DELETE THE BLOCK BELOW ########
"""@app.get("/predict")
async def cnn_prediction():
    file_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
    prediction_result = classifier.predict(file_path)
    return {"prediction": prediction_result}"""
######## DO NOT DELETE ########

@app.post("/predict")
async def cnn_prediction(file: UploadFile = File(...)):

    image_bytes = await file.read()

    file_path = os.path.join(os.path.dirname(__file__), "temp_image.jpg")
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    prediction_result = classifier.predict(file_path)
    return {"prediction": prediction_result}

# Service URL for Image: https://detect-ai-content-image-api-334152645738.europe-west1.run.app
# To test locally: uvicorn detect_ai_content.api.aban371818_api.image_classifier_api:app --reload
