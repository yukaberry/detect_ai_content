.DEFAULT_GOAL := default

reinstall_package:
	@pip uninstall -y detect_ai_content
	@pip install -e .

load_model:
	python3 -c 'from detect_ai_content.ml_logic.registry import load_model; load_model()'

run_pred:
	python3 -c 'from detect_ai_content.interface.main import pred; pred()'

run_preprocess:
	python3 -c 'from detect_ai_content.ml_logic.preprocess import preprocess_text; preprocess_text("my essay to classify")'

run_local_fast_api:
	fastapi dev detect_ai_content/api/fast.py

run_local_streamlit:
	streamlit run ./detect_ai_content/streamlit/app_v0.py

## run api locally
run_local_uvicorn:
	uvicorn detect_ai_content.api.fast:app --host 0.0.0.0

# TODO to decide if we use spacy or not
## run api locally with spacy
run_local_uvicorn_spacy:
	python -m spacy download en_core_web_sm
	uvicorn detect_ai_content.api.fast:app --host 0.0.0.0

## TESTS

run_tests:
	python3 -m unittest discover -s tests

run_cnn_tests:
	python tests/test_image_modeling_TrueNetImageUsinCustomCNN.py

run_one_test:
	python tests/test_texts_modeling_using_ml_features.py TestTextModeling.test_preprocessing

## RE TRAIN MODELS

run_retrain_TrueNetTextLogisticRegression:
	python3 -c 'from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextLogisticRegression import *; TrueNetTextLogisticRegression.retrain_full_model()'

run_retrain_TrueNetTextTfidfNaiveBayesClassifier:
	python3 -c 'from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextTfidfNaiveBayesClassifier import *; TrueNetTextTfidfNaiveBayesClassifier.retrain_full_model()'

run_retrain_TrueNetTextDecisionTreeClassifier:
	python3 -c 'from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextDecisionTreeClassifier import *; TrueNetTextDecisionTreeClassifier.retrain_full_model()'

run_retrain_TrueNetTextKNeighborsClassifier:
	python3 -c 'from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextKNeighborsClassifier import *; TrueNetTextKNeighborsClassifier.retrain_full_model()'

run_retrain_TrueNetTextSVC:
	python3 -c 'from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextSVC import *; TrueNetTextSVC.retrain_full_model()'

run_retrain_TrueNetTextUsingBERTMaskedPredictions:
	python3 -c 'from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextUsingBERTMaskedPredictions import TrueNetTextUsingBERTMaskedPredictions; TrueNetTextUsingBERTMaskedPredictions.retrain_full_model()'

run_retrain_TrueNetImageCNN_JM:
	python3 -c 'from detect_ai_content.ml_logic.for_images.TrueNetImageCNN_JM import TrueNetImageCNN_JM; TrueNetImageCNN_JM.retrain_full_model()'

run_retrain_TrueNetImageCNN_vgg16_JM:
	python3 -c 'from detect_ai_content.ml_logic.for_images.TrueNetImageCNN_JM import TrueNetImageCNN_vgg16_JM; TrueNetImageCNN_vgg16_JM.retrain_full_model()'

run_retrain_TrueNetTextRNN:
	python3 -c 'from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextRNN import TrueNetTextRNN; TrueNetTextRNN.retrain_full_model()'

run_retrain_all_text_models:
	make run_retrain_TrueNetTextLogisticRegression
	make run_retrain_TrueNetTextTfidfNaiveBayesClassifier
	make run_retrain_TrueNetTextDecisionTreeClassifier
	make run_retrain_TrueNetTextKNeighborsClassifier
	make run_retrain_TrueNetTextSVC
	make run_retrain_TrueNetTextUsingBERTMaskedPredictions
	make run_retrain_TrueNetTextRNN

run_train_production_pipelines:
	python3 -c 'from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextLogisticRegression import *; TrueNetTextLogisticRegression.retrain_production_pipeline()'
	python3 -c 'from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextDecisionTreeClassifier import *; TrueNetTextDecisionTreeClassifier.retrain_production_pipeline()'
	python3 -c 'from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextKNeighborsClassifier import *; TrueNetTextKNeighborsClassifier.retrain_production_pipeline()'
	python3 -c 'from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextRNN import *; TrueNetTextRNN.retrain_production_pipeline()'
	python3 -c 'from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextSVC import *; TrueNetTextSVC.retrain_production_pipeline()'
	python3 -c 'from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextTfidfNaiveBayesClassifier import *; TrueNetTextTfidfNaiveBayesClassifier.retrain_production_pipeline()'
	python3 -c 'from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextUsingBERTMaskedPredictions import *; TrueNetTextUsingBERTMaskedPredictions.retrain_production_pipeline()'
	python3 -c 'from detect_ai_content.ml_logic.for_images.TrueNetImageResNet50 import *; TrueNetImageResNet50.retrain_full_model()'


## TOOLS

run_generate_text_sample_datasets:
	python3 -c 'from detect_ai_content.ml_logic.data import generate_dataset_sample; generate_dataset_sample(1000)'
	python3 -c 'from detect_ai_content.ml_logic.data import generate_dataset_sample; generate_dataset_sample(10000)'
	python3 -c 'from detect_ai_content.ml_logic.data import generate_dataset_sample; generate_dataset_sample(50000)'

### DOCKER and gcloud
### run them in order from the top then you can deploy!

# build image
run_docker_build:
	docker build --tag=${IMAGE}:dev . -f Dockerfile

# run the image
run_docker_run:
	docker run -it -e PORT=8000 -p 8000:8000 ${IMAGE}:dev

# optional
# run the image more flexible, loading from your .env file
run_docker_run_env:
	docker run -it -e PORT=8000 -p 8000:8000 --env-file .env ${IMAGE}:dev

# optional
# if you make a new repo on artifacts
run_create_new_repo:
	gcloud artifacts repositories create ${ARTIFACTSREPO} --repository-format=docker --location=${GCP_REGION}

run_docker_build_production:
	docker build --platform linux/amd64 -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod .

run_docker_deploy_production:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod
	gcloud run deploy \
  --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
  --memory ${MEMORY} \
  --region ${GCP_REGION}


#### prefect flow

# test for specific script, TrueNetImageUsinCustomCNN.py
run_prefect_workflow:
	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m detect_ai_content.ml_logic.for_images.TrueNetImageUsinCustomCNN

#run_prefect_workflow:
#	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m detect_ai_content.interface.ml_flow_images
