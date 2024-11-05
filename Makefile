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


## TOOLS

run_generate_text_sample_datasets:
	python3 -c 'from detect_ai_content.ml_logic.data import generate_dataset_sample; generate_dataset_sample(1000)'
	python3 -c 'from detect_ai_content.ml_logic.data import generate_dataset_sample; generate_dataset_sample(10000)'
	python3 -c 'from detect_ai_content.ml_logic.data import generate_dataset_sample; generate_dataset_sample(50000)'

## DOCKER

run_docker_build:
	docker build --tag=${IMAGE}:dev . -f Dockerfile

run_docker_run:
	docker run -it -e PORT=8000 -p 8000:8000 ${IMAGE}:dev

run_docker_build_production:
	docker build --platform linux/amd64 -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod .

run_docker_deploy_production:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod
	gcloud run deploy \
  --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
  --memory ${MEMORY} \
  --region ${GCP_REGION}
