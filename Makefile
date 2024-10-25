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
