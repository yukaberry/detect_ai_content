

## Datasets
- **source**: is the original source of the data HuggingFace, Kaggle, etc. It's interesting to know that to have a look to the possible articles & notebooks
- **public url**: is a public url i created (using GCP) to simplify our tests on Google Colab for example

### Texts
- [source](https://huggingface.co/datasets/dmitva/human_ai_generated_text), [public url](https://storage.googleapis.com/human-ai-detection-raw-data/hugging_face_human_ai_generated_text/model_training_dataset.csv.zip)
- [source](https://www.kaggle.com/code/syedali110/ai-generated-vs-human-text-95-accuracy/notebook), [public url](https://storage.googleapis.com/human-ai-detection-raw-data/kaggle-ai-generated-vs-human-text/AI_Human.csv.zip)


## Baseline - Ben "MVP" for short App Architecture
- [x] : Train a Baseline model
- [x] : Loading a trained model (locally saved 😜)
- [x] : Expose some shortcuts using Makefile
- [?] : API not needed ... the model is store on the project
- [ ] : Expose minimalist interface (Streamlit)

### Run API

```
make run_local_fast_api
```

### Run API

```
make run_local_fast_api
```
