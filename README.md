

## Datasets
- **source**: is the original source of the data HuggingFace, Kaggle, etc. It's interesting to know that to have a look to the possible articles & notebooks
- **public url**: is a public url i created (using GCP) to simplify our tests on Google Colab for example

### Texts
- [HuggingFace - human_ai_generated_text](https://huggingface.co/datasets/dmitva/human_ai_generated_text), [dataset - public url](https://storage.googleapis.com/detect-human-ai-generated-raw-data/hugging_face_human_ai_generated_text/model_training_dataset.csv.zip)
- [Kaggle - ai-generated-vs-human-text-95-accuracy](https://www.kaggle.com/code/syedali110/ai-generated-vs-human-text-95-accuracy/notebook), [dataset - public url](https://storage.googleapis.com/detect-human-ai-generated-raw-data/kaggle-ai-generated-vs-human-text/AI_Human.csv.zip)
- [Kaggle - daigt-v2-train-dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset), [dataset - public url](https://storage.googleapis.com/detect-human-ai-generated-raw-data/kaggle-daigt-v2-train-dataset/train_v2_drcat_02.csv.zip)

## Documentations

- [How this repo is organized?](./documentations/git_repo_structure.md)
- [Git branch Management](./documentations/git_branches.md)
- [MLFlow & models](./documentations/mlflow.md)

### How to use our work ?

### Use the package

- Make a local copy
```
git@github.com:yukaberry/detect_ai_content.git
```

- Install the package & dependencies
```
pip install -e .
```

- Load a model and use it for predictions ?
```Python
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextLogisticRegression import TrueNetTextLogisticRegression
from detect_ai_content.ml_logic.data import load_dataset

# Load the Pipeline ! Everything is already package
# Features Engineering + Scaler + model architecture + tuned hyperparameters
pipeline = TrueNetTextLogisticRegression().local_trained_pipeline()

# Pred using your dataframe or use on local datasets
df = load_dataset(source="llama2_chat")
preds = pipeline.predict(text_df)
```

### Use the API

Call your endpoint

```Python
import requests
headers = {
   'accept': 'application/json',
}
params = {
  "text":text
}
response = requests.get('https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/text_single_predict', headers=headers, params=params)
```

### Use the Website

(Visit our website)[https://lewagon1705.streamlit.app/text_genAi_detection]

## Modeling

All our models are based on pipelines.
The pipeline compress all the steps to do a prediction (enrich data - preprocess - scaler - and finally a prediction)

![alt text](https://github.com/yukaberry/detect_ai_content/blob/master/images/pipeline_example.png)

### 1st batch of features
Those model are build around the same features.
- repetitions_ratio
- punctuations_ratio
- text_corrections_ratio
- average_sentence_lenght
- average_neg_sentiment_polarity
- lexical_diversity
- smog_index
- flesch_reading_ease
- avg_word_length

- **TrueNetTextLogisticRegression**
```Python
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextLogisticRegression import TrueNetTextLogisticRegression
```

- **TrueNetTextDecisionTreeClassifier**
```Python
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextDecisionTreeClassifier import TrueNetTextDecisionTreeClassifier
```

- **TrueNetTextSVC**
```Python
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextSVC import TrueNetTextSVC
```

- **TrueNetTextRNN**
```Python
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextRNN import TrueNetTextRNN
```

- **TrueNetTextKNeighborsClassifier**
```Python
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextKNeighborsClassifier import TrueNetTextKNeighborsClassifier
```

## No feature, just the text itself
In this model we use a [TF-IDF](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) and a [Naive Bayes method](https://scikit-learn.org/dev/modules/naive_bayes.html) to analyze the text features.

- **TrueNetTextTfidfNaiveBayesClassifier**
```Python
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextTfidfNaiveBayesClassifier import TrueNetTextTfidfNaiveBayesClassifier
```

## One feature
The funny idea of this model is to masked some words of the text and use a [BERT Masked model](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/bert#transformers.BertForMaskedLM) to predict the removed words. If the transformer is able to find the correct missing word, it's a successfull prediction. In our study we discover that we are making less correct predictions for Human texts.

![alt text](https://github.com/yukaberry/detect_ai_content/blob/master/images/f1st_batch_features_shap.png)

- **TrueNetTextUsingBERTMaskedPredictions**
```Python
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextUsingBERTMaskedPredictions import TrueNetTextUsingBERTMaskedPredictions
```

### 2nd batch of features
Those model are build around the same features.
- stopwords_ratio
- punctuation_ratio
- repetition_ratio
- dependency_ratio
- spelling_errors_ratio
- pos__ratio
- avg_word_length
- lexical_diversity
- flesch_reading_ease
- smog_index
- flesch_kincaid_grade
- sentiment

- **lgbm_internal**
```Python
from detect_ai_content.ml_logic.for_texts.lgbm_internal import LgbmInternal
```

## Model benchmark !

Prediction benchmark has been computed using 50 texts (103,010 letters)

![alt text](https://github.com/yukaberry/detect_ai_content/blob/master/images/predictors_by_accuracy.png)
